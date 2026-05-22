import mujoco
import mujoco.viewer
import glfw
import pickle
import numpy as np
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = True


class MjInfer(MJInferBase):
    def __init__(
        self, model_path: str, reference_data: str, onnx_model_path: str, standing: bool
    ):
        super().__init__(model_path)

        self.standing = standing
        self.head_control_mode = self.standing

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)
        self.obs_size = self.policy.ort_session.get_inputs()[0].shape[1]

        self.COMMANDS_RANGE_X = [-0.15, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        snap_size = 6 + 2 * self.num_dofs
        if self.obs_size == 101:
            self.history_len = 0
            self.has_linvel_gravity = False
            self.has_phase = True
        elif self.obs_size == 209:
            self.has_linvel_gravity = True
            self.has_phase = True
            base_obs = 107
            self.history_len = max(0, (self.obs_size - base_obs) // snap_size)
        else:
            self.has_linvel_gravity = True
            self.has_phase = False
            base_obs = 105
            self.history_len = max(0, (self.obs_size - base_obs) // snap_size)
        self.proprio_history = np.zeros(self.history_len * snap_size)

        print(f"Policy obs_size: {self.obs_size}, history_len: {self.history_len}, "
              f"linvel/gravity: {self.has_linvel_gravity}, phase: {self.has_phase}")

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        self.held_keys = set()
        self.push_magnitude = 2.0
        self.pending_push_dir = None

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        if self.history_len > 0:
            proprio_snapshot = np.concatenate([
                gyro,
                accelerometer,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
            ])
            snap_len = len(proprio_snapshot)
            self.proprio_history = np.roll(self.proprio_history, snap_len)
            self.proprio_history[:snap_len] = proprio_snapshot

        shared = [
            command,
            joint_angles - self.default_actuator,
            joint_vel * self.dof_vel_scale,
            self.last_action,
            self.last_last_action,
            self.last_last_last_action,
            self.motor_targets,
            contacts,
        ]

        if self.has_linvel_gravity:
            linvel_addr = self.model.sensor_adr[self.linvel_id]
            linvel = data.sensordata[linvel_addr:linvel_addr + 3]
            gravity = data.site_xmat[self.imu_site_id].reshape(3, 3).T @ np.array([0, 0, -1])
            parts = [linvel, gyro, accelerometer, gravity] + shared
        else:
            parts = [gyro, accelerometer] + shared

        if self.has_phase:
            parts.append(self.imitation_phase)

        if self.history_len > 0:
            parts.append(self.proprio_history)

        obs = np.concatenate(parts)
        return obs

    OPPOSITES = {265: 264, 264: 265, 263: 262, 262: 263, 81: 69, 69: 81}

    def key_callback(self, keycode):
        if keycode == glfw.KEY_H:
            self.head_control_mode = not self.head_control_mode
            return
        if keycode == glfw.KEY_P:
            self.phase_frequency_factor += 0.1
            return
        if keycode == glfw.KEY_M:
            self.phase_frequency_factor -= 0.1
            return
        if keycode == glfw.KEY_SPACE:
            self.held_keys.clear()
            self.commands[:] = [0.0] * 7
            return
        if keycode == glfw.KEY_LEFT_BRACKET:
            self.push_magnitude = max(0.1, self.push_magnitude - 0.1)
            return
        if keycode == glfw.KEY_RIGHT_BRACKET:
            self.push_magnitude = min(3.0, self.push_magnitude + 0.1)
            return

        ctrl_held = glfw.get_key(glfw.get_current_context(), glfw.KEY_LEFT_CONTROL) == glfw.PRESS or \
                    glfw.get_key(glfw.get_current_context(), glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
        if ctrl_held and keycode in (glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT):
            direction_map = {
                glfw.KEY_UP: 0.0,
                glfw.KEY_DOWN: np.pi,
                glfw.KEY_LEFT: np.pi / 2,
                glfw.KEY_RIGHT: -np.pi / 2,
            }
            self.pending_push_dir = direction_map[keycode]
            return

        if keycode in self.held_keys:
            self.held_keys.discard(keycode)
        else:
            self.held_keys.discard(self.OPPOSITES.get(keycode, -1))
            self.held_keys.add(keycode)

        self._update_commands()

    def _update_commands(self):
        if not self.head_control_mode:
            lin_vel_x = 0.0
            lin_vel_y = 0.0
            ang_vel = 0.0
            if 265 in self.held_keys:  # arrow up - forward
                lin_vel_x += self.COMMANDS_RANGE_X[1]
            if 264 in self.held_keys:  # arrow down - backward
                lin_vel_x += self.COMMANDS_RANGE_X[0]
            if 263 in self.held_keys:  # arrow left - turn left
                ang_vel += self.COMMANDS_RANGE_THETA[1]
            if 262 in self.held_keys:  # arrow right - turn right
                ang_vel += self.COMMANDS_RANGE_THETA[0]
            if 81 in self.held_keys:  # q - strafe left
                lin_vel_y += self.COMMANDS_RANGE_Y[1]
            if 69 in self.held_keys:  # e - strafe right
                lin_vel_y += self.COMMANDS_RANGE_Y[0]
            self.commands[0] = lin_vel_x
            self.commands[1] = lin_vel_y
            self.commands[2] = ang_vel
        else:
            neck_pitch = 0.0
            head_pitch = 0.0
            head_yaw = 0.0
            head_roll = 0.0
            if 265 in self.held_keys:
                head_pitch = self.NECK_PITCH_RANGE[1]
            if 264 in self.held_keys:
                head_pitch = self.NECK_PITCH_RANGE[0]
            if 263 in self.held_keys:
                head_yaw = self.HEAD_YAW_RANGE[1]
            if 262 in self.held_keys:
                head_yaw = self.HEAD_YAW_RANGE[0]
            if 81 in self.held_keys:
                head_roll = self.HEAD_ROLL_RANGE[1]
            if 69 in self.held_keys:
                head_roll = self.HEAD_ROLL_RANGE[0]
            self.commands[3] = neck_pitch
            self.commands[4] = head_pitch
            self.commands[5] = head_yaw
            self.commands[6] = head_roll

    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while viewer.is_running():

                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    # Clamp base velocity to max 3 m/s
                    base_vel = self.data.qvel[0:3]
                    speed = np.linalg.norm(base_vel)
                    if speed > 3.0:
                        self.data.qvel[0:3] = base_vel * (3.0 / speed)

                    counter += 1

                    if counter % self.decimation == 0:
                        if not self.standing:
                            self.imitation_i += 1.0 * self.phase_frequency_factor
                            self.imitation_i = (
                                self.imitation_i % self.PRM.nb_steps_in_period
                            )
                            self.imitation_phase = np.array(
                                [
                                    np.cos(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                    np.sin(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                ]
                            )
                        obs = self.get_obs(
                            self.data,
                            self.commands,
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        # self.action_filter.push(action)
                        # action = self.action_filter.get_filtered_action()

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        self.motor_targets = (
                            self.default_actuator + action * self.action_scale
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                                self.prev_motor_targets
                                + self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                            )

                            self.prev_motor_targets = self.motor_targets.copy()

                        # head_targets = self.commands[3:]
                        # self.motor_targets[5:9] = head_targets
                        self.data.ctrl = self.motor_targets.copy()

                    if self.pending_push_dir is not None:
                        trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_assembly")
                        mass = np.sum([self.model.body_mass[i] for i in range(self.model.nbody)])
                        fx = mass * self.push_magnitude * np.cos(self.pending_push_dir)
                        fy = mass * self.push_magnitude * np.sin(self.pending_push_dir)
                        self.data.xfrc_applied[trunk_id, :3] = [fx, fy, 0.0]
                        self._push_steps_remaining = int(0.1 / self.model.opt.timestep)
                        self.pending_push_dir = None

                    if hasattr(self, '_push_steps_remaining') and self._push_steps_remaining > 0:
                        self._push_steps_remaining -= 1
                        if self._push_steps_remaining == 0:
                            self.data.xfrc_applied[:] = 0.0

                    vel = np.linalg.norm(self.data.qvel[0:3])
                    xfrc = self.data.xfrc_applied
                    force_mag = np.sum(np.linalg.norm(xfrc[:, :3], axis=1))
                    mode = "Head" if self.head_control_mode else "Walk"
                    status = f"Mode: {mode}  Push: {self.push_magnitude:.1f} m/s  Vel: {vel:.2f} m/s"
                    if force_mag > 0.01:
                        status += f"  FORCE: {force_mag:.1f} N"
                    viewer.set_texts([
                        (mujoco.mjtGridPos.mjGRID_TOPLEFT, None, "Status", status),
                    ])

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path, args.reference_data, args.onnx_model_path, args.standing
    )
    mjinfer.run()
