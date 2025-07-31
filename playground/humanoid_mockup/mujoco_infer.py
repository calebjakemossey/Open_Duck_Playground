import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.humanoid_mockup.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = False


class MjInfer(MJInferBase):
    def __init__(self, model_path: str, reference_data: str, onnx_model_path: str):
        super().__init__(model_path)

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 1.0
        self.action_scale = 1.0

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        self.PRM = PolyReferenceMotion(reference_data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.COMMANDS_RANGE_X = [-0.3, 0.3]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0]

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []
        self.torques = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        self.left_knee_id = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee") + 6
        )
        self.right_knee_id = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee")
            + 6
        )

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

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

        # AFTER
        gravity = np.array(data.site_xmat[self.get_site_id_from_name("imu")]).reshape(
            (3, 3)
        ).T @ np.array([0, 0, -1])

        obs = np.concatenate(
            [
                gyro,
                gravity,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                self.imitation_phase,
            ]
        )

        return obs

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        if keycode == 265:  # arrow up
            lin_vel_x = self.COMMANDS_RANGE_X[1]
        if keycode == 264:  # arrow down
            lin_vel_x = self.COMMANDS_RANGE_X[0]
        if keycode == 263:  # arrow left
            lin_vel_y = self.COMMANDS_RANGE_Y[1]
        if keycode == 262:  # arrow right
            lin_vel_y = self.COMMANDS_RANGE_Y[0]
        if keycode == 81:  # a
            ang_vel = self.COMMANDS_RANGE_THETA[1]
        if keycode == 69:  # e
            ang_vel = self.COMMANDS_RANGE_THETA[0]
        if keycode == 80:  # p
            self.data.qvel[0] = 1.0
            # self.phase_frequency_factor += 0.1
        if keycode == 59:  # m
            self.data.qvel[0] = -1.0
            # self.phase_frequency_factor -= 0.1

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel

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
                while True:
                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    counter += 1

                    if counter % self.decimation == 0:
                        left_knee_torque = self.data.qfrc_actuator[self.left_knee_id]
                        right_knee_torque = self.data.qfrc_actuator[self.right_knee_id]
                        print(
                            f"Left knee torque: {np.around(left_knee_torque, 2)}, Right knee torque: {np.around(right_knee_torque, 2)}"
                        )

                        self.torques.append(
                            np.array(
                                [
                                    left_knee_torque,
                                    right_knee_torque,
                                ]
                            )
                        )

                        self.imitation_i += 1.0 * self.phase_frequency_factor
                        self.imitation_i = (
                            self.imitation_i % self.PRM.nb_steps_in_period
                        )
                        # print(self.PRM.nb_steps_in_period)
                        # exit()
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
                        # self.data.ctrl = self.PRM.get_reference_motion(0, 0, 0, self.imitation_i)[:10]

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))
            pickle.dump(self.torques, open("mujoco_torques.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/humanoid_mockup/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/humanoid_mockup/xmls/scene_flat_terrain.xml",
    )

    args = parser.parse_args()

    mjinfer = MjInfer(args.model_path, args.reference_data, args.onnx_model_path)
    mjinfer.run()
