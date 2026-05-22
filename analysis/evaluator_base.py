"""Shared base class for all headless policy evaluation.

Single source of truth for observation construction, policy stepping,
reset, and fall detection. HeadlessEvaluator and _QuickAssessor both
inherit from this to prevent divergence when the obs space changes.
"""
import numpy as np
import mujoco
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase


class PolicyEvaluatorBase(MJInferBase):
    """Base for headless evaluation - obs construction, stepping, fall detection."""

    DOF_VEL_SCALE = 0.05
    ACTION_SCALE = 0.25
    MAX_MOTOR_VELOCITY = 5.24
    ACCEL_BIAS = 1.3
    UPDATE_IMITATION_PHASE = True
    HISTORY_LEN = 0
    HAS_LINVEL_GRAVITY = True
    HAS_PHASE = False

    def __init__(self, model_path, reference_data):
        super().__init__(model_path)
        self.PRM = PolyReferenceMotion(reference_data)
        self.upvec_addr = self.model.sensor_adr[self.gravity_id]

        self.left_foot_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"
        )
        self.right_foot_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"
        )

        self._reset_action_state()

    def _reset_action_state(self):
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        self.imitation_i = 0
        self.imitation_phase = np.array([0.0, 0.0])
        snap_size = 6 + 2 * self.num_dofs
        self.proprio_history = np.zeros(self.HISTORY_LEN * snap_size)

    def reset(self):
        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.qvel[:] = 0
        self.data.ctrl[:] = self.default_actuator
        mujoco.mj_forward(self.model, self.data)
        self._reset_action_state()

    def build_obs(self, data, command):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += self.ACCEL_BIAS

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        contacts = self.get_feet_contacts(data)

        if self.HISTORY_LEN > 0:
            proprio_snapshot = np.concatenate([
                gyro, accelerometer,
                joint_angles - self.default_actuator,
                joint_vel * self.DOF_VEL_SCALE,
            ])
            snap_len = len(proprio_snapshot)
            self.proprio_history = np.roll(self.proprio_history, snap_len)
            self.proprio_history[:snap_len] = proprio_snapshot

        shared = [
            command,
            joint_angles - self.default_actuator,
            joint_vel * self.DOF_VEL_SCALE,
            self.last_action,
            self.last_last_action,
            self.last_last_last_action,
            self.motor_targets,
            contacts,
        ]

        if self.HAS_LINVEL_GRAVITY:
            linvel = self.get_base_lin_vel_local()
            gravity = self.data.site_xmat[self.imu_site_id].reshape(3, 3).T @ np.array([0, 0, -1])
            parts = [linvel, gyro, accelerometer, gravity] + shared
        else:
            parts = [gyro, accelerometer] + shared

        if self.HAS_PHASE:
            parts.append(self.imitation_phase)

        if self.HISTORY_LEN > 0:
            parts.append(self.proprio_history)

        return np.concatenate(parts)

    def step_policy(self, policy, command):
        obs = self.build_obs(self.data, command)
        action = policy.infer(obs)

        self.last_last_last_action = self.last_last_action.copy()
        self.last_last_action = self.last_action.copy()
        self.last_action = action.copy()

        self.motor_targets = self.default_actuator + action * self.ACTION_SCALE
        self.motor_targets = np.clip(
            self.motor_targets,
            self.prev_motor_targets - self.MAX_MOTOR_VELOCITY * (self.sim_dt * self.decimation),
            self.prev_motor_targets + self.MAX_MOTOR_VELOCITY * (self.sim_dt * self.decimation),
        )
        self.prev_motor_targets = self.motor_targets.copy()
        self.data.ctrl[:] = self.motor_targets

        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        if self.UPDATE_IMITATION_PHASE:
            self.imitation_i = (self.imitation_i + 1) % self.PRM.nb_steps_in_period
            phase = self.imitation_i / self.PRM.nb_steps_in_period
            self.imitation_phase = np.array([np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)])

        return action

    def get_upvec(self):
        return self.data.sensordata[self.upvec_addr:self.upvec_addr + 3]

    def is_fallen(self):
        return self.data.qpos[2] < 0.08 or self.get_upvec()[2] < 0.5

    def get_base_lin_vel_local(self):
        world_vel = self.data.qvel[0:3].copy()
        R = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
        return R.T @ world_vel

    def get_base_ang_vel(self):
        return self.data.qvel[3:6].copy()

    def get_foot_heights(self):
        return (
            float(self.data.site_xpos[self.left_foot_site][2]),
            float(self.data.site_xpos[self.right_foot_site][2]),
        )

    def apply_push(self, magnitude_ns, direction_rad):
        push_dir = np.array([np.cos(direction_rad), np.sin(direction_rad)])
        self.data.qvel[0:2] += push_dir * magnitude_ns
        mujoco.mj_forward(self.model, self.data)

    def apply_body_push(self, body_name, magnitude, direction_rad):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        push_dir = np.array([np.cos(direction_rad), np.sin(direction_rad), 0.0])
        total_mass = float(self.model.body_subtreemass[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_assembly")
        ])
        force = push_dir * magnitude * total_mass / (self.sim_dt * self.decimation)
        self.data.xfrc_applied[body_id, 0:3] = force
        mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[body_id, 0:3] = 0.0

    def apply_angular_push(self, magnitude, direction_rad):
        push_dir = np.array([np.cos(direction_rad), np.sin(direction_rad), 0.0])
        self.data.qvel[3:6] += push_dir * magnitude
        mujoco.mj_forward(self.model, self.data)
