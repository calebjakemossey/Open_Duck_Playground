import jax
import jax.numpy as jp

def reward_imitation(
    base_qpos: jax.Array,
    base_qvel: jax.Array,
    joints_qpos: jax.Array,
    joints_qvel: jax.Array,
    contacts: jax.Array,
    reference_frame: jax.Array,
    cmd: jax.Array,
    use_imitation_reward: bool = False,
) -> jax.Array:
    if not use_imitation_reward:
        return jp.nan_to_num(0.0)

    # TODO don't reward for moving when the command is zero.
    cmd_norm = jp.linalg.norm(cmd[:3])

    w_torso_pos = 1.0
    w_torso_orientation = 1.0
    w_lin_vel_xy = 1.0
    w_lin_vel_z = 1.0
    w_ang_vel_xy = 0.5
    w_ang_vel_z = 0.5
    w_joint_pos = 15.0
    w_joint_vel = 1.0e-3
    w_contact = 1.0

    # Polynomial reference layout (14 joints, antennas merged - new symmetric URDF):
    # [0:14]   joint_pos    (5 left leg + 4 head + 5 right leg)
    # [14:28]  joint_vel
    # [28:30]  foot_contacts (left, right)
    # [30:33]  base linear_vel (xyz)
    # [33:36]  base angular_vel (xyz)
    # NB: root_quat is NOT in the polynomial output - torso_orientation_rew is unused.
    joint_pos_slice_start = 0
    joint_pos_slice_end = 14

    joint_vels_slice_start = 14
    joint_vels_slice_end = 28

    foot_contacts_slice_start = 28
    foot_contacts_slice_end = 30

    linear_vel_slice_start = 30
    linear_vel_slice_end = 33

    angular_vel_slice_start = 33
    angular_vel_slice_end = 36

    # Quaternion slice is dead - not in polynomial. Kept for code below; produces zeros.
    root_quat_slice_start = 0
    root_quat_slice_end = 0

    # ref_base_pos = reference_frame[root_pos_slice_start:root_pos_slice_end]
    # base_pos = qpos[:3]

    # Quaternion not present in current polynomial layout; use identity quat as no-op
    # (torso_orientation_rew is unused in the final reward sum anyway).
    ref_base_orientation_quat = jp.array([1.0, 0.0, 0.0, 0.0])
    base_orientation = base_qpos[3:7]
    base_orientation = base_orientation / jp.linalg.norm(base_orientation)

    ref_base_lin_vel = reference_frame[linear_vel_slice_start:linear_vel_slice_end]
    base_lin_vel = base_qvel[:3]

    ref_base_ang_vel = reference_frame[angular_vel_slice_start:angular_vel_slice_end]
    base_ang_vel = base_qvel[3:6]

    ref_joint_pos = reference_frame[joint_pos_slice_start:joint_pos_slice_end]
    # Joint layout: [0-4] left leg, [5-8] head, [9-13] right leg.
    # Drop head joints for the imitation comparison (policy + reference both 14 dims).
    ref_joint_pos = jp.concatenate([ref_joint_pos[:5], ref_joint_pos[9:]])
    joint_pos = jp.concatenate([joints_qpos[:5], joints_qpos[9:]])

    ref_joint_vels = reference_frame[joint_vels_slice_start:joint_vels_slice_end]
    ref_joint_vels = jp.concatenate([ref_joint_vels[:5], ref_joint_vels[9:]])
    joint_vel = jp.concatenate([joints_qvel[:5], joints_qvel[9:]])

    # ref_left_toe_pos = reference_frame[left_toe_pos_slice_start:left_toe_pos_slice_end]
    # ref_right_toe_pos = reference_frame[right_toe_pos_slice_start:right_toe_pos_slice_end]

    ref_foot_contacts = reference_frame[
        foot_contacts_slice_start:foot_contacts_slice_end
    ]

    # reward
    # torso_pos_rew = jp.exp(-200.0 * jp.sum(jp.square(base_pos[:2] - ref_base_pos[:2]))) * w_torso_pos

    # real quaternion angle doesn't have the expected  effect, switching back for now
    # torso_orientation_rew = jp.exp(-20 * self.quaternion_angle(base_orientation, ref_base_orientation_quat)) * w_torso_orientation

    # TODO ignore yaw here, we just want xy orientation
    torso_orientation_rew = (
        jp.exp(-20.0 * jp.sum(jp.square(base_orientation - ref_base_orientation_quat)))
        * w_torso_orientation
    )

    lin_vel_xy_rew = (
        jp.exp(-8.0 * jp.sum(jp.square(base_lin_vel[:2] - ref_base_lin_vel[:2])))
        * w_lin_vel_xy
    )
    lin_vel_z_rew = (
        jp.exp(-8.0 * jp.sum(jp.square(base_lin_vel[2] - ref_base_lin_vel[2])))
        * w_lin_vel_z
    )

    ang_vel_xy_rew = (
        jp.exp(-2.0 * jp.sum(jp.square(base_ang_vel[:2] - ref_base_ang_vel[:2])))
        * w_ang_vel_xy
    )
    ang_vel_z_rew = (
        jp.exp(-2.0 * jp.sum(jp.square(base_ang_vel[2] - ref_base_ang_vel[2])))
        * w_ang_vel_z
    )

    joint_pos_rew = -jp.sum(jp.square(joint_pos - ref_joint_pos)) * w_joint_pos
    joint_vel_rew = -jp.sum(jp.square(joint_vel - ref_joint_vels)) * w_joint_vel

    ref_foot_contacts = jp.where(
        ref_foot_contacts > 0.5,
        jp.ones_like(ref_foot_contacts),
        jp.zeros_like(ref_foot_contacts),
    )
    contact_rew = jp.sum(contacts == ref_foot_contacts) * w_contact

    reward = (
        lin_vel_xy_rew
        + lin_vel_z_rew
        + ang_vel_xy_rew
        + ang_vel_z_rew
        + joint_pos_rew
        + joint_vel_rew
        + contact_rew
        # + torso_orientation_rew
    )

    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return jp.nan_to_num(reward)