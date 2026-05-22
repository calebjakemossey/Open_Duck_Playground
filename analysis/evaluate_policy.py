"""
Headless policy evaluation for Open Duck Mini.

Runs a trained ONNX policy through standardised tests WITHOUT opening a viewer window.
Computes smoothness, stability, tracking, gait, and push-recovery metrics. Outputs JSON.

Usage:
    uv run analysis/evaluate_policy.py -o checkpoints/baseline/BEST_WALK_ONNX.onnx
    uv run analysis/evaluate_policy.py -o policy.onnx --output results.json --num_episodes 20
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from playground.common.onnx_infer import OnnxInfer
from analysis.evaluator_base import PolicyEvaluatorBase


class HeadlessEvaluator(PolicyEvaluatorBase):
    def __init__(self, model_path, reference_data, onnx_model_path):
        super().__init__(model_path, reference_data)
        self.policy = OnnxInfer(onnx_model_path, awd=True)
        obs_size = self.policy.ort_session.get_inputs()[0].shape[1]
        snap_size = 6 + 2 * self.num_dofs
        if obs_size == 101:
            self.HAS_LINVEL_GRAVITY = False
            self.HAS_PHASE = True
            self.HISTORY_LEN = 0
        elif obs_size == 209:
            self.HAS_LINVEL_GRAVITY = True
            self.HAS_PHASE = True
            base_obs = 107
            self.HISTORY_LEN = max(0, (obs_size - base_obs) // snap_size)
        else:
            self.HAS_LINVEL_GRAVITY = True
            self.HAS_PHASE = False
            base_obs = 105
            self.HISTORY_LEN = max(0, (obs_size - base_obs) // snap_size)
        self._reset_action_state()

    def run_episode(self, command, max_steps=1000, pushes=None):
        """Run one episode and collect rich telemetry.

        Args:
            command: 7-vector command [vx, vy, wz, ...head...]
            max_steps: max steps
            pushes: list of (step, magnitude_ns, direction_rad) tuples to apply
        """
        self.reset()
        push_schedule = {p[0]: (p[1], p[2]) for p in (pushes or [])}

        actions = []
        positions = []          # base xy position
        base_heights = []       # base z
        upvecs = []             # torso up-vector (for orientation)
        lin_vel_local = []      # body-frame linear velocity
        ang_vel = []            # angular velocity
        left_heights = []
        right_heights = []
        left_contacts = []
        right_contacts = []
        actuator_forces = []    # for CoT
        joint_velocities = []   # for CoT
        left_foot_xy = []       # for foot slip
        right_foot_xy = []      # for foot slip

        for step in range(max_steps):
            if step in push_schedule:
                mag, direction = push_schedule[step]
                self.apply_push(mag, direction)

            action = self.step_policy(self.policy, command)

            actions.append(action.copy())
            positions.append(self.data.qpos[0:2].copy())
            base_heights.append(float(self.data.qpos[2]))
            upvecs.append(self.get_upvec().copy())
            lin_vel_local.append(self.get_base_lin_vel_local())
            ang_vel.append(self.get_base_ang_vel())

            lh, rh = self.get_foot_heights()
            left_heights.append(lh)
            right_heights.append(rh)
            lc, rc = self.get_feet_contacts(self.data)
            left_contacts.append(bool(lc))
            right_contacts.append(bool(rc))
            actuator_forces.append(self.data.actuator_force.copy())
            joint_velocities.append(self.get_actuator_joints_qvel(self.data.qvel))
            left_foot_xy.append(self.data.site_xpos[self.left_foot_site][:2].copy())
            right_foot_xy.append(self.data.site_xpos[self.right_foot_site][:2].copy())

            if self.is_fallen():
                break

        return {
            "actions": np.array(actions),
            "positions": np.array(positions),
            "base_heights": np.array(base_heights),
            "upvecs": np.array(upvecs),
            "lin_vel_local": np.array(lin_vel_local),
            "ang_vel": np.array(ang_vel),
            "left_heights": np.array(left_heights),
            "right_heights": np.array(right_heights),
            "left_contacts": np.array(left_contacts),
            "right_contacts": np.array(right_contacts),
            "actuator_forces": np.array(actuator_forces),
            "joint_velocities": np.array(joint_velocities),
            "left_foot_xy": np.array(left_foot_xy),
            "right_foot_xy": np.array(right_foot_xy),
            "steps": len(actions),
        }

    def compute_metrics(self, ep, command):
        """Compute all metrics from a single episode."""
        m = {"steps": ep["steps"]}
        actions = ep["actions"]
        n = ep["steps"]

        if n < 4:
            return m

        # ===== Smoothness =====
        action_diffs = np.diff(actions, axis=0)
        m["action_rate"] = float(np.mean(np.abs(action_diffs)))

        action_accels = np.diff(action_diffs, axis=0)
        m["jerk"] = float(np.mean(np.abs(action_accels)))

        spectrum = np.abs(np.fft.rfft(actions, axis=0))
        total_energy = float(np.sum(spectrum)) + 1e-10
        cutoff_idx = max(1, int(n * 10 / 50))  # 10 Hz cutoff at 50 Hz sample rate
        m["hf_ratio"] = float(np.sum(spectrum[cutoff_idx:]) / total_energy)

        # ===== Velocity tracking =====
        # Use second half of episode after gait stabilises
        skip = min(100, n // 4)
        ach_lin = ep["lin_vel_local"][skip:]
        ach_ang = ep["ang_vel"][skip:]
        cmd_vx, cmd_vy, cmd_wz = command[0], command[1], command[2]

        m["achieved_vx_mean"] = float(np.mean(ach_lin[:, 0]))
        m["achieved_vy_mean"] = float(np.mean(ach_lin[:, 1]))
        m["achieved_wz_mean"] = float(np.mean(ach_ang[:, 2]))

        # Tracking error: signed mean and abs error
        m["vx_err_mean"] = float(np.mean(ach_lin[:, 0] - cmd_vx))
        m["vy_err_mean"] = float(np.mean(ach_lin[:, 1] - cmd_vy))
        m["wz_err_mean"] = float(np.mean(ach_ang[:, 2] - cmd_wz))

        # Tracking accuracy as a [0, 1] score (1 = perfect, 0 = totally wrong)
        # Score relative to command magnitude; if command is 0, expect achieved near 0
        def tracking_score(achieved, cmd, scale=0.1):
            if abs(cmd) < 1e-3:
                return float(np.exp(-(np.mean(achieved) ** 2) / (scale ** 2)))
            return float(np.exp(-((np.mean(achieved) - cmd) ** 2) / (scale ** 2)))

        m["vx_tracking_score"] = tracking_score(ach_lin[:, 0], cmd_vx, 0.05)
        m["vy_tracking_score"] = tracking_score(ach_lin[:, 1], cmd_vy, 0.05)
        m["wz_tracking_score"] = tracking_score(ach_ang[:, 2], cmd_wz, 0.2)

        # ===== Displacement =====
        if n > 50:
            net_disp = ep["positions"][-1] - ep["positions"][skip]
            duration_s = (n - skip) * 0.02  # ctrl_dt = 0.02s
            m["net_speed_x"] = float(net_disp[0] / duration_s)
            m["net_speed_y"] = float(net_disp[1] / duration_s)
            m["total_distance"] = float(np.sum(np.linalg.norm(np.diff(ep["positions"], axis=0), axis=1)))

        # ===== Orientation (matches training cost_orientation) =====
        # cost = sum of squared x,y components of torso up-vector
        m["orientation_cost"] = float(np.mean(np.sum(ep["upvecs"][:, :2] ** 2, axis=1)))
        # Tilt angle in degrees (max during episode)
        tilt = np.arccos(np.clip(ep["upvecs"][:, 2], -1, 1)) * 180 / np.pi
        m["max_tilt_deg"] = float(np.max(tilt))
        m["mean_tilt_deg"] = float(np.mean(tilt))

        # ===== Base height stability =====
        m["base_height_mean"] = float(np.mean(ep["base_heights"]))
        m["base_height_std"] = float(np.std(ep["base_heights"]))

        # ===== Gait quality (only meaningful when walking) =====
        # Step height: max foot height during swing phase
        # Swing = not in contact
        lh = ep["left_heights"]
        rh = ep["right_heights"]
        lc = ep["left_contacts"]
        rc = ep["right_contacts"]

        # Foot lift = peak height when foot is in air (during a swing)
        left_swing_heights = lh[~lc] if (~lc).sum() > 0 else np.array([0])
        right_swing_heights = rh[~rc] if (~rc).sum() > 0 else np.array([0])
        m["left_swing_peak"] = float(np.max(left_swing_heights))
        m["right_swing_peak"] = float(np.max(right_swing_heights))
        m["mean_swing_peak"] = float((m["left_swing_peak"] + m["right_swing_peak"]) / 2)

        # Step frequency: count contact transitions (rising edges)
        left_strikes = np.sum(np.diff(lc.astype(int)) > 0)
        right_strikes = np.sum(np.diff(rc.astype(int)) > 0)
        m["left_step_count"] = int(left_strikes)
        m["right_step_count"] = int(right_strikes)
        duration_s = n * 0.02
        m["step_frequency_hz"] = float((left_strikes + right_strikes) / duration_s)

        # Air time ratio: how much time at least one foot is off the ground
        any_air = ~(lc & rc)  # at least one foot off the ground
        both_contact = lc & rc
        m["air_time_ratio"] = float(np.mean(any_air))
        m["double_support_ratio"] = float(np.mean(both_contact))

        # ===== Gait symmetry (detects "pivoting on one leg") =====
        # Per-foot air time (fraction of episode each foot is off the ground)
        left_air_ratio = float(np.mean(~lc))
        right_air_ratio = float(np.mean(~rc))
        m["left_air_ratio"] = left_air_ratio
        m["right_air_ratio"] = right_air_ratio

        # Symmetry scores in [0, 1]: 1.0 = perfectly symmetric, 0.0 = one leg never moves
        # If both legs equally airborne, ratio is 1. If one stays planted, ratio approaches 0.
        def sym_score(a, b):
            tot = a + b
            return float(min(a, b) / (max(a, b) + 1e-6)) if tot > 1e-6 else 0.0

        m["step_count_symmetry"] = sym_score(int(left_strikes), int(right_strikes))
        m["air_time_symmetry"] = sym_score(left_air_ratio, right_air_ratio)
        m["swing_peak_symmetry"] = sym_score(m["left_swing_peak"], m["right_swing_peak"])

        # Overall pivot-detection score: low = one leg is barely moving = pivoting like a broken leg
        m["gait_symmetry_score"] = float(
            (m["step_count_symmetry"] + m["air_time_symmetry"] + m["swing_peak_symmetry"]) / 3
        )

        # ===== Standing quality (body sway) =====
        ang = ep["ang_vel"]
        m["body_roll_rate"] = float(np.mean(np.abs(ang[:, 0])))
        m["body_pitch_rate"] = float(np.mean(np.abs(ang[:, 1])))
        m["body_yaw_rate"] = float(np.mean(np.abs(ang[:, 2])))
        m["body_sway_rate"] = float(np.mean(np.sqrt(ang[:, 0]**2 + ang[:, 1]**2)))
        if n > 20:
            pos = ep["positions"]
            lateral_disp = np.std(pos[:, 1])
            m["lateral_drift_std"] = float(lateral_disp)
            m["com_drift_total"] = float(np.linalg.norm(pos[-1] - pos[0]))

        # ===== Cost of Transport (energy efficiency) =====
        forces = ep["actuator_forces"]
        jvel = ep["joint_velocities"]
        power = np.abs(forces * jvel)
        m["mean_mechanical_power"] = float(np.mean(np.sum(power, axis=1)))
        if n > 50:
            dist = m.get("total_distance", 0)
            if dist > 0.01:
                total_energy = float(np.sum(np.sum(power, axis=1)) * 0.02)
                m["cost_of_transport"] = round(total_energy / dist, 4)

        # ===== Stride duration CV (gait periodicity) =====
        for side, contacts in [("left", lc), ("right", rc)]:
            strikes = np.where(np.diff(contacts.astype(int)) > 0)[0]
            if len(strikes) >= 3:
                stride_durations = np.diff(strikes) * 0.02
                m[f"{side}_stride_cv"] = float(np.std(stride_durations) / (np.mean(stride_durations) + 1e-6))
        if "left_stride_cv" in m and "right_stride_cv" in m:
            m["stride_cv"] = round((m["left_stride_cv"] + m["right_stride_cv"]) / 2, 5)

        # ===== Foot slip (foot velocity during ground contact) =====
        for side, foot_xy, contacts in [
            ("left", ep["left_foot_xy"], lc),
            ("right", ep["right_foot_xy"], rc),
        ]:
            if n > 2:
                foot_vel = np.linalg.norm(np.diff(foot_xy, axis=0), axis=1) / 0.02
                contact_mask = contacts[1:]
                if contact_mask.sum() > 0:
                    slip_vel = foot_vel[contact_mask]
                    m[f"{side}_slip_mean"] = float(np.mean(slip_vel))
                    m[f"{side}_slip_max"] = float(np.max(slip_vel))
        if "left_slip_mean" in m and "right_slip_mean" in m:
            m["foot_slip_mean"] = round((m["left_slip_mean"] + m["right_slip_mean"]) / 2, 5)

        # ===== Bang-bang detection (actions saturated at limits) =====
        action_abs = np.abs(actions)
        m["bang_bang_ratio"] = float(np.mean(action_abs > 0.95))
        m["action_mean_abs"] = float(np.mean(action_abs))

        return m

    def run_phased_episode(self, phases, max_steps_total=2000):
        """Run an episode with a sequence of (n_steps, command) phases.

        Returns full telemetry plus phase boundary indices.
        Used to test command-gating and stand-then-command failure modes.
        """
        self.reset()
        actions, positions, base_heights = [], [], []
        upvecs, lin_vel_local, ang_vel = [], [], []
        left_heights, right_heights = [], []
        left_contacts, right_contacts = [], []
        phase_boundaries = [0]

        total_steps = 0
        for n_steps, cmd in phases:
            for _ in range(n_steps):
                if total_steps >= max_steps_total: break
                self.step_policy(self.policy, cmd)
                actions.append(self.last_action.copy())
                positions.append(self.data.qpos[0:2].copy())
                base_heights.append(float(self.data.qpos[2]))
                upvecs.append(self.get_upvec().copy())
                lin_vel_local.append(self.get_base_lin_vel_local())
                ang_vel.append(self.get_base_ang_vel())
                lh, rh = self.get_foot_heights()
                left_heights.append(lh); right_heights.append(rh)
                lc, rc = self.get_feet_contacts(self.data)
                left_contacts.append(bool(lc)); right_contacts.append(bool(rc))
                total_steps += 1
                if self.is_fallen(): break
            phase_boundaries.append(total_steps)
            if self.is_fallen() or total_steps >= max_steps_total: break

        return {
            "actions": np.array(actions),
            "positions": np.array(positions),
            "base_heights": np.array(base_heights),
            "upvecs": np.array(upvecs),
            "lin_vel_local": np.array(lin_vel_local),
            "ang_vel": np.array(ang_vel),
            "left_heights": np.array(left_heights),
            "right_heights": np.array(right_heights),
            "left_contacts": np.array(left_contacts),
            "right_contacts": np.array(right_contacts),
            "steps": total_steps,
            "phase_boundaries": phase_boundaries,
        }

    def phase_velocity(self, ep, phase_idx, cmd, skip_initial=20):
        """Get achieved velocity during a specific phase, with tracking score AND temporal analysis.

        Reports:
        - early/steady velocity: first 1/3 vs last 1/3 of phase (catches startup delays)
        - time_to_motion: first step where velocity reaches 50% of commanded magnitude
        """
        b = ep["phase_boundaries"]
        if phase_idx + 1 >= len(b): return None
        phase_start = b[phase_idx]
        start = phase_start + skip_initial
        end = b[phase_idx + 1]
        if start >= end: start = phase_start
        if start >= end: return None

        lin_x = ep["lin_vel_local"][start:end, 0]
        lin_y = ep["lin_vel_local"][start:end, 1]
        ang_z = ep["ang_vel"][start:end, 2]
        phase_len = end - start

        vx = float(np.mean(lin_x))
        vy = float(np.mean(lin_y))
        wz = float(np.mean(ang_z))

        # Early vs steady-state - reveals startup delays
        third = max(20, phase_len // 3)
        vx_early = float(np.mean(lin_x[:third]))
        vx_steady = float(np.mean(lin_x[-third:]))
        vy_early = float(np.mean(lin_y[:third]))
        vy_steady = float(np.mean(lin_y[-third:]))
        wz_early = float(np.mean(ang_z[:third]))
        wz_steady = float(np.mean(ang_z[-third:]))

        # Time-to-motion: first step where commanded axis reaches 50% threshold
        def time_to_motion(signal, target):
            if abs(target) < 1e-3: return None  # no command on this axis
            threshold = abs(target) * 0.5
            for i, v in enumerate(signal):
                if (target > 0 and v >= threshold) or (target < 0 and v <= -threshold):
                    return int(i)
            return None  # never reached

        ttm_vx = time_to_motion(lin_x, cmd[0])
        ttm_vy = time_to_motion(lin_y, cmd[1])
        ttm_wz = time_to_motion(ang_z, cmd[2])

        # Tracking score in [0,1]
        def s(achieved, c, scale):
            return float(np.exp(-((achieved - c) ** 2) / (scale ** 2)))

        return {
            "vx": round(vx, 5), "vy": round(vy, 5), "wz": round(wz, 5),
            "vx_score": round(s(vx, cmd[0], 0.05), 3),
            "vy_score": round(s(vy, cmd[1], 0.05), 3),
            "wz_score": round(s(wz, cmd[2], 0.2), 3),
            # Temporal analysis
            "vx_early": round(vx_early, 5), "vx_steady": round(vx_steady, 5),
            "vy_early": round(vy_early, 5), "vy_steady": round(vy_steady, 5),
            "wz_early": round(wz_early, 5), "wz_steady": round(wz_steady, 5),
            "time_to_vx_steps": ttm_vx, "time_to_vy_steps": ttm_vy, "time_to_wz_steps": ttm_wz,
            "duration_steps": phase_len,
        }

    def evaluate(self, num_episodes=10):
        # Locomotion tests with achievable command magnitudes (matching joystick ranges)
        tests = {
            # Single-axis tests
            "forward_walk":         {"cmd": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            "backward_walk":        {"cmd": [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            "lateral_left":         {"cmd": [0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0]},
            "lateral_right":        {"cmd": [0.0, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0]},
            "turn_left":            {"cmd": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]},
            "turn_right":           {"cmd": [0.0, 0.0, -0.8, 0.0, 0.0, 0.0, 0.0]},
            # Combined commands (the user's real-world usage pattern)
            "fwd_strafe_left":      {"cmd": [0.1, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0]},
            "fwd_strafe_right":     {"cmd": [0.1, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0]},
            "fwd_turn_left":        {"cmd": [0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]},
            "fwd_turn_right":       {"cmd": [0.1, 0.0, -0.8, 0.0, 0.0, 0.0, 0.0]},
            "back_turn_left":       {"cmd": [-0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]},
            "back_turn_right":      {"cmd": [-0.1, 0.0, -0.8, 0.0, 0.0, 0.0, 0.0]},
            "standing":             {"cmd": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        }

        results = {}

        for test_name, cfg in tests.items():
            print(f"  Running {test_name} ({num_episodes} eps)...")
            all_metrics = []
            for ep_idx in range(num_episodes):
                ep = self.run_episode(cfg["cmd"], max_steps=1000)
                m = self.compute_metrics(ep, cfg["cmd"])
                all_metrics.append(m)

            # Aggregate: mean across episodes for each metric
            agg = {}
            for key in all_metrics[0].keys():
                vals = [m[key] for m in all_metrics if key in m]
                if vals:
                    agg[key] = round(float(np.mean(vals)), 5)
                    if key == "steps":
                        agg["survival_rate"] = round(float(np.mean([v >= 990 for v in vals])), 3)
            results[test_name] = agg

        # ===== Push recovery tests =====
        # Apply a single push at step 300 (after gait is established), various magnitudes
        print(f"  Running push recovery tests...")
        push_results = {}
        for magnitude in [0.3, 0.6, 1.0, 1.5]:
            for direction_label, direction in [("front", 0.0), ("back", np.pi), ("side", np.pi / 2)]:
                key = f"push_{magnitude}_{direction_label}"
                survived = 0
                tilts = []
                for ep_idx in range(5):  # 5 trials per push
                    ep = self.run_episode(
                        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # gentle forward walk
                        max_steps=1000,
                        pushes=[(300, magnitude, direction)],
                    )
                    if ep["steps"] >= 990:
                        survived += 1
                    if ep["steps"] > 350:  # measure tilt 50 steps after push
                        post_push_upvec = ep["upvecs"][350:min(450, ep["steps"])]
                        max_tilt = float(np.max(np.arccos(np.clip(post_push_upvec[:, 2], -1, 1)) * 180 / np.pi))
                        tilts.append(max_tilt)

                push_results[key] = {
                    "survival_rate": survived / 5,
                    "max_tilt_after_push_deg": round(float(np.mean(tilts)) if tilts else 90.0, 1),
                }
        results["push_recovery"] = push_results

        # ===== Responsiveness tests - the failure mode our averaging missed =====
        # User scenario: viewer has imitation_phase ticking while standing; they then press a key.
        # Test whether each command works AFTER a standstill period (the realistic user experience).
        print(f"  Running responsiveness tests (stand-then-command)...")
        stand_n = 200  # 4 sec of standing first
        active_n = 800  # then 16 sec of the test command (long enough to catch delayed responses)

        responsiveness_tests = {
            # Single axis
            "stand_then_forward":            [0.10, 0.00, 0.00, 0,0,0,0],
            "stand_then_backward":           [-0.10, 0.00, 0.00, 0,0,0,0],
            "stand_then_strafe_left":        [0.00, 0.15, 0.00, 0,0,0,0],
            "stand_then_strafe_right":       [0.00, -0.15, 0.00, 0,0,0,0],
            "stand_then_turn_left":          [0.00, 0.00, 0.80, 0,0,0,0],
            "stand_then_turn_right":         [0.00, 0.00, -0.80, 0,0,0,0],
            # Combined commands AFTER standing (user's real-world test pattern)
            "stand_then_fwd_strafe_left":    [0.10, 0.15, 0.00, 0,0,0,0],
            "stand_then_fwd_strafe_right":   [0.10, -0.15, 0.00, 0,0,0,0],
            "stand_then_fwd_turn_left":      [0.10, 0.00, 0.80, 0,0,0,0],
            "stand_then_fwd_turn_right":     [0.10, 0.00, -0.80, 0,0,0,0],
            "stand_then_back_turn_left":     [-0.10, 0.00, 0.80, 0,0,0,0],
            "stand_then_back_turn_right":    [-0.10, 0.00, -0.80, 0,0,0,0],
            "stand_then_strafe_turn_left":   [0.00, 0.15, 0.80, 0,0,0,0],
            "stand_then_strafe_turn_right":  [0.00, -0.15, -0.80, 0,0,0,0],
        }
        resp = {}
        for name, cmd in responsiveness_tests.items():
            ep = self.run_phased_episode([(stand_n, [0,0,0,0,0,0,0]), (active_n, cmd)])
            phase2 = self.phase_velocity(ep, 1, cmd, skip_initial=20)
            resp[name] = phase2 if phase2 else {"vx": 0, "vy": 0, "wz": 0,
                                                "vx_score": 0, "vy_score": 0, "wz_score": 0}
        results["responsiveness"] = resp

        # ===== Command-gating tests - is motion gated on having a turn input? =====
        print(f"  Running command-gating tests...")
        gating_tests = {}

        # Strafe alone after strafe+turn: does strafing stop when turn removed?
        ep = self.run_phased_episode([
            (200, [0.00, 0.15, 0.80, 0,0,0,0]),  # strafe+turn
            (300, [0.00, 0.15, 0.00, 0,0,0,0]),  # strafe alone
        ])
        gating_tests["strafe_persists_without_turn_left"] = {
            "with_turn": self.phase_velocity(ep, 0, [0,0.15,0.8,0,0,0,0]),
            "turn_removed": self.phase_velocity(ep, 1, [0,0.15,0,0,0,0,0]),
        }
        ep = self.run_phased_episode([
            (200, [0.00, -0.15, -0.80, 0,0,0,0]),
            (300, [0.00, -0.15, 0.00, 0,0,0,0]),
        ])
        gating_tests["strafe_persists_without_turn_right"] = {
            "with_turn": self.phase_velocity(ep, 0, [0,-0.15,-0.8,0,0,0,0]),
            "turn_removed": self.phase_velocity(ep, 1, [0,-0.15,0,0,0,0,0]),
        }

        # Forward+strafe → strafe alone
        ep = self.run_phased_episode([
            (200, [0.10, 0.15, 0.00, 0,0,0,0]),
            (300, [0.00, 0.15, 0.00, 0,0,0,0]),
        ])
        gating_tests["strafe_persists_without_forward"] = {
            "with_forward": self.phase_velocity(ep, 0, [0.1,0.15,0,0,0,0,0]),
            "forward_removed": self.phase_velocity(ep, 1, [0,0.15,0,0,0,0,0]),
        }

        results["command_gating"] = gating_tests

        # ===== Aggregate locomotion =====
        locomotion_tests = ["forward_walk", "backward_walk", "lateral_left", "lateral_right",
                            "turn_left", "turn_right",
                            "fwd_strafe_left", "fwd_strafe_right",
                            "fwd_turn_left", "fwd_turn_right",
                            "back_turn_left", "back_turn_right"]
        agg_keys = ["action_rate", "jerk", "hf_ratio", "orientation_cost", "mean_swing_peak",
                    "step_frequency_hz", "vx_tracking_score", "vy_tracking_score",
                    "wz_tracking_score", "survival_rate", "mean_tilt_deg",
                    "gait_symmetry_score", "step_count_symmetry",
                    "cost_of_transport", "stride_cv", "foot_slip_mean",
                    "bang_bang_ratio", "mean_mechanical_power"]
        agg = {}
        for k in agg_keys:
            vals = [results[t].get(k) for t in locomotion_tests if results[t].get(k) is not None]
            if vals:
                agg[k] = round(float(np.mean(vals)), 5)
        results["aggregate_locomotion"] = agg

        # ===== Responsiveness aggregate =====
        # For each test, compute composite score across all commanded axes (vx, vy, wz).
        # A test counts each non-zero command axis equally.
        resp_aggregate = {}
        for name, cmd_full in responsiveness_tests.items():
            v = resp[name]
            scores = []
            if abs(cmd_full[0]) > 1e-3: scores.append(v.get("vx_score", 0))
            if abs(cmd_full[1]) > 1e-3: scores.append(v.get("vy_score", 0))
            if abs(cmd_full[2]) > 1e-3: scores.append(v.get("wz_score", 0))
            resp_aggregate[name + "_score"] = round(float(np.mean(scores)) if scores else 0.0, 3)
        resp_aggregate["mean_responsiveness"] = round(
            float(np.mean([v for k, v in resp_aggregate.items() if k != "mean_responsiveness"])), 4)
        results["responsiveness_aggregate"] = resp_aggregate

        return results


def _grade(val, good, ok, bad_label="FAIL"):
    if val >= good: return "GOOD"
    if val >= ok: return "OK"
    return bad_label


def _grade_low(val, good, ok, bad_label="FAIL"):
    """Lower is better."""
    if val <= good: return "GOOD"
    if val <= ok: return "OK"
    return bad_label


def print_verdict(results):
    """Print a human-readable verdict table."""
    agg = results.get("aggregate_locomotion", {})
    stand = results.get("standing", {})
    push = results.get("push_recovery", {})
    resp_agg = results.get("responsiveness_aggregate", {})

    print("\n" + "=" * 70)
    print("  VERDICT SUMMARY")
    print("=" * 70)

    # Walking
    print("\n  WALKING")
    walk_tests = {
        "forward_walk": ("vx", 0.10), "backward_walk": ("vx", -0.10),
        "lateral_left": ("vy", 0.15), "lateral_right": ("vy", -0.15),
        "turn_left": ("wz", 0.80), "turn_right": ("wz", -0.80),
    }
    for test, (axis, target) in walk_tests.items():
        r = results.get(test, {})
        key = f"achieved_{axis}_mean" if axis != "wz" else "achieved_wz_mean"
        achieved = r.get(key, 0)
        score_key = f"{axis}_tracking_score"
        score = r.get(score_key, 0)
        grade = _grade(score, 0.8, 0.5, "WEAK")
        print(f"    {test:<20} {grade:<5} {achieved:+.3f} (target {target:+.2f})  tracking={score:.2f}")

    cot = agg.get("cost_of_transport")
    if cot: print(f"    CoT:                 {_grade_low(cot, 30, 80):<5} {cot:.1f} J/m")
    scv = agg.get("stride_cv")
    if scv is not None:
        label = "regular" if scv < 0.15 else "irregular" if scv < 0.3 else "stumbling"
        print(f"    Stride CV:           {_grade_low(scv, 0.15, 0.3):<5} {scv:.3f} ({label})")
    slip = agg.get("foot_slip_mean")
    if slip is not None: print(f"    Foot slip:           {_grade_low(slip, 0.02, 0.05):<5} {slip:.4f} m/s")
    bb = agg.get("bang_bang_ratio")
    if bb is not None: print(f"    Bang-bang:           {_grade_low(bb, 0.05, 0.15):<5} {bb*100:.1f}%")

    # Standing
    print("\n  STANDING")
    step_count = stand.get("left_step_count", 0) + stand.get("right_step_count", 0)
    duration_s = stand.get("steps", 1000) * 0.02
    steps_per_10s = step_count / duration_s * 10
    drift = stand.get("com_drift_total", 0)
    sway = stand.get("body_sway_rate", 0)
    print(f"    Step count:          {_grade_low(steps_per_10s, 5, 20):<5} {steps_per_10s:.0f} steps/10s (target < 5)")
    print(f"    CoM drift:           {_grade_low(drift, 0.05, 0.2):<5} {drift:.3f}m (target < 0.05m)")
    print(f"    Body sway:           {_grade_low(sway, 0.2, 0.5):<5} {sway:.3f} rad/s")

    # Push recovery
    print("\n  PUSH RECOVERY")
    for mag in [0.3, 0.6, 1.0, 1.5]:
        directions = ["front", "back", "side"]
        survived = 0
        tested = 0
        for d in directions:
            key = f"push_{mag}_{d}"
            if key in push:
                tested += 1
                if push[key]["survival_rate"] >= 0.5:
                    survived += 1
        if tested > 0:
            grade = "PASS" if survived == tested else "PARTIAL" if survived > 0 else "FAIL"
            fails = [d for d in directions if f"push_{mag}_{d}" in push and push[f"push_{mag}_{d}"]["survival_rate"] < 0.5]
            fail_str = f" ({', '.join(fails)} FAIL)" if fails else ""
            print(f"    {mag} N*s:            {grade:<8} {survived}/{tested} directions{fail_str}")

    # Responsiveness
    print("\n  RESPONSIVENESS")
    mean_resp = resp_agg.get("mean_responsiveness", 0)
    n_tests = len([k for k in resp_agg if k != "mean_responsiveness"])
    print(f"    Mean score:          {_grade(mean_resp, 0.7, 0.4):<5} {mean_resp:.3f}  ({n_tests} tests)")
    worst = sorted([(k, v) for k, v in resp_agg.items() if k != "mean_responsiveness"], key=lambda x: x[1])
    if worst:
        print(f"    Weakest:             {worst[0][0].replace('_score','')} {worst[0][1]:.3f}", end="")
        if len(worst) > 1:
            print(f", {worst[1][0].replace('_score','')} {worst[1][1]:.3f}")
        else:
            print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Headless policy evaluation")
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument("--reference_data", type=str,
                        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
    parser.add_argument("--model_path", type=str,
                        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Evaluating: {args.onnx_model_path}")
    evaluator = HeadlessEvaluator(args.model_path, args.reference_data, args.onnx_model_path)

    start = time.time()
    results = evaluator.evaluate(num_episodes=args.num_episodes)
    elapsed = time.time() - start

    results["metadata"] = {
        "onnx_path": args.onnx_model_path,
        "num_episodes": args.num_episodes,
        "evaluation_time_seconds": round(elapsed, 1),
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output} ({elapsed:.0f}s)")
    else:
        print(json.dumps(results, indent=2))

    print_verdict(results)


if __name__ == "__main__":
    main()
