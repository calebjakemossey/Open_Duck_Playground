"""Lightweight walking assessment hooked into training - logs real walking metrics to TB.

Called once per checkpoint. Runs deterministic forward/strafe/turn/combined scenarios
and returns metrics suitable for logging. Targets <10 seconds per call.
"""
import numpy as np
from playground.common.onnx_infer import OnnxInfer
from analysis.evaluator_base import PolicyEvaluatorBase


_BASE_ASSESSOR = None


class _QuickAssessor(PolicyEvaluatorBase):
    """Minimal walking assessor - keeps the env in memory between calls."""

    def __init__(self, model_path, reference_data):
        super().__init__(model_path, reference_data)

    def run_phased(self, policy, phases, collect_standing=False):
        self.reset()
        records = []
        phase_starts = []
        standing_data = [] if collect_standing else None
        idx = 0
        for n_steps, cmd in phases:
            phase_starts.append(idx)
            for _ in range(n_steps):
                self.step_policy(policy, cmd)
                local_vel = self.get_base_lin_vel_local()
                wz = float(self.get_base_ang_vel()[2])
                records.append((float(local_vel[0]), float(local_vel[1]), wz))
                if collect_standing:
                    lc, rc = self.get_feet_contacts(self.data)
                    ang = self.get_base_ang_vel()
                    pos = self.data.qpos[0:2].copy()
                    standing_data.append((bool(lc), bool(rc), float(ang[0]), float(ang[1]), float(ang[2]), float(pos[0]), float(pos[1])))
                idx += 1
                if self.is_fallen(): break
            if self.is_fallen(): break
        return records, phase_starts, standing_data


def get_assessor(model_path, reference_data):
    global _BASE_ASSESSOR
    if _BASE_ASSESSOR is None:
        _BASE_ASSESSOR = _QuickAssessor(model_path, reference_data)
    return _BASE_ASSESSOR


def assess_walking(onnx_path,
                   model_path="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
                   reference_data="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"):
    """Run a fast walking assessment, return dict of metrics for TB logging."""
    ev = get_assessor(model_path, reference_data)
    policy = OnnxInfer(onnx_path, awd=True)

    obs_size = policy.ort_session.get_inputs()[0].shape[1]
    snap_size = 6 + 2 * ev.num_dofs
    if obs_size == 101:
        ev.HAS_LINVEL_GRAVITY = False
        ev.HAS_PHASE = True
        ev.HISTORY_LEN = 0
    elif obs_size == 209:
        ev.HAS_LINVEL_GRAVITY = True
        ev.HAS_PHASE = True
        base_obs = 107
        ev.HISTORY_LEN = max(0, (obs_size - base_obs) // snap_size)
    else:
        ev.HAS_LINVEL_GRAVITY = True
        ev.HAS_PHASE = False
        base_obs = 105
        ev.HISTORY_LEN = max(0, (obs_size - base_obs) // snap_size)
    ev._reset_action_state()

    metrics = {}

    # Cold-start single-axis tests (500 steps each, ~10 sec sim)
    # 500 steps gives strafe/backward time to converge to steady-state
    scenarios = {
        "forward":      ([0.10, 0.0, 0.0, 0,0,0,0], 0, 0.10, 0.05),
        "backward":     ([-0.10, 0.0, 0.0, 0,0,0,0], 0, -0.10, 0.05),
        "strafe_left":  ([0.0, 0.15, 0.0, 0,0,0,0], 1, 0.15, 0.05),
        "strafe_right": ([0.0, -0.15, 0.0, 0,0,0,0], 1, -0.15, 0.05),
        "turn_left":    ([0.0, 0.0, 0.80, 0,0,0,0], 2, 0.80, 0.2),
        "turn_right":   ([0.0, 0.0, -0.80, 0,0,0,0], 2, -0.80, 0.2),
    }
    for name, (cmd, ax_idx, target, scale) in scenarios.items():
        records, _, _ = ev.run_phased(policy, [(500, cmd)])
        if records and len(records) > 100:
            arr = np.array(records[100:])
            achieved = float(np.mean(arr[:, ax_idx]))
            score = float(np.exp(-((achieved - target) ** 2) / (scale ** 2)))
            metrics[f"walking/cold_{name}_score"] = score

    # Stand-then-command tests (catches standstill failure mode)
    # 250 steps standing (5s) + 500 steps command (10s)
    # collect_standing=True to get standing quality metrics from the standstill phase
    stand_tests = {
        "stand_then_fwd":           ([0.10, 0.0, 0.0, 0,0,0,0], 0, 0.10, 0.05),
        "stand_then_back":          ([-0.10, 0.0, 0.0, 0,0,0,0], 0, -0.10, 0.05),
        "stand_then_strafe":        ([0.0, 0.15, 0.0, 0,0,0,0], 1, 0.15, 0.05),
        "stand_then_turn":          ([0.0, 0.0, 0.80, 0,0,0,0], 2, 0.80, 0.2),
        "stand_then_fwd_turn":      ([0.10, 0.0, 0.80, 0,0,0,0], 0, 0.10, 0.05),
        "stand_then_fwd_strafe":    ([0.10, 0.15, 0.0, 0,0,0,0], 1, 0.15, 0.05),
    }
    all_standing_data = []
    for name, (cmd, ax_idx, target, scale) in stand_tests.items():
        records, phases, sd = ev.run_phased(policy, [
            (250, [0,0,0,0,0,0,0]),
            (500, cmd),
        ], collect_standing=True)
        if len(phases) > 1 and len(records) > phases[1] + 50:
            arr = np.array(records[phases[1]+50:])
            achieved = float(np.mean(arr[:, ax_idx]))
            score = float(np.exp(-((achieved - target) ** 2) / (scale ** 2)))
            metrics[f"walking/{name}_score"] = score
        else:
            metrics[f"walking/{name}_score"] = 0.0
        if sd and len(phases) > 1:
            all_standing_data.append(sd[:phases[1]])

    # Standing quality metrics from the standstill phases
    if all_standing_data:
        step_counts = []
        drifts = []
        sway_rates = []
        air_ratios = []
        for phase_data in all_standing_data:
            if not phase_data:
                continue
            sd_arr = np.array(phase_data)
            lc = sd_arr[:, 0].astype(bool)
            rc = sd_arr[:, 1].astype(bool)
            ang_x, ang_y, ang_z = sd_arr[:, 2], sd_arr[:, 3], sd_arr[:, 4]
            pos_x, pos_y = sd_arr[:, 5], sd_arr[:, 6]

            left_strikes = int(np.sum(np.diff(lc.astype(int)) > 0))
            right_strikes = int(np.sum(np.diff(rc.astype(int)) > 0))
            step_counts.append(left_strikes + right_strikes)

            drift = float(np.sqrt((pos_x[-1] - pos_x[0])**2 + (pos_y[-1] - pos_y[0])**2))
            drifts.append(drift)

            sway = float(np.mean(np.sqrt(ang_x**2 + ang_y**2)))
            sway_rates.append(sway)

            any_air = ~(lc & rc)
            air_ratios.append(float(np.mean(any_air)))

        metrics["standing/step_count"] = float(np.mean(step_counts))
        metrics["standing/drift_m"] = float(np.mean(drifts))
        metrics["standing/sway_rate"] = float(np.mean(sway_rates))
        metrics["standing/air_time_ratio"] = float(np.mean(air_ratios))

        step_score = float(np.exp(-(np.mean(step_counts) / 5.0) ** 2))
        drift_score = float(np.exp(-(np.mean(drifts) / 0.05) ** 2))
        sway_score = float(np.exp(-(np.mean(sway_rates) / 0.3) ** 2))
        metrics["standing/STANDING_SCORE"] = round((step_score + drift_score + sway_score) / 3, 5)

    # Composites
    cold_scores = [v for k, v in metrics.items() if k.startswith("walking/cold_")]
    stand_scores = [v for k, v in metrics.items() if k.startswith("walking/stand_")]
    metrics["walking/WALKING_SCORE"] = float(np.mean(cold_scores)) if cold_scores else 0.0
    metrics["walking/responsiveness_avg"] = float(np.mean(stand_scores)) if stand_scores else 0.0

    w_score = metrics["walking/WALKING_SCORE"]
    r_score = metrics["walking/responsiveness_avg"]
    metrics["walking/HEADLINE"] = 0.5 * w_score + 0.5 * r_score

    return metrics
