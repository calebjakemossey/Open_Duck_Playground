"""Lightweight standing assessment hooked into training - logs standing metrics to TB.

Called once per checkpoint during standing policy training. Runs deterministic
standstill, push recovery, and head-tracking scenarios. Targets <10 seconds per call.
"""
import numpy as np
from playground.common.onnx_infer import OnnxInfer
from analysis.evaluator_base import PolicyEvaluatorBase


_BASE_ASSESSOR = None

_PUSH_BODIES = ["trunk_assembly", "hip_roll_assembly", "hip_roll_assembly_2",
                "left_roll_to_pitch_assembly", "right_roll_to_pitch_assembly"]


class _StandingAssessor(PolicyEvaluatorBase):
    """Minimal standing assessor - keeps the env in memory between calls."""

    UPDATE_IMITATION_PHASE = False

    def __init__(self, model_path, reference_data):
        super().__init__(model_path, reference_data)

    def run_standing(self, policy, n_steps, push_at=None, push_mag=0, push_dir=0,
                     push_body=None, angular_mag=0):
        self.reset()
        contacts_l, contacts_r = [], []
        ang_vels = []
        positions = []
        for i in range(n_steps):
            if push_at is not None and i == push_at:
                if push_body is not None:
                    self.apply_body_push(push_body, push_mag, push_dir)
                else:
                    self.apply_push(push_mag, push_dir)
                if angular_mag > 0:
                    self.apply_angular_push(angular_mag, push_dir)
            self.step_policy(policy, [0, 0, 0, 0, 0, 0, 0])
            lc, rc = self.get_feet_contacts(self.data)
            contacts_l.append(bool(lc))
            contacts_r.append(bool(rc))
            ang_vels.append(self.get_base_ang_vel().copy())
            positions.append(self.data.qpos[0:2].copy())
            if self.is_fallen():
                break
        return {
            "steps": len(contacts_l),
            "contacts_l": np.array(contacts_l),
            "contacts_r": np.array(contacts_r),
            "ang_vels": np.array(ang_vels),
            "positions": np.array(positions),
        }


def get_assessor(model_path, reference_data):
    global _BASE_ASSESSOR
    if _BASE_ASSESSOR is None:
        _BASE_ASSESSOR = _StandingAssessor(model_path, reference_data)
    return _BASE_ASSESSOR


def assess_standing(onnx_path,
                    model_path="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
                    reference_data="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"):
    """Run a fast standing assessment, return dict of metrics for TB logging."""
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

    # Standstill test: 500 steps (10s) at zero command
    ep = ev.run_standing(policy, 500)
    n = ep["steps"]
    lc, rc = ep["contacts_l"], ep["contacts_r"]
    ang = ep["ang_vels"]
    pos = ep["positions"]

    left_strikes = int(np.sum(np.diff(lc.astype(int)) > 0))
    right_strikes = int(np.sum(np.diff(rc.astype(int)) > 0))
    step_count = left_strikes + right_strikes
    metrics["standing/step_count"] = float(step_count)

    drift = float(np.linalg.norm(pos[-1] - pos[0])) if n > 1 else 0.0
    metrics["standing/drift_m"] = drift

    sway = float(np.mean(np.sqrt(ang[:, 0]**2 + ang[:, 1]**2)))
    metrics["standing/sway_rate"] = sway

    any_air = ~(lc & rc)
    metrics["standing/air_time_ratio"] = float(np.mean(any_air))

    metrics["standing/survived"] = float(n >= 499)

    # Component scores
    step_score = float(np.exp(-(step_count / 5.0) ** 2))
    drift_score = float(np.exp(-(drift / 0.05) ** 2))
    sway_score = float(np.exp(-(sway / 0.3) ** 2))
    metrics["standing/stillness_score"] = round((step_score + drift_score + sway_score) / 3, 5)

    # Base velocity push recovery: 6 magnitudes x 4 directions
    push_survival = {}
    for mag in [0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        survived = 0
        tested = 0
        for direction, label in [(0, "front"), (np.pi, "back"), (np.pi/2, "left"), (-np.pi/2, "right")]:
            ep = ev.run_standing(policy, 400, push_at=100, push_mag=mag, push_dir=direction)
            tested += 1
            if ep["steps"] >= 399:
                survived += 1
            metrics[f"standing/push_{mag}_{label}"] = float(ep["steps"] >= 399)
        push_survival[mag] = survived / tested
        metrics[f"standing/push_{mag}_rate"] = push_survival[mag]

    push_score = (
        0.15 * push_survival[0.4]
        + 0.15 * push_survival[0.6]
        + 0.20 * push_survival[0.8]
        + 0.25 * push_survival[1.0]
        + 0.15 * push_survival[1.5]
        + 0.10 * push_survival[2.0]
    )
    metrics["standing/push_score"] = round(push_score, 5)

    # Body push recovery: push applied to individual body parts at 1.0 m/s
    body_survived = 0
    body_tested = 0
    for body in _PUSH_BODIES:
        for mag in [0.6, 0.8]:
            for direction, label in [(0, "front"), (np.pi, "back"), (np.pi/2, "left"), (-np.pi/2, "right")]:
                ep = ev.run_standing(policy, 400, push_at=100, push_mag=mag,
                                     push_dir=direction, push_body=body)
                body_tested += 1
                if ep["steps"] >= 399:
                    body_survived += 1
                metrics[f"standing/body_{body}_{mag}_{label}"] = float(ep["steps"] >= 399)
    metrics["standing/body_push_rate"] = round(body_survived / body_tested, 5)

    # Angular perturbation recovery: combined linear + angular push at multiple magnitudes
    ang_survived = 0
    ang_tested = 0
    for ang_mag in [1.0, 2.0, 3.0]:
        for direction, label in [(0, "front"), (np.pi, "back"), (np.pi/2, "left"), (-np.pi/2, "right")]:
            ep = ev.run_standing(policy, 400, push_at=100, push_mag=0.5,
                                 push_dir=direction, angular_mag=ang_mag)
            ang_tested += 1
            if ep["steps"] >= 399:
                ang_survived += 1
            metrics[f"standing/angular_{ang_mag}_{label}"] = float(ep["steps"] >= 399)
        metrics[f"standing/angular_{ang_mag}_rate"] = round(
            sum(1 for d in ["front","back","left","right"]
                if metrics[f"standing/angular_{ang_mag}_{d}"] >= 1.0) / 4, 5)
    metrics["standing/angular_push_rate"] = round(ang_survived / ang_tested, 5)

    # HEADLINE: weighted toward hard pushes
    metrics["standing/HEADLINE"] = round(
        0.30 * metrics["standing/stillness_score"]
        + 0.40 * metrics["standing/push_score"]
        + 0.15 * metrics["standing/body_push_rate"]
        + 0.15 * metrics["standing/angular_push_rate"], 5
    )

    return metrics
