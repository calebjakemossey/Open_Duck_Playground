"""
Comprehensive training run analysis - one command to understand a run.

Given a training directory, this script:
1. Reads TensorBoard events to show training dynamics (plateau, regression, trends)
2. Finds the best checkpoint by composite walking score
3. Evaluates the best checkpoint with full metrics + verdict
4. Optionally runs temporal debug traces

Usage:
    PYTHONPATH=. uv run python analysis/analyse_run.py checkpoints/E14_imitation_gating
    PYTHONPATH=. uv run python analysis/analyse_run.py checkpoints/E14_imitation_gating --skip_eval
    PYTHONPATH=. uv run python analysis/analyse_run.py checkpoints/E14_imitation_gating --debug
    PYTHONPATH=. uv run python analysis/analyse_run.py checkpoints/E14_imitation_gating --compare checkpoints/E5_curriculum_phase2
"""

import argparse
import glob
import json
import time
import numpy as np
from pathlib import Path


# ===== TB Events Analysis =====

def load_events(events_path):
    import tensorflow as tf
    metrics = {}
    for event in tf.compat.v1.train.summary_iterator(events_path):
        for v in event.summary.value:
            metrics.setdefault(v.tag, []).append((event.step, v.simple_value))
    for tag in metrics:
        metrics[tag].sort(key=lambda x: x[0])
    return metrics


def best_and_final(values, lower_is_better=False):
    steps = np.array([v[0] for v in values])
    vals = np.array([v[1] for v in values])
    best_idx = int(np.argmin(vals) if lower_is_better else np.argmax(vals))
    drop = float(vals[best_idx] - vals[-1]) if not lower_is_better else float(vals[-1] - vals[best_idx])
    drop_pct = float(drop / (abs(vals[best_idx]) + 1e-8) * 100)
    return {
        "best_step": int(steps[best_idx]),
        "best_val": float(vals[best_idx]),
        "final_step": int(steps[-1]),
        "final_val": float(vals[-1]),
        "peak_fraction": float(steps[best_idx] / steps[-1]) if steps[-1] > 0 else 0.0,
        "drop_pct": drop_pct,
    }


def trend_late(values, last_n=5):
    if len(values) < last_n:
        return 0.0
    last = np.array([v[1] for v in values[-last_n:]])
    steps = np.array([v[0] for v in values[-last_n:]])
    if (steps[-1] - steps[0]) == 0:
        return 0.0
    return float((last[-1] - last[0]) / (steps[-1] - steps[0]))


def trend_classify(slope, scale):
    norm = slope * 1e7 / (abs(scale) + 1e-6)
    if abs(norm) < 0.05: return "FLAT"
    return "RISING" if norm > 0 else "FALLING"


def analyse_training(run_dir):
    events_files = glob.glob(str(Path(run_dir) / "events.out.tfevents.*"))
    if not events_files:
        print(f"  No TF events found in {run_dir}")
        return None
    events_path = sorted(events_files)[-1]

    metrics = load_events(events_path)
    total_steps = max(v[0] for vs in metrics.values() for v in vs)

    print(f"  Total steps: {total_steps:,}  |  Evaluations: {len(next(iter(metrics.values())))}  |  Metrics: {len(metrics)}")

    key_groups = {
        "OUTCOME": [
            "eval/episode_reward", "eval/avg_episode_length",
        ],
        "TRACKING REWARDS (higher is better)": [
            "eval/episode_reward/tracking_lin_vel", "eval/episode_reward/tracking_ang_vel",
            "eval/episode_reward/imitation", "eval/episode_reward/alive",
        ],
        "COSTS (lower is better)": [
            "eval/episode_cost/action_rate", "eval/episode_cost/torques",
            "eval/episode_cost/orientation", "eval/episode_cost/stand_still",
        ],
        "WALKING EVAL": [
            "walking/WALKING_SCORE", "walking/responsiveness_avg", "walking/HEADLINE",
            "standing/STANDING_SCORE", "standing/HEADLINE",
        ],
        "TRAINING HEALTH": [
            "training/kl_mean", "training/policy_loss", "training/v_loss", "training/entropy_loss",
        ],
    }

    print(f"\n  {'metric':<44} {'best':>9} {'final':>9} {'@peak%':>7} {'drop':>8} {'trend':>8}")
    print("  " + "-" * 90)

    for group_name, tags in key_groups.items():
        present = [t for t in tags if t in metrics]
        if not present:
            continue
        is_cost = "cost" in group_name.lower() or "lower is better" in group_name.lower()
        print(f"\n  {group_name}")
        for tag in present:
            r = best_and_final(metrics[tag], lower_is_better=is_cost)
            slope = trend_late(metrics[tag])
            scale = abs(r["best_val"]) + abs(r["final_val"]) + 1e-6
            trend = trend_classify(slope, scale)
            print(f"  {tag:<44} {r['best_val']:>9.3f} {r['final_val']:>9.3f} "
                  f"{r['peak_fraction']*100:>6.0f}% {r['drop_pct']:>7.1f}% {trend:>8}")

    # Regression check
    regressions = []
    for tag in ["eval/episode_reward", "eval/episode_reward/tracking_lin_vel",
                "eval/episode_reward/tracking_ang_vel", "eval/episode_reward/imitation",
                "eval/episode_reward/alive", "eval/avg_episode_length"]:
        if tag in metrics:
            r = best_and_final(metrics[tag])
            if r["drop_pct"] > 5 and r["peak_fraction"] < 0.95:
                regressions.append((tag, r))
    if regressions:
        print(f"\n  REGRESSIONS (peak before final, drop > 5%):")
        for tag, r in regressions:
            print(f"    {tag}: best {r['best_val']:.2f} @ step {r['best_step']:,}, "
                  f"final {r['final_val']:.2f} (-{r['drop_pct']:.1f}%)")

    # Best checkpoint recommendation
    reward_vals = metrics.get("eval/episode_reward", [])
    if reward_vals:
        best_reward_step = best_and_final(reward_vals)["best_step"]
        print(f"\n  Best checkpoint (by reward): step {best_reward_step:,}")

    return metrics


# ===== Checkpoint Sweep =====

def step_from_name(p):
    try:
        return int(Path(p).stem.split("_")[-1])
    except ValueError:
        return -1


def find_best_checkpoint(run_dir, skip_early=50_000_000, env="joystick"):
    onnx_files = sorted(
        [f for f in glob.glob(str(Path(run_dir) / "*.onnx"))
         if step_from_name(f) >= skip_early],
        key=step_from_name,
    )
    if not onnx_files:
        print(f"  No ONNX files above step {skip_early:,}")
        return None, None

    print(f"\n  Found {len(onnx_files)} checkpoints above step {skip_early:,}")

    if env == "standing":
        from analysis.quick_standing_eval import assess_standing
        best_path = None
        best_headline = -1
        for f in onnx_files:
            step = step_from_name(f)
            try:
                qm = assess_standing(f)
                headline = qm.get("standing/HEADLINE", 0)
                marker = ""
                if headline > best_headline:
                    best_headline = headline
                    best_path = f
                    marker = " <-- BEST"
                print(f"    step {step:>11,}  HEADLINE={headline:.3f}  "
                      f"still={qm.get('standing/stillness_score', 0):.3f}  "
                      f"push={qm.get('standing/push_score', 0):.3f}{marker}")
            except Exception as exc:
                print(f"    step {step:>11,}  FAILED: {exc}")
    else:
        from analysis.quick_walking_eval import assess_walking
        best_path = None
        best_headline = -1
        for f in onnx_files:
            step = step_from_name(f)
            try:
                qm = assess_walking(f)
                headline = qm.get("walking/HEADLINE", 0)
                marker = ""
                if headline > best_headline:
                    best_headline = headline
                    best_path = f
                    marker = " <-- BEST"
                print(f"    step {step:>11,}  HEADLINE={headline:.3f}  "
                      f"walk={qm.get('walking/WALKING_SCORE', 0):.3f}  "
                      f"resp={qm.get('walking/responsiveness_avg', 0):.3f}{marker}")
            except Exception as exc:
                print(f"    step {step:>11,}  FAILED: {exc}")

    if best_path:
        print(f"\n  Best checkpoint: {best_path} (HEADLINE={best_headline:.3f})")
    return best_path, best_headline


# ===== Standing Full Evaluation =====

def evaluate_standing(onnx_path, model_path, reference_data):
    """Full standing evaluation: stillness + base/body/angular push recovery."""
    from analysis.quick_standing_eval import get_assessor, _PUSH_BODIES
    from playground.common.onnx_infer import OnnxInfer

    metrics = {}
    ev = get_assessor(model_path, reference_data)
    policy = OnnxInfer(onnx_path, awd=True)

    obs_size = policy.ort_session.get_inputs()[0].shape[1]
    base_obs_size = 107
    snap_size = 6 + 2 * ev.num_dofs
    if obs_size > base_obs_size:
        ev.HISTORY_LEN = (obs_size - base_obs_size) // snap_size
    else:
        ev.HISTORY_LEN = 0
    ev._reset_action_state()

    # Standstill test
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

    step_score = float(np.exp(-(step_count / 5.0) ** 2))
    drift_score = float(np.exp(-(drift / 0.05) ** 2))
    sway_score = float(np.exp(-(sway / 0.3) ** 2))
    metrics["standing/stillness_score"] = round((step_score + drift_score + sway_score) / 3, 5)

    # Base velocity push recovery: extended range
    all_mags = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    push_survival = {}
    for mag in all_mags:
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

    push_score = (push_survival[0.4] * 0.4 + push_survival[0.6] * 0.4 + push_survival[0.8] * 0.2)
    metrics["standing/push_score"] = round(push_score, 5)

    # Body push recovery: force applied to individual body parts
    body_survived = 0
    body_tested = 0
    for body in _PUSH_BODIES:
        for mag in [0.4, 0.6, 0.8]:
            for direction, label in [(0, "front"), (np.pi, "back"), (np.pi/2, "left"), (-np.pi/2, "right")]:
                ep = ev.run_standing(policy, 400, push_at=100, push_mag=mag,
                                     push_dir=direction, push_body=body)
                body_tested += 1
                if ep["steps"] >= 399:
                    body_survived += 1
                metrics[f"standing/body_{body}_{mag}_{label}"] = float(ep["steps"] >= 399)
    metrics["standing/body_push_rate"] = round(body_survived / body_tested, 5)

    # Angular perturbation recovery
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
            sum(1 for l in ["front","back","left","right"]
                if metrics.get(f"standing/angular_{ang_mag}_{l}", 0) >= 0.5) / 4, 5)
    metrics["standing/angular_push_rate"] = round(ang_survived / ang_tested, 5)

    metrics["standing/HEADLINE"] = round(
        0.4 * metrics["standing/stillness_score"]
        + 0.3 * metrics["standing/push_score"]
        + 0.15 * metrics["standing/body_push_rate"]
        + 0.15 * metrics["standing/angular_push_rate"], 5
    )

    return metrics


def print_standing_verdict(results):
    """Print verdict for a standing policy evaluation."""
    m = results if not isinstance(results, dict) or "metadata" not in results else results
    metrics = m.get("standing_metrics", m)

    print("\n" + "=" * 70)
    print("  STANDING POLICY VERDICT")
    print("=" * 70)

    print("\n  STILLNESS")
    step_count = metrics.get("standing/step_count", 0)
    drift = metrics.get("standing/drift_m", 0)
    sway = metrics.get("standing/sway_rate", 0)
    air_ratio = metrics.get("standing/air_time_ratio", 0)
    survived = metrics.get("standing/survived", 0)
    stillness = metrics.get("standing/stillness_score", 0)

    def grade_low(val, good, ok):
        if val <= good: return "GOOD"
        if val <= ok: return "OK"
        return "FAIL"

    def grade_high(val, good, ok):
        if val >= good: return "GOOD"
        if val >= ok: return "OK"
        return "FAIL"

    print(f"    Survived 10s:        {'PASS' if survived else 'FAIL'}")
    print(f"    Step count:          {grade_low(step_count, 3, 10):<5} {step_count:.0f} (target < 3)")
    print(f"    Drift:               {grade_low(drift, 0.03, 0.1):<5} {drift:.4f}m (target < 0.03m)")
    print(f"    Sway rate:           {grade_low(sway, 0.2, 0.5):<5} {sway:.3f} rad/s")
    print(f"    Air time ratio:      {grade_low(air_ratio, 0.05, 0.2):<5} {air_ratio:.3f}")
    print(f"    Stillness score:     {grade_high(stillness, 0.8, 0.5):<5} {stillness:.3f}")

    print("\n  PUSH RECOVERY")
    for mag in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        rate_key = f"standing/push_{mag}_rate"
        if rate_key not in metrics:
            continue
        rate = metrics[rate_key]
        dirs = []
        for label in ["front", "back", "left", "right"]:
            k = f"standing/push_{mag}_{label}"
            if k in metrics:
                dirs.append((label, metrics[k]))
        fails = [d for d, v in dirs if v < 0.5]
        grade = "PASS" if rate >= 1.0 else "PARTIAL" if rate > 0 else "FAIL"
        fail_str = f"  ({', '.join(fails)} FAIL)" if fails else ""
        print(f"    {mag} m/s:            {grade:<8} {rate*100:.0f}%{fail_str}")

    push_score = metrics.get("standing/push_score", 0)
    print(f"\n    Push score:          {grade_high(push_score, 0.8, 0.5):<5} {push_score:.3f}")

    # Body push recovery
    body_rate = metrics.get("standing/body_push_rate")
    if body_rate is not None:
        print(f"\n  BODY PUSH RECOVERY")
        print(f"    Overall:             {grade_high(body_rate, 0.8, 0.5):<5} {body_rate*100:.0f}%")

    # Angular perturbation recovery
    ang_rate = metrics.get("standing/angular_push_rate")
    if ang_rate is not None:
        print(f"\n  ANGULAR PERTURBATION RECOVERY")
        for ang_mag in [1.0, 2.0, 3.0]:
            rate_key = f"standing/angular_{ang_mag}_rate"
            if rate_key in metrics:
                rate = metrics[rate_key]
                grade = "PASS" if rate >= 1.0 else "PARTIAL" if rate > 0 else "FAIL"
                print(f"    {ang_mag} rad/s + 0.5 m/s: {grade:<8} {rate*100:.0f}%")
        print(f"    Overall:             {grade_high(ang_rate, 0.8, 0.5):<5} {ang_rate*100:.0f}%")

    headline = metrics.get("standing/HEADLINE", 0)
    print(f"\n    HEADLINE:            {grade_high(headline, 0.8, 0.5):<5} {headline:.3f}")
    print("=" * 70)


# ===== Debug Traces =====

def run_debug_traces(ev):
    from analysis.evaluate_policy import HeadlessEvaluator

    print("\n  TEMPORAL VELOCITY TRACES")
    print("  " + "-" * 60)
    traces = [
        ("forward",       [0.1, 0.0, 0.0, 0,0,0,0]),
        ("backward",      [-0.1, 0.0, 0.0, 0,0,0,0]),
        ("strafe_left",   [0.0, 0.15, 0.0, 0,0,0,0]),
        ("turn_left",     [0.0, 0.0, 0.8, 0,0,0,0]),
    ]
    for label, cmd in traces:
        ep = ev.run_episode(cmd, max_steps=500)
        n = ep["steps"]
        vx = ep["lin_vel_local"][:, 0]
        vy = ep["lin_vel_local"][:, 1]
        wz = ep["ang_vel"][:, 2]
        print(f"\n    {label}  cmd={cmd[:3]}  steps={n}")
        for wlabel, lo, hi in [("0-50", 0, 50), ("50-200", 50, 200), ("200-500", 200, 500)]:
            if lo >= n: continue
            hi = min(hi, n)
            print(f"      {wlabel:<8}  vx={np.mean(vx[lo:hi]):+.4f}  vy={np.mean(vy[lo:hi]):+.4f}  wz={np.mean(wz[lo:hi]):+.4f}")

    print("\n  PUSH RECOVERY")
    print("  " + "-" * 60)
    for name, mag, direction in [
        ("front_0.4", 0.4, 0.0), ("front_0.6", 0.6, 0.0), ("front_0.8", 0.8, 0.0),
        ("back_0.4", 0.4, np.pi), ("back_0.6", 0.6, np.pi),
        ("side_0.4", 0.4, np.pi/2), ("side_0.6", 0.6, np.pi/2),
    ]:
        ep = ev.run_episode([0, 0, 0, 0, 0, 0, 0], max_steps=500, pushes=[(100, mag, direction)])
        survived = ep["steps"] >= 499
        post = ep["steps"] - 100 if ep["steps"] > 100 else 0
        print(f"    {name:<14}  survived={survived}  post_push_steps={post}")


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(description="Comprehensive training run analysis")
    parser.add_argument("run_dir", help="Training output directory")
    parser.add_argument("--env", type=str, default="joystick", choices=["joystick", "standing"],
                        help="Environment type (joystick=walking, standing=standing)")
    parser.add_argument("--skip_early", type=int, default=50_000_000)
    parser.add_argument("--skip_eval", action="store_true", help="Skip full policy evaluation")
    parser.add_argument("--debug", action="store_true", help="Include temporal debug traces")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="Save full eval JSON to this path")
    parser.add_argument("--compare", nargs="*", default=[], help="Additional run dirs to compare against")
    args = parser.parse_args()

    run_name = Path(args.run_dir).name
    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS: {run_name}")
    print(f"{'=' * 80}")

    # 1. Training dynamics
    print(f"\n{'=' * 80}")
    print(f"  TRAINING DYNAMICS")
    print(f"{'=' * 80}")
    analyse_training(args.run_dir)

    # 2. Find best checkpoint
    print(f"\n{'=' * 80}")
    print(f"  CHECKPOINT SWEEP ({args.env})")
    print(f"{'=' * 80}")
    best_onnx, best_headline = find_best_checkpoint(args.run_dir, args.skip_early, env=args.env)

    if best_onnx and not args.skip_eval:
        # 3. Full evaluation of best checkpoint
        print(f"\n{'=' * 80}")
        print(f"  FULL EVALUATION: {Path(best_onnx).name}")
        print(f"{'=' * 80}")

        model_path = "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
        reference_data = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"

        start = time.time()

        if args.env == "standing":
            standing_metrics = evaluate_standing(best_onnx, model_path, reference_data)
            elapsed = time.time() - start

            results = {
                "standing_metrics": standing_metrics,
                "metadata": {
                    "onnx_path": best_onnx,
                    "run_dir": args.run_dir,
                    "env": "standing",
                    "evaluation_time_seconds": round(elapsed, 1),
                },
            }

            out_path = args.output or f"analysis/results/{run_name}_analysis.json"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Results saved to {out_path} ({elapsed:.0f}s)")

            print_standing_verdict(results)
        else:
            from analysis.evaluate_policy import HeadlessEvaluator, print_verdict
            ev = HeadlessEvaluator(model_path, reference_data, best_onnx)

            results = ev.evaluate(num_episodes=args.num_episodes)
            elapsed = time.time() - start

            results["metadata"] = {
                "onnx_path": best_onnx,
                "run_dir": args.run_dir,
                "env": "joystick",
                "num_episodes": args.num_episodes,
                "evaluation_time_seconds": round(elapsed, 1),
            }

            out_path = args.output or f"analysis/results/{run_name}_analysis.json"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Results saved to {out_path} ({elapsed:.0f}s)")

            print_verdict(results)

            if args.debug:
                print(f"\n{'=' * 80}")
                print(f"  DEBUG TRACES")
                print(f"{'=' * 80}")
                run_debug_traces(ev)

    # 5. Cross-run comparison
    if args.compare:
        print(f"\n{'=' * 80}")
        print(f"  CROSS-RUN COMPARISON")
        print(f"{'=' * 80}")
        from analysis.compare_policies import compare_section

        all_jsons = []
        names = []
        primary_json = args.output or f"analysis/results/{run_name}_analysis.json"
        if Path(primary_json).exists():
            with open(primary_json) as f:
                all_jsons.append(json.load(f))
            names.append(run_name)

        for comp_dir in args.compare:
            comp_name = Path(comp_dir).name
            comp_json = f"analysis/results/{comp_name}_analysis.json"
            if Path(comp_json).exists():
                with open(comp_json) as f:
                    all_jsons.append(json.load(f))
                names.append(comp_name)
            else:
                print(f"  No analysis JSON for {comp_name} - run analyse_run.py on it first")

        if len(all_jsons) >= 2:
            compare_section(all_jsons, names, "aggregate_locomotion", 0)


if __name__ == "__main__":
    main()
