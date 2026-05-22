"""
Compare evaluation results from multiple policies side-by-side.

Reads all metric keys from the JSON files themselves - no hardcoded test names.

Usage:
    uv run analysis/compare_policies.py results/A.json results/B.json
    uv run analysis/compare_policies.py results/A.json results/B.json --reference 0
"""

import argparse
import json
import sys
from pathlib import Path

LOWER_IS_BETTER = {
    "action_rate", "jerk", "hf_ratio", "orientation_cost", "mean_tilt_deg",
    "max_tilt_deg", "body_sway_rate", "body_roll_rate", "body_pitch_rate",
    "body_yaw_rate", "lateral_drift_std", "com_drift_total",
    "cost_of_transport", "stride_cv", "left_stride_cv", "right_stride_cv",
    "foot_slip_mean", "left_slip_mean", "right_slip_mean",
    "left_slip_max", "right_slip_max", "bang_bang_ratio",
    "mean_mechanical_power", "base_height_std",
    "max_tilt_after_push_deg",
}

HIGHLIGHT_KEYS = {
    "vx_tracking_score", "vy_tracking_score", "wz_tracking_score",
    "gait_symmetry_score", "survival_rate", "cost_of_transport",
    "stride_cv", "foot_slip_mean", "bang_bang_ratio", "steps",
    "mean_responsiveness",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(val, ref_val=None, higher_is_better=True):
    if val is None:
        return "N/A"
    s = f"{val:.4f}"
    if ref_val is not None:
        diff = val - ref_val
        pct = (diff / (abs(ref_val) + 1e-8)) * 100
        good = (diff > 0) == higher_is_better
        if abs(pct) < 1:
            marker = "~"
        elif good:
            marker = "BETTER"
        else:
            marker = "WORSE"
        arrow = "+" if diff > 0 else ""
        s += f" ({arrow}{pct:.1f}% {marker})"
    return s


def compare_section(results_list, names, section, ref_idx):
    datas = [r.get(section, {}) for r in results_list]
    if all(not d for d in datas):
        return

    all_keys = []
    seen = set()
    for d in datas:
        if isinstance(d, dict):
            for k in d:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

    if not all_keys:
        return

    numeric_keys = [k for k in all_keys if any(
        isinstance(d.get(k), (int, float)) for d in datas
    )]
    if not numeric_keys:
        return

    print(f"\n{'=' * 70}")
    print(f"  {section.upper().replace('_', ' ')}")
    print("=" * 70)

    col_w = max(22, max(len(n) for n in names) + 2)
    header = f"{'metric':<30}"
    for name in names:
        header += f"  {name:>{col_w}}"
    print(header)
    print("-" * (30 + (col_w + 2) * len(names)))

    for key in numeric_keys:
        hib = key not in LOWER_IS_BETTER
        ref_val = datas[ref_idx].get(key) if ref_idx is not None else None
        row = f"{key:<30}"
        for i, d in enumerate(datas):
            val = d.get(key)
            ref = ref_val if (i != ref_idx and ref_val is not None) else None
            cell = fmt(val, ref, hib)
            row += f"  {cell:>{col_w}}"
        highlight = " *" if key in HIGHLIGHT_KEYS else ""
        print(f"{row}{highlight}")


def compare_push(results_list, names, ref_idx):
    pushes = [r.get("push_recovery", {}) for r in results_list]
    if all(not p for p in pushes):
        return

    all_keys = sorted(set(k for p in pushes for k in p))
    if not all_keys:
        return

    print(f"\n{'=' * 70}")
    print("  PUSH RECOVERY")
    print("=" * 70)
    col_w = max(22, max(len(n) for n in names) + 2)
    header = f"{'push':<24}"
    for name in names:
        header += f"  {name:>{col_w}}"
    print(header)
    print("-" * (24 + (col_w + 2) * len(names)))

    for key in all_keys:
        row = f"{key:<24}"
        for i, p in enumerate(pushes):
            entry = p.get(key, {})
            sr = entry.get("survival_rate")
            if sr is not None:
                row += f"  {sr*100:>{col_w-1}.0f}%"
            else:
                row += f"  {'N/A':>{col_w}}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Compare policy evaluation results")
    parser.add_argument("results", nargs="+", help="JSON result files to compare")
    parser.add_argument("--reference", type=int, default=0)
    args = parser.parse_args()

    results_list = []
    names = []
    for path in args.results:
        p = Path(path)
        if not p.exists():
            print(f"Not found: {path}", file=sys.stderr)
            sys.exit(1)
        results_list.append(load(path))
        names.append(p.stem)

    print(f"Comparing {len(results_list)} policies (reference: {names[args.reference]})")

    test_sections = [k for k in results_list[0]
                     if k not in ("push_recovery", "responsiveness", "command_gating",
                                  "responsiveness_aggregate", "metadata", "aggregate_locomotion")]
    for section in test_sections:
        compare_section(results_list, names, section, args.reference)

    compare_section(results_list, names, "aggregate_locomotion", args.reference)
    compare_push(results_list, names, args.reference)
    compare_section(results_list, names, "responsiveness_aggregate", args.reference)


if __name__ == "__main__":
    main()
