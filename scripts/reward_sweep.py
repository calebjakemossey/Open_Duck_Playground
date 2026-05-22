"""Stage 1 Latin Hypercube screening driver.

Generates 15 LHS samples in the 7-D reward config space, runs a 25M cold-start
training for each, and appends the result to docs/18_reward_tuning_report.md.

Run from Open_Duck_Playground/.
"""
import os
import re
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from tensorboard.backend.event_processing import event_accumulator


PLAYGROUND = Path("/home/lakieb/Documents/open_duck_mini_research/Open_Duck_Playground")
JOYSTICK = PLAYGROUND / "playground/open_duck_mini_v2/joystick.py"
RUNNER = PLAYGROUND / "playground/open_duck_mini_v2/runner.py"
REPORT = Path("/home/lakieb/Documents/open_duck_mini_research/docs/18_reward_tuning_report.md")
CHECKPOINTS_ROOT = PLAYGROUND / "checkpoints"
NUM_SAMPLES = 15
STEPS_PER_RUN = 50_000_000  # 25M is too early - walking emergence happens between 30-50M

# 7-D search box. Sampling scale flagged for log-space dims.
BOUNDS = [
    ("tracking_sigma",    0.05, 0.5,  "log"),
    ("alive",             1.0,  10.0, "linear"),
    ("tracking_lin_vel",  1.0,  6.0,  "linear"),
    ("tracking_ang_vel",  1.0,  4.0,  "linear"),
    ("action_rate",      -0.5, -0.05, "logmag"),  # negative, log-spaced magnitude
    ("stand_still",      -2.0, -0.2,  "linear"),
    ("imitation",         0.5,  2.0,  "linear"),
]


def lhs_samples(n: int, seed: int = 0) -> list[dict]:
    sampler = qmc.LatinHypercube(d=len(BOUNDS), seed=seed)
    u = sampler.random(n)  # shape (n, d), each col uniform in [0, 1]
    configs = []
    for row in u:
        cfg = {}
        for i, (name, lo, hi, scale) in enumerate(BOUNDS):
            val_u = row[i]
            if scale == "linear":
                v = lo + val_u * (hi - lo)
            elif scale == "log":
                v = lo * (hi / lo) ** val_u
            elif scale == "logmag":
                amag_lo, amag_hi = abs(hi), abs(lo)  # smaller magnitude is "hi" here
                m = amag_lo * (amag_hi / amag_lo) ** val_u
                v = -m
            else:
                raise ValueError(scale)
            cfg[name] = float(round(v, 4))
        configs.append(cfg)
    return configs


def patch_joystick(cfg: dict):
    """Edit joystick.py in-place with the new config values."""
    src = JOYSTICK.read_text()
    patches = [
        (r"tracking_lin_vel=-?[0-9.eE+]+",  f"tracking_lin_vel={cfg['tracking_lin_vel']}"),
        (r"tracking_ang_vel=-?[0-9.eE+]+",  f"tracking_ang_vel={cfg['tracking_ang_vel']}"),
        (r"action_rate=-?[0-9.eE+]+",       f"action_rate={cfg['action_rate']}"),
        (r"stand_still=-?[0-9.eE+]+",       f"stand_still={cfg['stand_still']}"),
        (r"alive=-?[0-9.eE+]+",             f"alive={cfg['alive']}"),
        (r"imitation=-?[0-9.eE+]+",         f"imitation={cfg['imitation']}"),
        (r"tracking_sigma=-?[0-9.eE+]+",    f"tracking_sigma={cfg['tracking_sigma']}"),
    ]
    for pat, repl in patches:
        new_src, n = re.subn(pat, repl, src, count=1)
        if n != 1:
            raise RuntimeError(f"Failed to apply patch {pat!r} - matched {n} times")
        src = new_src
    JOYSTICK.write_text(src)


def read_final_metrics(run_dir: Path) -> dict:
    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    tags = ea.Tags()["scalars"]

    def last(k):
        if k not in tags:
            return None
        evs = ea.Scalars(k)
        return evs[-1].value if evs else None

    def last_step(k):
        if k not in tags:
            return None
        evs = ea.Scalars(k)
        return evs[-1].step if evs else None

    return {
        "step": last_step("walking/HEADLINE"),
        "HEADLINE": last("walking/HEADLINE"),
        "cold": last("walking/cold_start_avg"),
        "resp": last("walking/responsiveness_avg"),
        "reward": last("eval/episode_reward"),
    }


def append_report_row(idx: int, cfg: dict, metrics: dict, run_dir: Path):
    """Insert one Markdown table row into the Stage 1 section of the report."""
    md = REPORT.read_text()
    sentinel_header = "| # | sigma | alive | lin | ang | act_rate | stand | imit | step | HEADLINE | cold | resp | vx_tr | reward |"
    sentinel_sep = "|---|-------|-------|-----|-----|----------|-------|------|------|----------|------|------|-------|--------|"

    row = (
        f"| {idx} | {cfg['tracking_sigma']:.3f} | {cfg['alive']:.2f} | "
        f"{cfg['tracking_lin_vel']:.2f} | {cfg['tracking_ang_vel']:.2f} | "
        f"{cfg['action_rate']:.3f} | {cfg['stand_still']:.2f} | {cfg['imitation']:.2f} | "
        f"{(metrics.get('step') or 0):>9} | "
        f"{(metrics.get('HEADLINE') or 0):.3f} | "
        f"{(metrics.get('cold') or 0):.3f} | "
        f"{(metrics.get('resp') or 0):.3f} | "
        f"- | "
        f"{(metrics.get('reward') or 0):.1f} |"
    )
    # Insert row right after the separator line.
    if sentinel_sep not in md:
        raise RuntimeError("report sentinel separator missing")
    md = md.replace(sentinel_sep, sentinel_sep + "\n" + row, 1)
    REPORT.write_text(md)


def update_report_observation(text: str):
    md = REPORT.read_text()
    needle = "### Stage 1 observations"
    if needle not in md:
        return
    md = md.replace(needle, needle + "\n" + text + "\n", 1)
    REPORT.write_text(md)


def run_one(idx: int, cfg: dict):
    print(f"\n=== Run {idx}/{NUM_SAMPLES} ===")
    print(json.dumps(cfg, indent=2))
    patch_joystick(cfg)

    out_dir = CHECKPOINTS_ROOT / f"lhs_{idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = f"/tmp/lhs_{idx:02d}.log"

    cmd = [
        "uv", "run", "python", str(RUNNER),
        "--output_dir", str(out_dir.relative_to(PLAYGROUND)),
        "--num_timesteps", str(STEPS_PER_RUN),
    ]
    t0 = time.time()
    with open(log_path, "w") as logf:
        subprocess.run(cmd, cwd=PLAYGROUND, stdout=logf, stderr=subprocess.STDOUT, check=False)
    dt = time.time() - t0

    # Find the events file directory (out_dir itself)
    metrics = read_final_metrics(out_dir)
    metrics["dt_sec"] = dt
    append_report_row(idx, cfg, metrics, out_dir)
    # Save config alongside checkpoints
    (out_dir / "lhs_config.json").write_text(json.dumps({"cfg": cfg, "metrics": metrics}, indent=2))
    print(f"Done in {dt/60:.1f} min. Metrics: {metrics}")


def main():
    start_from = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    configs = lhs_samples(NUM_SAMPLES, seed=42)
    (CHECKPOINTS_ROOT / "lhs_configs.json").write_text(json.dumps(configs, indent=2))
    print(f"Generated {len(configs)} LHS samples. Starting from run {start_from}.")

    for i, cfg in enumerate(configs):
        if i < start_from:
            continue
        run_one(i, cfg)

    print("\nLHS screening complete.")


if __name__ == "__main__":
    main()
