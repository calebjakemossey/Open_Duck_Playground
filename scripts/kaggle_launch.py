"""
Kaggle parallel training launcher for Open Duck Mini V2.

Usage
-----
    # Smoke test (1M steps):
    python scripts/kaggle_launch.py --run-name smoke-test --timesteps 1000000

    # Full run with custom reward weights:
    python scripts/kaggle_launch.py \
        --run-name "tracking-2.5-imitation-20" \
        --timesteps 50000000 \
        --rewards '{"tracking_lin_vel": 2.5, "imitation": 20.0}'

    # Push dataset update first (after local code changes):
    python scripts/kaggle_launch.py --update-dataset

    # Poll until done and download results:
    python scripts/kaggle_launch.py --run-name smoke-test --wait --download

How it works
------------
1. Writes a kernel-metadata.json into /tmp with the run config.
2. Pushes the kernel (which points at the lakieb/open-duck-playground-src dataset).
3. Optionally polls for completion and downloads results.

Prerequisites
-------------
- kaggle CLI installed: `uv tool install kaggle`
- ~/.kaggle/access_token containing the KGAT token
- Dataset already uploaded: `python scripts/kaggle_launch.py --update-dataset`
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
KAGGLE_USERNAME = "twixters"
DATASET_ID = f"{KAGGLE_USERNAME}/open-duck-playground-src"
NOTEBOOK_SCRIPT = Path(__file__).parent / "kaggle_notebook.py"
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_STAGING = PROJECT_ROOT / ".kaggle_dataset"


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def update_dataset() -> None:
    """Re-sync playground source and pyproject.toml into the dataset staging dir,
    then push to Kaggle."""
    print("Syncing dataset files...")
    DATASET_STAGING.mkdir(exist_ok=True)

    # Sync playground source (exclude __pycache__ and .pyc)
    run([
        "rsync", "-a", "--delete",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        str(PROJECT_ROOT / "playground") + "/",
        str(DATASET_STAGING / "playground") + "/",
    ])
    shutil.copy(PROJECT_ROOT / "pyproject.toml", DATASET_STAGING / "pyproject.toml")

    # Write/refresh dataset metadata
    meta = {
        "title": "open-duck-playground-src",
        "id": DATASET_ID,
        "licenses": [{"name": "Apache 2.0"}],
    }
    (DATASET_STAGING / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    # Check if dataset already exists
    result = run(
        ["kaggle", "datasets", "list", "--user", KAGGLE_USERNAME, "--search", "open-duck-playground-src"],
        capture=True,
        check=False,
    )
    dataset_exists = "open-duck-playground-src" in (result.stdout or "")

    if dataset_exists:
        print("Updating existing dataset...")
        run(["kaggle", "datasets", "version", "-p", str(DATASET_STAGING), "-m", "auto-update", "--dir-mode", "zip"])
    else:
        print("Creating new dataset...")
        run(["kaggle", "datasets", "create", "-p", str(DATASET_STAGING)])

    print("Dataset upload complete.")


def sanitise_run_name(name: str) -> str:
    """Kaggle kernel slugs: lowercase alphanumeric + hyphens, max 50 chars."""
    slug = name.lower().replace("_", "-").replace(" ", "-")
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    return slug[:50]


def launch_kernel(
    run_name: str,
    num_timesteps: int,
    reward_overrides: dict | None,
    env_name: str,
    task_name: str,
    use_tpu: bool = False,
) -> str:
    """Push a kernel to Kaggle and return the kernel ref (username/slug)."""
    slug = sanitise_run_name(run_name)
    kernel_ref = f"{KAGGLE_USERNAME}/{slug}"

    env_vars: dict[str, str] = {
        "NUM_TIMESTEPS": str(num_timesteps),
        "ENV_NAME": env_name,
        "TASK_NAME": task_name,
    }
    if use_tpu:
        env_vars["USE_TPU"] = "1"
    if reward_overrides:
        env_vars["REWARD_CONFIG"] = json.dumps(reward_overrides)

    notebook_src = NOTEBOOK_SCRIPT.read_text()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        meta = {
            "id": kernel_ref,
            "title": run_name[:50],
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": not use_tpu,
            "enable_tpu": use_tpu,
            "enable_internet": True,
            "dataset_sources": [DATASET_ID],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

        env_block = "\n".join(
            f'os.environ.setdefault({k!r}, {v!r})'
            for k, v in env_vars.items()
        )
        full_src = f"import os\n{env_block}\n\n{notebook_src}"
        (tmp / "kernel.py").write_text(full_src)

        accel = "TPU" if use_tpu else "GPU"
        print(f"Pushing kernel '{kernel_ref}' ({accel})...")
        run(["kaggle", "kernels", "push", "-p", str(tmp)])

    print(f"Kernel pushed: https://www.kaggle.com/code/{kernel_ref}")
    return kernel_ref


def wait_for_kernel(kernel_ref: str, poll_interval: int = 60) -> str:
    """Poll until the kernel finishes. Returns final status string."""
    print(f"Polling kernel {kernel_ref} every {poll_interval}s...")
    while True:
        result = run(
            ["kaggle", "kernels", "status", kernel_ref],
            capture=True,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        print(f"  Status: {output.strip()}")

        # Kaggle CLI prints something like: "lakieb/smoke-test has status complete"
        if "complete" in output.lower():
            return "complete"
        if "error" in output.lower() or "cancel" in output.lower():
            return "error"

        time.sleep(poll_interval)


def download_output(kernel_ref: str, dest_dir: str) -> None:
    """Download kernel output files to dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    run(["kaggle", "kernels", "output", kernel_ref, "-p", dest_dir])
    print(f"Downloaded outputs to {dest_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Open Duck Mini V2 training on Kaggle")
    parser.add_argument("--run-name", default="odm-v2-run", help="Human-readable run name (becomes the kernel slug)")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Number of training timesteps")
    parser.add_argument("--rewards", type=str, default=None, help='JSON dict of reward weight overrides, e.g. \'{"imitation": 20.0}\'')
    parser.add_argument("--env", default="joystick", help="Environment name (joystick or standing)")
    parser.add_argument("--task", default="flat_terrain", help="Task name (flat_terrain, rough_terrain, ...)")
    parser.add_argument("--update-dataset", action="store_true", help="Sync and upload the dataset before launching")
    parser.add_argument("--wait", action="store_true", help="Poll until the kernel completes")
    parser.add_argument("--download", action="store_true", help="Download outputs after kernel completes (implies --wait)")
    parser.add_argument("--download-dir", default=None, help="Where to save downloaded outputs (default: ./kaggle_outputs/<run-name>)")
    parser.add_argument("--tpu", action="store_true", help="Use TPU instead of GPU (Kaggle TPU V5e)")
    parser.add_argument("--no-launch", action="store_true", help="Only update dataset, do not launch a kernel")

    args = parser.parse_args()

    if args.download:
        args.wait = True

    if args.update_dataset or args.no_launch:
        update_dataset()

    if args.no_launch:
        print("Dataset updated. Exiting (--no-launch).")
        return

    reward_overrides = None
    if args.rewards:
        try:
            reward_overrides = json.loads(args.rewards)
        except json.JSONDecodeError as e:
            print(f"ERROR: --rewards must be valid JSON: {e}", file=sys.stderr)
            sys.exit(1)

    kernel_ref = launch_kernel(
        run_name=args.run_name,
        num_timesteps=args.timesteps,
        reward_overrides=reward_overrides,
        env_name=args.env,
        task_name=args.task,
        use_tpu=args.tpu,
    )

    if args.wait:
        status = wait_for_kernel(kernel_ref)
        print(f"Kernel finished with status: {status}")
        if args.download:
            dest = args.download_dir or str(PROJECT_ROOT / "kaggle_outputs" / sanitise_run_name(args.run_name))
            download_output(kernel_ref, dest)
    else:
        print(f"Kernel launched. Check status with:")
        print(f"  kaggle kernels status {kernel_ref}")
        print(f"Or download outputs later with:")
        print(f"  kaggle kernels output {kernel_ref} -p ./kaggle_outputs/{sanitise_run_name(args.run_name)}")


if __name__ == "__main__":
    main()
