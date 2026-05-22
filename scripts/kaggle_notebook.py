"""
Kaggle notebook script for Open Duck Mini V2 RL training.

This script is uploaded as a Kaggle kernel and runs training on GPU or TPU.
The dataset 'lakieb/open-duck-playground-src' must be attached to the kernel
before running.

Reward weights are injected via the REWARD_CONFIG environment variable
(JSON string) or left at defaults defined in joystick.py.
"""

import os
import sys
import json
import subprocess
import shutil

# ── 1. Install dependencies ──────────────────────────────────────────────────
USE_TPU = os.environ.get("USE_TPU", "0") == "1"

if USE_TPU:
    JAX_DEP = "jax[tpu]>=0.5.0,<0.7"
else:
    JAX_DEP = "jax[cuda12]>=0.5.0,<0.7"

DEPS = [
    JAX_DEP,
    "jaxlib>=0.5.0,<0.7",
    "jaxtyping>=0.2.38",
    "playground==0.0.5",
    "tensorboardX>=2.6",
    "tf2onnx>=1.16.1",
    "tensorflow>=2.18.0",
    "orbax-checkpoint",
    "mediapy>=1.2.2",
]

print(f"Installing dependencies ({'TPU' if USE_TPU else 'GPU'} mode)...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet"] + DEPS,
    env={**os.environ, "PIP_PROGRESS_BAR": "off"},
)
print("Dependencies installed.")

# ── 2. Set up working directory ──────────────────────────────────────────────
DATASET_ROOT = "/kaggle/input/open-duck-playground-src"
WORK_DIR = "/kaggle/working"
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Debug: show what Kaggle mounted
print(f"Dataset root contents ({DATASET_ROOT}):")
if os.path.exists(DATASET_ROOT):
    for item in os.listdir(DATASET_ROOT):
        full = os.path.join(DATASET_ROOT, item)
        kind = "dir" if os.path.isdir(full) else "file"
        print(f"  {item} ({kind})")
else:
    print(f"  [NOT FOUND] - listing /kaggle/input/ instead:")
    for item in os.listdir("/kaggle/input"):
        print(f"  {item}")

# Find playground source - Kaggle may nest files differently
SRC_PLAYGROUND = None
for candidate in [
    os.path.join(DATASET_ROOT, "playground"),
    os.path.join(DATASET_ROOT, "open-duck-playground-src", "playground"),
]:
    if os.path.isdir(candidate):
        SRC_PLAYGROUND = candidate
        break

# Fallback: search recursively for the playground dir
if SRC_PLAYGROUND is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if "playground" in dirs:
            candidate = os.path.join(root, "playground")
            if os.path.exists(os.path.join(candidate, "open_duck_mini_v2")):
                SRC_PLAYGROUND = candidate
                break

if SRC_PLAYGROUND is None:
    raise FileNotFoundError(
        f"Could not find playground source dir under {DATASET_ROOT} or /kaggle/input/. "
        "Check dataset upload."
    )

print(f"Found playground source at: {SRC_PLAYGROUND}")

DEST_PLAYGROUND = os.path.join(WORK_DIR, "playground")
if not os.path.exists(DEST_PLAYGROUND):
    shutil.copytree(SRC_PLAYGROUND, DEST_PLAYGROUND)
    print(f"Copied playground source to {DEST_PLAYGROUND}")

# Also copy pyproject.toml if present
for name in ["pyproject.toml"]:
    for candidate in [os.path.join(DATASET_ROOT, name), os.path.join(os.path.dirname(SRC_PLAYGROUND), name)]:
        if os.path.isfile(candidate):
            shutil.copy(candidate, os.path.join(WORK_DIR, name))
            print(f"Copied {name}")
            break

# Make the playground package importable
sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)

# ── 3. Apply reward config overrides (injected by kaggle_launch.py) ──────────
REWARD_CONFIG_JSON = os.environ.get("REWARD_CONFIG", "")
REWARD_OVERRIDES = {}
if REWARD_CONFIG_JSON:
    try:
        REWARD_OVERRIDES = json.loads(REWARD_CONFIG_JSON)
        print(f"Reward overrides: {REWARD_OVERRIDES}")
    except json.JSONDecodeError as e:
        print(f"[WARN] Could not parse REWARD_CONFIG env var: {e}")

# ── 4. Patch joystick.py reward scales if overrides provided ────────────────
if REWARD_OVERRIDES:
    import re
    joystick_path = os.path.join(WORK_DIR, "playground", "open_duck_mini_v2", "joystick.py")
    with open(joystick_path) as f:
        src = f.read()
    for key, value in REWARD_OVERRIDES.items():
        # Replace "key=<number>" in the reward scales config block
        pattern = rf"({re.escape(key)}\s*=\s*)[-\d.e+]+"
        replacement = rf"\g<1>{value}"
        new_src = re.sub(pattern, replacement, src)
        if new_src == src:
            print(f"[WARN] Reward key '{key}' not found or unchanged in joystick.py")
        else:
            print(f"  Patched {key} -> {value}")
            src = new_src
    with open(joystick_path, "w") as f:
        f.write(src)

# ── 5. JAX compilation cache ─────────────────────────────────────────────────
import jax
CACHE_DIR = os.path.join(WORK_DIR, ".tmp", "jax_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
os.environ["JAX_COMPILATION_CACHE_DIR"] = CACHE_DIR

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

# ── 6. Training config ───────────────────────────────────────────────────────
NUM_TIMESTEPS = int(os.environ.get("NUM_TIMESTEPS", "1000000"))
ENV_NAME = os.environ.get("ENV_NAME", "joystick")
TASK_NAME = os.environ.get("TASK_NAME", "flat_terrain")

print(f"Training config: env={ENV_NAME} task={TASK_NAME} num_timesteps={NUM_TIMESTEPS}")
print(f"Output dir: {OUTPUT_DIR}")

# ── 7. Run training ───────────────────────────────────────────────────────────
import argparse
import functools

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import joystick, standing


class OpenDuckMiniV2Runner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (joystick, joystick.Joystick),
            "standing": (standing, standing.Standing),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")
        self.env_file = available_envs[args.env]
        self.env_config = self.env_file[0].default_config()
        self.env = self.env_file[1](task=args.task)
        self.eval_env = self.env_file[1](task=args.task)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(self.env.observation_size["state"][0])
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")


args = argparse.Namespace(
    output_dir=OUTPUT_DIR,
    num_timesteps=NUM_TIMESTEPS,
    env=ENV_NAME,
    task=TASK_NAME,
    restore_checkpoint_path=None,
)

runner = OpenDuckMiniV2Runner(args)
runner.train()

print("Training complete.")
print(f"Outputs in {OUTPUT_DIR}:")
for f in os.listdir(OUTPUT_DIR):
    print(f"  {f}")
