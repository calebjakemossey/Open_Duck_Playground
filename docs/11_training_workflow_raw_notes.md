# Training Workflow Raw Notes

> **Superseded.** These were initial reading notes compiled before hands-on work began. The accurate, up-to-date references are:
> - `01_training_pipeline.md` - technical training pipeline reference
> - `WALKTHROUGH.md` - step-by-step training guide with validated commands
> - `03_reference_motions.md` - reference motion pipeline
>
> The notes below are preserved for historical context but may contain outdated file paths and line numbers.

# Open Duck Mini V2 - End-to-End Policy Training

Raw notes compiled from direct source reading. All file paths are absolute. Line numbers reference the state of the repo at time of reading.

---

## 1. Repository Layout

```
Open_Duck_Playground/
├── pyproject.toml                              # package manifest and dependencies
├── README.md                                   # top-level instructions
├── playground/
│   ├── common/
│   │   ├── runner.py                           # BaseRunner - PPO training loop
│   │   ├── export_onnx.py                      # ONNX conversion logic
│   │   ├── onnx_infer.py                       # ONNX inference wrapper (CPU)
│   │   ├── rewards.py                          # shared reward functions
│   │   ├── randomize.py                        # domain randomisation
│   │   ├── poly_reference_motion.py            # JAX reference motion (training)
│   │   ├── poly_reference_motion_numpy.py      # NumPy reference motion (inference)
│   │   └── utils.py                            # LowPassActionFilter
│   └── open_duck_mini_v2/
│       ├── runner.py                           # robot-specific entry point
│       ├── joystick.py                         # Joystick env (main walking task)
│       ├── standing.py                         # Standing env (head control task)
│       ├── base.py                             # OpenDuckMiniV2Env base class
│       ├── constants.py                        # scene paths, joint names, sensor names
│       ├── mujoco_infer.py                     # interactive MuJoCo visualisation
│       ├── mujoco_infer_base.py                # shared inference base class
│       ├── ref_motion_viewer.py                # visualise reference motion only
│       ├── data/
│       │   └── polynomial_coefficients.pkl     # reference motion data (required)
│       └── xmls/
│           ├── scene_flat_terrain.xml
│           ├── scene_flat_terrain_backlash.xml
│           ├── scene_rough_terrain_backlash.xml
│           ├── open_duck_mini_v2.xml
│           ├── open_duck_mini_v2_backlash.xml
│           └── assets/
```

---

## 2. Installation

### Tool: uv (not pip, not conda)

Source: `README.md` lines 6-9 and `pyproject.toml`.

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`uv` manages the virtual environment and dependencies automatically. There is no `conda` environment or `requirements.txt`. The `.venv` directory is already present in the repo (Python 3.12.2 via uv).

### Python version requirement

`pyproject.toml` line 6: `requires-python = ">=3.11"`

The `.venv/pyvenv.cfg` shows uv resolved to **Python 3.12.2**.

### Dependencies (from `pyproject.toml` lines 7-19)

```toml
dependencies = [
    "framesviewer>=1.0.2",
    "jax[cuda12]>=0.5.0",       # JAX with CUDA 12 support - GPU required for practical training
    "jaxlib>=0.5.0",
    "jaxtyping>=0.2.38",
    "matplotlib>=3.10.0",
    "mediapy>=1.2.2",
    "onnxruntime>=1.20.1",
    "playground>=0.0.3",        # mujoco_playground package (Google DeepMind)
    "pygame>=2.6.1",
    "tensorflow>=2.18.0",       # used only for ONNX export, not training
    "tf2onnx>=1.16.1",
]
```

Key observations:
- `jax[cuda12]` - training uses JAX/MJX on GPU. CUDA 12 is explicitly required.
- `tensorflow` and `tf2onnx` are only used in `export_onnx.py` during checkpoint callbacks - not during the RL training itself.
- `onnxruntime` is used at inference time only (CPU execution provider).
- `playground>=0.0.3` refers to the `mujoco_playground` package from Google DeepMind/kscale, not this local package.

### No explicit CUDA version documentation

There is no README section specifying CUDA driver version requirements beyond what JAX implies. JAX >=0.5.0 with `cuda12` extra implies CUDA 12.x and a compatible NVIDIA driver.

---

## 3. Starting a Training Run

### Primary command (from `README.md` line 97 - the documented "current win")

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```

`uv run` automatically activates the managed venv. Run from the repo root (`Open_Duck_Playground/`).

### All available CLI arguments

Source: `playground/open_duck_mini_v2/runner.py` lines 36-56.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--output_dir` | str | `"checkpoints"` | Directory for checkpoints and ONNX files |
| `--num_timesteps` | int | `150000000` | Total environment steps (150M default, 300M recommended) |
| `--env` | str | `"joystick"` | Environment class: `"joystick"` or `"standing"` |
| `--task` | str | `"flat_terrain"` | Scene XML to load (see task map below) |
| `--restore_checkpoint_path` | str | `None` | Path to an Orbax checkpoint directory to resume from |

### Available tasks (from `playground/open_duck_mini_v2/constants.py` lines 28-34)

| `--task` value | XML loaded |
|---|---|
| `flat_terrain` | `xmls/scene_flat_terrain.xml` |
| `rough_terrain` | `xmls/scene_rough_terrain.xml` |
| `flat_terrain_backlash` | `xmls/scene_flat_terrain_backlash.xml` |
| `rough_terrain_backlash` | `xmls/scene_rough_terrain_backlash.xml` |

The `backlash` variants include joint backlash modelling for better sim-to-real transfer.

### Example commands

```bash
# Minimal run (default 150M steps, flat terrain, no backlash)
uv run playground/open_duck_mini_v2/runner.py

# Recommended full run (300M steps, backlash model)
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000

# Custom output directory
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000 --output_dir runs/experiment_01

# Standing policy (head control only, no locomotion)
uv run playground/open_duck_mini_v2/runner.py --env standing --task flat_terrain

# Resume from checkpoint
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000 --restore_checkpoint_path checkpoints/2025_05_11_143000_5000000
```

---

## 4. What Happens During Training

### Training stack

Source: `playground/common/runner.py` lines 1-118.

- Algorithm: **Proximal Policy Optimisation (PPO)** via `brax.training.agents.ppo`
- Simulation: **MuJoCo MJX** (GPU-accelerated JAX-based MuJoCo)
- PPO config: loaded from `mujoco_playground.config.locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")` - this is the Berkeley Humanoid config, used as-is (noted as TODO in source, line 88)

### JAX compilation cache

Source: `playground/common/runner.py` lines 45-54.

On first run, JAX JIT-compiles the training step. This is cached in `.tmp/jax_cache/`. Subsequent runs with the same config will be faster.

### PPO network architecture

Source: `playground/common/export_onnx.py` lines 74-89, and `playground/common/runner.py` line 100.

- Default: `ppo_networks.make_ppo_networks` with hidden layer sizes from the Berkeley Humanoid config
- The ONNX export uses `hidden_layer_sizes` from `ppo_params.network_factory.policy_hidden_layer_sizes`

---

## 5. Monitoring Training Progress

### TensorBoard

Source: `README.md` lines 26-28.

```bash
uv run tensorboard --logdir=checkpoints
# or with a custom dir:
uv run tensorboard --logdir=runs/experiment_01
```

TensorBoard logs are written to `--output_dir` (default `checkpoints/`). The `SummaryWriter` is initialised in `playground/common/runner.py` line 39.

### Stdout progress

Source: `playground/common/runner.py` lines 56-66 (`progress_callback`).

At each evaluation interval, stdout prints:

```
-----------
STEP: <num_steps> reward: <eval/episode_reward> reward_std: <eval/episode_reward_std>
-----------
```

### TensorBoard metrics logged

All entries in `metrics` dict are logged. Key scalars (visible in TensorBoard):

- `eval/episode_reward` - mean episode reward (primary metric to watch)
- `eval/episode_reward_std` - standard deviation across eval episodes
- `reward/tracking_lin_vel` - how well the robot tracks commanded linear velocity
- `reward/tracking_ang_vel` - how well the robot tracks commanded angular velocity
- `reward/alive` - alive reward (robot still standing)
- `reward/imitation` - imitation reward (if enabled)
- `cost/torques` - torque usage penalty
- `cost/action_rate` - action smoothness penalty
- `cost/stand_still` - penalty for moving when no command is given
- `swing_peak` - peak foot height during swing phase

Source: `playground/open_duck_mini_v2/joystick.py` lines 305-312 (metric initialisation) and lines 470-477 (metric update per step).

---

## 6. Checkpoints

### Format

Checkpoints use **Orbax** (`orbax.checkpoint`), Google's JAX checkpoint library.

Source: `playground/common/runner.py` lines 68-84 (`policy_params_fn`).

### Save frequency

Checkpoints are saved by the `policy_params_fn` callback, which is called by Brax PPO at its internal eval interval (defined by the PPO config, not a user-facing argument in this repo).

### Checkpoint naming

```
<output_dir>/<YYYY_MM_DD_HHMMSS>_<step>/
```

Example: `checkpoints/2025_05_11_143000_5000000/`

Source: `playground/common/runner.py` lines 73-76.

### Resuming training

Pass the checkpoint directory path to `--restore_checkpoint_path`:

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --task flat_terrain_backlash \
  --num_timesteps 300000000 \
  --restore_checkpoint_path checkpoints/2025_05_11_143000_5000000
```

Source: `playground/common/runner.py` line 112 - `restore_checkpoint_path` is passed directly to `ppo.train`.

---

## 7. ONNX Export

### When it happens

ONNX export is **automatic** - it fires at every checkpoint save, alongside the Orbax checkpoint.

Source: `playground/common/runner.py` lines 77-84.

```python
onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
export_onnx(
    params,
    self.action_size,
    self.ppo_params,
    self.obs_size,
    output_path=onnx_export_path
)
```

### Output files per checkpoint

Two ONNX files are produced per checkpoint save:

1. `<output_dir>/<YYYY_MM_DD_HHMMSS>_<step>.onnx` - timestamped, per-checkpoint file
2. `ONNX.onnx` - always overwritten in the **current working directory** (hardcoded at line 176-178 of `export_onnx.py`)

Source: `playground/common/export_onnx.py` lines 171-178. Comment in source: "For Antoine :)" - the `ONNX.onnx` file in the working directory is a convenience copy.

### What the ONNX file contains

- Input: `obs` - shape `(1, obs_size)`, dtype `float32`
- Output: `continuous_actions` - shape `(1, action_size)` after `tanh` activation
- Network: MLP with architecture from `ppo_params.network_factory.policy_hidden_layer_sizes`
- Normalisation: observation mean/std are **baked in** as non-trainable variables (lines 35-38, 91-95)
- ONNX opset: 11

### Export mechanism

JAX params are converted via TensorFlow Keras then to ONNX using `tf2onnx`. The conversion:
1. Reconstructs the policy MLP in Keras
2. Transfers weights from JAX parameter dict to Keras layers via `transfer_weights()` (lines 108-148)
3. Bakes in observation normalisation stats (`mean`, `std` from `params[0]`)
4. Converts to ONNX via `tf2onnx.convert.from_keras`

Source: `playground/common/export_onnx.py` lines 1-179.

---

## 8. Simulation Visualisation / Inference

### MuJoCo interactive viewer (policy evaluation)

Source: `playground/open_duck_mini_v2/mujoco_infer.py`.

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <path_to_onnx>
```

Full argument list (lines 246-261):

| Argument | Default | Description |
|---|---|---|
| `-o` / `--onnx_model_path` | required | Path to `.onnx` file |
| `--reference_data` | `playground/open_duck_mini_v2/data/polynomial_coefficients.pkl` | Reference motion data file |
| `--model_path` | `playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml` | MuJoCo scene XML |
| `--standing` | `False` | Use standing policy mode (no imitation phase) |

Example commands:

```bash
# Test the latest checkpoint ONNX
uv run playground/open_duck_mini_v2/mujoco_infer.py -o checkpoints/2025_05_11_143000_5000000.onnx

# Use backlash scene to match training
uv run playground/open_duck_mini_v2/mujoco_infer.py \
  -o checkpoints/2025_05_11_143000_5000000.onnx \
  --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml

# Standing policy
uv run playground/open_duck_mini_v2/mujoco_infer.py -o ONNX.onnx --standing
```

### Interactive controls in the viewer

Source: `playground/open_duck_mini_v2/mujoco_infer.py` lines 105-154 (`key_callback`).

Default mode (locomotion):
- Arrow up/down: forward/backward velocity command
- Arrow left/right: lateral velocity command
- `q`/`e`: yaw turn command
- `p`/`m`: increase/decrease phase frequency factor (gait speed)
- `h`: toggle head control mode

Head control mode (after pressing `h`):
- Arrow up/down: head/neck pitch
- Arrow left/right: head yaw
- `q`/`e`: head roll

On `Ctrl+C`, the viewer saves observation history to `mujoco_saved_obs.pkl` (line 241).

### Reference motion viewer (no policy)

```bash
uv run playground/open_duck_mini_v2/ref_motion_viewer.py \
  --reference_data playground/open_duck_mini_v2/data/polynomial_coefficients.pkl \
  --scene flat_terrain
```

Source: `playground/open_duck_mini_v2/ref_motion_viewer.py`.

---

## 9. Reward Functions and Weights

### Reward weights (Joystick env, from `joystick.py` lines 77-88)

All weights are in `default_config().reward_config.scales`:

```python
reward_config=config_dict.create(
    scales=config_dict.create(
        tracking_lin_vel=2.5,      # positive: reward for matching commanded x/y velocity
        tracking_ang_vel=6.0,      # positive: reward for matching commanded yaw rate
        torques=-1.0e-3,           # negative: penalise joint torques
        action_rate=-0.5,          # negative: penalise rapid action changes
        stand_still=-0.2,          # negative: penalise movement when command is zero
        alive=20.0,                # positive: constant reward for remaining upright
        imitation=1.0,             # positive: reward for matching reference motion
    ),
    tracking_sigma=0.01,           # tracking reward sharpness
)
```

Source: `playground/open_duck_mini_v2/joystick.py` lines 77-88.

### How to change reward weights

The current runner does not expose config overrides as CLI arguments. The only way to change reward weights is to **edit `joystick.py` directly**.

Edit `playground/open_duck_mini_v2/joystick.py`, `default_config()` function (lines 49-102).

### Reward functions (source code)

Rewards are applied per simulation step. The final reward is:

```python
reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
```

Source: `playground/open_duck_mini_v2/joystick.py` line 447.

Note: `self.dt` is `ctrl_dt = 0.02` (50 Hz control). Each reward value is the raw function output multiplied by its scale weight, then by `dt`.

Individual reward functions in `playground/common/rewards.py`:

- `reward_tracking_lin_vel` (lines 11-22): exponential kernel on x-velocity error plus clipped y-velocity error
- `reward_tracking_ang_vel` (lines 25-32): exponential kernel on yaw rate error
- `cost_torques` (lines 68-69): sum of squared actuator forces
- `cost_action_rate` (lines 77-79): sum of squared action deltas
- `cost_stand_still` (lines 93-117): sum of pose + velocity deviation when command norm < 0.01
- `reward_alive` (lines 124-125): always returns 1.0

Imitation reward in `playground/open_duck_mini_v2/custom_rewards.py` (lines 1-149):
- Compares base orientation, linear/angular velocities, joint positions, joint velocities, and foot contacts against reference motion
- Internal sub-weights: `w_joint_pos=15.0` (dominant term), `w_torso_orientation=1.0`, `w_lin_vel_xy=1.0`, etc.
- Zero reward when commanded velocity norm < 0.01 (stand-still case)

### USE_IMITATION_REWARD flag

Source: `playground/open_duck_mini_v2/joystick.py` line 46.

```python
USE_IMITATION_REWARD = True
```

Setting this to `False`:
- Skips loading reference motion data
- Sets the imitation reward to 0
- Removes the imitation phase signal from observations
- Allows training without the reference motion data

---

## 10. Observation Space

### Policy input (Joystick env `state` observation)

Source: `playground/open_duck_mini_v2/joystick.py` lines 570-589 (`_get_obs`).

```
noisy_gyro                  [3]   - body angular velocity (rad/s)
noisy_accelerometer         [3]   - body linear acceleration (m/s^2), with +1.3 x-bias added
command                     [7]   - [lin_vel_x, lin_vel_y, ang_vel_yaw, neck_pitch, head_pitch, head_yaw, head_roll]
noisy_joint_angles - default[10]  - joint position errors (rad), scaled by noise
noisy_joint_vel * 0.05      [10]  - joint velocities scaled by dof_vel_scale=0.05
last_act                    [10]  - previous action
last_last_act               [10]  - action 2 steps ago
last_last_last_act          [10]  - action 3 steps ago
motor_targets               [10]  - current motor position targets
contact                     [2]   - binary foot contact flags
imitation_phase             [2]   - [cos(phase), sin(phase)] of gait cycle
```

Total: 3+3+7+10+10+10+10+10+10+2+2 = **77 values** (when USE_IMITATION_REWARD=True, imitation_phase is 2 values)

Note: the `onnx_infer.py` standalone test hardcodes `obs_size = 46` (line 33) which is an older value. The actual size is determined at training time from `self.env.observation_size["state"][0]` in `runner.py` line 29.

### Termination condition

Source: `playground/open_duck_mini_v2/joystick.py` lines 483-485.

```python
def _get_termination(self, data):
    fall_termination = self.get_gravity(data)[-1] < 0.0
    return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
```

Episode terminates when the up-vector z-component goes negative (robot has fallen past horizontal) or NaN detected.

### Action space

10 continuous actions (one per actuated joint). Actions are scaled by `action_scale=0.25` and added to the home-keyframe default position.

```python
motor_targets = self._default_actuator + action_w_delay * self._config.action_scale
```

Source: `playground/open_duck_mini_v2/joystick.py` line 405.

Motor velocity is also clamped: `max_motor_velocity=5.24 rad/s` per control step.

### Joints (10 actuated, from `constants.py` lines 65-76)

```
left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
```

---

## 11. Domain Randomisation

Applied automatically during training via `playground/common/randomize.py`.

Source: `playground/common/randomize.py` lines 26-146.

Parameters randomised per environment reset:
- Floor friction: `U(0.5, 1.0)`
- Joint friction loss: `*U(0.9, 1.1)`
- Joint armature: `*U(1.0, 1.05)`
- Centre of mass position: `+U(-0.05, 0.05)` per axis
- All link masses: `*U(0.9, 1.1)`
- Torso mass: `+U(-0.1, 0.1)` kg
- Initial joint positions (qpos0): `+U(-0.03, 0.03)` rad
- PD controller KP gains: `*U(0.9, 1.1)`

Randomisation is applied at the MJX model level using `jax.vmap` across parallel environments.

---

## 12. Imitation Reward / Reference Motion

### Requirement

If `USE_IMITATION_REWARD = True` (default), the file `playground/open_duck_mini_v2/data/polynomial_coefficients.pkl` must exist.

The file is already present in the repo. To regenerate or extend it, use the [Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator) repo.

### What the reference motion provides

The `PolyReferenceMotion` class (`playground/common/poly_reference_motion.py`) stores polynomial approximations of gait cycles indexed by `(dx, dy, dtheta)` velocity commands. At each step it evaluates the polynomial at the current gait phase `t` to produce a 40-dimensional reference state vector.

Reference vector dimensions (from comments in `poly_reference_motion.py` lines 6-51):
- [0:16] joint positions (left leg [0:5], head [5:11], right leg [11:16])
- [16:32] joint velocities (same order)
- [32:34] foot contact flags
- [34:37] base linear velocity
- [37:40] base angular velocity

---

## 13. Standing Policy (alternative environment)

Source: `playground/open_duck_mini_v2/standing.py`.

```bash
uv run playground/open_duck_mini_v2/runner.py --env standing --task flat_terrain
```

Differences from Joystick env:
- No locomotion commands - only head commands
- `USE_IMITATION_REWARD = False` (hardcoded)
- Different reward config: `orientation=-0.5, head_pos=-2.0, alive=20.0, ...` (lines 73-86)
- Command is 7-dim but `lin_vel_x/y/ang_vel_yaw` are always 0.0 (lines 647-661)
- No `imitation_phase` in observation

---

## 14. Key Simulation Parameters

From `joystick.py` `default_config()` (lines 49-102):

| Parameter | Value | Meaning |
|---|---|---|
| `ctrl_dt` | 0.02 s | Policy control frequency: 50 Hz |
| `sim_dt` | 0.002 s | Physics simulation timestep: 500 Hz |
| `episode_length` | 1000 steps | Max episode length = 20 s at 50 Hz |
| `action_repeat` | 1 | Actions repeated each control step |
| `action_scale` | 0.25 | Action output multiplier (rad) |
| `dof_vel_scale` | 0.05 | Joint velocity observation scaling |
| `max_motor_velocity` | 5.24 rad/s | Per-step velocity clamp |

n_substeps = ctrl_dt / sim_dt = 10 physics steps per policy step.

---

## 15. ONNX File Usage for Deployment

The exported ONNX file uses **CPU execution provider** in `onnxruntime`.

Source: `playground/common/onnx_infer.py` lines 6-9.

```python
self.ort_session = onnxruntime.InferenceSession(
    self.onnx_model_path, providers=["CPUExecutionProvider"]
)
```

Input/output contract:
- Input name: `"obs"`, shape `(1, obs_size)`, dtype `float32`
- Output name: `"continuous_actions"`, shape `(1, action_size)`, range `[-1, 1]` (tanh output)

The `awd=True` flag in `OnnxInfer.__init__` (used by `mujoco_infer.py`) wraps the input in a list and unwraps the first element of the output.

---

## 16. Summary of File Locations for Each Step

| Step | File |
|---|---|
| Install deps | `pyproject.toml` |
| Start training | `playground/open_duck_mini_v2/runner.py` |
| PPO training loop | `playground/common/runner.py` |
| Env config / rewards | `playground/open_duck_mini_v2/joystick.py` |
| Reward functions | `playground/common/rewards.py`, `playground/open_duck_mini_v2/custom_rewards.py` |
| Domain randomisation | `playground/common/randomize.py` |
| ONNX export | `playground/common/export_onnx.py` |
| Visualise in sim | `playground/open_duck_mini_v2/mujoco_infer.py` |
| Run ONNX standalone | `playground/common/onnx_infer.py` |
| Scene XMLs | `playground/open_duck_mini_v2/xmls/` |
| Reference motion data | `playground/open_duck_mini_v2/data/polynomial_coefficients.pkl` |
