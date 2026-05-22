# Open Duck Playground - Training Pipeline Technical Reference

Source: `/home/lakieb/Documents/open_duck_mini_research/Open_Duck_Playground`

---

## 1. RL Algorithm

The pipeline uses **Proximal Policy Optimisation (PPO)** from Google's [Brax](https://github.com/google/brax) library, running on hardware-accelerated MJX (MuJoCo XLA). The training is GPU-vectorised: thousands of environments run in parallel on a single GPU using JAX's `vmap` (`vmap` is JAX's way of running the same function on many inputs simultaneously - here it means 'run this function 8,192 times in parallel, one per simulated robot').

### PPO Hyperparameters

Taken from `mujoco_playground.config.locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")` - this is hard-coded in `playground/common/runner.py` and is the config used for the Open Duck despite the Berkeley Humanoid name (a TODO in the code).

| Parameter | Value | Plain-English meaning |
|---|---|---|
| `num_timesteps` | 150,000,000 (default); 300,000,000 recommended in README | Total number of simulation steps across all environments |
| `num_evals` | 15 | How many times to pause training and evaluate the current policy |
| `num_envs` | 8,192 | Number of robots simulated in parallel |
| `unroll_length` | 20 steps | How many steps of experience are collected before doing a weight update |
| `num_minibatches` | 32 | How many times the same batch of experience is reused to squeeze out more learning |
| `num_updates_per_batch` | 4 | How many times the same batch of experience is reused to squeeze out more learning |
| `batch_size` | 256 | Number of experience samples processed together in each gradient update |
| `discounting` (gamma) | 0.97 | How much the policy values future rewards vs immediate ones (1.0 = cares equally about all future; 0.0 = cares only about now) |
| `learning_rate` | 3e-4 | Step size for each weight update |
| `entropy_cost` | 0.005 | A small reward for trying diverse actions, preventing the policy from committing too early to one approach |
| `max_grad_norm` | 1.0 | A safety limit on how much the weights can change in a single step - prevents catastrophic updates |
| `clipping_epsilon` | 0.2 | The PPO 'proximal' constraint - how much the policy is allowed to change per update (0.2 = 20%) |
| `reward_scaling` | 1.0 | Multiplier applied to all rewards before computing gradient updates |
| `normalize_observations` | True | Whether to rescale observations to zero mean and unit variance during training |
| `action_repeat` | 1 | How many physics steps to repeat each action before querying the policy again |
| `num_resets_per_eval` | 10 | How many fresh episode starts are used when evaluating the policy |

### Network Configuration

The policy and value networks both share the architecture defined in `network_factory`:

- **Policy network**: reads from `"state"` observation key
- **Value network**: reads from `"privileged_state"` observation key (asymmetric actor-critic)
- **Policy hidden layers**: `(512, 256, 128)`
- **Value hidden layers**: `(512, 256, 128)`

### Training Loop Structure

```
runner.train()
  -> ppo.train(environment, eval_env, wrap_env_fn, ...)
     -> 8192 parallel envs (vmap over reset/step)
     -> Collect unroll_length=20 steps per env
     -> Compute advantages (GAE with gamma=0.97)  # GAE (Generalised Advantage Estimation) is a technique for estimating 'was this action better or worse than average?' that balances between looking only at immediate reward vs looking at the whole future. The gamma value controls this trade-off.
     -> Update policy with 4 gradient steps over 32 minibatches
     -> Every N steps: checkpoint + ONNX export + eval
```

Training checkpoints are saved via Orbax and simultaneously exported to ONNX for hardware deployment. The JAX compilation cache is written to `.tmp/jax_cache/`.

---

## 2. Reward Function

### Joystick Task (Walking Policy)

Defined in `playground/open_duck_mini_v2/joystick.py`, `_get_reward()`. The total reward per step is:

```
reward = clip(sum(r_i * scale_i) * dt, 0.0, 10000.0)
```

where `dt = ctrl_dt = 0.02 s`.

#### Reward Terms and Weights

| Term | Scale | Sign | Formula |
|---|---|---|---|
| `tracking_lin_vel` | 2.5 | + | Exponential tracking of X/Y velocity commands |
| `tracking_ang_vel` | 6.0 | + | Exponential tracking of yaw rate command |
| `alive` | 20.0 | + | Constant 1.0 per step |
| `imitation` | 1.0 | + | Multi-component reference motion matching |
| `torques` | -1.0e-3 | - | Sum of squared actuator forces |
| `action_rate` | -0.5 | - | Sum of squared action deltas |
| `stand_still` | -0.2 | - | Pose + velocity deviation when zero command |

#### Exact Formulations

**`reward_tracking_lin_vel`** - with asymmetric Y tolerance:

```python
y_tol = 0.1
error_x = (commands[0] - local_vel[0])^2
error_y = clip(|local_vel[1] - commands[1]| - y_tol, 0, inf)^2
lin_vel_error = error_x + error_y
reward = exp(-lin_vel_error / tracking_sigma)   # tracking_sigma = 0.1
```

> **tracking_sigma** controls how sharply the reward falls off as tracking error increases. At 0.01 (the original value), the reward is effectively binary - the robot gets almost zero reward unless it perfectly matches the commanded velocity. At 0.1 (the current value, industry standard range 0.1-0.25), the robot gets partial credit for being close, which produces a smoother learning gradient. The original 0.01 created an "overshoot trap" where standing still scored higher than attempting to walk imperfectly.

**`reward_tracking_ang_vel`**:

```python
ang_vel_error = (commands[2] - gyro[2])^2
reward = exp(-ang_vel_error / tracking_sigma)   # tracking_sigma = 0.1
```

**`cost_torques`**:

```python
cost = sum(actuator_force^2)
```

**`cost_action_rate`**:

```python
cost = sum((action - last_action)^2)
```

**`cost_stand_still`** - only active when `norm(commands[:3]) < 0.01`:

```python
pose_cost = sum(|joint_qpos - default_pose|)
vel_cost = sum(|joint_qvel|)
cost = (pose_cost + vel_cost) * (cmd_norm < 0.01)
```

**`reward_alive`**:

```python
reward = 1.0  # constant
```

### Standing Task Reward Weights (for reference)

| Term | Scale |
|---|---|
| `alive` | 20.0 |
| `orientation` | -0.5 |
| `torques` | -1.0e-3 |
| `action_rate` | -0.375 |
| `stand_still` | -0.3 |
| `head_pos` | -2.0 |

---

## 3. Imitation Reward (Disney BDX)

### Motivation

Inspired directly by the [BD-X paper (Disney Research)](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf). Instead of hand-crafting a gait pattern, reference motions are generated offline (using the [Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator) repo which uses the Placo walk engine) and then used to compute a dense tracking reward during training.

### Toggle

```python
# playground/open_duck_mini_v2/joystick.py
USE_IMITATION_REWARD = True
```

Setting this to `False` skips all polynomial coefficient loading and sets the imitation reward to 0.

### Reference Motion Format

The reference data is stored in `playground/open_duck_mini_v2/data/polynomial_coefficients.pkl`. It is a dictionary keyed by `"dx_dy_dtheta"` velocity triplets. Each entry contains polynomial coefficients that parameterise the full-period motion as a function of normalised time `t in [0, 1]`. The trajectory covers 40 state dimensions:

```
Indices 0-15:   joint positions (all 16 joints including head/antennas)
Indices 16-31:  joint velocities (same ordering)
Indices 32-33:  foot contacts [left, right]
Indices 34-36:  base linear velocity [x, y, z]
Indices 37-39:  base angular velocity [x, y, z]
```

The `PolyReferenceMotion` class (`playground/common/poly_reference_motion.py`) handles lookup by finding the nearest velocity grid point and evaluating the polynomial at the current phase:

```python
def get_reference_motion(self, dx, dy, dtheta, i):
    ix, iy, itheta = self.vel_to_index(dx, dy, dtheta)
    t = (i % nb_steps_in_period) / nb_steps_in_period
    return vmap(lambda c: polyval(c, t))(self.data_array[ix][iy][itheta])
```

The phase counter `imitation_i` increments every control step and wraps at `nb_steps_in_period`.

### Phase Encoding in Observation

The phase is encoded as a 2D unit vector and passed into the policy observation:

```python
imitation_phase = [cos(2*pi * i/T), sin(2*pi * i/T)]
```

This lets the policy know where it is in the gait cycle without discontinuity.

### Imitation Reward Computation

Defined in `playground/open_duck_mini_v2/custom_rewards.py`. The full reward signal is:

```python
def reward_imitation(...):
    # --- Internal weights ---
    w_lin_vel_xy   = 1.0
    w_lin_vel_z    = 1.0
    w_ang_vel_xy   = 0.5
    w_ang_vel_z    = 0.5
    w_joint_pos    = 15.0
    w_joint_vel    = 1.0e-3
    w_contact      = 1.0
    # w_torso_orientation = 1.0  (computed but commented out of final sum)

    # --- Per-component formulas ---

    # Torso orientation (DISABLED in sum, TODO note in code)
    torso_orientation_rew = exp(-20.0 * sum((q_base - q_ref)^2)) * 1.0

    # XY linear velocity tracking
    lin_vel_xy_rew = exp(-8.0 * sum((v_base[:2] - v_ref[:2])^2)) * 1.0

    # Z linear velocity tracking
    lin_vel_z_rew  = exp(-8.0 * (v_base[2] - v_ref[2])^2) * 1.0

    # XY angular velocity tracking
    ang_vel_xy_rew = exp(-2.0 * sum((w_base[:2] - w_ref[:2])^2)) * 0.5

    # Z angular velocity tracking
    ang_vel_z_rew  = exp(-2.0 * (w_base[2] - w_ref[2])^2) * 0.5

    # Joint position MSE (penalty, not exponential)
    joint_pos_rew  = -sum((q_joints - q_ref_joints)^2) * 15.0

    # Joint velocity MSE (penalty, not exponential)
    joint_vel_rew  = -sum((dq_joints - dq_ref_joints)^2) * 1e-3

    # Foot contact matching
    # ref contacts binarised at threshold 0.5
    contact_rew    = sum(contacts == ref_contacts) * 1.0

    reward = (lin_vel_xy_rew + lin_vel_z_rew + ang_vel_xy_rew
              + ang_vel_z_rew + joint_pos_rew + joint_vel_rew
              + contact_rew)

    reward *= (norm(cmd[:3]) > 0.01)  # zero out for zero commands
```

**Joint masking**: Head joints (indices 5-8) and antennas (indices 9-10 in the 16-joint reference) are excluded from the joint position/velocity terms. Only 10 joints are compared: left leg (0-4) and right leg (11-15 in reference = 5-9 in actuator array).

**Important caveats noted in code**:
- The `torso_orientation_rew` term is computed but excluded from the sum (commented out with "TODO ignore yaw here")
- The slices mapping reference frame indices to simulation state are marked `# TODO: double check if the slices are correct`
- The function is marked `# FIXME, this reward is so adhoc...` in the calling code

The imitation reward is scaled by `1.0` in the overall reward config, meaning its contribution is added directly before multiplying by `dt`.

---

## 4. Observation Space

### Policy Observation ("state" key)

This is what the deployed policy actually receives at runtime. Total dimensions: **61**.

| Component | Dims | Notes |
|---|---|---|
| Gyroscope (body frame) | 3 | Noisy: uniform noise `+/- 0.1 rad/s` |
| Accelerometer (body frame) | 3 | Noisy: uniform noise `+/- 0.05 m/s^2`; **+1.3 bias applied to X axis** |
| Command vector | 3 | `[lin_vel_x, lin_vel_y, ang_vel_yaw]` (head commands stripped) |
| Joint angles - default | 10 | Relative to home pose; per-joint noise (hip 0.03, knee 0.05, ankle 0.08 rad) |
| Joint velocities * 0.05 | 10 | Scaled by `dof_vel_scale=0.05`; noise `+/- 2.5 rad/s` |
| Last action | 10 | Action at `t-1` |
| Last-last action | 10 | Action at `t-2` |
| Last-last-last action | 10 | Action at `t-3` |
| Motor targets | 10 | Actual clamped targets sent to joints at `t-1` |
| Foot contacts | 2 | Boolean: left and right foot |
| Imitation phase | 2 | `[cos(2*pi*i/T), sin(2*pi*i/T)]` |

**Command vector note**: The full command is 7D (`[vx, vy, vyaw, neck_pitch, head_pitch, head_yaw, head_roll]`) but only the first 3 elements are passed to the policy observation. Head commands are sampled but not directly in state (they were used in an earlier version and some dead code remains).

**Accelerometer bias**: A +1.3 offset is deliberately applied to the X axis of the accelerometer reading. This is a known issue - issue #24 on GitHub is a PR to remove it, noting "JAX bug: the `.at[0].set()` call does not mutate in-place so the bias has no effect during training, but was naively added in the numpy inference path." This is a sim-to-real discrepancy.

**Noise model**: All sensor noise uses uniform distribution `U(-scale, +scale) * noise_level` where `noise_level=1.0`. Noise is also applied with random delay (action delay 0-3 steps, IMU delay 0-3 steps).

**Backlash model**: Backlash is the mechanical slack (play/looseness) in a gear train. When you reverse direction, the gears need to travel a small dead zone before they engage and start actually moving the output. This means the joint position reported by the motor's sensor can lag behind the actual mechanical position. The simulation models this as extra 'passive' joints that can wiggle freely by about ±0.5 degrees. In the backlash variants, the observed joint angles have the backlash joint offset added: `joint_angles_obs = actuator_joint_qpos + backlash_joint_qpos`.

### Privileged Observation ("privileged_state" key)

Used only by the value network during training; not available at deployment. Extends "state" with:

| Additional Component | Dims |
|---|---|
| True gyro (no noise) | 3 |
| True accelerometer (no noise) | 3 |
| True gravity vector | 3 |
| True local linear velocity | 3 |
| True global angular velocity | 3 |
| True joint angles - default | 10 |
| True joint velocities | 10 |
| Root height (z) | 1 |
| Actuator forces | 10 |
| Foot contacts (duplicate) | 2 |
| Foot linear velocities | 6 (2 feet x 3) |
| Feet air time | 2 |
| Current reference motion frame | 40 |
| Imitation phase index (scalar) | 1 |
| Imitation phase (cos/sin) | 2 |

---

## 5. Action Space

The policy outputs **10 continuous actions**, one per actuated joint. The joint order (defined in `constants.JOINTS_ORDER_NO_HEAD`) is:

```
0: left_hip_yaw
1: left_hip_roll
2: left_hip_pitch
3: left_knee
4: left_ankle
5: right_hip_yaw
6: right_hip_roll
7: right_hip_pitch
8: right_knee
9: right_ankle
```

Head/neck/antenna joints (4 joints) are present in the MJCF but are **not controlled by the RL policy** in the current default configuration. They receive zero commands.

### Action Interpretation

Actions are **position offsets relative to the home pose**, scaled and clamped for motor velocity:

```python
motor_targets = default_actuator + action * action_scale   # action_scale = 0.25
# Motor velocity clamping (USE_MOTOR_SPEED_LIMITS = True):
motor_targets = clip(
    motor_targets,
    prev_targets - max_motor_velocity * dt,   # max_motor_velocity = 5.24 rad/s
    prev_targets + max_motor_velocity * dt,
)
```

The clamping limits the maximum joint velocity change per control step to `5.24 * 0.02 = 0.1048 rad` per step.

The MJCF uses **position actuators** (PD controllers), so `motor_targets` are target joint positions fed directly to `data.ctrl`. The PD gains are:

```xml
<!-- joints_properties.xml -->
<position kp="17.8" kv="0.0" forcerange="-3.35 3.35"/>
```

Torques are capped at ±3.35 Nm. The `kv=0.0` means damping is handled by the `joint damping` attribute, not the actuator.

The policy output (raw neural network output, before `tanh`) is split: the first half is the mean action (`loc`), and `tanh` is applied to bound it to `[-1, 1]`.

Terminology: `logits` = raw unnormalised network outputs before any squishing; `loc` = the centre/mean of the action distribution; `tanh` = a squishing function that maps any number to the range -1 to +1.

In plain terms: the network outputs raw numbers, `tanh` squishes them into -1 to +1, then they are scaled by 0.25 to give the final joint offsets.

```python
logits = mlp(obs)
loc, _ = tf.split(logits, 2, axis=-1)
action = tf.tanh(loc)
```

---

## 6. Policy Network Architecture

The policy MLP architecture (from `export_onnx.py` and the PPO config):

```
Input: obs (61-dim "state" vector)
  -> LayerNorm/mean-std normalisation (running stats from training)
  -> Dense(512, activation=swish)
  -> Dense(256, activation=swish)
  -> Dense(128, activation=swish)
  -> Dense(20, activation=None)   # 10 actions * 2 (mean + log_std)
  -> split -> loc (10,), log_std (10,)  [log_std unused at inference]
  -> tanh(loc) -> action (10,)
```

**Activation**: `swish` (SiLU) is used in the ONNX export. The Brax PPO default is also swish for the policy network.

**Observation normalisation**: Running mean and standard deviation are tracked during training and baked into the exported ONNX model as non-trainable variables.

**Value network** (critic, not exported): 3-layer MLP `(512, 256, 128)` with `swish`, reading from `privileged_state`. Not deployed to hardware.

The exported ONNX model uses opset 11 (matching Isaac Lab convention).

---

## 7. Domain Randomisation

Defined in `playground/common/randomize.py`, applied per-environment at reset via `jax.vmap`. The following parameters are randomised every episode:

| Parameter | Distribution | Notes |
|---|---|---|
| Floor friction (sliding) | `U(0.5, 1.0)` | Applied to `geom_friction[floor, 0]` |
| Joint friction loss | `* U(0.9, 1.1)` | Multiplicative, per actuated DOF |
| Joint armature | `* U(1.0, 1.05)` | Multiplicative, per actuated DOF |
| Torso CoM position | `+ U(-0.05, 0.05)` | 3D offset in metres |
| All link masses | `* U(0.9, 1.1)` | Multiplicative, per body |
| Torso mass | `+ U(-0.1, 0.1)` | Additive in kg |
| Joint home position (qpos0) | `+ U(-0.03, 0.03)` | Per joint in radians |
| PD gain Kp | `* U(0.9, 1.1)` | Both `actuator_gainprm` and `actuator_biasprm` updated together |

**Note on backlash DOFs**: Friction loss randomisation is only applied to DOFs where `dof_hasfrictionloss == True`. Backlash joints have `frictionloss=0` (deliberately excluded from randomisation).

### Episode Reset Randomisation

At each episode reset (in `joystick.py`):

| Parameter | Distribution |
|---|---|
| Base XY position | `+ U(-0.05, 0.05) m` |
| Base yaw | `U(-pi, pi)` |
| Joint positions | `* U(0.5, 1.5)` (relative to home) |
| Base velocities (6D) | `U(-0.05, 0.05)` |

Push perturbations are applied during episodes every `U(5.0, 10.0)` seconds with magnitude `U(0.1, 1.0)` in a random horizontal direction.

---

## 8. MuJoCo/MJX Setup

### Timesteps and Control

| Parameter | Value |
|---|---|
| `sim_dt` | 0.002 s (500 Hz physics) |
| `ctrl_dt` | 0.02 s (50 Hz control) |
| `n_substeps` | 10 (sim steps per control step) |
| `episode_length` | 1000 steps (20 seconds) |

### Solver Settings

From `open_duck_mini_v2.xml`:

```xml
<option iterations="1" ls_iterations="5">
    <flag eulerdamp="disable"/>
</option>
```

- `iterations=1`: one Newton step per physics step (fast, acceptable for MJX training)
- `ls_iterations=5`: 5 line-search iterations
- Euler damping disabled

### Contact Model

- **Collision**: only `foot_bottom_tpu` geoms (TPU-material foot pads) are enabled for contact; all other geoms have `contype=0`/`conaffinity=0`
- **Floor friction**: `friction="0.6" condim="3"` in flat terrain scenes; `friction="1.0"` in rough terrain
- **Contact dimension**: `condim=3` (sliding friction only, no torsional)

### Actuator/Joint Properties

From `joints_properties.xml` (most up-to-date; the inline XML in `open_duck_mini_v2.xml` is overridden):

```xml
<joint damping="0.60" frictionloss="0.052" armature="0.028"/>
<position kp="17.8" kv="0.0" forcerange="-3.35 3.35"/>
```

Backlash joints: `range="-0.008726 0.008726"` (+/- 0.5 degrees), `damping=0.01`, `frictionloss=0`, `armature=0.01`.

### Terrain Variants

| Task | XML | Notes |
|---|---|---|
| `flat_terrain` | `scene_flat_terrain.xml` | No backlash joints |
| `flat_terrain_backlash` | `scene_flat_terrain_backlash.xml` | With backlash joints (recommended) |
| `rough_terrain_backlash` | `scene_rough_terrain_backlash.xml` | Heightfield terrain from `hfield.png` |

The README states the recommended training command is:
```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```

---

## 9. Reference Motion System

### Generation

Reference motions are generated externally via [Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator). That repo uses the **Placo** walk engine to generate kinematically valid walking motions for a grid of velocity commands `(dx, dy, dtheta)`. Each motion clip is then compressed into polynomial coefficients via least-squares fitting.

The output is `polynomial_coefficients.pkl` placed at `playground/open_duck_mini_v2/data/polynomial_coefficients.pkl`.

### Data Structure

Each `(dx, dy, dtheta)` entry stores polynomial coefficients for 40 dimensions (joint positions, velocities, contacts, base velocities) over one gait period. The period and FPS are stored alongside the coefficients. The `startend_double_support_ratio` defines a double-support phase at start/end of each cycle.

### Runtime Lookup

`PolyReferenceMotion` (JAX version) and `PolyReferenceMotionNumpy` (NumPy version for inference) both work identically:

1. Clip the commanded velocity to the covered grid range
2. Find the nearest grid point via `argmin` of absolute distance
3. Compute normalised time `t = (i % T) / T`
4. Evaluate all 40 polynomial channels at `t` using `jnp.polyval`

The nearest-neighbour lookup means there is no interpolation between velocity levels - the policy learns to deal with this discretisation as part of its training distribution.

### Phase Counter Management

```python
# At each step:
imitation_i = (imitation_i + 1) % nb_steps_in_period

# Command changes (every 500 steps):
# new reference motion is fetched for updated command
# imitation_i continues from current position (no phase reset)
```

Commands are re-sampled every 500 control steps (10 seconds); the reference motion pointer continues from its current phase position rather than resetting, which avoids discontinuities.

---

## 10. Config System

There is no YAML or JSON training config. All configuration is Python `ml_collections.ConfigDict` objects defined inline in each task file.

### `default_config()` in `joystick.py`

Key settings:

```python
ctrl_dt = 0.02              # control frequency (50 Hz)
sim_dt = 0.002              # physics frequency (500 Hz)
episode_length = 1000       # steps
action_scale = 0.25         # action magnitude scaling
dof_vel_scale = 0.05        # joint velocity scaling in obs
soft_joint_pos_limit_factor = 0.95  # soft limits at 95% of hard range
max_motor_velocity = 5.24   # rad/s motor velocity clamp

# Velocity command ranges
lin_vel_x = [-0.15, 0.15]   # m/s forward/back
lin_vel_y = [-0.2, 0.2]     # m/s lateral
ang_vel_yaw = [-1.0, 1.0]   # rad/s yaw

# Head command ranges (sampled but not currently used in policy obs)
neck_pitch_range = [-0.34, 1.1]
head_pitch_range = [-0.78, 0.78]
head_yaw_range = [-1.5, 1.5]
head_roll_range = [-0.5, 0.5]
```

### `xmls/config.json`

Not a training config - this is the [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) config for generating the MJCF from the Onshape CAD model. It controls which parts are visual vs collision, joint class assignments, and which supplementary XMLs to include.

### PPO Config

Loaded from `mujoco_playground.config.locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")`. The `num_timesteps` is overridden by the `--num_timesteps` CLI argument to the runner.

### Runner CLI Arguments

```bash
uv run playground/open_duck_mini_v2/runner.py \
    --output_dir checkpoints \
    --num_timesteps 150000000 \
    --env joystick \
    --task flat_terrain_backlash \
    --restore_checkpoint_path /path/to/checkpoint  # optional
```

---

## 11. Known Issues and TODOs

### Confirmed Bugs

**Accelerometer bias (issue #24)**: The code in `joystick.py` contains:
```python
accelerometer.at[0].set(accelerometer[0] + 1.3)
```
This is a JAX bug - `.at[].set()` on a temporary array does not mutate. The bias **has no effect during training** but was added (incorrectly) to the NumPy inference path (`mujoco_infer.py`) as `accelerometer[0] += 1.3`. This creates a sim-to-real discrepancy where the deployed policy sees a biased accelerometer that it was never trained on. PR #24 proposes removing the bias from the inference path.

### TODOs and FIXMEs in Code

From `playground/common/rewards.py`:
- Multiple reward functions are marked `# FIXME` and not used in the active training config: `cost_joint_deviation_hip`, `cost_joint_deviation_knee`, `cost_pose`, `cost_feet_slip`, `cost_feet_clearance`, `cost_feet_height`, `reward_feet_air_time`, `reward_feet_phase`
- `cost_stand_still`: `# TODO no hard coded slices`

From `playground/open_duck_mini_v2/custom_rewards.py`:
- `# TODO don't reward for moving when the command is zero.` (for imitation reward - currently only suppressed by `cmd_norm > 0.01` threshold)
- `# TODO: double check if the slices are correct` - the reference frame slice indices have not been formally verified
- `# TODO ignore yaw here, we just want xy orientation` - torso orientation component is computed but disabled
- `# FIXME, this reward is so adhoc...` - in the calling site in `joystick.py`

From `playground/common/runner.py`:
- `# TODO` comment on the PPO config: hard-coded to `BerkeleyHumanoidJoystickFlatTerrain` regardless of the actual robot/task

From `playground/open_duck_mini_v2/joystick.py`:
- `info["last_act"] = action  # was` / `# state.info["last_act"] = motor_targets  # became` - there is uncertainty about whether last_act should store the raw network action or the clamped motor targets
- Head joint control is partially wired but commented out: `# motor_targets.at[5:9].set(state.info["command"][3:])`

### Open GitHub Issues (apirrone/Open_Duck_Playground)

| Issue | Title | Summary |
|---|---|---|
| #24 | Removed accelerometer bias | PR: removes training/inference discrepancy caused by JAX in-place mutation bug |
| #23 | Mac OS compatibility | PR: platform-specific CUDA dependency marker for Mac CPU-only use |
| #20 | MuJoCo XML validation | Feature: add schema validation for MJCF files |
| #19 | num_envs increase | User question about increasing parallel environments |
| #18/#13 | runner.py errors | User-reported CUDA/import errors |
| #17 | FastAPI + voice control | Feature addition from community |
| #16 | RTX 5xxx CUDA error | TF/CUDA incompatibility with RTX 5000 series |
| #14 | Custom robot training | Issues using custom polynomial coefficients |
| #12/#11 | Contact-aided invariant EKF | PR: adds EKF state estimator to mujoco_infer for better velocity estimation |
| #8 | `AttributeError: 'dict' object has no attribute 'policy'` | ONNX export bug |

---

## Summary of Key Architecture Decisions

1. **Asymmetric actor-critic**: policy sees only noisy "state" (61-dim); critic sees "privileged_state" with ground-truth sensors - standard technique for sim-to-real
2. **3-step action history in obs**: the policy receives `last_act`, `last_last_act`, `last_last_last_act` (30 dims total) which gives it implicit knowledge of its recent control history
3. **Motor target in obs**: `motor_targets` (after velocity clamping) is also included, distinguishing target from action
4. **Imitation reward disabled for zero commands**: prevents the reference motion from pulling the robot out of a standing pose
5. **Velocity clamping at 5.24 rad/s**: limits mechanical stress during training; matched in inference
6. **Backlash modelling**: dedicated passive joints with ±0.5° range model real servo backlash; included in observation
7. **Command resampling every 500 steps** within an episode: trains the policy to handle mid-episode command changes without episode termination
8. **Termination condition**: only falls (gravity vector Z < 0) or NaN states trigger episode end; no time limit termination signal
