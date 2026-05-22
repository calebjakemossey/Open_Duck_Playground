# Motion Smoothness Improvements for Open Duck Mini

*Research report - May 2026 | Sources: 30+ papers and repositories*

---

## Executive Summary

The Open Duck Mini's characteristic "waddling" gait stems from several compounding problems: the policy explores high-frequency joint oscillations because nothing penalises them sufficiently; cheap Dynamixel XL-330 servos introduce backlash and compliance that the sim doesn't fully capture; and the reference motions used for imitation learning may themselves encode a duck-like lateral sway that the policy faithfully reproduces. The literature identifies six mutually-compatible levers for improvement, ordered here by implementation effort and expected impact. The most impactful single change is strengthening the action-smoothness reward terms. The most underexplored opportunity in the current codebase is adding second-order smoothness (action acceleration, i.e. jerk) and Lipschitz-constrained policy (LCP) gradient penalty training.

---

## 1. Action Smoothness Reward Terms

### What the literature shows

Action smoothness is the most consistently applied technique across the field. The standard toolkit has three levels of temporal penalty:

| Penalty | Formula | What it suppresses |
|---------|---------|-------------------|
| Action rate (1st order) | `‖a_t - a_{t-1}‖²` | Step-to-step jumps |
| Action acceleration (2nd order) | `‖a_t - 2a_{t-1} + a_{t-2}‖²` | Direction reversals / zig-zag |
| Joint velocity | `‖q̇‖²` | High-frequency oscillation |
| Joint acceleration | `‖q̈‖²` | Abrupt torque changes |
| Torque | `‖τ‖²` | Energy waste, servo heating |

Papers that combine all five (e.g. [arXiv 2401.16889] bipedal skill learning, [arXiv 2011.01387] periodic reward framework) consistently report visibly smoother real-world motion compared to using only velocity tracking rewards.

The second-order term `‖a_t - 2a_{t-1} + a_{t-2}‖²` is specifically noted in [arXiv 2509.09106] as an "action smoothness" term distinct from the first-order "action rate" term - this distinction is important because zig-zagging actions can have low first-order change but high second-order change.

### Current state of the Open Duck codebase

`playground/common/rewards.py` defines `cost_action_rate` as `‖a_t - a_{t-1}‖²`. The joystick config sets its scale to `-0.5` (reduced from a previous `-1.5`). The following terms are **absent**:
- Second-order action smoothness (acceleration / jerk)
- Joint velocity cost
- Joint acceleration cost

`last_last_act` and `last_last_last_act` are already stored in `info`, so the infrastructure for second-order penalties exists.

### Actionable changes

**In `playground/common/rewards.py`**, add:

```python
def cost_action_rate2(act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
    """Second-order action smoothness (penalises acceleration / direction changes)."""
    return jp.nan_to_num(jp.sum(jp.square(act - 2 * last_act + last_last_act)))

def cost_joint_vel(qvel: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(qvel)))

def cost_joint_acc(qvel: jax.Array, last_qvel: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(qvel - last_qvel)))
```

**In `joystick.py`**, start with these additional scales and tune from there:

```python
reward_config=config_dict.create(
    scales=config_dict.create(
        tracking_lin_vel=2.5,
        tracking_ang_vel=6.0,
        torques=-1.0e-3,
        action_rate=-0.5,
        action_rate2=-0.1,      # NEW: second-order smoothness
        joint_vel=-1.0e-3,      # NEW: suppress oscillation
        stand_still=-0.2,
        alive=20.0,
        imitation=1.0,
    ),
)
```

The weight for `action_rate2` should be roughly 5-10x smaller than `action_rate` initially. The joint velocity penalty must be kept small (1e-3 to 1e-4 range) to avoid killing legitimate gait dynamics.

**Why this works**: the policy gradient is pushed toward action trajectories that are smooth in both value and gradient, not just value. The duck can no longer exploit fast reversals that happen to have low per-step delta but accumulate into visible jitter.

---

## 2. Lipschitz-Constrained Policies (LCP)

### What the literature shows

Chen et al. 2024 (arXiv:2410.11825, IROS 2025 - "Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies") demonstrate that a single gradient penalty term added to the PPO loss eliminates the need for both smoothness reward tuning and low-pass filters. The constraint bounds how much the policy output can change per unit change in observation - enforcing a global smoothness on the mapping itself rather than penalising individual action transitions.

> **A Lipschitz constant** is an upper bound on how fast a function's output can change relative to its input change. If a function has a Lipschitz constant of 2, then doubling the input change can at most double the output change - the function cannot 'explode'. For a robot policy, constraining the Lipschitz constant means: small changes in sensor readings can only cause small changes in servo commands. This is precisely what 'smooth' means mathematically - the policy cannot make sudden large moves in response to small sensor fluctuations.

Critically, this was validated on the Berkeley Humanoid (a small, ~1m humanoid robot comparable in scale challenge to the Duck), Unitree H1, and two Fourier GR1 variants - all zero-shot sim-to-real. The paper is cited by [arXiv on SUBO-2] (a low-backdrivability bipedal robot similar in character to the Duck's XL-330 chain drives) as the smoothness technique that resolved policy instability when combined with actuator network training.

The penalty is:

```
L_LCP = λ_gp · E[‖∇_obs π(obs)‖²]
```

where `∇_obs π` is the Jacobian of the action with respect to the observation, computed via standard autograd.

> **A Jacobian** is a matrix that captures 'if I nudge each input by a tiny amount, how much does each output change?' It has one row per output and one column per input. The gradient penalty penalises having any entry in this matrix be large - meaning no single input change should be able to cause a disproportionately large output change.

### Implementation for Open Duck Playground

The playground uses JAX/Flax. Adding LCP requires patching the PPO training loop in `playground/common/runner.py`. The gradient penalty requires differentiating the actor through its inputs:

```python
import jax

def lcp_gradient_penalty(params, apply_fn, obs, lambda_gp=0.002):
    """Lipschitz constraint as gradient penalty on policy output w.r.t. observations."""
    def policy_fn(o):
        return apply_fn({'params': params}, o)
    
    # Jacobian of actions w.r.t. observations
    jac = jax.jacfwd(policy_fn)(obs)  # shape: [action_dim, obs_dim]
    grad_norm_sq = jp.sum(jp.square(jac))
    return lambda_gp * grad_norm_sq
```

This is added as an auxiliary loss during PPO actor updates. The paper reports `λ_gp = 0.002` for the Berkeley Humanoid (comparable size to the Duck).

> **In PPO**, the 'actor loss' is the main scoring function being optimised - it measures how much better or worse the policy's recent actions were compared to average, and nudges the weights accordingly. An 'auxiliary loss' is an additional term bolted on to encourage a specific property (in this case, smoothness). The total score being optimised becomes: main PPO objective + value estimation accuracy + exploration encouragement + smoothness penalty.

**Why this is better than reward engineering alone**: reward penalties only see the action trajectory after training; the gradient penalty directly shapes the policy's functional form. It is differentiable, scales automatically with the magnitude of the observation space, and requires tuning only one scalar.

---

## 3. Low-Pass Filtering of Policy Outputs

### What the literature shows

**Two distinct approaches exist, with opposing trade-offs:**

**Option A: Filter at inference time only (post-training)**
Apply a first-order exponential low-pass filter to the raw policy output before converting to motor targets. Used extensively in practice (arXiv:2409.19795 on passive locomotion at 50 Hz uses this), and visible in the commented-out `LowPassActionFilter` code in `joystick.py`.

The filter: `filtered_a_t = α · a_t + (1 - α) · filtered_a_{t-1}`

At 50 Hz control rate, `α = 0.6` gives a cutoff of roughly 12 Hz.

**Warning from CAPS paper (Mysore et al., ICRA 2021)**: naively applying filters to a policy trained without them changes the dynamical response the network expects. The network was trained assuming its previous output was applied directly; now a damped version is applied, breaking the Markov assumption. They observed EMA filters causing moderate oscillation and FIR filters causing catastrophic control failure on neural network policies.

> **The Markov assumption** in RL is the assumption that the current observation contains all the information needed to make a good decision - you don't need to remember the history. When you apply a filter that the policy wasn't trained with, the policy receives modified versions of its previous commands (because the filter blends them), but it doesn't know about this modification. This breaks the 'current observation is sufficient' assumption and can cause unpredictable behaviour.

**The correct approach is to train with the filter in the loop**: include the filtered previous action as part of the observation, and apply the filter both during training and inference. The `LowPassActionFilter` is already imported (commented out) in `joystick.py` - it is commented out because the filter was not included in the observation history during training.

**Option B: Train with filter in the loop**
The filter must be included in the simulation step during training. The observation must include `filtered_last_act` rather than `last_act`. This is more work but ensures consistency between training and deployment.

### Actionable change

Enable the `LowPassActionFilter` in `joystick.py` but also change the observation to pass `motor_targets` (the filtered and clamped target positions) rather than the raw `action` as the "last action" history. This is actually already partially done - the observation state includes `info["motor_targets"]` as a separate entry distinct from `info["last_act"]`. The motor velocity clamping (`USE_MOTOR_SPEED_LIMITS`) already acts as a soft low-pass filter on the action space.

The simplest correct option: **lower `kp` on the real robot**. The current recommended inference config is `kp = 22` (down from training `kp = 32`). Further reducing to `kp = 15-18` increases effective servo compliance, which acts as a physical low-pass filter on the motion. This is already noted in the playground README as a key sim-to-real tweak.

---

## 4. Actuator-Aware Training (Backlash and Servo Dynamics)

### What the literature shows

The playground already has backlash modelling - the `flat_terrain_backlash` task includes dummy backlash joints in the MJCF and the `base.py` class explicitly handles separating actuator joints from backlash joints. The community's recommended training command is:

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```

However, research on extended friction models for servo actuators (arXiv:2410.08650) shows that the standard Coulomb-Viscous friction model used in MuJoCo significantly underestimates servo friction - especially the static-to-kinetic friction transition and velocity-dependent viscous friction at low speeds. For Dynamixel MX-106 (a similar series to XL-330), extended models reduce MAE by over 2x versus the standard model.

The Open Duck project already uses Rhoban's BAM (Backlash-Aware Model) for actuator identification. The key identified parameters exported to MuJoCo units are: `damping`, `kp`, `frictionloss`, `armature`, `forcerange`. What BAM does **not** capture is the velocity-dependent friction transition (stick-slip) that causes micro-oscillation at low velocities.

The SUBO-2 paper (Springer 2026) specifically addresses low-backdrivability bipedal robots and finds the combination of:
1. Unsupervised Actuator Network (UAN) for corrective torque modelling
2. Lipschitz-Constrained Policy (LCP) for smoothness

reduces action jerk, joint position jerk, and joint angular velocity compared to conventional smoothing on a robot with gear reduction - directly analogous to the Duck's XL-330 servo gearboxes.

### Actionable changes

**Increase friction randomisation range**: In the domain randomisation config, widen the `frictionloss` randomisation range by 2-3x the current range. Servos at low speed exhibit significantly higher effective friction than the nominal BAM value. Training the policy to be robust to this range will reduce the tendency to produce small oscillatory corrections.

**Add motor velocity as observation noise**: The current noise scale for `joint_vel` is `2.5 rad/s`. For the XL-330 at low speeds this is appropriate, but consider also randomising the sign of the velocity offset (the servo encoder can report a small non-zero velocity when stationary due to gear lash settling).

**Consider UAN if BAM results are insufficient**: Record 5-10 minutes of diverse open-loop joint trajectories on the real robot (sinusoids at 0.5-3 Hz, varying amplitude). Train a small MLP to predict the corrective torque δτ that minimises the discrepancy between the simulated and real joint position trajectory. Insert this network into the MuJoCo simulation during training. This requires no torque sensing - only the joint encoder difference between sim and real.

---

## 5. Reference Motion Quality

### What the literature shows

The imitation reward is the single largest driver of gait naturalness in the current architecture. The reference motions are generated by `Open_Duck_reference_motion_generator` using `auto_waddle.py` based on a parametric walk engine (Placo). The name `auto_waddle` is revealing - the generator is explicitly designed around the duck's natural waddling gait, which includes lateral body sway that the policy will imitate.

The Disney BDX paper (which the imitation reward is based on) used hand-crafted reference motions for the full-sized BDX robot, not a parametric approximation. The polynomial fit used in `polynomial_coefficients.pkl` approximates these motions as a function of `(vx, vy, ω, phase)`. The quality of this approximation directly bounds how smooth the learned gait can be.

Key findings from the reference motion literature:
- Smoother reference motions produce smoother learned policies, but the policy will always slightly degrade naturalness from the reference (it must be robust to perturbations the reference never encounters)
- The gait playground script (`gait_playground.py`) allows interactive tuning of the walk parameters - **reducing lateral sway amplitude is the most direct lever for reducing waddling**

### Actionable changes

**Reduce lateral sway in reference generation**: In `auto_gait.json`, find the parameters controlling lateral body oscillation (likely named `lateral_amplitude` or `body_sway`). Reduce this to 30-50% of current value and regenerate `polynomial_coefficients.pkl`. Retrain. The policy will then try to imitate a more upright gait.

**Add a lateral velocity cost**: The current reward does not explicitly penalise lateral CoM oscillation. Add:

```python
def cost_lin_vel_y_rms(global_linvel: jax.Array) -> jax.Array:
    """Penalise excessive lateral body oscillation."""
    return jp.nan_to_num(jp.square(global_linvel[1]))
```

at a small weight (-0.1 to -0.5). This is distinct from the lateral velocity tracking reward (which rewards tracking the commanded `vy`) - this term specifically penalises unintended lateral sway when `vy` command is zero.

**Improve polynomial fit quality**: The current fit may have poor accuracy at intermediate velocities if the training sweep doesn't cover the full command space densely. Use `--sweep` with a finer grid, or compare the polynomial predictions against the raw recordings using `plot_poly_fit.py`. If the fit error exceeds ~5-10% of the motion amplitude at any point in the gait cycle, that error contributes to reward noise.

---

## 6. Curriculum Learning

### What the literature shows

Several papers demonstrate that curriculum learning leads to qualitatively smoother gaits, not just higher reward, because it prevents the policy from converging to a jerky local optimum that "gets the job done" at the cost of smoothness.

The key insight from "Learning Symmetric and Low-Energy Locomotion" (Ling et al., SIGGRAPH 2018): providing modulated physical assistance (external forces supporting balance) during early training, then gradually removing it, allows the energy penalty to be set much higher than would otherwise be feasible. Without the curriculum, a high energy penalty causes the policy to fail to learn to walk; with it, the policy learns a genuinely low-energy, smooth gait.

A two-stage curriculum that applies directly to Open Duck:

**Stage 1**: Train with:
- Higher `alive` reward weight (encourage survival)
- Lower smoothness penalty weights (allow exploration)
- Smaller command range (only forward walking, low speed)
- Higher imitation weight (force the policy to track reference motions closely)
- Optionally: add a small assistive upward force on the torso

**Stage 2**: Transfer weights and continue with:
- Reduced imitation weight
- Increased smoothness penalty weights (now safe to enforce, since the policy already knows how to walk)
- Full command range
- Remove any assistive force

In the playground, this is implemented by training two policies sequentially and using the first as the starting checkpoint for the second. Alternatively, the reward weights can be scheduled as a function of training step count.

### Actionable change

Add a `training_stage` parameter to `default_config()` (or pass via command-line override) and adjust reward scales accordingly:

```python
# Stage 1: get it walking
scales_stage1 = dict(action_rate=-0.1, action_rate2=-0.01, imitation=3.0, alive=30.0)

# Stage 2: refine smoothness
scales_stage2 = dict(action_rate=-0.8, action_rate2=-0.2, imitation=1.0, alive=20.0)
```

Train stage 1 for 100-150M steps, then continue from checkpoint with stage 2 scales for another 150-200M steps.

---

## 7. Control Frequency Considerations

### What the literature shows

Research from Oxford (arXiv:2209.14887, "Learning Low-Frequency Motion Control") is the most rigorous study of this question. Key findings:

- **Low-frequency policies (8-25 Hz) are significantly less sensitive** to actuation latency and servo dynamics variation
- A policy trained at 10 Hz can transfer sim-to-real without any dynamics randomisation on a quadruped
- A policy trained at 200 Hz requires explicit joint state history to model the servo dynamics implicitly; without it, the policy exploits fast contact switching (jitter)
- **The optimal range for sim-to-real on non-ideal actuators is 10-25 Hz**

The current Open Duck config uses `ctrl_dt=0.02` (50 Hz). This is on the higher end of what transfers cleanly with low-backdrivability servos. The Dynamixel XL-330 has a maximum communication rate of ~200 Hz (USB2Dynamixel limits this further in practice) and a mechanical bandwidth far below 50 Hz.

A complementary paper on bipedal torque control (Seoul National University) found that for torque-based bipedal control, lower frequencies were more robust to system delays even on unexpected terrain.

### Actionable change

**Try retraining at `ctrl_dt=0.05` (20 Hz)**: Halving the control rate forces the policy to make coarser decisions, which naturally reduces high-frequency oscillation. The action at each step represents a larger committed motion, so the policy learns to be more deliberate. The sim_dt can remain at 0.002 (500 Hz physics).

The tradeoff: slower reaction time to perturbations. Given the Duck's modest walking speed (0.15 m/s), 20 Hz provides 18.75 mm of travel between updates - likely sufficient.

If moving to 20 Hz, the `max_motor_velocity` clamp (`5.24 rad/s × ctrl_dt`) needs adjusting: at 20 Hz, the maximum single-step target change becomes `5.24 × 0.05 = 0.262 rad` per step vs. `0.105 rad` at 50 Hz. This larger window gives the servo more time to reach each target, reducing the tendency to issue targets the servo cannot track.

---

## 8. Teacher-Student Distillation

### What the literature shows

The standard teacher-student paradigm for locomotion smoothness:
1. Train a **teacher policy** with access to privileged information (ground truth contact forces, terrain height, true friction, etc.) - this policy can be robust but may be jerky
2. Train a **student policy** that only receives the same observations available on real hardware, by imitating the teacher's actions (DAgger) or behaviour distribution (GAIL)

The key smoothness benefit: the student, lacking privileged information, cannot exploit the same high-frequency cues the teacher uses. It must learn a smoother function of available observations. This is documented in the Distillation-PPO paper (arXiv:2503.08299) which notes that teacher-to-student transfer tends to produce policies with lower action variance.

> **DAgger** (Dataset Aggregation) = the student policy runs the robot, an expert identifies mistakes, and the student is trained on the expert's corrected actions. **GAIL** (Generative Adversarial Imitation Learning) = a more sophisticated approach where the student learns to match the overall pattern of the expert's behaviour rather than copying individual actions. Both are ways of transferring knowledge from a capable 'teacher' policy to a smoother 'student' policy.

For Open Duck, the teacher could observe:
- True motor torques (not available on real hardware without force sensing)
- Exact backlash joint positions (not directly observable)
- Ground truth feet contact forces

The student (deployed policy) only sees: gyro, accelerometer, joint angles + velocities, last actions.

The current architecture already has a `privileged_state` observation vector in the environment that includes true torques, gravity vector, foot velocities etc. - the infrastructure for asymmetric actor-critic training exists.

### Actionable change

The playground uses an asymmetric actor-critic structure where the **critic** sees `privileged_state` but the **actor** sees only `state`. This is already implemented and is the correct approach for the deployment phase.

To add proper teacher-student distillation: train a **teacher actor** that also sees `privileged_state`. After convergence, distil to the student actor (which only sees `state`) using behaviour cloning with the teacher's actions as supervision targets, then fine-tune with RL. The MimicKit framework (referenced by the LCP paper) provides a clean implementation for this pattern in JAX.

For a simpler first step: ensure the `privileged_state` is being fully used by the critic (check that the critic network is larger/wider than the actor), and add `data.actuator_force` to the actor's observation only during training (not deployment) as an auxiliary signal. This won't help at inference but trains a stronger policy that the student can imitate.

---

## 9. Techniques from Humanoid Robotics Papers

### Direct applicability to Open Duck

| Technique | Source | Applies to Duck | Notes |
|-----------|--------|----------------|-------|
| LCP gradient penalty | arXiv:2410.11825 (Berkeley/SFU) | High | Validated on Berkeley Humanoid (small robot) |
| Action acceleration penalty | arXiv:2509.09106, arXiv:2401.16889 | High | Already has infrastructure in info dict |
| Decoupled velocity tracking | arXiv:2509.09106 | Medium | Stability-prioritised reward fusion |
| Gait-conditioned reward routing | arXiv:2505.20619 (Unitree G1) | Low | Duck has one gait mode currently |
| Mirror symmetry loss | Ling et al. 2018 | High | Duck gait should be L-R symmetric; adding symmetry loss to actor update is trivial |
| Causal transformer policy | Berkeley Humanoid (Science Robotics 2024) | Medium | Replaces MLP; captures history implicitly |
| Variable stiffness in action space | arXiv:2502.09436 | Medium | Adds per-joint kp to action space |

**Mirror symmetry loss** is particularly low-cost and applicable. A symmetric gait is inherently smoother because it suppresses asymmetric oscillations. The loss adds a term to the PPO objective that penalises the difference between the policy's output for the current observation and the output for the mirror-reflected observation:

```python
# Assuming joints are ordered [left_hip, left_knee, left_ankle, ..., right_hip, right_knee, right_ankle, ...]
def mirror_symmetry_loss(actor_params, apply_fn, obs):
    action = apply_fn({'params': actor_params}, obs)
    mirrored_obs = mirror_observation(obs)  # flip left/right joints and lateral velocity sign
    mirrored_action = apply_fn({'params': actor_params}, mirrored_obs)
    mirrored_action_expected = mirror_action(action)  # flip left/right outputs
    return jp.sum(jp.square(mirrored_action - mirrored_action_expected))
```

---

## 10. Open Duck Community: What Has Been Tried

### From the repositories and documentation

**Backlash modelling**: The project's "current win" is training with `--task flat_terrain_backlash` for 300M steps. Backlash is modelled as dummy passive joints in the MJCF. The observation adds the backlash joint position to the actuator joint position before presenting it to the policy - this teaches the policy to expect the servo's reported position to differ from its commanded position due to gear lash.

**BAM actuator identification**: The project uses Rhoban's BAM tool to identify `damping`, `kp`, `frictionloss`, `armature`, `forcerange` for each servo. This is documented in `docs/sim2real.md` as critical for sim-to-real transfer.

**Motor velocity clamping**: `USE_MOTOR_SPEED_LIMITS = True` in `joystick.py` clamps each motor target to within `±max_motor_velocity × ctrl_dt` of the previous target. This acts as a hard first-order smoothness constraint on the motor targets (distinct from the raw policy output). This is a valuable implicit smoother.

**Action delay randomisation**: The environment randomises 0-3 step delays on both action application and IMU readings. This teaches the policy to be robust to communication latency, which in turn tends to produce more conservative, smoother actions.

**kp reduction at inference**: The readme notes that running inference with `kp = 22` instead of the training value of `kp = 32` produces better real-world motion. Lower kp means the servo tracks more softly - the gearbox compliance absorbs more of the policy's jitter before it becomes visible motion.

**What has not yet been tried** (based on codebase analysis):
- LCP gradient penalty on the actor
- Mirror symmetry loss
- Lateral CoM velocity penalty
- Control frequency reduction (currently fixed at 50 Hz)
- Teacher-student distillation with the full asymmetric actor-critic structure

---

## 11. Our Experimental Results

### Experiment s01: Combined smoothness penalties (REJECTED)

Changed three weights simultaneously from the baseline:
- `action_rate`: -0.5 to -1.0 (doubled)
- `jerk`: 0 to -0.3 (new second-order penalty)
- `orientation`: 0 to -0.2 (new torso orientation penalty)

Trained 150M steps, 8192 envs (~27 min on RTX 5080 Mobile).

**Result**: The robot effectively stopped walking. Forward velocity dropped to 0.0002 m/s (vs 0.079 m/s baseline). Step frequency fell from 3.67 Hz to 0.10 Hz. The gait symmetry score collapsed to 0.25 (vs 0.68 baseline) - the robot pivoted on one leg instead of walking.

The smoothness numbers looked excellent (-63% action rate) because the robot barely moved. This was the key lesson: **smoothness metrics alone are meaningless without locomotion quality metrics**. A policy that stands still has perfect smoothness.

### Key lessons learned

1. **Never change multiple reward weights simultaneously.** Changing three at once made it impossible to determine which change caused the failure. One-factor-at-a-time (OFAT) experiments are essential.

2. **Training reward is unreliable for checkpoint selection.** The training reward is dominated by the `alive` weight (20.0), making it noisy. The "best by reward" checkpoint often barely walks - it scores highly by staying alive in place. We built a walking quality evaluator (`analysis/evaluate_policy.py`) that tests actual locomotion to select the best checkpoint.

3. **tracking_sigma matters enormously.** The original value (0.01) was 25x tighter than industry standard (0.1-0.25). At sigma=0.01, the tracking reward is effectively binary - the robot gets almost zero reward unless it perfectly matches the commanded velocity. This creates an "overshoot trap" where standing still is safer than attempting to walk. Changed to 0.1.

4. **CAD symmetry is a prerequisite.** A 37mm asymmetry in the right lower leg of the original CAD caused the MJCF model, reference motions, and trained policies to all be asymmetric. This was fixed at the source (OnShape) before further training experiments. See `14_cad_symmetry_fix.md` for details.

5. **Stand-then-command is the real test.** Cold-start tests (where velocity is commanded from frame 0) mask responsiveness issues. The real-world scenario is: the robot stands idle, then receives a command. Many policies that walk fine from cold start fail to respond from standstill. Our evaluation tool now tests both scenarios.

### Current approach (after s01)

Rather than aggressive smoothness penalties, the revised strategy is:
1. Fix the model first (symmetric CAD, correct reference motions, correct tracking_sigma)
2. Train an anchor policy with original reward weights on the fixed model
3. Apply one-factor-at-a-time sensitivity sweeps with mild changes (action_rate -0.5 to -0.75, jerk 0 to -0.05)
4. Evaluate each variant with the comprehensive walking quality tool before deciding

---

## Prioritised Implementation Roadmap

Given the analysis above, the recommended order of implementation is:

### Tier 1 - Low effort, high expected impact (implement first)

1. **Add second-order action penalty** (`cost_action_rate2`) at scale `-0.1`. Store `last_qvel` in info and add joint velocity cost at `-1e-3`. These require two new reward functions and two new lines in `_get_reward`.

2. **Reduce lateral sway in reference motions**: Open `gait_playground.py`, reduce the lateral body oscillation parameter, regenerate `polynomial_coefficients.pkl`, retrain.

3. **Add mirror symmetry loss** to the PPO actor update. Requires implementing `mirror_observation()` and `mirror_action()` functions for the Duck's joint ordering.

### Tier 2 - Moderate effort, likely high impact

4. **Add LCP gradient penalty** (`λ_gp = 0.002`) to the PPO actor loss. Requires modifying `runner.py` to compute the Jacobian penalty during actor updates.

5. **Try 20 Hz control rate**: Change `ctrl_dt=0.05`, adjust `max_motor_velocity` clamp, retrain 300M steps.

6. **Add lateral velocity penalty** at `-0.2` to suppress unintended lateral sway.

### Tier 3 - Higher effort, worthwhile if Tier 1/2 insufficient

7. **Two-stage curriculum**: Train stage 1 (high imitation, low smoothness) then stage 2 (lower imitation, high smoothness).

8. **UAN actuator correction**: Record real-world joint trajectories, train a corrective torque network, insert into sim during training.

9. **Enable the low-pass action filter consistently** in both training and inference, passing `motor_targets` (post-filter) as the "last action" observation.

---

## Key References

1. Chen et al. "Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies" arXiv:2410.11825 (IROS 2025) - https://lipschitz-constrained-policy.github.io
2. Mysore et al. "Regularizing Action Policies for Smooth Control with Reinforcement Learning (CAPS)" ICRA 2021 - https://cs-people.bu.edu/rmancuso/files/papers/ICRA21_1616_FI.pdf
3. Ankur et al. "Learning Low-Frequency Motion Control for Robust and Dynamic Robot Locomotion" arXiv:2209.14887 - https://arxiv.org/pdf/2209.14887v2.pdf
4. Lee et al. "Reinforcement Learning Framework for Improving Real-World Performance of SUBO-2" Springer 2026 - https://link.springer.com/article/10.1007/s12541-026-01489-6
5. Benchmarking Smoothness and Reducing High-Frequency Oscillations arXiv:2410.16632 - https://arxiv.org/abs/2410.16632
6. Open Duck Playground (rewards.py, joystick.py) - https://github.com/apirrone/Open_Duck_Playground
7. Open Duck Mini sim2real docs - https://github.com/apirrone/Open_Duck_Mini/blob/v2/docs/sim2real.md
8. Ling et al. "Learning Symmetric and Low-Energy Locomotion" SIGGRAPH 2018 - https://ar5iv.labs.arxiv.org/html/1801.08093
9. Extended friction models for servo actuators arXiv:2410.08650 - https://arxiv.org/pdf/2410.08650
10. Variable Stiffness for Robust Locomotion arXiv:2502.09436 - https://arxiv.org/abs/2502.09436

---

## Methodology

Searched 25+ queries across: arXiv preprints, GitHub repositories, project documentation, and robotics conference papers. Analysed source code of Open Duck Playground (rewards.py, joystick.py, base.py). Sub-questions investigated: action smoothness reward terms, low-pass filtering techniques, curriculum learning for locomotion, teacher-student distillation, actuator-aware training, reference motion quality, control frequency, impedance/compliance control, humanoid robotics papers (MIT, Berkeley, Unitree), Open Duck community activity.
