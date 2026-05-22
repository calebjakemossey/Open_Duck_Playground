# Experiment Catalogue

All training experiments on Open Duck Mini V2. Each entry records what was changed, why, the quantitative results, and the verdict.

---

## Baselines

Two pre-trained policies provided by the upstream project (apirrone/tnkr.ai). These are the reference points for all experiments.

| Metric | baseline_1 (BEST_WALK_ONNX.onnx) | baseline_2 (BEST_WALK_ONNX_2.onnx) |
|--------|-----------------------------------|--------------------------------------|
| Forward vx (cmd 0.1 m/s) | 0.079 m/s | 0.034 m/s |
| vx tracking score | 0.81 | 0.71 |
| Step frequency | 3.78 Hz | 2.94 Hz |
| Foot lift | 0.009 m | 0.007 m |
| Mean tilt | 2.2 deg | 3.2 deg |
| Action rate (smoothness) | 0.072 | 0.057 |
| Jerk | 0.072 | 0.062 |
| HF ratio | 0.226 | 0.203 |
| Push survival (0.6 N-s) | All directions | Front + side only |
| Strafe from standstill | Works (left good, right slow) | Does not strafe |
| Backward from standstill | Does not work | Does not work |

**baseline_1** is functionally better (faster, more responsive, handles pushes from all directions, strafes). **baseline_2** is smoother but at the cost of locomotion capability.

---

## Phase 1: s01 - Combined Smoothness Penalties

**Date**: 2026-05-10
**Model**: Original asymmetric MJCF (pre-CAD fix)
**Changes from baseline**:
- `action_rate`: -0.5 to -1.0
- `jerk`: 0 to -0.3 (new)
- `orientation`: 0 to -0.2 (new)

**Training**: 150M steps, 8192 envs, ~27 min

### Results

| Metric | baseline_1 | s01_final | Change |
|--------|-----------|-----------|--------|
| Forward vx | 0.079 | 0.0002 | -99.7% |
| vx tracking | 0.81 | 0.59 | -27% |
| Step frequency | 3.78 | 1.29 | -66% |
| Foot lift | 0.009 | 0.003 | -67% |
| Mean tilt | 2.2 | 4.4 | +100% |
| Action rate | 0.072 | 0.026 | -64% |
| Push 0.6 N-s front | 100% | 0% | lost |
| Gait symmetry | 0.68 | 0.25 | -63% |

**Verdict: REJECT.** Over-penalised. Robot effectively stopped walking. Changed too many things at once - impossible to determine which change caused the failure.

**Lessons**:
1. One-factor-at-a-time experiments only
2. Smoothness metrics are meaningless without locomotion quality metrics
3. Training reward (dominated by alive=20.0) is unreliable for checkpoint selection

---

## Phase 2: Anchor v1 - Reproduction with Original Weights

**Date**: 2026-05-11
**Model**: Original asymmetric MJCF (pre-CAD fix)
**Changes**: None - exact original reward config, trained from scratch
**Training**: 150M steps, 8192 envs, ~30 min

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| HEADLINE score | 0.27 | vs estimated baseline_1 ~0.50 |
| Forward cold start | Works (vx=0.058, gait_sym=0.88) | |
| Stand-then-forward | Broken (vx=0.0001) | Same failure as user-reported s01 issue |
| Turn-left cold start | Works well | |
| Stand-then-turn-left | Works | |
| Backward | Broken (cold and stand) | |
| Stand-then-strafe | Broken | |

**Verdict: PARTIAL.** Walks forward from cold start but fails to respond from standstill. Same limitation seen across all our locally trained policies. baseline_1 is the only policy that responds well from standstill - suggesting its training config or duration differed from what we can reproduce.

**Key insight**: The "best by reward" checkpoint barely walked. Walking quality evaluation (not training reward) is essential for checkpoint selection. Walking score was still rising at step 140M, suggesting undertraining.

---

## Infrastructure Built

### Analysis Tools (in `analysis/`)

| Tool | Purpose |
|------|---------|
| `evaluate_policy.py` (v4) | 13 cold-start + 14 stand-then-command + push recovery + gait symmetry tests |
| `quick_walking_eval.py` | 0.5s walking assessment hooked into training callbacks |
| `find_best_checkpoint.py` | Sweeps all checkpoints by walking quality, not training reward |
| `analyse_training.py` | Reads TensorBoard events, reports plateau/regression |
| `debug_policy.py` | Per-step velocity traces for temporal failure analysis |
| `replay_reference.py` | Visual playback of reference motions in MuJoCo viewer |
| `backfill_walking_metrics.py` | Adds walking metrics to existing TF event files post-hoc |

### Model Fixes

| Fix | Impact |
|-----|--------|
| CAD symmetry (37mm right leg offset) | Fixed at OnShape source, re-exported MJCF/URDF |
| Home keyframe | Recalculated for symmetric model, both feet at identical height |
| Reference motion grid (zero-entry bug) | auto_gait.json fixed: dy and dtheta grids now include zero |
| Reference motions regenerated | 210 motions on symmetric URDF |
| tracking_sigma 0.01 to 0.1 | Industry standard, prevents overshoot trap |
| get_gravity() sensor bug | Was using sensor ID instead of address |
| Dependency pins | playground==0.0.5, jax<0.7 |

---

## Stage 2: Disney-Style Experiments

### Disney Run 1: imit=15, alive=20, action_rate=-1.5, action_accel=-0.45

**Date**: 2026-05-14
**Training**: 50M steps, 8192 envs, local RTX 5080
**Config**: `tracking_lin_vel=2.5, tracking_ang_vel=2.5, action_rate=-1.5, action_accel=-0.45, stand_still=-1.0, alive=20.0, imitation=15.0, tracking_sigma=0.25`

**Result: BEST EVER** (peak at 36M, regressed after)

| Metric | Run 6 | Disney Run 1 (36M) | Change |
|--------|-------|---------------------|--------|
| vx_tracking | 0.595 | **0.699** | +17% |
| gait_symmetry | 0.697 | **0.826** | +19% |
| st_fwd | 0.993 | **0.986** | same |
| st_back | 0.288 | **0.465** | +62% |
| st_turn_R | 0.570 | 0.144 | -75% |
| action_rate | 0.072 | **0.069** | best ever |
| HEADLINE | 0.532 | **0.604** | +14% |

**Verdict**: KEEP (36M checkpoint). Best forward/backward/symmetry/smoothness. Turning regression needs fixing.

### Disney Run 2: imit=8 (stability test)

**Date**: 2026-05-14
**Training**: 50M steps, 8192 envs, local RTX 5080
**Config**: Same as Run 1 except `imitation=8.0`

**Result: REJECTED** - halving imitation destroyed responsiveness

| Metric | Disney Run 1 (36M) | Disney Run 2 (50M) | Change |
|--------|---------------------|---------------------|--------|
| st_fwd | 0.986 | 0.131 | -87% |
| vx_tracking | 0.699 | 0.123 | -82% |
| mean_responsiveness | 0.517 | 0.135 | -74% |
| action_rate | 0.069 | 0.059 | -14% |

**Verdict**: DISCARD. Confirms imitation=15.0 is necessary with action_rate=-1.5. The imit/alive ratio=0.40 is insufficient when action_rate is this restrictive.

### Disney Run 3: ang_vel=4.0 (fix turning) - REJECTED

**Date**: 2026-05-14
**Training**: 50M steps, 8192 envs, local RTX 5080
**Config**: Disney Run 1 base with `tracking_ang_vel=4.0` (doubled from 2.5)
**Hypothesis**: Higher angular velocity reward fixes Disney Run 1's turning weakness

**Result: REJECTED** - turning got WORSE despite doubling the ang_vel reward

| Metric | Disney Run 1 (36M) | Disney Run 3 (50M) | Change |
|--------|---------------------|---------------------|--------|
| st_turn_L | 0.600 | 0.000 | **-100%** |
| st_turn_R | 0.144 | 0.033 | -77% |
| wz_tracking | 0.587 | 0.478 | -19% |
| gait_symmetry | 0.826 | 0.891 | +8% (only win) |
| st_fwd | 0.986 | 0.946 | -4% |
| st_back | 0.465 | 0.196 | -58% |
| mean_responsiveness | 0.517 | 0.351 | -32% |
| HEADLINE (training) | 0.604 | 0.365 | -40% |

**Verdict**: DISCARD. Boosting ang_vel created conflicting gradients with imitation reward. wz_tracking dropped 19% despite doubling the ang_vel weight - the opposite of expected. Neither loosening action_rate (Run 4) nor boosting ang_vel (Run 3) fixes turning in the Disney config. The Disney branch may have a fundamental turning ceiling at imit=15/alive=20/act_rate=-1.5.

**Key insight**: The Disney config's tight action_rate + high imitation creates a strong attractor for the reference walking gait. Turning requires deviating from this reference (asymmetric leg movements), but the imitation reward penalises deviation. Increasing ang_vel competes directly with imitation rather than complementing it.

### Disney Run 4: action_rate=-0.5 (loosen smoothness)

**Date**: 2026-05-14
**Training**: 50M steps, 8192 envs, local RTX 5080
**Config**: Disney Run 1 base with `action_rate=-0.5` (loosened from -1.5)
**Hypothesis**: Loosening smoothness constraint recovers turning while keeping quality

**Result: REJECTED** - worse on almost everything

| Metric | Disney Run 1 (36M) | Disney Run 4 (50M) | Change |
|--------|---------------------|---------------------|--------|
| action_rate | 0.069 | 0.078 | +14% (less smooth) |
| jerk | 0.050 | 0.065 | +31% (less smooth) |
| gait_symmetry | 0.826 | 0.861 | +4% |
| mean_tilt_deg | 2.67 | 2.01 | -25% (more stable) |
| vx_tracking | 0.699 | 0.685 | -2% |
| wz_tracking | 0.587 | 0.528 | -10% |
| st_fwd | 0.986 | 0.949 | -4% |
| st_turn_L | 0.600 | 0.134 | **-78%** |
| st_turn_R | 0.144 | 0.229 | +59% |
| mean_responsiveness | 0.517 | 0.424 | -18% |
| HEADLINE (training) | 0.604 | 0.477 | -21% |

**Verdict**: DISCARD. Loosening action_rate did NOT help turning - left turning got catastrophically worse. Smoothness regressed as expected. The turning problem is definitively about angular velocity reward weight, not smoothness constraint. This validates Disney Run 3 (ang_vel=4.0) as the correct next experiment.

**Key insight**: action_rate=-1.5 is actively HELPING turning (Run 1 st_turn_L=0.600 vs Run 4 st_turn_L=0.134). The tight smoothness constraint forces the policy to learn more deliberate, committed turns rather than jittery half-turns.

---

## Kaggle Infrastructure - ABANDONED

Kaggle GPU (P100) validated but 7x slower than local RTX 5080. Kaggle TPU failed to allocate (kernel ran on CPU). All 3 running kernels (disney-angvel-4, run6-angvel-25, disney-angvel-4-tpu) abandoned. All future training runs local-only.

---

## Training Stability Research (2026-05-14 to 2026-05-15)

Full details in `docs/19_training_stability_research.md`. Summary of experiments below.

### E3: LayerNorm Only - FAILED

**Date**: 2026-05-14
**Config**: ExpA + LayerNorm on policy network
**Result**: Policy never learned to walk (HEADLINE 0.000-0.085 over 120M steps). Reward climbed but no locomotion. LayerNorm is incompatible with this architecture/hyperparameter combination.

### E1: Reward Rescale + LayerNorm + Adaptive KL - FAILED

**Date**: 2026-05-14
**Config**: ExpA with all weights /10, LayerNorm, Adaptive KL (desired_kl=0.01)
**Result**: Same failure as E3. LayerNorm is the common factor killing learning.

### E2a: Lower imit/alive, NO LayerNorm - COMPLETED (best stability so far)

**Date**: 2026-05-14
**Config**: imit=5, alive=10, no LayerNorm, no Adaptive KL

| Metric | ExpA (43M peak) | E2a (150M final) |
|--------|-----------------|------------------|
| HEADLINE | 0.692 | 0.481 |
| st_fwd | 0.935 | 0.836 |
| st_turn_L | 0.721 | 0.149 |
| action_rate | 0.069 | 0.068 |
| jerk | 0.050 | 0.050 |
| collapse? | Yes @ 75M | **No** |

**Verdict**: KEEP as reference. Smoothest policy ever. No training collapse. But turning from standstill still broken. Learned slowly (needed ~120M to reach quality).

### E4: Bounded Imitation, Sharp Kernel (w=15) - FAILED

**Date**: 2026-05-15
**Config**: ExpA weights + `joint_pos_rew` changed to `exp(-15 * sum_sq_error)` (bounded [0,1])
**Result**: Policy never learned (HEADLINE=0.008 for 64M steps). Exp-kernel too sharp, no gradient signal at distance.

### E4b: Bounded Imitation, Soft Kernel (w=2) - FAILED

**Date**: 2026-05-15
**Config**: Same as E4 but `w_joint_pos=2.0` for broader kernel
**Result**: Peaked at 0.110 at 54M, collapsed by 97M. Same collapse pattern as ExpA, just weaker.

**Consolidated finding**: The training collapse is driven by outer reward magnitudes (imit=15 + alive=20 drowning tracking=2.5), not the inner kernel shape. Only E2a (imit=5, alive=10) avoided collapse.

### E5: Imitation Curriculum (two-phase training) - COMPLETE

**Date**: 2026-05-15
**Strategy**: Phase 1: imit=15 for 40M (fast learning). Phase 2: warm-start with imit=5 for 110M (stable refinement).

**Result**: Best all-round policy we've trained. Curriculum prevented catastrophic collapse.

| Metric | E2a (150M) | E5 best (65M) | Baseline 1 |
|--------|-----------|---------------|------------|
| st_forward | 0.836 | **0.997** | 0.809 |
| st_strafe_L | 0.778 | **0.982** | 0.001 |
| st_turn_L | 0.149 | 0.032 | **0.647** |
| st_turn_R | 0.110 | **0.318** | **0.747** |
| gait_symmetry | 0.705 | **0.885** | 0.668 |
| jerk | 0.050 | **0.049** | 0.071 |
| mean_tilt_deg | 1.92 | **1.68** | 2.45 |

**Verdict**: KEEP as best trained policy. The curriculum approach works - prevents collapse while retaining fast early learning. Turning from standstill remains unsolved across all experiments. Phase 2 shows skill oscillation (0.39-0.63 range) - best checkpoint selection is necessary.

**Best checkpoint**: `checkpoints/E5_curriculum_phase2/2026_05_15_013604_23592960.onnx` (65M total)

---

## Infrastructure: ONNX Export Rewrite

**Date**: 2026-05-15
**Problem**: TF-based ONNX export took 7 min per checkpoint (PTX JIT for RTX 5080 compute 12.0a) and crashed training by competing for GPU memory.
**Fix**: Rewrote `export_onnx.py` using `onnx.helper` - no TF dependency. 0.08s per export (5,250x faster). Output matches old exporter within float32 tolerance.

---

## Stage 3: Turning-From-Standstill Investigation

### Walking-Turning Trade-off Discovery

**Date**: 2026-05-15

Quadrant analysis of all 57 evaluated policies revealed a near-perfect inverse correlation between walking-from-standstill and turning-from-standstill:

| Group | Count | Config | st_fwd | st_turn_L/R | wz_tracking |
|-------|-------|--------|--------|-------------|-------------|
| Walks + Turns | 1 | baseline_1 (unknown) | 0.809 | 0.647/0.747 | 0.847 |
| Walks only | 13 | Disney (sigma=0.25, act_rate=-1.5, act_accel=-0.45) | 0.8-1.0 | 0.0-0.15 | 0.49-0.53 |
| Turns only | 43 | Original (sigma=0.1, act_rate=-0.5, no act_accel) | 0.018-0.33 | 0.8-1.0 | 0.94-0.99 |

Three config variables changed together between the two groups: tracking_sigma (0.1 vs 0.25), action_rate (-0.5 vs -1.5), action_accel (0 vs -0.45). Baseline_1 is the only policy that bridges both capabilities.

### E6: Biased Command Sampling - BEST WALKER

**Date**: 2026-05-15
**Config**: Disney base (imit=15, alive=20, action_rate=-1.5, action_accel=-0.45, sigma=0.25) + 12% pure-turn command sampling (vx=0, vy=0, wz from full range) in `sample_command`.
**Training**: 50M steps, 8192 envs, local RTX 5080
**Change**: Added biased sampling to `joystick.py sample_command()` - 12% pure turn, 10% stand still, 78% normal. Previously only ~1.3% of commands were pure-turn due to uniform sampling.

| Metric | E5 best (65M) | E6 best (46M) | Change |
|--------|--------------|---------------|--------|
| st_forward | 0.997 | **0.996** | same |
| st_strafe_R | - | **0.955** | new |
| st_turn_L | 0.032 | 0.013 | same |
| st_turn_R | 0.318 | 0.267 | -16% |
| st_fwd_turn_L | - | **0.500** | new |
| st_fwd_turn_R | - | **0.583** | new |
| st_strafe_turn_R | - | **0.486** | new |
| mean_responsiveness | 0.376 | **0.496** | +32% |
| gait_symmetry | 0.885 | 0.870 | -2% |
| action_rate | 0.049 | 0.070 | +43% |

**Verdict**: KEEP as best overall policy. Biased sampling significantly improved combined movements (turning while walking/strafing) and overall responsiveness. Pure in-place turning still broken. Smoothness slightly worse than E5 (expected with Disney action_rate=-1.5 vs E5's curriculum).

**Best checkpoint**: `checkpoints/E6_biased_sampling/2026_05_15_084635_46858240.onnx`

### E7: Biased Sampling + Reduced w_joint_pos - REJECTED

**Date**: 2026-05-15
**Config**: E6 base + w_joint_pos reduced from 15.0 to 5.0 in `custom_rewards.py`
**Training**: 50M steps, 8192 envs, local RTX 5080
**Hypothesis**: Loosening the joint position penalty would let the policy explore asymmetric leg movements needed for in-place pivoting.

| Metric | E6 (46M) | E7 (36M) | Change |
|--------|----------|----------|--------|
| st_forward | 0.996 | 0.974 | -2% |
| st_turn_L | 0.013 | 0.004 | worse |
| st_turn_R | 0.267 | 0.001 | -99% |
| gait_symmetry | 0.870 | 0.556 | -36% |
| mean_responsiveness | 0.496 | 0.272 | -45% |

**Verdict**: REJECT. Reducing w_joint_pos made everything worse. The joint position weight was helping maintain structured gait, not blocking turning. w_joint_pos restored to 15.0.

### E8: Conditional Imitation Suppression - CANCELLED

**Date**: 2026-05-15
**Config**: E6 base + imitation reward scaled by `1.0 - 0.9 * turn_ratio` (pure turns get 10% imitation)
**Hypothesis**: Suppress imitation for pure-turn commands since reference motions produce only 3% of commanded angular velocity.
**Status**: Cancelled during training after discovering the angular velocity recording bug (see below). The suppression was a workaround for bad reference data, not a fundamental fix.

### Angular Velocity Recording Bug in Placo gait_generator.py

**Date discovered**: 2026-05-15
**File**: `Open_Duck_reference_motion_generator/open_duck_reference_motion_generator/gait_generator.py` line 180

**Bug**: `compute_angular_velocity()` double-multiplies by the rotation angle:

```python
axis, angle = r_rel.as_rotvec(), np.linalg.norm(r_rel.as_rotvec())
angular_velocity = axis * (angle / dt)  # axis is already axis_unit * angle
```

`as_rotvec()` returns `axis_unit * angle`. Multiplying by `angle/dt` gives `axis_unit * angle^2 / dt` instead of the correct `axis_unit * angle / dt`.

**Fix**: `angular_velocity = r_rel.as_rotvec() / dt`

**Impact**: All `world_angular_vel` values in recordings and polynomial fits are 8-12x too small. Evidence from recording `254_0.0_0.0_1.037.json`:

| Source | ang_vel_z (deg/s) |
|--------|-------------------|
| Quaternion trajectory (ground truth) | 12.57 |
| Recorded world_angular_vel_z | 1.55 |
| Metadata Yaw field | 1.62 |
| Ratio (truth/recorded) | 8.1x |

This means the imitation reward's `ang_vel_z_rew` sub-term (w=0.5) has been comparing against bogus reference values for all training runs. During a turn, the sub-term penalises actual rotation (rewards matching ~0.03 rad/s instead of the true ~0.22 rad/s). While the sub-term weight is small (0.5), it provides a directionally wrong gradient signal.

**Status**: Fixed. `gait_generator.py` line 180 corrected. Existing recordings patched via `scripts/fix_angular_vel.py` (recomputes ang_vel from quaternion finite differences). Polynomials refitted. Verified: recording `254_0.0_0.0_1.037.json` now shows `mean ang_vel_z = 1.10 rad/s` (was 0.13 rad/s).

---

### E9: Fixed Angular Velocity References + E6 Config (50M)

**Date**: 2026-05-15
**Config**: Identical to E6 (Disney base + biased command sampling). Only change: polynomial_coefficients.pkl regenerated from patched recordings with correct angular velocities.
**Steps**: 50,000,000 (8192 envs)
**Checkpoint dir**: `checkpoints/E9_fixed_angvel/`
**Best checkpoint**: `2026_05_15_103501_50462720.onnx` (50M, final)

**Hypothesis**: The angular velocity bug caused the imitation reward to penalise actual turning. Fixing it should unblock turn-from-standstill without any reward weight changes.

**Result**: **NEW BEST OVERALL**. Turn-from-standstill fixed. HEADLINE=0.714 (previous best: Disney R1 0.604).

| Metric | E6 (best prior) | Disney R1 36M | **E9 50M** | Change vs E6 |
|--------|-----------------|---------------|------------|--------------|
| HEADLINE | 0.453 | 0.604 | **0.714** | +58% |
| st_forward | ~0.85 | 0.699 | **0.998** | +17% |
| st_turn_L | 0.013 | 0.144 | **0.678** | +5115% |
| st_turn_R | 0.013 | - | **0.925** | +7015% |
| st_strafe_L | - | - | **0.744** | - |
| st_strafe_R | - | - | **1.000** | - |
| st_fwd_turn_L | - | - | **0.889** | - |
| st_fwd_turn_R | - | - | **0.953** | - |
| gait_symmetry | 0.870 | 0.826 | **0.882** | +1% |
| survival | 1.000 | 1.000 | **1.000** | same |
| wz_tracking | - | - | **0.733** | - |
| action_rate | 0.062 | 0.069 | 0.074 | +19% |
| st_backward | - | - | 0.240 | - |
| st_back_turn_L | - | - | 0.127 | - |
| st_back_turn_R | - | - | 0.024 | - |
| mean_responsiveness | - | - | **0.709** | - |

**HEADLINE progression** (quick eval per checkpoint):

| Steps | HEADLINE | cold_start | responsive |
|-------|----------|------------|------------|
| 0M | 0.004 | 0.005 | 0.003 |
| 14.4M | 0.015 | 0.014 | 0.017 |
| 21.6M | 0.126 | 0.160 | 0.093 |
| 25.2M | 0.411 | 0.440 | 0.381 |
| 28.8M | 0.588 | 0.595 | 0.582 |
| 32.4M | 0.691 | 0.722 | 0.659 |
| 36.0M | 0.697 | 0.727 | 0.668 |
| 43.2M | 0.700 | 0.690 | 0.710 |
| 50.0M | 0.714 | 0.717 | 0.712 |

**Extended training (300M, completed)**: Training was extended from 50M checkpoint to 300M. HEADLINE oscillated significantly (0.365-0.757) but eventually produced a new best at checkpoint 14 (~328M total).

| Total Steps | HEADLINE | cold_start | responsive |
|-------------|----------|------------|------------|
| ~50M (restored) | 0.726 | 0.735 | 0.717 |
| ~70M | 0.606 | 0.607 | 0.605 |
| ~90M | 0.675 | 0.738 | 0.613 |
| ~110M | 0.409 | 0.467 | 0.352 |
| ~136M | 0.365 | 0.364 | 0.366 |
| ~157M | 0.450 | 0.518 | 0.382 |
| ~178M | 0.588 | 0.615 | 0.561 |
| ~200M | 0.622 | 0.630 | 0.613 |
| ~221M | 0.516 | 0.600 | 0.431 |
| ~242M | 0.643 | 0.592 | 0.694 |
| ~264M | 0.599 | 0.713 | 0.484 |
| ~285M | 0.703 | 0.707 | 0.700 |
| **~328M** | **0.757** | **0.814** | 0.700 |
| ~350M (final) | 0.684 | 0.748 | 0.619 |

**Best checkpoint**: `E9_fixed_angvel_300M/2026_05_15_111900_279019520.onnx` (HEADLINE=0.757, checkpoint 14 at ~328M total). This is our best policy across all experiments.

**Verdict**: KEEP. Bug fix was the root cause of turn-from-standstill failure across all 57+ prior experiments. No reward weight changes needed - the same E6 config produces dramatically better turning with correct reference data. Extended training (300M) does eventually beat the 50M peak, but only by cherry-picking the best checkpoint from a heavily oscillating trajectory. The final checkpoint (0.684) is worse than the best (0.757).

**Remaining weaknesses**:
- Backward walking (0.240) and backward+turn (0.024-0.127) - likely a genuine limitation of the Placo reference motions for backward gaits
- Training instability - HEADLINE oscillates wildly (0.365-0.757) rather than converging, requiring checkpoint selection rather than using the final model

**Full evaluation of best checkpoint (CP14, 279M steps)**:

Cold-start locomotion (100% survival across all scenarios):

| Scenario | Achieved | Target | Notes |
|----------|----------|--------|-------|
| forward | vx=0.099 | 0.10 | Near-perfect |
| turn_left | wz=0.841 | 0.80 | Excellent |
| turn_right | wz=-0.762 | -0.80 | Good |
| strafe_right | vy=-0.135 | -0.15 | Good |
| strafe_left | vy=0.129 | 0.15 | Decent |
| backward | vx=-0.048 | -0.10 | Half-speed |

Responsiveness (stand-then-command):

| Test | Score | Issue |
|------|-------|-------|
| stand_then_forward | 0.999 | - |
| stand_then_turn_L/R | 0.950/0.951 | - |
| stand_then_strafe_right | 0.906 | - |
| stand_then_fwd_turn_L/R | 0.858/0.897 | - |
| stand_then_fwd_strafe_L/R | 0.884/0.670 | Right side weaker |
| stand_then_backward | 0.334 | Half-speed |
| stand_then_back_turn_L/R | 0.115/0.203 | Very weak |
| stand_then_strafe_left | 0.000 | Completely broken (right works fine) |

---

### E10: Tight Tracking Sigma + Value Clipping (150M)

**Date**: 2026-05-15
**Config**: Same as E9 (Disney base + biased sampling + fixed ang_vel references) with two changes:
1. `tracking_sigma`: 0.25 to 0.05 (aligns training reward with evaluation scoring)
2. `clipping_epsilon_value`: None to 0.2 (prevents value function overshooting)

**Steps**: 150,000,000 (8192 envs, from scratch)
**Checkpoint dir**: `checkpoints/E10_tight_tracking/`

**Why these changes**:

The training reward didn't correlate with walking quality in E9 - at the HEADLINE trough (0.365), training reward was 607 (above average). Root cause: `tracking_sigma=0.25` is so loose that a robot achieving 50% of commanded velocity gets 0.99 training reward. The evaluation uses `scale=0.05` (5x tighter). With sigma=0.25, the optimiser has no gradient to push the policy from "roughly moving" to "precisely tracking."

Value function clipping (`clipping_epsilon_value=0.2`) is standard PPO practice that prevents the critic from overshooting on each update. Brax supports it natively but it wasn't enabled.

**What we hope to learn**:
1. Does tighter tracking_sigma eliminate the HEADLINE oscillation seen in E9?
2. Does the policy converge monotonically rather than oscillating?
3. Does early training still work (tight sigma = sparser reward signal initially)?

**Success criteria** (vs E9 best, HEADLINE=0.757):
- HEADLINE >= 0.75
- Monotonic HEADLINE improvement (no oscillation > 0.1)
- Final checkpoint within 5% of best checkpoint (convergence, not luck)

**Results**:

| CP | Steps | HEADLINE | cold_start | responsive | Training reward |
|----|-------|----------|------------|------------|-----------------|
| 1  | 0 | 0.004 | 0.005 | 0.003 | - |
| 2  | 7.2M | 0.008 | 0.006 | 0.011 | - |
| 3  | 10.8M | 0.009 | 0.006 | 0.012 | - |
| 4  | 21.6M | 0.255 | 0.250 | 0.259 | 438 |
| **5** | **32.4M** | **0.641** | **0.623** | **0.659** | **477** |
| 6  | 43.3M | 0.479 | 0.571 | 0.387 | 558 |
| 7  | 54.1M | 0.440 | 0.512 | 0.368 | 605 |
| 8  | 64.9M | 0.300 | 0.204 | 0.396 | 545 |
| 9  | 86.5M | 0.438 | 0.483 | 0.393 | 609 |
| 10 | 97.3M | 0.514 | 0.594 | 0.433 | 625 |
| 11 | 108.1M | 0.544 | 0.523 | 0.564 | 586 |
| 12 | 118.9M | 0.493 | 0.480 | 0.506 | 589 |
| 13 | 129.8M | 0.497 | 0.475 | 0.518 | 584 |
| 14 | 140.6M | 0.452 | 0.457 | 0.446 | 649 |
| 15 | 151.4M | 0.481 | 0.474 | 0.487 | 501 |

**Best checkpoint**: CP5 at 32.4M steps (HEADLINE=0.641)
**Best ONNX**: `2026_05_15_120353_32440320.onnx`

**Full eval of CP5** (via `evaluate_policy.py`):

| Scenario | vx | vy | wz | Notes |
|----------|-----|-----|-----|-------|
| Forward walk | 0.144 | 0.888 | 0.134 | Very poor forward tracking |
| Backward walk | 0.119 | 0.898 | 0.998 | Can't walk backward |
| Lateral left | 0.965 | 0.004 | 0.680 | Can't strafe left |
| Lateral right | 0.749 | 0.626 | 0.437 | Partial |
| Turn left | 0.398 | 0.680 | 0.449 | Weak |
| Turn right | 0.723 | 0.951 | 0.006 | Can't turn right |
| Survival | 100% | - | - | Never falls |
| Responsiveness mean | 0.304 | - | - | Much worse than E9 (0.757) |

**Verdict**: FAILED all three success criteria.
1. HEADLINE (0.641) below target (0.75) and below E9's best (0.757)
2. Oscillation persists: 0.641 -> 0.300 -> 0.544 -> 0.452 (swings of 0.34)
3. Final checkpoint (0.481) is 25% below best - no convergence

**Why it failed**: Tightening tracking_sigma made precision harder to earn, but alive=20 still dominates. Training reward and HEADLINE are negatively correlated after CP5 - the optimiser improved survival (reward 477 -> 649) while walking quality degraded (HEADLINE 0.641 -> 0.452). The alive:tracking ratio (~4:1 in reward magnitude) means the optimiser gets far more signal from staying upright than from tracking velocity precisely.

**What we learned**:
- Tighter sigma alone makes things worse without rebalancing the alive:tracking ratio
- Value clipping (0.2) didn't prevent oscillation
- The alive-dominance problem is the primary remaining issue: alive earns ~400/episode, tracking earns ~100/episode max
- Training reward going UP while HEADLINE goes DOWN is the clearest possible evidence of reward misalignment

---

### E11: Reduced Alive Weight (150M)

**Date**: 2026-05-15
**Config**: Same as E9 (Disney base + biased sampling + fixed ang_vel references) with one change:
1. `alive`: 20.0 to 5.0 (reduces alive dominance from ~4:1 to ~1:1 vs tracking)

E10's changes (tracking_sigma=0.05, clipping_epsilon_value=0.2) are reverted to isolate the alive variable against the E9 baseline.

**Steps**: 150,000,000 (8192 envs, from scratch)
**Checkpoint dir**: `checkpoints/E11_reduced_alive/`

**Why this change**:

E10 proved that alive dominance is the primary issue. Training reward went UP while HEADLINE went DOWN - the optimiser maximised survival (alive=20 earns ~400/episode) at the expense of velocity tracking (tracking=2.5 earns ~100/episode max). Reducing alive to 5.0 brings the ratio to ~1:1 (alive ~100/episode vs tracking ~100/episode).

Prior evidence supports this:
- LHS screening: best runs (Run 6, Run 14) both had alive=3.2-3.4
- E2a (alive=10, imit=5): only run that avoided training collapse entirely
- Neither of those had the angular velocity bug fix (+0.30 HEADLINE)

**What we hope to learn**:
1. Does reducing alive eliminate training oscillation?
2. Does the policy still survive (alive=5.0 is above the LHS floor of 3.5)?
3. With alive rebalanced, does the ang_vel bug fix push HEADLINE past E9's 0.757?

**Success criteria** (vs E9 best, HEADLINE=0.757):
- HEADLINE >= 0.75
- No training collapse (survival stays at 100%)
- Less oscillation than E9/E10

**Results**:

| CP | Steps | HEADLINE | cold_start | responsive | Training reward |
|----|-------|----------|------------|------------|-----------------|
| 1  | 0 | 0.004 | 0.005 | 0.003 | 0.4 |
| 2  | 10.8M | 0.007 | 0.005 | 0.009 | 272 |
| 3  | 10.8M | 0.054 | 0.072 | 0.037 | 272 |
| 4  | 21.6M | 0.614 | 0.540 | 0.688 | 366 |
| 5  | 32.4M | 0.864 | 0.869 | 0.860 | 357 |
| 6  | 43.3M | 0.820 | 0.825 | 0.815 | 429 |
| 7  | 54.1M | 0.740 | 0.698 | 0.783 | 439 |
| **8** | **64.9M** | **0.876** | **0.887** | **0.865** | **390** |
| 9  | 75.7M | 0.793 | 0.805 | 0.780 | 429 |
| 10 | 86.5M | 0.735 | 0.724 | 0.745 | 440 |
| 11 | 97.3M | 0.792 | 0.801 | 0.783 | 480 |
| 12 | 108.1M | 0.863 | 0.845 | 0.880 | 423 |
| 13 | 118.9M | 0.815 | 0.804 | 0.825 | 443 |
| 14 | 129.8M | 0.771 | 0.760 | 0.782 | 414 |
| 15 | 140.6M | 0.759 | 0.725 | 0.792 | 455 |
| 16 | 151.4M | 0.759 | 0.725 | 0.792 | 377 |

**Best checkpoint**: CP8 at 64.9M steps (HEADLINE=0.876) - NEW ALL-TIME BEST
**Best ONNX**: `2026_05_15_125015_64880640.onnx`

**Full eval of CP8** (via `evaluate_policy.py`):

| Capability | E9 CP14 | E11 CP8 | Change |
|-----------|---------|---------|--------|
| Forward walk (vx) | 0.999 | 0.882 | -0.117 |
| Backward walk (vx) | 0.334 | 0.761 | **+0.427** |
| Strafe left (stand-then) | 0.000 | 0.983 | **+0.983** |
| Strafe right (stand-then) | 0.906 | 1.000 | +0.094 |
| Turn left (stand-then) | 0.950 | 0.351 | -0.599 |
| Turn right (stand-then) | 0.951 | 0.583 | -0.368 |
| Fwd + turn left | 0.897 | 0.996 | +0.099 |
| Fwd + turn right | 0.858 | 0.922 | +0.064 |
| Back + turn left | 0.115 | 0.206 | +0.091 |
| Back + turn right | 0.203 | 0.259 | +0.056 |
| Strafe + turn left | - | 0.974 | NEW |
| Strafe + turn right | - | 0.980 | NEW |
| Survival | 100% | 100% | same |
| Gait symmetry | - | 0.845 | - |
| HEADLINE | 0.757 | **0.876** | **+0.119** |

**Verdict**: PASSED two of three criteria, partial on third.
1. HEADLINE (0.876) exceeds target (0.75) and E9's best (0.757) - **PASS**
2. Survival stays at 100% despite alive=5 - **PASS**
3. Oscillation reduced but not eliminated: amplitude ~0.13 vs E10's ~0.34 - **PARTIAL**

**What we learned**:
- Reducing alive from 20 to 5 is the single most impactful tuning change we've made (after the ang_vel bug fix)
- Training reward and HEADLINE are now more synchronised (no systematic divergence)
- Fixes strafe-left asymmetry (0.000 → 0.983) and backward walking (0.334 → 0.761) without any targeted changes
- Pure turning from standstill regressed (0.95 → 0.35-0.58) - trade-off of lower alive
- Oscillation persists (~0.13 amplitude, ~20M step cycle) - likely needs adaptive KL or LR decay to converge
- Best checkpoint at 65M, not end of training - still need checkpoint selection strategy

---

### E12: Stable Training + Standing Fix (150M)

**Date**: 2026-05-15
**Config**: Same as E11 (alive=5, Disney base + biased sampling + fixed ang_vel references) with three changes:
1. `num_updates_per_batch`: 4 to 2 (reduces data staleness causing oscillation)
2. `clipping_epsilon_value`: None to 0.2 (prevents value function overshooting)
3. `stand_still`: -1.0 to -5.0 (fixes standing bopping/yaw drift - imitation=15 was overwhelming stand_still=-1)

**Steps**: 150,000,000 (8192 envs, from scratch)
**Checkpoint dir**: `checkpoints/E12_stable_training/`

**Why these changes**:

E11 achieved best-ever HEADLINE (0.876) but had two remaining issues:
1. Training oscillates ~0.13 amplitude over ~20M step cycles. Root cause: 4 PPO epochs per rollout push policy far from data-collection policy, causing value function lag and advantage oscillation. PPO paper recommends 1-3 epochs for continuous control.
2. Duck bops and rotates right when standing (wz=-0.465 rad/s, wz_tracking=0.004). stand_still=-1.0 is 15x weaker than imitation=15.0.

Value clipping was tested in E10 but masked by alive dominance. With alive=5 the reward balance is better, so clipping should now help.

**What we hope to learn**:
1. Does halving PPO epochs reduce oscillation amplitude?
2. Does the duck stand still properly with stand_still=-5.0?
3. Does the combination maintain or improve E11's HEADLINE (0.876)?

**Success criteria** (vs E11 best, HEADLINE=0.876):
- HEADLINE >= 0.85
- Oscillation amplitude < 0.08 (vs E11's ~0.13)
- Standing yaw drift < 0.1 rad/s (vs E11's 0.465)
- Survival 100%

**Results**:

| CP | Steps | HEADLINE | cold_start | responsive | Training reward |
|----|-------|----------|------------|------------|-----------------|
| 5  | 32.4M | 0.676 | 0.692 | 0.660 | 362 |
| 6  | 43.3M | 0.796 | 0.766 | 0.826 | 448 |
| 7  | 54.1M | 0.813 | 0.824 | 0.801 | 457 |
| 8  | 64.9M | 0.790 | 0.744 | 0.835 | 430 |
| 9  | 75.7M | 0.782 | 0.754 | 0.810 | 451 |
| 10 | 86.5M | 0.793 | 0.786 | 0.800 | 441 |
| **11** | **97.3M** | **0.837** | **0.814** | **0.860** | **448** |
| 12 | 108.1M | 0.729 | 0.688 | 0.769 | 446 |
| 13 | 118.9M | 0.743 | 0.697 | 0.788 | 448 |
| 14 | 129.8M | 0.821 | 0.798 | 0.843 | 439 |
| 15 | 140.6M | 0.774 | 0.781 | 0.768 | 446 |
| 16 | 151.4M | 0.774 | 0.781 | 0.768 | 376 |

**Best checkpoint**: CP11 at 97.3M steps (HEADLINE=0.837)
**Best ONNX**: `2026_05_15_133156_97320960.onnx`

**Verdict**: PARTIAL - improved stability but missed peak and standing targets.
1. HEADLINE (0.837) below target (0.85) and below E11's best (0.876) - **FAIL**
2. Oscillation amplitude ~0.10 (vs E11's ~0.14) - improved but didn't meet < 0.08 target - **PARTIAL**
3. Standing yaw drift -0.473 rad/s - unchanged from E11 (-0.465) - **FAIL**
4. Survival 100% - **PASS**

**What we learned**:
- `num_updates_per_batch=2` reduces oscillation moderately (0.14 → 0.10) but doesn't eliminate it
- Forward walk precision improved significantly (0.882 → 0.957) and backward walking improved (0.761 → 0.807)
- `stand_still=-5.0` had zero effect on standing behaviour - imitation=15 completely overwhelms it. The duck still rotates at -0.47 rad/s and walks 3.4m during standstill. Need either much stronger penalty (-15 to -20) or conditional imitation gating
- Peak HEADLINE lower (0.837 vs 0.876) - fewer PPO epochs means slower learning, and the policy may not have peaked yet at 150M steps
- Best checkpoint at 97M (later than E11's 65M) - consistent with slower but more stable learning
- Push recovery weakened for front pushes (0.6N: 60% → 0%) but improved for back pushes (20% → 100%)

---

### E13: Adaptive KL + Value Clipping (150M)

**Date**: 2026-05-15
**Config**: Same as E11 (alive=5, num_updates_per_batch=4, stand_still=-1.0) with two additions:
1. `--adaptive_kl` enabled (LRSchedule.ADAPTIVE_KL, desired_kl=0.01)
2. `clipping_epsilon_value=0.2` (value function clipping)

Reverted E12's `num_updates_per_batch=2` and `stand_still=-5.0` changes. E12 showed that reducing PPO epochs lowered peak HEADLINE. Adaptive KL addresses overshooting by throttling the learning rate dynamically rather than removing gradient signal.

**Steps**: 150,000,000 (8192 envs, from scratch)
**Checkpoint dir**: `checkpoints/E13_adaptive_kl/`

**What we hope to learn**:
1. Does adaptive KL give E11's peak (0.876) with reduced oscillation?
2. Does the best checkpoint appear later in training (convergence)?

**Success criteria** (vs E11 best, HEADLINE=0.876):
- HEADLINE >= 0.87
- Oscillation amplitude < 0.08
- Best checkpoint after 100M steps (convergence, not early peak)

**Results**:

| Metric | E11 CP8 (best) | E13 CP15 (best) | Change |
|--------|---------------|-----------------|--------|
| **HEADLINE** | **0.876** | **0.785** | **-0.091** |
| Forward walk vx | 0.882 | 0.887 | +0.005 |
| Backward walk vx | 0.761 | 0.829 | +0.068 |
| Strafe left vy | 0.983 | 0.963 | -0.020 |
| Strafe right vy | 1.000 | 0.990 | -0.010 |
| Turn left wz | 0.351 | 0.254 | -0.097 |
| Turn right wz | 0.583 | 0.635 | +0.052 |
| Fwd+turn left wz | 0.922 | 0.919 | -0.003 |
| Fwd+turn right wz | 0.996 | 0.954 | -0.042 |
| Back+turn left wz | 0.206 | 0.000 | -0.206 |
| Back+turn right wz | 0.259 | 0.000 | -0.259 |
| Standing yaw drift | -0.465 rad/s | 0.315 rad/s | direction flipped, still drifts |
| Push 0.6N front | 60% | 0% | -60% |
| Push 0.6N back | 20% | 20% | same |
| Gait symmetry | 0.845 | 0.808 | -0.037 |
| Oscillation amplitude | ~0.14 | ~0.09 | improved |

**HEADLINE progression during training**:
```
CP3  (32M):  0.711
CP5  (43M):  0.782  <-- early peak
CP7  (65M):  0.774
CP9  (86M):  0.765
CP11 (108M): 0.772
CP13 (130M): 0.699  <-- trough
CP15 (151M): 0.785  <-- final/best
```

**Verdict**: FAIL - adaptive KL with desired_kl=0.01 is too conservative. It successfully reduced oscillation amplitude (0.14 to 0.09) but capped peak HEADLINE at 0.785, well below E11's 0.876. The LR throttling prevents the policy from reaching E11's quality. Back+turn completely broken (0.000). Convergence partially improved (best at final checkpoint rather than early peak), but the ceiling is too low to be useful.

**Key learning**: The oscillation is a symptom we can live with - use checkpoint selection instead. Both convergence fixes (E12: fewer epochs, E13: adaptive KL) traded peak quality for stability. The better path is to improve the reward structure directly and pick the best checkpoint.

---

### E14: Imitation Gating (150M)

**Date**: 2026-05-15
**Config**: Same as E11 (alive=5, imitation=15, tracking_sigma=0.25, clipping=0.2) with one change:
- Imitation reward multiplied by `clip(sqrt(vx^2 + vy^2 + wz^2) / 0.05, 0, 1)` - gates imitation to zero when velocity command magnitude is near zero

**Why**: E12 proved that stand_still=-5.0 cannot compete with imitation=15 during standstill. The duck walks in place and rotates right because the imitation reward actively encourages the reference gait regardless of command. Rather than increasing stand_still further (diminishing returns against imitation=15), we gate imitation off entirely when the duck should be standing still.

**Steps**: 150,000,000 (8192 envs, from scratch)
**Checkpoint dir**: `checkpoints/E14_imitation_gating/`

**What we hope to learn**:
1. Does gating fix standing behaviour (yaw drift, walking in place)?
2. Does it maintain E11's walking quality (HEADLINE >= 0.876)?
3. Does it affect turn-from-standstill (where cmd_vel is non-zero, so imitation should still be active)?

**Success criteria** (vs E11 best, HEADLINE=0.876):
- Standing yaw drift < 0.1 rad/s (vs current 0.315-0.465)
- Standing COM drift < 0.5m (vs current 2.3-3.4m)
- HEADLINE >= 0.85 (walking quality maintained)
- Turn-from-standstill not degraded

**Results**:

| Metric | E11 CP8 | E14 CP9 | Change |
|--------|---------|---------|--------|
| **HEADLINE** | **0.876** | **0.832** | **-0.044** |
| Forward walk vx | 0.882 | 0.996 | **+0.114** |
| Backward walk vx | 0.761 | 0.329 | **-0.432** |
| Strafe left vy | 0.983 | 0.932 | -0.051 |
| Strafe right vy | 1.000 | 0.992 | -0.008 |
| Turn left wz | 0.351 | 0.962 | **+0.611** |
| Turn right wz | 0.583 | 0.977 | **+0.394** |
| Fwd+turn left wz | 0.922 | 0.998 | +0.076 |
| Fwd+turn right wz | 0.996 | 0.601 | -0.395 |
| Back+turn left wz | 0.206 | 0.000 | -0.206 |
| Back+turn right wz | 0.259 | 0.000 | -0.259 |
| Standing yaw drift | -0.465 rad/s | 0.053 rad/s | **89% reduction** |
| Standing COM drift | 3.4m | 0.96m | **72% reduction** |
| Push 0.6N front | 60% | 0% | -60% |
| Push 0.6N back | 20% | 100% | **+80%** |
| Gait symmetry | 0.845 | 0.808 | -0.037 |

**HEADLINE progression during training**:
```
CP3  (32M):  0.492
CP5  (43M):  0.673
CP7  (65M):  0.800
CP8  (76M):  0.769
CP9  (87M):  0.832  <-- best
CP10 (97M):  0.758
CP11 (108M): 0.830
CP12 (119M): 0.716
CP13 (130M): 0.722
CP14 (141M): 0.790
CP15 (151M): 0.805
```

**Verdict**: PARTIAL SUCCESS - imitation gating achieved its primary goal. Standing behaviour dramatically improved (yaw drift 89% lower, COM drift 72% lower). Pure turning recovered from E11's regression (0.35-0.58 to 0.96-0.98). Forward walking also improved (0.882 to 0.996). However, backward walking regressed hard (0.761 to 0.329) and fwd+turn right dropped (0.996 to 0.601). Overall HEADLINE 0.832 vs E11's 0.876.

**Key learning**: Imitation gating works for standing but the gate threshold (0.05) interacts with backward walking. When walking backward at -0.1 m/s, the velocity magnitude is 0.1/0.05 = 2.0 so the gate is fully open - so the regression isn't from the gate being too aggressive. More likely the gating changes the reward landscape enough that the optimiser finds a different equilibrium where backward walking is deprioritised. The turning recovery is a strong positive signal - removing imitation during standstill apparently freed up capacity for better turning.

---

### E15: Imitation Gating + Backward-Biased Sampling (150M)

**Date**: 2026-05-15
**Config**: Same as E14 (imitation gating, alive=5, imitation=15, tracking_sigma=0.25, clipping=0.2) with one change:
- Command sampling: 10% pure turn, 10% standstill, 10% backward-biased (vx forced negative), 70% normal
- Previously: 12% pure turn, 10% standstill, 78% normal

**Why**: E14 fixed standing and turning but backward walking regressed from 0.761 to 0.329. The duck moves backward at half speed. Adding dedicated backward commands ensures the optimiser sees enough backward training data. The backward commands use the full velocity range but force vx to be negative.

**Steps**: 150,000,000 (8192 envs, from scratch)
**Checkpoint dir**: `checkpoints/E15_imit_gate_backward_bias/`

**What we hope to learn**:
1. Does backward-biased sampling recover backward walking while keeping E14's gains?
2. Do we maintain standing improvement and turning recovery?

**Success criteria** (vs E14 best, HEADLINE=0.832):
- Backward walk vx >= 0.70 (recovered from E14's 0.329)
- Standing yaw drift < 0.10 (maintained from E14)
- Turn left/right wz >= 0.90 (maintained from E14)
- Forward walk vx >= 0.90 (maintained from E14)
- HEADLINE >= 0.86

**Results**:

| Metric | E11 CP8 | E14 CP9 | E15 CP15 | E15 vs E14 |
|--------|---------|---------|----------|------------|
| **HEADLINE** | **0.876** | **0.832** | **0.782** | **-0.050** |
| Forward walk vx | 0.882 | 0.996 | 1.000 | +0.004 |
| Backward walk vx | 0.761 | 0.329 | 0.173 | **-0.156** |
| Strafe left vy | 0.983 | 0.932 | 0.968 | +0.036 |
| Turn left wz | 0.351 | 0.962 | 0.824 | -0.138 |
| Turn right wz | 0.583 | 0.977 | 0.989 | +0.012 |
| Standing yaw drift | -0.465 | 0.053 | 0.089 | similar |
| Standing COM drift | 3.4m | 0.96m | 0.74m | improved |

**Verdict**: FAIL - backward-biased sampling made backward walking *worse* (0.329 to 0.173). The problem is not data starvation. Overall HEADLINE dropped to 0.782. Turn left also regressed from E14 (0.962 to 0.824). Standing gains preserved.

**Key learning**: The backward walking regression is caused by the imitation gating changing the reward landscape, not by insufficient backward training data. More backward commands didn't help because the reference motion is a forward gait - imitation actively conflicts with backward tracking. The biased sampling also diluted the normal command distribution (70% vs 78%), which may have slightly hurt other capabilities.

---

### E16: Directional Imitation Gate (150M)

**Date**: 2026-05-15
**Config**: Same as E14 (imitation gating, alive=5, imitation=15) but with directional gate:
- Gate: `clip(sqrt(max(vx, 0)^2 + vy^2 + wz^2) / 0.05, 0, 1)`
- Uses `max(vx, 0)` instead of `vx` - negative vx contributes zero to the gate magnitude
- Reverted E15's backward-biased sampling back to 12% turn / 10% still / 78% normal

**Why**: E14/E15 showed backward walking regresses with imitation gating. The reference motion is a forward gait, so imitation conflicts with backward velocity tracking. By gating imitation off when vx < 0 (and no lateral/angular command), the policy can learn backward walking without the forward-gait reference interfering.

**Steps**: 150,000,000 (8192 envs, from scratch)
**Checkpoint dir**: `checkpoints/E16_directional_imit_gate/`

**What we hope to learn**:
1. Does directional gating recover backward walking?
2. Does it preserve standing and turning improvements from E14?
3. Can we beat E11's HEADLINE (0.876) with a better overall profile?

**Success criteria**:
- Backward walk vx >= 0.70 (recovered)
- Standing yaw drift < 0.10 (maintained from E14)
- Turn left/right wz >= 0.90 (maintained from E14)
- Forward walk vx >= 0.90 (maintained from E14)
- HEADLINE >= 0.87 (beat E11)

**Results**: ABANDONED - killed after discovering the reference motion contained backward motions (not just forward). The directional gate was based on a wrong assumption. Investigation led to discovering the standing reference bug (Placo generates walking-in-place at dx=0).

---

### E17: Fixed Standing Reference (150M)

**Date**: 2026-05-15
**Config**: Same as E11 (alive=5, imitation=15, tracking_sigma=0.25, no imitation gate, no adaptive KL). Only change: polynomial_coefficients.pkl regenerated with static standing pose at dx=0.

**Why**: Investigation revealed the root cause of standing behaviour: Placo's walk engine generates a full walking-in-place gait at dx=0 (22-degree knee swings, alternating foot contacts). Fixed by auto-enabling `--stand` in gait_generator.py when dx=0, dy=0, dtheta=0, which freezes joints at initial pose with both feet on ground.

**Steps**: 150,000,000 (8192 envs, from scratch)
**Checkpoint dir**: `checkpoints/E17_fixed_standing_ref/`

**Peak HEADLINE**: 0.812 (CP 140M)

| Metric | E17 best | E11 CP8 | Change |
|--------|----------|---------|--------|
| Forward | 1.000 (140M) | 0.882 | +0.118 |
| Backward | 0.238 (64M) | 0.761 | -0.523 |
| Strafe L/R | 0.999/0.999 (75M) | 0.983/1.000 | similar |
| Turn L/R | 0.973/0.922 (140M) | 0.305/0.584 | +0.668/+0.338 |
| Stand-then-turn | 0.961/0.933 (140M) | 0.351/0.583 | +0.610/+0.350 |
| Standing steps | 73 | 73 | unchanged |

**Verdict**: PARTIAL - fixed reference didn't hurt training (HEADLINE comparable to E11). Turning improved substantially but backward regressed. Standing stepping behaviour unchanged - the duck still walks in place with 73 steps/10s because no reward signal tells it to stop stepping.

**Key learning**: The reference data fix was correct (wrong data is wrong data) but insufficient for standing behaviour. The imitation gate (E14) was doing the heavy lifting for standing yaw reduction, not the reference data.

---

### E18: Imitation Gate + Fixed Standing Reference (150M)

**Date**: 2026-05-15
**Config**: Same as E14 (imitation gate, alive=5, imitation=15, tracking_sigma=0.25) + fixed polynomial_coefficients.pkl with static standing pose.

**Why**: Test whether the imitation gate and fixed reference are complementary. E14 had the gate with old reference, E17 had the fixed reference without gate.

**Steps**: 150,000,000 (8192 envs, from scratch)
**Checkpoint dir**: `checkpoints/E18_imit_gate_fixed_ref/`

**Peak HEADLINE**: 0.846 (CP 64M)

| Metric | E18 best | E14 best | Change |
|--------|----------|----------|--------|
| Forward | 1.000 (140M) | 0.996 | +0.004 |
| Backward | 0.637 (64M) | 0.329 | +0.308 |
| Strafe L/R | 0.999/0.999 (75M) | 0.932/0.992 | +0.067/+0.007 |
| Turn L/R | 0.973/0.922 (140M) | 0.962/0.977 | +0.011/-0.055 |
| Standing steps | 73-78 | 73 | unchanged |
| Standing wz | 3.6 deg/s (64M) | 3.0 deg/s | similar |

**Verdict**: PARTIAL - backward walking recovered significantly vs E14 (0.637 vs 0.329). The fixed reference + gate combination is slightly better than either alone. But standing stepping behaviour is identical. The backward-vs-turning oscillation persists across checkpoints.

**Key learning**: Gate + fixed reference are complementary for backward walking. Neither addresses the fundamental stepping-in-place problem.

---

### E19: Strong stand_still Penalty (150M)

**Date**: 2026-05-16
**Config**: Same as E18 (imitation gate + fixed ref) but stand_still increased from -1.0 to -15.0.

**Why**: E12 tested stand_still=-5 but imitation=15 overwhelmed it. With the imitation gate active, imitation is zero at standstill - stand_still=-15 competes only against alive=5 (3:1 ratio).

**Steps**: 150,000,000 (8192 envs, from scratch)
**Checkpoint dir**: `checkpoints/E19_strong_stand_still/`

**Peak HEADLINE**: 0.854 (CP 97M)

| Metric | E19 CP97M | E14 best | Change |
|--------|-----------|----------|--------|
| Forward | 1.000 | 0.996 | +0.004 |
| Backward | 0.449 | 0.329 | +0.120 |
| Strafe L/R | 0.999/0.851 | 0.932/0.992 | mixed |
| Turn L/R | 0.838/0.966 | 0.962/0.977 | -0.124/-0.011 |
| Standing steps | 82 | 73 | unchanged |
| Standing air% | 65% | 52% | worse |
| Standing swing | 27.5mm | 25.0mm | unchanged |

**Verdict**: FAIL for standing. 15x penalty increase had no measurable effect on stepping behaviour. All checkpoints: 55-109 steps, 21-65% air time. The alive reward (5.0/step) makes rhythmic stepping the optimal survival strategy regardless of joint deviation penalties.

**Key learning**: Penalising joint deviation from default pose cannot stop stepping. The policy has learned that rhythmic weight-shifting IS the most stable strategy. The cost function targets the wrong thing - it penalises joint positions, not foot contact state.

---

### E20: Foot Contact Reward During Standing (150M)

**Date**: 2026-05-16
**Config**: Same as E19 (stand_still=-15, imitation gate, fixed ref) plus new reward: stand_contact=+5.0. Rewards both feet on ground when cmd_vel near zero.

**Why**: stand_still penalises joint deviation (indirect). stand_contact directly rewards feet being planted (the actual behaviour we want). With imitation gated off and both rewards active, standing mode gets: no walking reference, strong joint-freeze penalty, positive feet-down reward.

**Steps**: 150,000,000 (8192 envs, from scratch)
**Checkpoint dir**: `checkpoints/E20_stand_contact/`

**Peak HEADLINE**: 0.848 (CP 97M from training callback)

| Metric | E20 best | E14 best | Change |
|--------|----------|----------|--------|
| Forward | 1.000 (140M) | 0.996 | +0.004 |
| Backward | 0.478 (54M) | 0.329 | +0.149 |
| Strafe L/R | 0.998/0.994 (86M/32M) | 0.932/0.992 | similar |
| Turn L/R | 0.999/0.999 (140M) | 0.962/0.977 | +0.037/+0.022 |
| Standing steps | 73-87 | 73 | unchanged |
| Standing air% | 44-65% | 52% | marginal |
| Standing swing | 17-29mm | 25mm | marginal |

**Verdict**: FAIL for standing. Foot contact reward (+5 when both feet down during standstill) did not stop stepping. Best checkpoint (129M) had 44% air time and 17.4mm swing - slightly better than E14 but still 74 steps. The reward fires only during the 10% standstill training time and cannot overcome the deeply ingrained stepping-is-stable prior.

Minor positive: push recovery improved slightly (survives 1.0 N*s side push, E14 couldn't).

**Key learning**: Neither penalty (stand_still) nor reward (stand_contact) at the values tested can stop stepping in a single policy. The alive reward creates a strong prior that stepping = stability. The 10% standstill training fraction means standing-specific rewards get minimal gradient signal. Next test: mode-conditional reward routing (E24) as the final single-policy attempt before considering separate policies.
