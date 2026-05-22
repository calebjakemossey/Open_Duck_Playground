# Reward Tuning - LHS Screening Data

Raw data from the Latin Hypercube Screening phase (14 runs). For conclusions and current understanding, see `docs/19_what_we_know.md`. Note: some conclusions below (especially points 6-10 about "fundamental turning problem") were written before the angular velocity bug was discovered and are now outdated.

## Cumulative Findings (updated 2026-05-14)

What we know so far from 14 LHS runs + 4 Disney experiments (Runs 1-4):

1. **`imitation` weight is the single most important variable.** Spearman r=0.58 (p<0.05) for both `st_fwd` and `gait_symmetry`. The anchor's imit=1.0 was catastrophically low.
2. **The imit/alive ratio controls the stand-to-walk transition.** Ratio >= 0.4 required. Disney BDX uses ~1.1. Our anchor was 0.05 (20x too low).
3. **`action_rate=-1.5` works but requires `imitation>=15.0` to push through.** Disney Run 2 proved that imit=8.0 is catastrophically insufficient with this smoothness penalty. No middle ground.
4. **Disney-style config produces our best-ever policy** (Run 1 36M peak): HEADLINE=0.604, best forward tracking (0.699), best symmetry (0.826), best smoothness. One weakness: turning (st_turn_R=0.144).
5. **Training instability above 36M steps** with Disney config. HEADLINE regressed 27% from 36M to 50M. The high imitation gradient may cause oscillation between local optima. Best checkpoint selection is critical.
6. **`tracking_ang_vel` is the strongest predictor of turning quality** (Spearman r=-0.64). Disney Run 1's turning weakness is likely due to ang_vel=2.5 being too low. Run 3 (ang_vel=4.0) tests this.
7. **`action_rate=-1.5` actively helps turning** (Disney Run 4 finding). Loosening to -0.5 made left turning 78% worse (0.600 to 0.134). The tight smoothness constraint forces committed, deliberate turns. Turning is an ang_vel problem, not a smoothness problem.
8. **Kaggle GPU (P100) is 7x slower than local RTX 5080.** Kaggle TPU failed to allocate. All training now local-only.
9. **Boosting `tracking_ang_vel` hurts, not helps** (Disney Run 3). Doubling ang_vel from 2.5 to 4.0 made turning WORSE (st_turn_L 0.600 to 0.000) and wz_tracking dropped 19%. The higher ang_vel competes with imitation reward rather than complementing it.
10. **The Disney config has a fundamental turning problem.** Neither loosening action_rate (Run 4) nor boosting ang_vel (Run 3) fixes it. High imitation (15.0) creates a strong attractor for the symmetric walking reference gait, penalising the asymmetric leg movements needed for turning. The Run 6 low-alive branch may be more promising for turning.

## Training Infrastructure

| Platform | GPU/TPU | Steps/sec | 50M wall time | Use |
|---|---|---|---|---|
| Local RTX 5080 | 16GB VRAM | ~58,500 | ~15-20 min | Evaluation, viewer, quick tests |
| Kaggle GPU (P100) | 16GB HBM2 | ~8,400 | ~1h 46min | Validated but too slow for production runs |
| Kaggle TPU V5e | 8 chips | FAILED | N/A | `enable_tpu: true` not honoured - Kaggle allocated no accelerator, JAX fell back to CPU. Needs investigation. |

**TPU failure (2026-05-14)**: Kernel ran with `Accelerator: None` and `JAX devices: [CpuDevice(id=0)]`. The `enable_tpu` flag in kernel-metadata.json was ignored. Possible causes: Kaggle requires separate TPU verification, or the API flag has changed. Parked for now - GPU parallelism is sufficient.

**Strategy (revised)**: All training, evaluation, and analysis local on RTX 5080. Kaggle abandoned - too slow for iteration speed needed. 50M step runs take ~15-20 min locally vs ~1h 46min on Kaggle P100.

## Anchor baseline (reference for all comparisons)

| Metric | Value |
|---|---|
| Steps | 140,574,720 (best checkpoint of fresh_150m_postfix) |
| `walking/HEADLINE` | 0.327 |
| `walking/cold_start_avg` | 0.339 |
| `walking/responsiveness_avg` | 0.315 |
| `eval/episode_reward` | ~307 |
| `vx_tracking_score` | 0.329 |
| `wz_tracking_score` | 0.827 |
| `gait_symmetry_score` | 0.276 |
| `stand_then_forward` | 0.021 (broken) |
| `stand_then_backward` | 0.018 (broken) |
| `stand_then_strafe_left/right` | 0.000 / 0.000 (broken) |
| `stand_then_turn_left/right` | 0.999 / 0.996 (great) |
| Survival rate | 1.0 |

**Anchor config:**
```
tracking_lin_vel=2.5, tracking_ang_vel=6.0, alive=20.0,
action_rate=-0.5, stand_still=-0.2, imitation=1.0, tracking_sigma=0.02
```
Plus hard-coded `y_tol=0.1` in `rewards.py:18`.

## Failed warm-start experiments (pre-methodology)

| ID | Change from anchor | Result |
|---|---|---|
| s_lin | tracking_lin_vel 2.5 -> 5.0 | NEGATIVE. HEADLINE flat at 0.327 through 36M warm; reward climbed (weight scaled) but behaviour unchanged. |
| s_alive | alive 20.0 -> 5.0 | NEGATIVE. HEADLINE flat at 0.327 through 18M warm; reward dropped (weight scaled) but behaviour unchanged. |
| s_sigma | tracking_sigma 0.1 -> 0.02 | NEGATIVE for warm-start. cold_start crashed 0.339 -> 0.163 then clawed back to 0.332 by 50M. Confirms sigma is high-leverage but cannot be tested warm. |

**Lesson learned**: reward weight and shape changes require fresh cold-start runs. Switching to LHS screening methodology.

## Stage 1 - Latin Hypercube Screening

Status: COMPLETE (14/15 runs; run 8 permanently lost to PTX crash, not re-run).

### Run table (sorted by HEADLINE)

| # | sigma | alive | lin | ang | act_rate | stand | imit | imit/alv | HEAD | vx_tr | wz_tr | sym | st_fwd | st_tL | m_resp |
|---|-------|-------|-----|-----|----------|-------|------|----------|------|-------|-------|-----|--------|-------|--------|
| **6** | 0.330 | 3.21 | 2.72 | 1.64 | -0.075 | -0.95 | 1.83 | **0.570** | **0.532** | **0.595** | 0.688 | **0.697** | **0.993** | 0.426 | **0.499** |
| **14** | 0.106 | 3.42 | 1.03 | 2.66 | -0.260 | -2.00 | 1.52 | **0.445** | **0.404** | **0.465** | 0.754 | **0.561** | **0.710** | 0.740 | **0.457** |
| 10 | 0.193 | 5.55 | 1.38 | 3.15 | -0.197 | -0.35 | 1.07 | 0.193 | 0.328 | 0.343 | 0.829 | 0.368 | 0.018 | 0.987 | 0.358 |
| 12 | 0.125 | 9.35 | 3.76 | 3.91 | -0.122 | -0.74 | 1.68 | 0.180 | 0.320 | 0.340 | 0.818 | 0.293 | 0.018 | 0.909 | 0.364 |
| 4 | 0.167 | 4.79 | 3.09 | 3.21 | -0.065 | -0.48 | 1.25 | 0.262 | 0.314 | 0.332 | 0.817 | 0.416 | 0.018 | 0.956 | 0.357 |
| 5 | 0.225 | 6.32 | 4.17 | 2.55 | -0.053 | -1.57 | 0.62 | 0.098 | 0.302 | 0.346 | 0.775 | 0.378 | 0.019 | 0.991 | 0.311 |
| 3 | 0.064 | 8.22 | 4.37 | 3.64 | -0.090 | -1.46 | 1.20 | 0.145 | 0.272 | 0.363 | 0.760 | 0.311 | 0.019 | 0.784 | 0.328 |
| 0 | 0.444 | 1.34 | 5.71 | 2.26 | -0.493 | -1.16 | 0.92 | 0.691 | 0.250 | 0.346 | 0.806 | 0.324 | 0.019 | 0.807 | 0.307 |
| 2 | 0.253 | 8.06 | 4.82 | 2.19 | -0.129 | -0.88 | 1.72 | 0.214 | 0.228 | 0.353 | 0.738 | 0.458 | 0.021 | 0.441 | 0.288 |
| 1 | 0.070 | 7.52 | 2.18 | 1.53 | -0.093 | -0.28 | 0.72 | 0.095 | 0.202 | 0.353 | 0.744 | 0.284 | 0.019 | 0.702 | 0.217 |
| 13 | 0.385 | 1.93 | 5.54 | 1.34 | -0.389 | -1.20 | 1.99 | 1.030 | 0.141 | 0.384 | 0.542 | 0.502 | 0.031 | 0.299 | 0.195 |
| 9 | 0.082 | 4.22 | 3.48 | 3.49 | -0.301 | -0.56 | 0.86 | 0.203 | 0.083 | 0.371 | 0.665 | 0.267 | 0.019 | 0.056 | 0.140 |
| 7 | 0.143 | 9.88 | 2.00 | 2.84 | -0.332 | -1.72 | 0.52 | 0.053 | 0.067 | 0.386 | 0.633 | 0.412 | 0.018 | 0.099 | 0.117 |
| 11 | 0.285 | 6.67 | 5.07 | 1.07 | -0.160 | -1.38 | 1.38 | 0.208 | 0.044 | 0.386 | 0.533 | 0.390 | 0.020 | 0.000 | 0.078 |

All runs: 50M steps, cold start, 8192 envs. Run 8 lost to GPU crash (PTX compilation hang), not included.

### Full responsiveness table (v4 eval)

| # | vx_tr | wz_tr | sym | st_fwd | st_back | st_strL | st_strR | st_tL | st_tR | m_resp |
|---|-------|-------|-----|--------|---------|---------|---------|-------|-------|--------|
| **Anchor 140M** | 0.329 | **0.827** | 0.276 | 0.021 | 0.018 | 0.000 | 0.000 | **0.999** | **0.996** | 0.348 |
| **6** | **0.595** | 0.688 | **0.697** | **0.993** | **0.288** | **0.361** | **0.437** | 0.426 | 0.570 | **0.499** |
| **14** | **0.465** | 0.754 | **0.561** | **0.710** | 0.051 | 0.107 | 0.227 | 0.740 | **0.964** | **0.457** |
| 0 | 0.346 | 0.806 | 0.324 | 0.019 | 0.018 | 0.000 | 0.000 | 0.807 | 0.899 | 0.307 |
| 1 | 0.353 | 0.744 | 0.284 | 0.019 | 0.019 | 0.000 | 0.000 | 0.702 | 0.251 | 0.217 |
| 2 | 0.353 | 0.738 | 0.458 | 0.021 | 0.018 | 0.000 | 0.000 | 0.441 | 0.991 | 0.288 |
| 3 | 0.363 | 0.760 | 0.311 | 0.019 | 0.019 | 0.000 | 0.000 | 0.784 | 0.809 | 0.328 |
| 4 | 0.332 | 0.817 | 0.416 | 0.018 | 0.018 | 0.000 | 0.000 | 0.956 | 0.988 | 0.357 |
| 5 | 0.346 | 0.775 | 0.378 | 0.019 | 0.019 | 0.000 | 0.000 | 0.991 | 0.743 | 0.311 |
| 7 | 0.386 | 0.633 | 0.412 | 0.018 | 0.018 | 0.000 | 0.000 | 0.099 | 0.000 | 0.117 |
| 9 | 0.371 | 0.665 | 0.267 | 0.019 | 0.019 | 0.000 | 0.000 | 0.056 | 0.089 | 0.140 |
| 10 | 0.343 | 0.829 | 0.368 | 0.018 | 0.018 | 0.000 | 0.000 | 0.987 | 0.983 | 0.358 |
| 11 | 0.386 | 0.533 | 0.390 | 0.020 | 0.018 | 0.000 | 0.000 | 0.000 | 0.043 | 0.078 |
| 12 | 0.340 | 0.818 | 0.293 | 0.018 | 0.019 | 0.000 | 0.000 | 0.909 | 1.000 | 0.364 |
| 13 | 0.384 | 0.542 | 0.502 | 0.031 | 0.033 | 0.001 | 0.147 | 0.299 | 0.173 | 0.195 |

### Stage 1 complete analysis (N=14)

#### Spearman rank correlations

| Param | vs HEAD | vs vx_tr | vs sym | vs st_fwd | vs m_resp |
|---|---|---|---|---|---|
| `tracking_sigma` | -0.02 | +0.03 | **+0.52** | +0.35 | -0.07 |
| `alive` | -0.24 | -0.14 | -0.33 | **-0.48** | -0.20 |
| `tracking_lin_vel` | -0.42 | -0.18 | -0.09 | +0.22 | -0.39 |
| `tracking_ang_vel` | +0.32 | -0.43 | -0.43 | **-0.64*** | +0.37 |
| `action_rate` | +0.43 | -0.25 | +0.02 | -0.07 | +0.43 |
| `stand_still` | +0.09 | **-0.56*** | -0.42 | -0.35 | +0.09 |
| `imitation` | +0.32 | +0.17 | **+0.58*** | **+0.58*** | +0.36 |
| `imit/alive ratio` | +0.23 | +0.14 | **+0.56*** | **+0.58*** | +0.23 |

*\* = p < 0.05*

**Sensitivity ranking** (revised from N=5):
1. **`imitation`** (and `imit/alive` ratio) - only variable reaching statistical significance for both `st_fwd` and `gait_symmetry`. This was barely visible at N=5.
2. **`tracking_ang_vel`** - significant for `st_fwd` (negative: lower ang_vel helps responsiveness). Anchor's 6.0 was grossly over-weighted.
3. **`stand_still`** - significant for `vx_tracking` (negative: more negative hurts tracking).
4. **`alive`** - moderate negative correlation with `st_fwd` (-0.48), not reaching significance.
5. **`action_rate`** - moderate positive with HEAD (+0.43), but the interaction with other params matters more than the marginal effect.
6. **`tracking_sigma`** - only significant for `gait_symmetry`. Not the key lever the plan predicted.
7. **`tracking_lin_vel`** - weak or negative. Not important.

#### The three-variable recipe

Only 2 of 14 runs fix the stand-then-forward failure mode. Both share three properties:

| Requirement | Run 6 | Run 14 | Nearest failure (Run 0) |
|---|---|---|---|
| `imitation/alive` >= ~0.4 | 0.570 | 0.445 | 0.691 (has it) |
| `alive` <= ~3.5 | 3.21 | 3.42 | 1.34 (has it) |
| `action_rate` >= ~-0.26 | -0.075 | -0.260 | **-0.493 (FAILS)** |

Run 0 has the ratio (0.691) and the low alive (1.34) but its `action_rate=-0.493` is too restrictive. Run 13 has the highest ratio (1.030) and imitation (1.99) but `action_rate=-0.389` kills it. All three variables must be in range simultaneously.

The recipe: **imitation/alive >= 0.4, alive <= 3.5, action_rate >= -0.26.**

Why this works mechanically:
- **High imitation/alive ratio** rebalances the reward landscape. With anchor's 1.0/20.0=0.05, standing still collects 95% of max reward (dominated by alive). With 1.83/3.21=0.57, the imitation term has comparable weight to alive, creating gradient pressure to match the walking reference motion even during command transitions.
- **Low alive** directly reduces the "do nothing" attractor. Standing still with alive=3.21 gives ~3.2 reward/step. With alive=20.0, it gives ~20.0/step - overwhelming any tracking signal.
- **Loose action_rate** allows the policy to make the large joint angle changes needed to transition from standing to walking. With action_rate=-0.5, the penalty for a step initiation (~0.15 rad change across 14 joints) costs ~0.5 * 14 * 0.15^2 = 0.16/step - comparable to the tracking reward gained.

#### Run 6 vs Run 14: two different trade-off profiles

| Metric | Anchor 140M | Run 6 50M | Run 14 50M |
|---|---|---|---|
| `vx_tracking` | 0.329 | **0.595** (+81%) | **0.465** (+41%) |
| `wz_tracking` | 0.827 | 0.688 (-17%) | 0.754 (-9%) |
| `gait_symmetry` | 0.276 | **0.697** (+152%) | **0.561** (+103%) |
| `stand_then_forward` | 0.021 | **0.993** (FIXED) | **0.710** (mostly fixed) |
| `stand_then_backward` | 0.018 | **0.288** (partial) | 0.051 (still broken) |
| `stand_then_strafe_L/R` | 0.000 | **0.361/0.437** | 0.107/0.227 (partial) |
| `stand_then_turn_L/R` | 0.999/0.996 | 0.426/0.570 (-53%) | **0.740/0.964** (kept!) |
| `mean_responsiveness` | 0.348 | **0.499** (+43%) | **0.457** (+31%) |

Run 6 is the better overall robot but loses turning-from-stand. Run 14 preserves turning ability (0.964 right turn) while gaining forward responsiveness (0.710). This suggests the trade-off between turning and forward responsiveness is not binary - a config between these two could achieve both.

Run 14's key difference from Run 6: higher `ang_vel` (2.66 vs 1.64, preserves turning) and tighter `action_rate` (-0.260 vs -0.075, partially suppresses the stand-to-walk transition but keeps turn transitions intact).

#### Visual testing observations (user-reported, runs 6 and 13)

**Run 6**: Forward walk slightly slow but works. Strafing from standstill works well. Left/right turns drag the inside foot (improved over anchor but not clean). Robot drifts forward when given no input (standing-still broken - creeps forward over time). Head wobbles more than anchor. For 50M steps this is impressive - the anchor's original ONNX was trained on 300M.

**Run 13**: Not tested in viewer due to wrong ONNX path, but metrics show it walks poorly (HEAD=0.141). High imitation but crippled by action_rate=-0.389.

#### Cross-reference with Disney BDX paper (arXiv:2501.05204)

The Disney BDX paper (Table I, page 6) provides the most relevant published reward configuration for this class of robot. Key comparison:

| Reward term | Anchor | Run 6 | Disney BDX | Notes |
|---|---|---|---|---|
| Survival/alive | **20.0** | 3.21 | **20.0** | Disney uses 20.0 BUT with 11 imitation terms summing to ~22+ effective weight |
| Leg joint imitation | 1.0 | 1.83 | **15.0** | Disney's imitation is 15x our anchor. This is the gap. |
| Action rate | -0.5 | -0.075 | **-1.5** | Disney's is 3x MORE restrictive than anchor |
| Action acceleration | 0.0 | 0.0 | **-0.45** | Disney adds second-derivative smoothing. We have none. |
| Effective imit/alive ratio | 0.050 | 0.570 | **~1.1** (22/20) | Disney's ratio is 22x our anchor |

The critical insight: **Disney's alive=20.0 works because their combined imitation weights (~22.0) exceed it.** The anchor's problem is not that alive=20 is too high - it is that imitation=1.0 is too low relative to alive. Disney solves the same stand-to-walk problem with the same mechanism: making the imitation reward dominate the alive reward during walking, so "stand still" is never the optimal strategy.

Disney also uses `action_rate=-1.5` (restrictive) alongside `action_acceleration=-0.45` (jerk penalty). Our run 6 achieves responsiveness with loose action_rate (-0.075) instead. This is a different solution: Disney constrains action rate but has massive imitation gradient to push through it; we loosen action rate to reduce the barrier. A future experiment should test Disney's approach: high imitation (10-15), alive=20, action_rate=-1.5, plus an action_acceleration term.

#### Cross-reference with tuning plan predictions (`docs/17_reward_tuning_plan.md`)

| Plan prediction | Actual result |
|---|---|
| `tracking_sigma` is the key lever | WRONG. sigma has no significant correlation with HEADLINE or responsiveness |
| `alive` reduction helps | PARTIALLY RIGHT. Helps but is necessary, not sufficient |
| `action_rate` loosening helps | RIGHT. But only in combination with high imitation |
| `ang_vel` overweighted at 6.0 | RIGHT. Optimal range is 1.5-3.5 |
| `imitation` has modest impact | WRONG. Imitation (and imit/alive ratio) is THE most important variable |
| `stand_still` matters | MIXED. Significant for vx_tracking but not for responsiveness |
| `tracking_lin_vel` matters | WRONG. Weakest correlation of all 7 parameters |

The plan's biggest miss: framing the problem as "tracking sigma is too tight" when the real problem was "imitation is too weak relative to alive". The plan's search ranges were well-chosen though - the LHS box covered the winning region.

#### Cross-reference with smoothness research (`docs/06_smoothness_improvements.md`)

Our findings align with the literature on action rate penalties:
- The smoothness doc recommends action_rate in [-0.01, -0.1] range (IsaacLab defaults). Run 6's -0.075 is squarely in this range.
- The doc recommends adding action acceleration (jerk) penalty. Disney uses -0.45. We have not tested this. Stage 2 should include it.
- The doc mentions low-pass filtering of actions at the actuator level. Disney does this (37.5 Hz filter). Could be added post-training to any policy.

#### Inferences and recommendations for Stage 2

1. **Skip OFAT sensitivity testing.** The Spearman correlations at N=14 give clear rankings. Running 7 more OFAT runs around the best point would take 6 hours and add marginal information.

2. **Go directly to targeted Bayesian optimisation** around the winning region:
   - Fix `tracking_lin_vel` at 2.5 (irrelevant) and `tracking_sigma` at 0.25 (moderate, not sensitive)
   - Optimise 4 variables: `imitation` [1.5, 10.0], `alive` [1.0, 5.0], `action_rate` [-0.3, -0.01], `tracking_ang_vel` [1.5, 4.0]
   - Target: keep Run 6's responsiveness (st_fwd > 0.9) while recovering turning (st_tL > 0.8)
   - Run 14 shows this is possible: its config preserves turning at the cost of some responsiveness

3. **Test the Disney-style config** as a separate experiment: alive=20.0, imitation=15.0, action_rate=-1.5, action_acceleration=-0.45, ang_vel=2.0. If this works, it validates that the anchor's failure was purely about the imit/alive ratio, not about alive being too high.

4. **Run 8 re-run is low priority.** With 14 data points and clear findings, one missing sample does not change the conclusions.

## Stage 2 - Disney-Style Targeted Experiments

Skipping OFAT and Bayesian per Stage 1 findings. The Disney BDX reward balance (high imitation, alive=20, action_rate=-1.5, action_accel=-0.45) proved more effective than the low-alive approach from Stage 1. Running targeted experiments around the Disney config.

### Disney Run 1: alive=20.0, imitation=15.0, action_rate=-1.5, action_accel=-0.45

Full config: `tracking_lin_vel=2.5, tracking_ang_vel=2.5, torques=-1e-3, action_rate=-1.5, action_accel=-0.45, stand_still=-1.0, alive=20.0, imitation=15.0, tracking_sigma=0.25`

Training curve peaked at 36M then regressed (instability):

| Step | HEADLINE | cold_start | responsive |
|---|---|---|---|
| 25M | 0.413 | 0.342 | 0.483 |
| 32M | 0.590 | 0.515 | 0.665 |
| 36M | **0.604** | **0.549** | 0.659 |
| 43M | 0.568 | 0.524 | 0.612 |
| 50M | 0.439 | 0.461 | 0.416 |

Full eval (36M peak checkpoint):

| Metric | Anchor 140M | Run 6 50M | Disney 36M | Change vs Run 6 |
|---|---|---|---|---|
| `vx_tracking` | 0.329 | 0.595 | **0.699** | +17% |
| `gait_symmetry` | 0.276 | 0.697 | **0.826** | +19% |
| `stand_then_fwd` | 0.021 | 0.993 | **0.986** | same |
| `stand_then_back` | 0.018 | 0.288 | **0.465** | +62% |
| `stand_then_strafe_R` | 0.000 | 0.437 | **0.744** | +70% |
| `stand_then_turn_L` | 0.999 | 0.426 | 0.600 | +41% |
| `stand_then_turn_R` | 0.996 | 0.570 | 0.144 | -75% (regression) |
| `wz_tracking` | 0.827 | 0.688 | 0.587 | -15% |
| `mean_responsiveness` | 0.348 | 0.499 | **0.517** | +4% |
| `action_rate` (smoothness) | - | - | **0.069** | best ever |
| `jerk` (smoothness) | - | - | **0.050** | best ever |

**Findings**:
1. Disney-style reward balance works. Forward tracking, backward responsiveness, gait symmetry, and smoothness are all best-ever results.
2. Turning is still weak (especially right turn at 0.144). The `tracking_ang_vel=2.5` may be too low, or `action_rate=-1.5` suppresses the rapid leg changes needed for turning.
3. Training instability: HEADLINE peaked at 36M then regressed 27% by 50M. The imitation=15.0 gradient may be too sharp, causing oscillation between local optima.
4. The 50M checkpoint lost st_strafe_L (0.339->0.000) and st_turn_L (0.600->0.000), confirming instability.

**Decision**: test imitation=8.0 (halved) with everything else unchanged. Hypothesis: lower imitation reduces gradient sharpness, stabilises training, may recover turning.

### Disney Run 2: imitation=8.0 (stability test) - FAILED

Config: same as Disney Run 1 except `imitation=8.0` (imit/alive ratio = 0.40 vs Disney Run 1's 0.75).

Full config: `tracking_lin_vel=2.5, tracking_ang_vel=2.5, torques=-1e-3, action_rate=-1.5, action_accel=-0.45, stand_still=-1.0, alive=20.0, imitation=8.0, tracking_sigma=0.25`

50M steps, 8192 envs, local RTX 5080.

| Metric | Disney Run 1 (36M) | Disney Run 2 (50M) | Change |
|---|---|---|---|
| `st_fwd (vx_score)` | **0.986** | 0.131 | -87% |
| `st_back (vx_score)` | **0.465** | 0.071 | -85% |
| `st_strafe_R (vy_score)` | **0.744** | 0.427 | -43% |
| `st_turn_L (wz_score)` | 0.600 | 0.000 | -100% |
| `st_turn_R (wz_score)` | 0.144 | 0.019 | -87% |
| `vx_tracking (cold)` | **0.699** | 0.123 | -82% |
| `gait_symmetry` | **0.826** | 0.846 | same |
| `action_rate` | 0.069 | **0.059** | -14% (smoother) |
| `mean_responsiveness` | **0.517** | 0.135 | -74% |

**Findings**:
1. **Halving imitation from 15.0 to 8.0 destroyed responsiveness.** The robot barely walks forward (vx=0.028 m/s vs commanded 0.1). It survived all episodes (alive=20 dominates) but did not meaningfully respond to velocity commands.
2. **The imit/alive ratio threshold is real and sharp.** Disney Run 2 has ratio 0.40, which barely meets the Stage 1 recipe's minimum. But with action_rate=-1.5 (far below the recipe's -0.26 threshold), the combination kills the stand-to-walk transition. The three-variable recipe requires ALL conditions met simultaneously.
3. **Smoothness improved marginally** (action_rate 0.069 to 0.059), but this is meaningless when the robot barely moves - it is smooth because it is standing still.
4. **Gait symmetry held** (0.846), which shows the robot isn't in a broken pivoting pattern - it is doing a proper symmetric gait, just at negligible amplitude.
5. **Turning completely lost.** Turn left wz_score=0.000 (achieved 0.11 rad/s vs commanded 0.8), turn right wz_score=0.019. Disney Run 1 already had weak turning; halving imitation made it worse.

**Key inference**: With `action_rate=-1.5`, you need `imitation>=15.0` to push through the smoothness barrier. There is no middle ground - 8.0 is catastrophically insufficient. This confirms Disney's design: their action_rate=-1.5 only works because their effective imitation weight (~22.0) massively exceeds alive.

**Implications for remaining experiments**:
- Testing `action_rate=-0.5` with Disney's other weights is the next critical test. If loosening action_rate from -1.5 to -0.5 recovers walking with imitation=15.0, it proves that the action_rate penalty is the binding constraint on responsiveness.
- Testing `tracking_ang_vel=4.0` addresses Disney Run 1's turning weakness without touching the imitation/action_rate balance that works.

### Disney Run 3: ang_vel=4.0 (fix turning) - IN PROGRESS (local)

Config: Disney Run 1 base with `tracking_ang_vel` increased 2.5 to 4.0. Tests whether stronger angular velocity reward signal fixes the turning deficit.

Full config: `tracking_lin_vel=2.5, tracking_ang_vel=4.0, torques=-1e-3, action_rate=-1.5, action_accel=-0.45, stand_still=-1.0, alive=20.0, imitation=15.0, tracking_sigma=0.25`

50M steps, 8192 envs, local RTX 5080.

**Hypothesis**: Disney Run 1 achieved wz_tracking=0.587 with ang_vel=2.5. Increasing to 4.0 should strengthen the turning gradient.

| Metric | Disney Run 1 (36M) | Disney Run 3 (50M) | Change |
|---|---|---|---|
| `st_turn_L` | **0.600** | 0.000 | **-100%** |
| `st_turn_R` | 0.144 | 0.033 | -77% |
| `wz_tracking` | **0.587** | 0.478 | -19% |
| `gait_symmetry` | 0.826 | **0.891** | +8% |
| `st_fwd` | **0.986** | 0.946 | -4% |
| `st_back` | **0.465** | 0.196 | -58% |
| `mean_responsiveness` | **0.517** | 0.351 | -32% |
| `HEADLINE (training)` | **0.604** | 0.365 | -40% |

**Findings**:
1. **Boosting ang_vel made turning WORSE, not better.** st_turn_L went from 0.600 to 0.000. wz_tracking dropped 19% despite doubling the angular velocity reward weight.
2. **The ang_vel reward competes with imitation**, not complements it. Turning requires asymmetric leg movements that deviate from the symmetric walking reference. Higher ang_vel pushes for more turning, but imitation=15.0 pushes back harder, creating conflicting gradients that the policy resolves by doing neither well.
3. **Gait symmetry improved** (+8%), consistent with the policy avoiding asymmetric gaits that would be needed for turning.
4. **All responsiveness metrics degraded**, not just turning. Backward (-58%), strafe (-74%), and even forward (-4%).

**Key inference**: The Disney config's turning problem is not about insufficient angular velocity reward. It is about **imitation reward penalising the asymmetric leg movements that turning requires**. The LHS Stage 1 correlation (tracking_ang_vel predicting turning) was valid within the LHS range (ang_vel 1.0-4.0, imitation 0.5-2.0), but does not hold at Disney's imitation=15.0. At high imitation, the reference gait attractor dominates.

### Disney Run 4: action_rate=-0.5 (loosen smoothness) - REJECTED

Config: Disney Run 1 base with `action_rate` loosened from -1.5 to -0.5. Tests whether the smoothness constraint was suppressing turning.

Full config: `tracking_lin_vel=2.5, tracking_ang_vel=2.5, torques=-1e-3, action_rate=-0.5, action_accel=-0.45, stand_still=-1.0, alive=20.0, imitation=15.0, tracking_sigma=0.25`

50M steps, 8192 envs, local RTX 5080.

| Metric | Disney Run 1 (36M) | Disney Run 4 (50M) | Change |
|---|---|---|---|
| `action_rate` | **0.069** | 0.078 | +14% (less smooth) |
| `jerk` | **0.050** | 0.065 | +31% (less smooth) |
| `gait_symmetry` | 0.826 | **0.861** | +4% |
| `mean_tilt_deg` | 2.67 | **2.01** | -25% (more stable) |
| `vx_tracking` | **0.699** | 0.685 | -2% |
| `wz_tracking` | **0.587** | 0.528 | -10% |
| `st_fwd` | **0.986** | 0.949 | -4% |
| `st_turn_L` | **0.600** | 0.134 | **-78%** |
| `st_turn_R` | 0.144 | **0.229** | +59% |
| `mean_responsiveness` | **0.517** | 0.424 | -18% |
| `HEADLINE (training)` | **0.604** | 0.477 | -21% |

**Findings**:
1. **Loosening action_rate did NOT help turning** - left turning got catastrophically worse (0.600 to 0.134). Right turning improved slightly (0.144 to 0.229) but net turning quality dropped.
2. **Smoothness regressed significantly** as expected: action_rate +14%, jerk +31%, HF ratio +11%.
3. **Body stability improved** (mean_tilt 2.67 to 2.01 deg) and **gait symmetry improved** (+4%).
4. **Training was more stable** than Disney Run 1 - HEADLINE didn't regress from peak. But peak was much lower (0.477 vs 0.604).
5. **Forward responsiveness held** (0.949 vs 0.986) - loosening action_rate didn't damage forward walking.

**Key insight**: `action_rate=-1.5` is actively HELPING turning. The tight smoothness constraint forces the policy to learn more deliberate, committed turns rather than jittery half-measures. Loosening it lets the policy find lazy, asymmetric solutions (pivot in place rather than turn). The turning problem is definitively about angular velocity reward weight (`tracking_ang_vel`), not smoothness constraint (`action_rate`). This validates Disney Run 3 (ang_vel=4.0) as the correct next experiment.

### Fallback: Run 6 branch (low-alive approach)

If Disney Runs 3-4 fail to fix turning, the alternative is Run 6's low-alive config with targeted improvements:

| Metric | Disney Run 1 (36M) | Run 6 (50M) | Run 6 advantage |
|---|---|---|---|
| st_turn_R | 0.144 | **0.570** | +296% |
| st_turn_L | 0.600 | 0.426 | Disney better |
| st_fwd | 0.986 | **0.993** | comparable |
| gait_symmetry | **0.826** | 0.697 | Disney better |
| action_rate | **0.069** | 0.072 | comparable |
| HEADLINE | **0.604** | 0.532 | Disney better |

Run 6 config: `alive=3.21, imitation=1.83, action_rate=-0.075, tracking_sigma=0.330, tracking_ang_vel=1.64, tracking_lin_vel=2.72, stand_still=-0.95`

Planned improvement: boost `tracking_ang_vel` from 1.64 to 2.5 (it was the lowest value in the LHS range; Spearman data says higher helps turning). Adding mild smoothness penalties (action_rate=-0.2, action_accel=-0.1) may also be possible since Run 6's loose action_rate leaves headroom.

The two branches solve the imit/alive ratio problem from opposite ends:
- **Disney**: crank imitation up to 15+ to overpower alive=20
- **Run 6**: drop alive to 3.2 so even modest imitation=1.83 dominates

## Stage 3 - Bayesian Fine-Tune

(pending Stage 2 completion)

## Stage 4 - Full Validation

(pending Stage 3 completion)

## Final recommendation

(to be written after Stage 4)
