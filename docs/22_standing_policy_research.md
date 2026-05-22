# Standing Policy Research and Testing Plan

This document captures everything we know about standing policy training, what has been tried, what works, and what to test next. It is the single source of truth for standing policy experiments.

---

## 1. Problem Statement

Train a standing balance policy for Open Duck Mini V2 (2.1kg, 0.26m CoM, 14 actuators) that:
- Stands still with minimal drift and zero unnecessary stepping
- Recovers from pushes up to 3.0 m/s in all directions
- Tracks head commands while standing

Current best (S20): HEADLINE=0.974, push recovery fails above 0.8 m/s backward and above 1.0 m/s in other directions. Stillness is excellent (0 steps, 14mm drift). The remaining gap is push recovery at higher forces.

---

## 2. Physical Limits (proven)

### Single-step kinematic limit
- Maximum backward foot reach: 11.76cm (hip_pitch forward range +0.524 rad / 30 degrees)
- Maximum forward foot reach: ~18cm (hip_pitch backward range -1.222 rad / 70 degrees)
- Single-step backward recovery limit: **0.84 m/s** (CP at 0.84 = 11.81cm, exceeds 11.76cm reach)
- Single-step forward recovery limit: ~1.4 m/s (more generous due to hip asymmetry)

### Multi-step recovery
- 1.0 m/s backward IS recoverable with two steps (residual velocity after first step is only 0.16 m/s)
- For pushes above 0.84 m/s, the policy MUST learn to take multiple recovery steps
- This requires wide enough CP sigma to provide gradient signal above the single-step limit

### Capture point formula
```
CP = CoM_xy + CoM_vel_xy / sqrt(g/h)
```
At h=0.26m, omega = sqrt(9.81/0.26) = 6.14 rad/s. So CP distance = velocity / 6.14.

| Push velocity | CP distance | Reachable in 1 step? |
|--------------|-------------|---------------------|
| 0.6 m/s | 9.8cm | Yes (all dirs) |
| 0.8 m/s | 13.0cm | Forward only |
| 0.84 m/s | 13.7cm | Forward only |
| 1.0 m/s | 16.3cm | Forward only |
| 1.2 m/s | 19.5cm | No |
| 1.5 m/s | 24.4cm | No |

---

## 3. Experiment History

### Results Table (S-series standing experiments)

All experiments use 50M steps unless noted. Push scores are survival rates at each magnitude.

| Exp | HEADLINE | Stillness | Push | Body | 0.8 | 1.0 | 1.2 | Steps | Drift(m) | Key change |
|-----|----------|-----------|------|------|-----|-----|-----|-------|----------|------------|
| S7  | 0.928 | 0.857 | 1.000 | N/A | 1.00 | 0.75 | 0.25 | 2 | 0.028 | Motor clamp fix |
| S8  | 0.928 | 0.857 | 1.000 | N/A | 1.00 | 0.75 | 0.25 | 2 | 0.028 | Harder push range |
| S9  | 0.954 | 0.908 | 1.000 | N/A | 1.00 | 0.25 | 0.00 | 2 | 0.018 | Body push + angular |
| S10 | 0.993 | 0.984 | 1.000 | 1.000 | 1.00 | 0.50 | 0.25 | 1 | 0.003 | Harder range + eval fix |
| S11 | 0.971 | 0.927 | 1.000 | 1.000 | 1.00 | 0.75 | 0.00 | 2 | 0.013 | Gated action rate |
| S12 | 0.974 | 0.935 | 1.000 | 1.000 | 1.00 | 1.00 | 0.00 | 0 | 0.023 | Capture point reward |
| S13 | 0.996 | 0.990 | 1.000 | 1.000 | 1.00 | 0.75 | 0.50 | 0 | 0.009 | High force CP |
| S13b| 0.997 | 0.992 | 1.000 | 1.000 | 1.00 | 0.75 | 0.50 | 0 | 0.007 | S13 continued |
| S14 | 0.994 | 0.984 | 1.000 | 1.000 | 1.00 | 0.75 | 0.25 | 1 | 0.002 | Freq curriculum |
| S15 | 0.997 | 0.992 | 1.000 | 1.000 | 1.00 | 0.75 | 0.50 | 0 | 0.007 | History + veldamp + footplace |
| S16 | 0.993 | 0.983 | 1.000 | 1.000 | 1.00 | 0.75 | 0.75 | 0 | 0.011 | Backward bias (40%) |
| S17 | 0.983 | 0.995 | 0.950 | 1.000 | 0.75 | 0.50 | 0.00 | 0 | 0.005 | CP sigma 0.04->0.12 (50M) |
| S18 | 0.902 | 0.754 | 1.000 | 1.000 | 1.00 | 0.75 | 0.75 | 5 | 0.011 | Wide sigma full (100M, no stand_still) |
| S19 | 0.885 | 0.970 | 0.700 | 1.000 | 0.50 | 0.50 | 0.00 | 0 | 0.014 | stand_still=-1.5, vel_damp=2.0 |
| S20 | 0.974 | 0.973 | 0.950 | 1.000 | 0.75 | 0.75 | 0.00 | 0 | 0.014 | stand_still=-0.5, vel_damp=1.5 |

### Key Findings

1. **CP sigma is the push recovery bottleneck**: sigma=0.04 (default) gives zero gradient for pushes above 0.84 m/s. S17/S18 proved that sigma=0.12 extends the recovery boundary.

2. **Stillness vs stepping tension**: This is the core trade-off.
   - S18 (no stand_still, vel_damp=1.0): Best push recovery (1.0 back, 0.75@1.2) but 5 steps, poor stillness (0.754)
   - S19 (stand_still=-1.5, vel_damp=2.0): Good stillness (0.970) but terrible push (0.700, failed at 0.6)
   - S20 (stand_still=-0.5, vel_damp=1.5): Good balance (still=0.973, push=0.950) but no 1.2 recovery

3. **Pre-sigma experiments (S7-S16) hit a wall at 0.8-1.0 m/s**: All had sigma=0.04 and couldn't break through. S16 was the best (0.75@1.2) but only because of backward push bias.

4. **Grace period gating works**: stand_still and action_rate costs ramp from 0 to full over 100 steps (2s) after a push, preventing these costs from fighting recovery.

5. **Reward contribution dominance matters**: In S19, stand_still cost (818) exceeded capture_point reward (784), completely killing push recovery. The lesson: measure `weight * E[|r_term|] * time_active` before running.

---

## 4. Reward Architecture Analysis

### Current reward terms (S20 config)

| Term | Weight | Type | Est. magnitude | Est. contribution | Purpose |
|------|--------|------|----------------|-------------------|---------|
| alive | +20.0 | reward | 1.0 | 20.0/step | Survival |
| capture_point | +5.0 | reward | 0.3-0.9 | 1.5-4.5/step | Step toward CP |
| foot_placement | +3.0 | reward | 0.1-0.5 | 0.3-1.5/step | Foot near CP on touchdown |
| foot_direction | +2.0 | reward | 0.0-0.2 | 0.0-0.4/step | Step in velocity direction |
| velocity_damping | +1.5 | reward | 0.5-1.0 | 0.75-1.5/step | Damp COM velocity |
| head_pos | -2.0 | cost | 0.1-0.5 | 0.2-1.0/step | Track head commands |
| orientation | -1.0 | cost | 0.05-0.2 | 0.05-0.2/step | Upright posture |
| stand_still | -0.5 | cost | 1.0-3.0 | 0.5-1.5/step (quiet only) | Joint freeze at zero cmd |
| action_rate | -0.375 | cost | 0.5-2.0 | 0.19-0.75/step | Smooth actions |
| torques | -0.001 | cost | 100-300 | 0.1-0.3/step | Energy efficiency |

### Observations

- **alive dominates**: 20.0/step is 4-13x any other term. The policy optimises for survival above all else.
- **capture_point and velocity_damping conflict during recovery**: CP rewards stepping toward the capture point (moving), while vel_damp rewards being still. After a push, the policy needs to step (high CP) but vel_damp penalises the resulting COM velocity.
- **stand_still only fires at cmd_norm < 0.01**: This is correct (doesn't interfere with push recovery), but it's competing with alive during quiet periods.

---

## 5. Root Cause Analysis: Why Push Recovery Stalls Above 1.0 m/s

Three compounding factors:

### Factor A: CP sigma too narrow
- sigma=0.12 is better than 0.04 but still gives low reward for large CP errors
- At 1.2 m/s push, CP distance is 19.5cm. Best foot placement might reach 11.7cm backward.
- Error = 19.5 - 11.7 = 7.8cm. With sigma=0.12: `exp(-(0.078)^2/(0.12)^2)` = 0.65
- First step reduces velocity to ~0.36 m/s. CP error drops to ~2cm. sigma=0.12 gives 0.97
- The gradient exists but is shallow for the first step. The policy needs to discover that a suboptimal first step leads to a high-reward second step.

### Factor B: alive dominates the reward landscape
- alive=20 per step means 60,000 over a 3000-step episode
- CP reward at weight=5 with mean 0.5 = 2.5 per step = 7,500 per episode
- The policy gets more reward from surviving (doing nothing risky) than from attempting aggressive recovery steps

### Factor C: velocity_damping fights recovery stepping
- During a 1.0 m/s push recovery, COM velocity is high for 10-20 steps
- vel_damp=1.5 with sigma=0.3 penalises this directly
- The optimal multi-step recovery involves accepting high COM velocity temporarily

---

## 6. Research: Systematic Approaches

### Approach 1: Velocity-gated reward routing
**Principle**: Different reward terms should be active in different modes. During quiet standing, penalise movement. During recovery, reward active stepping.

**Implementation**: Use `||base_vel_xy||` as a mode signal:
- **Quiet mode** (||v|| < threshold): activate stand_still, velocity_damping
- **Recovery mode** (||v|| > threshold): activate capture_point, foot_placement, foot_direction; suppress stand_still, reduce velocity_damping
- Smooth transition via sigmoid: `recovery_weight = sigmoid((||v|| - threshold) / temperature)`

**Why this should work**: Eliminates the fundamental tension. vel_damp and stand_still won't fight recovery. CP and foot_placement won't activate during quiet standing (reducing noise in the reward signal).

**Risk**: The transition threshold and temperature are new hyperparameters. If threshold is too low, every small perturbation triggers recovery mode. If too high, the policy doesn't get recovery reward early enough.

**Suggested values**: threshold=0.15 m/s (3x the typical quiet-standing COM drift), temperature=0.05.

### Approach 2: Reduce alive weight
**Principle**: alive=20 makes survival the dominant objective. The policy learns conservative strategies. Reducing alive lets other terms drive behaviour.

**Evidence from walking experiments**: E11 reduced alive from 20 to 5 - single most impactful change ever (+0.119 HEADLINE improvement). This is directly applicable to standing.

**Risk**: Too low and the policy may not learn to stay upright. LHS screening showed alive=3.2-3.4 worked for walking; standing may need more since pushes are harsher.

**Suggested values**: alive=5.0 (matching the proven E11 config).

### Approach 3: Wider CP sigma with foot_placement sigma
**Principle**: For multi-step recovery, the first step doesn't need to be perfect - it just needs to reduce the CP error enough for the second step to finish the job. Wider sigma gives credit for partial progress.

**Current**: CP sigma=0.12, foot_placement sigma=0.10
**Suggested**: CP sigma=0.20, foot_placement sigma=0.15

**Risk**: Too wide and the reward becomes insensitive to precise foot placement during small perturbations.

### Approach 4: Recovery curriculum
**Principle**: Start with easy pushes, gradually increase. The policy learns the stepping reflex first, then extends it to harder pushes.

**Current**: Push magnitude ramps from 40% to 100% over 2000 episode steps. This may not be aggressive enough - the policy may settle into a "stiffen" strategy during easy pushes and never discover stepping.

**Alternative**: Start at 60% magnitude, ramp to 100% over 1000 steps. Forces stepping earlier.

### Approach 5: Episode termination on high CP error
**Principle**: If the CP is far outside the support polygon and the robot hasn't started stepping, terminate the episode early. This provides a strong negative signal for "freezing" during a push.

**Risk**: May make training unstable if termination is too aggressive.

---

## 7. Decision Tree: What to Test

### Priority order (test one variable at a time)

```
START (S20 baseline: HEADLINE=0.974, push=0.950, no 1.2 recovery)
  |
  +-- T1: Reduce alive 20->5 (proven in walking, never tested in standing)
  |     |
  |     +-- SUCCESS (push >= 0.950, 1.0 rate improves)
  |     |     |
  |     |     +-- T2: Velocity-gated reward routing
  |     |     |     |
  |     |     |     +-- SUCCESS -> T4: Wider sigma (0.12->0.20)
  |     |     |     +-- FAIL -> T3: Wider sigma without routing
  |     |     |
  |     |     +-- T3: Wider sigma (0.12->0.20) alone
  |     |           |
  |     |           +-- SUCCESS -> T5: Push curriculum (60% start)
  |     |           +-- FAIL -> Revert sigma, try T2
  |     |
  |     +-- FAIL (push regresses below 0.900)
  |           |
  |           +-- T1b: alive=10 (halfway)
  |                 |
  |                 +-- SUCCESS -> Continue from T2
  |                 +-- FAIL -> alive is not the bottleneck, try T2 at alive=20
  |
  +-- [If T1-T3 don't break through 1.2 m/s]:
        |
        +-- T6: Velocity-gated routing at best config so far
        +-- T7: CP termination (terminate if CP error > 0.25m for 20 consecutive steps)
        +-- T8: Asymmetric push curriculum (harder backward pushes in training)
```

### Success criteria for each test

| Test | Run for | Success | Failure |
|------|---------|---------|---------|
| T1: alive=5 | 50M | push_score >= 0.950 AND stillness >= 0.900 | push_score < 0.900 |
| T1b: alive=10 | 50M | push_score >= 0.950 | push_score < 0.900 |
| T2: velocity-gated routing | 50M | 1.0 rate >= 0.75 all dirs AND stillness >= 0.950 | 1.0 rate < 0.50 OR stillness < 0.800 |
| T3: wider sigma | 50M | 1.0 rate >= 1.00 OR 1.2 rate > 0.25 | push_score regresses > 0.05 |
| T4: sigma + routing | 50M | 1.2 rate >= 0.50 | 1.2 rate = 0 |
| T5: push curriculum | 50M | 1.2 rate >= 0.50 | No change vs parent |
| T6: routing at best | 50M | 1.2 rate >= 0.50 AND stillness >= 0.950 | Fails either |
| T7: CP termination | 50M | 1.2 rate >= 0.50 | Training unstable |

### Pre-flight checklist (before EVERY experiment)

1. Calculate expected reward contributions: `weight * E[|term|]` for each term
2. Verify no single term exceeds 2x the next-largest (except alive, which we're testing)
3. Confirm the changed variable is different from S20 baseline
4. Set aside S20 ONNX for comparison
5. Run 50M steps (sufficient for this robot - S-series experiments plateau by 40M)

---

## 8. Contribution Measurement Protocol

Before each experiment, estimate reward magnitudes to prevent S19-style dominance bugs.

```python
# For each reward term, estimate:
# 1. Per-step magnitude during quiet standing
# 2. Per-step magnitude during push recovery
# 3. Total contribution = weight * magnitude * fraction_of_episode

# Example for T1 (alive=5):
# quiet:   alive=5*1.0=5.0, cp=5*0.95=4.75, vel_damp=1.5*0.98=1.47, stand_still=0.5*2.0=1.0
# push:    alive=5*1.0=5.0, cp=5*0.5=2.5, foot_place=3*0.3=0.9, foot_dir=2*0.1=0.2
# Ratio: alive 5.0 vs cp 4.75 during quiet -> 1.05:1 (GOOD, not dominated)
```

---

## 9. What Not to Test (dead ends)

These have been tried and conclusively failed:

| Approach | Why it failed | Evidence |
|----------|--------------|---------|
| stand_still weight increases | Doesn't stop stepping; the policy learned stepping = stability | E19 (stand_still=-15), E20 (foot contact+5) - walking experiments |
| Foot contact reward | Only fires 10% of time (standstill fraction), insufficient gradient | E20 |
| Backward push direction bias alone | Doesn't help without CP sigma fix | S16 (0.75@1.2 but only backward) |
| Adaptive KL | Caps peak quality for modest oscillation reduction | E13 |
| Fewer PPO epochs | Slower learning, lower peak | E12 |
| LayerNorm | Incompatible with this architecture | E1, E3 |

---

## 10. Experiment Results

### T1: alive=5 (FAIL)

**Config**: S20 baseline with alive 20->5. 50M steps.
**Result**: HEADLINE=0.904, stillness=0.967, push=0.850

| Metric | S20 (alive=20) | T1 (alive=5) | Change |
|--------|---------------|-------------|--------|
| HEADLINE | 0.974 | 0.904 | -0.070 |
| Stillness | 0.973 | 0.967 | -0.006 |
| Push score | 0.950 | 0.850 | -0.100 |
| 0.6 rate | 100% | 75% (back) | -25% |
| 0.8 rate | 75% | 75% | same |
| 1.0 rate | 75% | 50% | -25% |
| Angular | 100% | 75% | -25% |

**Verdict**: FAIL. Push recovery regressed significantly. alive=5 weakened the survival incentive too much for the standing domain. Unlike walking (where alive=5 was transformative in E11), standing relies more heavily on the survival signal because pushes are the primary challenge. The policy became less aggressive about maintaining balance, not more aggressive about recovery.

**Key learning**: The alive reduction that worked for walking does NOT transfer to standing. Walking and standing have fundamentally different reward dynamics - walking has tracking rewards that benefit from alive reduction, while standing has only balance rewards where alive IS the primary useful signal.

### T1b: alive=10 (FAIL)

**Config**: S20 baseline with alive 20->10. 50M steps.
**Result**: HEADLINE=0.872, stillness=0.992, push=0.750

| Metric | S20 (alive=20) | T1b (alive=10) | T1 (alive=5) |
|--------|---------------|---------------|-------------|
| HEADLINE | 0.974 | 0.872 | 0.904 |
| Stillness | 0.973 | 0.992 | 0.967 |
| Push score | 0.950 | 0.750 | 0.850 |
| 0.4 rate | 100% | 75% | 100% |
| 0.6 rate | 100% | 75% | 75% |
| 1.2 rate | 0% | 25% (left) | 0% |

**Verdict**: FAIL. Push recovery regressed further (0.750). Backward push fails even at 0.4 m/s. One bright spot: 25% survival at 1.2 m/s (left direction), suggesting that with less alive dominance the policy occasionally discovers aggressive stepping, but inconsistently.

**Conclusion on alive reduction**: Both T1 and T1b confirm alive reduction hurts standing. Unlike walking, standing has no tracking rewards to "unlock" by reducing alive. The survival signal IS the useful signal for balance. The decision tree now routes to T2: velocity-gated reward routing at alive=20.

### T2: Velocity-gated reward routing (SUCCESS)

**Config**: S20 baseline (alive=20) with velocity-gated routing:
- `recovery_gate = sigmoid((||com_vel_xy|| - 0.15) / 0.05)`
- `quiet_gate = 1 - recovery_gate`
- velocity_damping gated by quiet_gate
- foot_placement, foot_direction gated by recovery_gate
- stand_still gated by quiet_gate (stacks with existing grace period gate)
- capture_point always active

**Result**: HEADLINE=0.992, stillness=0.981, push=1.000. **BEST EVER.**

| Metric | S20 (baseline) | T2 (gating) | Change |
|--------|---------------|-------------|--------|
| HEADLINE | 0.974 | 0.992 | +0.018 |
| Stillness | 0.973 | 0.981 | +0.008 |
| Push score | 0.950 | 1.000 | +0.050 |
| 0.8 rate | 75% | 100% | +25% (backward fixed) |
| 1.0 rate | 75% | 100% | +25% (backward fixed) |
| 1.2 rate | 0% | 25% (right) | new |
| Drift | 14mm | 6mm | -57% |

**Verdict**: SUCCESS. Velocity gating eliminated the backward push recovery wall at 0.8 m/s. The key was removing velocity_damping's braking force during recovery - vel_damp was penalising the COM velocity that necessarily accompanies stepping. Stillness improved too because foot_placement noise during quiet standing was suppressed.

**Training trajectory**: Push=1.000 from 18M steps onward (all last 8 checkpoints). Stillness dipped to 0.640 mid-training then recovered to 0.981 by end. HEADLINE oscillated 0.856-0.998 but final checkpoint was near-peak.

### T4: Wider CP sigma (PARTIAL SUCCESS)

**Config**: T2 baseline (velocity gating) with CP sigma 0.12->0.20 and foot_placement sigma 0.10->0.15.
**Result**: HEADLINE=0.965, stillness=0.913, push=1.000

| Metric | T2 (gating) | T4 (gating + wide sigma) | Change |
|--------|-------------|-------------------------|--------|
| HEADLINE | 0.992 | 0.965 | -0.027 |
| Stillness | 0.981 | 0.913 | -0.068 |
| Push score | 1.000 | 1.000 | same |
| 1.0 rate | 100% | 75% (back fail) | -25% |
| 1.2 rate | 25% | 75% (back fail) | +50% |
| Drift | 6mm | 17mm | +11mm |

**Verdict**: PARTIAL. Wider sigma pushed 1.2 m/s from 25% to 75% but degraded stillness (0.981->0.913) and 1.0 backward (100%->75%). The wider sigma makes the CP reward noisier during quiet standing. Training was also less stable - stillness collapsed to 0.023 mid-training before recovering.

**Key learning**: CP sigma controls a stillness-vs-high-force tradeoff. sigma=0.12 (T2) gives better all-round balance. sigma=0.20 (T4) unlocks 1.2 m/s but at a cost. Backward direction is the consistent failure at every threshold due to the 11.76cm hip_pitch reach limit.

### T2b: Longer training 100M (SUCCESS)

**Config**: Same as T2 (velocity gating, CP sigma=0.12, alive=20). 100M steps.
**Best checkpoint**: 79M steps (HEADLINE=0.997)

| Metric | T2 (50M) | T2b (100M) | Change |
|--------|---------|-----------|--------|
| HEADLINE | 0.992 | 0.997 | +0.005 |
| Stillness | 0.981 | 0.993 | +0.012 |
| Push score | 1.000 | 1.000 | same |
| 1.0 rate | 100% | 75% (back) | -25% |
| 1.2 rate | 25% | 50% | +25% |
| 1.5 rate | 0% | 25% (right) | new |

**Verdict**: SUCCESS. Longer training improved 1.2 m/s (25%->50%), broke into 1.5 m/s (25%), and improved stillness (0.981->0.993, 0 steps). Best checkpoint at 79M confirms checkpoint selection still needed. The 1.0 backward regression is the only downside - may be stochastic given the kinematic limit at 0.84 m/s.

### Current Best: T2b (velocity-gated routing, CP sigma=0.12, 100M steps)

HEADLINE=0.997, stillness=0.993, push=1.000, 50% at 1.2 m/s, 25% at 1.5 m/s. Best checkpoint: `checkpoints/T2b_gating_100M/2026_05_18_172744_79298560.onnx`
