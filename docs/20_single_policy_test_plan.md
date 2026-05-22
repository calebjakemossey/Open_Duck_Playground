# Single Policy Test Plan

Exhaust single-model approaches before considering separate stand/walk policies.

## Base config for all tests

E14 config + fixed standing reference + imitation gate:
- alive=5.0, imitation=15.0, tracking_lin_vel=2.5, tracking_ang_vel=2.5
- action_rate=-1.5, action_accel=-0.45, orientation=-5.0, stand_still=-1.0
- tracking_sigma=0.25, clipping_epsilon_value=0.2
- Imitation gate: `clip(sqrt(vx^2 + vy^2 + wz^2) / 0.05, 0, 1)`
- Fixed polynomial_coefficients.pkl (static standing pose at dx=0)
- Biased sampling: 12% pure turn, 10% standstill, 78% normal
- 150M steps, 8192 envs

## What we're trying to solve

| Problem | Current state | Target | Measured by |
|---------|--------------|--------|-------------|
| Walking in place during standstill | 73 steps/10s, 52% air time, 25mm foot lifts across ALL models | < 10 steps, < 10% air time | step_count, air_time_ratio, mean_swing_peak |
| Yaw rotation during standstill | 3 deg/s (E14 best), 27 deg/s (E11 worst) | < 2 deg/s | achieved_wz_mean |
| Linear drift during standstill | 5 cm/s forward (E14) | < 2 cm/s | achieved_vx_mean, total_distance |
| Push recovery while standing | Survives 0.3 N*s, falls at 0.6 from front | Survive 0.6 from all directions | push_recovery survival_rate |
| Turning from standstill | 0.96/0.98 (E14), regresses with backward | > 0.90 both directions | turn_left/right wz_tracking_score |
| Backward walking | 0.33 (E14), 0.64 (E18) - trades off with turning | > 0.50 (acceptable if slow) | backward_walk vx_tracking_score |
| Forward precision | 0.91-1.00 depending on checkpoint | > 0.95 | forward_walk vx_tracking_score |

## Test sequence

### Block A: Fix standing (priority - most visible defect)

#### E19: Strong stand_still with gate

**Change**: stand_still -1.0 to -15.0

**Why**: E12 tested stand_still=-5 but WITHOUT the imitation gate. imitation=15 overwhelmed it. With the gate active, imitation is zero at standstill. stand_still=-15 now competes against alive=5 only (3:1 penalty-to-reward ratio). The stand_still cost function already gates itself on cmd_norm < 0.01, so it only fires during standstill - it won't affect walking.

**What could go wrong**: The cost penalises joint deviation from default pose AND joint velocity. At -15, the velocity penalty component might make the policy sluggish when transitioning from standing to walking (the first few steps get penalised until cmd_norm exceeds 0.01).

**Pass criteria**:
- Standing: step_count < 20, air_time_ratio < 0.20, achieved_wz < 3 deg/s
- Walking: forward vx > 0.90, strafe L/R > 0.90, turn L/R > 0.85
- Responsiveness: stand_then_forward > 0.90 (tests transition speed)

**If it passes**: Proceed to Block B with stand_still=-15.
**If standing partially improves but not enough**: Try E19b at -25.
**If walking/responsiveness regresses**: The transition penalty theory is correct. Try E20 instead.

#### E20: Foot contact reward during standing

**Change**: Add new reward term `stand_contact` that rewards both feet on ground when cmd_vel is near zero.

```python
def reward_stand_contact(contact, commands):
    cmd_norm = jp.linalg.norm(commands[:3])
    both_feet_down = contact[0] & contact[1]  # boolean AND
    return both_feet_down * (cmd_norm < 0.01)
```

Weight: start at +5.0 (positive reward, not penalty). This directly rewards the behaviour we want (feet planted) rather than penalising the behaviour we don't want (joint deviation).

**Why**: stand_still penalises joint positions deviating from default, which is indirect. The duck could satisfy stand_still by keeping joints at default while still lifting feet (if default pose has slight knee bend). A contact reward directly targets the stepping-in-place problem.

**What could go wrong**: The reward is binary (both feet down or not). Could cause jerky transitions as the policy learns to slam feet down rather than gently keeping them planted. May need a soft version using contact force magnitude.

**Pass criteria**: Same as E19 but specifically air_time_ratio < 0.10, mean_swing_peak < 5mm.

**Only run if**: E19 doesn't adequately reduce stepping.

#### E20b: Combined stand_still + foot contact

**Change**: E19's stand_still=-15 AND E20's stand_contact=+5.0 together.

**Why**: Belt and braces. Penalise joint movement AND reward feet on ground. Only needed if neither alone is sufficient.

### Block B: Improve precision and turning (independent of Block A)

#### E21: tracking_ang_vel=3.5

**Change**: tracking_ang_vel 2.5 to 3.5

**Why**: E11's turning regressed from 0.95 to 0.35-0.58 when alive dropped from 20 to 5. E14's gate recovered turning to 0.96, but turning still oscillates across checkpoints. Slightly boosting ang_vel reward (2.5 to 3.5, not 4.0 which competed with imitation in Disney Run 3) gives the optimiser a stronger gradient for turning precision.

**What could go wrong**: Disney Run 3 showed 4.0 made turning worse by competing with imitation. 3.5 is between the working value (2.5) and the broken value (4.0). The boundary might be sharp.

**Pass criteria**:
- Turn L/R > 0.92 in majority of checkpoints (not just one)
- Backward vx > 0.30 (not worse than E14)
- Forward vx > 0.90

**If it works**: Adopt 3.5 as the new base for subsequent tests.
**If turning gets worse**: The competition with imitation kicks in below 4.0. Keep 2.5.

#### E22: tracking_sigma=0.05 (tight precision)

**Change**: tracking_sigma 0.25 to 0.05

**Why**: The untested combination. E10 proved sigma=0.05 is necessary for velocity precision but failed with alive=20 (the tighter sigma made precision harder to earn, so the optimiser maximised survival instead). E11 proved alive=5 works. With alive=5, the survival reward is modest enough that tight sigma should create useful precision gradient rather than driving the optimiser away from tracking.

**What could go wrong**: Early training may be harder - sigma=0.05 means the policy gets almost zero tracking reward until it's already walking at roughly the right speed. Could slow learning or cause early-training instability.

**Pass criteria**:
- Forward vx_tracking > 0.98 (recovering E9-level precision)
- HEADLINE >= 0.85
- No capability regressions vs E14

**If it works**: This is the biggest potential gain. E9 had forward=0.999 with tight sigma.
**If early training stalls**: Try sigma=0.10 as a middle ground.

### Block C: Combined best (depends on A and B results)

#### E23: Best of Blocks A + B

**Change**: Combine whatever worked from E19-E22.

**Why**: Each block tests one change in isolation. If stand_still=-15 fixes standing and tracking_ang_vel=3.5 fixes turning, combining them should give both improvements. But reward interactions are non-linear - we need to verify.

**Pass criteria**: Best scores from each individual test maintained when combined.

### Block D: Structural reward changes (only if A+B insufficient)

#### E24: Mode-conditional reward routing

**Change**: Inspired by the Gait-Conditioned RL paper. Define two modes based on cmd_vel magnitude:

Standing mode (cmd_norm < 0.01):
- imitation=0 (already gated)
- stand_still=-15
- stand_contact=+5
- action_rate=-3.0 (strongly penalise any joint movement)

Walking mode (cmd_norm >= 0.01):
- imitation=15
- stand_still=0
- stand_contact=0
- action_rate=-1.5

Smooth blend between modes using the same gate function: `clip(cmd_norm / 0.05, 0, 1)`.

**Why**: Rather than tuning a single set of weights that compromise between standing and walking, give each mode its own reward balance. The gate already provides the blending mechanism.

**What could go wrong**: The transition zone (cmd_norm 0.01-0.05) gets a blended reward that might be confusing. The policy might learn to avoid the transition zone entirely.

#### E25: Increased push magnitude during standstill

**Change**: When cmd_vel is zero, increase push magnitude range from [0.1, 1.0] to [0.3, 1.5] N*s.

**Why**: Push recovery during standing is weak (fails at 0.6 N*s from front). Training with stronger pushes specifically during standstill forces the policy to learn active balance without stepping. This is what Disney's standing policy gets trained with.

**What could go wrong**: Stronger pushes might make the policy take defensive steps, directly conflicting with the "don't step" objective. May need to combine with strong stand_contact reward.

## Evaluation protocol

For every experiment, eval ALL checkpoints (not just the best HEADLINE) and report:

1. **Standing metrics**: step_count, air_time_ratio, mean_swing_peak, achieved_vx/vy/wz, com_drift_total
2. **Cold-start tracking**: forward, backward, strafe L/R, turn L/R (primary axis tracking score)
3. **Responsiveness**: all stand-then-command scores (tests transition quality)
4. **Push recovery**: survival rate at 0.3, 0.6, 1.0, 1.5 N*s from front/back/side
5. **Gait symmetry**: gait_symmetry_score
6. **HEADLINE**: for trend comparison only, not as the primary success metric

## Decision tree

```
E19 (stand_still=-15)
  |
  +-- Standing fixed? ----YES----> E21 + E22 (parallel, independent)
  |                                  |
  |                                  +-> E23 (combine winners)
  |                                       |
  |                                       +-> DONE (single policy exhausted)
  |
  +-- Standing better but not fixed? --> E20 (foot contact reward)
  |                                       |
  |                                       +-- Fixed? -> continue to E21/E22
  |                                       +-- Not fixed? -> E20b (combined)
  |                                             |
  |                                             +-- Fixed? -> continue
  |                                             +-- Not fixed? -> E24 (mode routing)
  |
  +-- Standing unchanged or walking regressed? --> E20 (skip penalty, try reward)
                                                    |
                                                    (follow same branch as above)

If Block D still can't fix standing:
  -> Separate standing/walking policies (Disney approach)
```

## What "exhausted" means

Single policy is exhausted when EITHER:
- All Block A-D tests fail to get standing step_count < 20 AND air_time_ratio < 0.20
- OR: fixing standing consistently breaks walking/turning with no combination resolving the conflict

At that point, the evidence says the single network can't represent both "don't move legs" and "move legs precisely" simultaneously, and separate policies are the right architecture.
