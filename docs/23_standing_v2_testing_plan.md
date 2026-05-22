# Standing Policy V2 - Testing Plan

## Problem Statement

T2b (current best) reports HEADLINE=0.997 but can't survive pushes above 1.0 m/s.
The HEADLINE metric is broken - it only tests up to 0.8 m/s in the push_score component.
In sim the robot looks fragile: any meaningful perturbation topples it.

Disney BDX reports zero falls over 10 hours of real-world deployment.

## Success Criteria (revised)

**Primary**: survive 1.5 m/s pushes in all 4 directions.
**Stretch**: survive 2.0 m/s in at least 3/4 directions.
**Constraint**: stillness score >= 0.95 (no regression from T2b).

## HEADLINE Fix (prerequisite)

Before any training, fix the eval so HEADLINE reflects reality:

```
push_score = (
    0.15 * push_survival[0.4]   # easy
  + 0.15 * push_survival[0.6]   # easy
  + 0.20 * push_survival[0.8]   # medium
  + 0.25 * push_survival[1.0]   # hard
  + 0.15 * push_survival[1.5]   # very hard
  + 0.10 * push_survival[2.0]   # extreme
)
```

Under this formula, T2b would score roughly:
  stillness: 0.40 * 0.99 = 0.396
  push:      0.30 * (0.15+0.15+0.20+0.25*0.75+0.15*0.25+0) = 0.30 * 0.725 = 0.218
  body:      0.15 * 1.0 = 0.15
  angular:   0.15 * 1.0 = 0.15
  HEADLINE:  ~0.914

That's more honest. Target: HEADLINE >= 0.95 under the new formula.

## T2b Baseline (current state)

| Push mag | Front | Back | Left | Right | Rate |
|----------|-------|------|------|-------|------|
| 0.4      | 1     | 1    | 1    | 1     | 100% |
| 0.6      | 1     | 1    | 1    | 1     | 100% |
| 0.8      | 1     | 1    | 1    | 1     | 100% |
| 1.0      | 1     | 0    | 1    | 1     | 75%  |
| 1.2      | 0     | 0    | 1    | 1     | 50%  |
| 1.5      | 0     | 0    | 0    | 1     | 25%  |
| 2.0      | 0     | 0    | 0    | 0     | 0%   |

Failure pattern: back first, then front, then left. Right is strongest.

## Testing Tree

```
U0: Fix HEADLINE metric + add 1.0/1.5/2.0 to quick eval
 |
U1: Clean baseline (207 dims - PRM stripped, phase removed)
 |  WHY: Dead code removal + Disney alignment. Standing has no gait.
 |       standing.py now produces 207-dim obs (was 209 with dead phase dims).
 |       joystick.py reverted to 101-dim E14 format.
 |  EXPECT: Similar or identical to T2b (phase was always [0,0])
 |  IF WORSE: Something unexpected was depending on those dims. Investigate.
 |  IF SAME: Confirms cleanup is safe. Proceed.
 |
 +---> U2: Domain randomisation
 |     WHY: Disney's key ingredient. Forces robust recovery strategies
 |           instead of memorising one specific dynamics model.
 |           The policy currently "knows" exact mass and friction, so
 |           it learns a narrow solution. Randomise and it must develop
 |           wider safety margins.
 |     WHAT: mass ±15%, friction ±30%, actuator Kp ±10%
 |     EXPECT: 1.0 m/s survival improves to 4/4. 1.5 m/s improves.
 |             May see slight stillness regression (policy is more cautious).
 |     IF BIG IMPROVEMENT: DR was the bottleneck. Proceed to U4.
 |     IF SMALL/NO IMPROVEMENT: Training distribution is the problem, not
 |           generalisation. Proceed to U3.
 |     IF STILLNESS REGRESSES BADLY (< 0.90): DR ranges too wide,
 |           halve them and retry.
 |
 +---> U3: Push curriculum (can run in parallel with U2 if GPU allows)
 |     WHY: Current push distribution is uniform 0.3-3.0 m/s.
 |           Most samples are 0.3-1.5 (easy range). The policy rarely
 |           practices the hard cases. Backward is physically hardest
 |           (hip pitch limit +0.524 rad) so needs overrepresentation.
 |     WHAT: Bias push magnitude distribution toward 0.8-2.0 range.
 |           Add 2x backward push frequency (sample back 40%, other 20% each).
 |           Increase push frequency (interval 1.0-3.0s instead of 1.5-4.0s).
 |     EXPECT: Backward 1.0 m/s should improve. 1.5 m/s should appear.
 |     IF BIG IMPROVEMENT: Training distribution was the bottleneck.
 |     IF NO IMPROVEMENT: The robot physically can't recover from these
 |           pushes with current reward/architecture. Go to U4.
 |
 +---> U4: Combine best of U2+U3 + wider network (3x512)
       WHY: If DR and push curriculum each help partially, combining
             them should compound. Wider network (3x512, +8 MB VRAM -
             negligible) gives more capacity for the harder task.
             Disney uses 3x512 with ELU activations.
       WHAT: Best DR ranges from U2 + best push config from U3 + 3x512 ELU.
       EXPECT: This is the "throw everything at it" run. Should be best.
       IF STILL FAILS AT 1.5: Physical limit of the robot at this mass/CoM.
             Document the wall and move on to dual-policy integration.
       IF SUCCEEDS: Ship it. This is our standing policy.
```

## What each change targets

| Change | Targets | Mechanism |
|--------|---------|-----------|
| Strip PRM/phase | Code hygiene | Removes dead obs dims, aligns with Disney |
| Domain randomisation | Push robustness | Forces wider safety margins in learned policy |
| Push curriculum | Backward recovery | More practice at the failure cases |
| 3x512 network | Recovery capacity | More representational power for complex recovery |

## Implementation notes

### Domain randomisation (already existed in randomize.py, ranges widened)
DR was already active via `playground/common/randomize.py`, used by both
standing and walking training. Ranges widened for U2+U3:
- Floor friction: U(0.3, 1.0) (was 0.5-1.0)
- Friction loss: *U(0.85, 1.15) (was 0.9-1.1)
- Armature: *U(0.95, 1.1) (was 1.0-1.05)
- Link masses: *U(0.85, 1.15) (was 0.9-1.1)
- Torso mass offset: +U(-0.15, 0.15) (was -0.1-0.1)
- Qpos jitter: +U(-0.04, 0.04) (was -0.03-0.03)
- KP gains: *U(0.85, 1.15) (was 0.9-1.1)

**WARNING for walking (Phase 3):** These wider DR ranges are shared via
`randomize.py`. When retraining walking, verify E14-level performance
is maintained. If walking regresses, either:
(a) Revert randomize.py to original ranges and make standing-specific
    DR ranges configurable per-env, or
(b) Accept the wider ranges if walking scores stay >= 0.838

### Push curriculum (U3, modified in standing.py)
Standing push config changes:
- Push magnitude: beta distribution biased toward high end, or
  explicit mixture (50% uniform 0.3-3.0, 50% uniform 1.0-3.0)
- Direction: 40% backward, 20% each for front/left/right
- Interval: uniform(1.0, 3.0) instead of (1.5, 4.0)

### Network architecture (U4)
In runner.py, add CLI flag or env-specific config:
- `policy_hidden_layer_sizes=(512, 512, 512)`
- `value_hidden_layer_sizes=(512, 512, 512)`
- Activation: ELU (check if brax PPO supports this)

## Training budget per experiment

- U1: 100M steps (~30 min) - just confirming cleanup is safe
- U2: 150M steps (~45 min) - needs longer to learn with randomisation
- U3: 100M steps (~30 min)
- U4: 200M steps (~60 min) - combined changes need more convergence time

Total: ~3 hours GPU time worst case.

## Parallel execution strategy

U2 and U3 test independent hypotheses (generalisation vs distribution).
If we had two GPUs we'd run them in parallel. With one GPU, run sequentially.
U1 must come first. U4 depends on U2+U3 results.

## Applicability to walking (Phase 3 reference)

When retraining walking with E14 reward config:

**Domain randomisation**: YES, keep the wider ranges. Walking also benefits
from robust sim-to-real transfer. Disney uses the same DR for both standing
and walking training. However, if walking WALKING_SCORE drops below 0.838,
the DR ranges in randomize.py may need to be reverted or made per-env
configurable. Test walking first with wider DR before committing.

**Push curriculum**: NO, not applicable. Walking has its own push config
(interval 3.0-8.0s, magnitude 0.5-2.0) tuned for mid-gait recovery. The
bimodal hard-push distribution is standing-specific. Walking push recovery
is a different problem (maintain gait vs recover to standstill).

**Network architecture**: Walking uses 512-256-128 with 101-dim input.
E14 proved this is sufficient. Only change if walking scores regress.
Standing may benefit from 3x512 due to the harder recovery task, but
this is tested separately in U4.

**Obs format**: Walking stays at 101-dim (E14 format). Standing at 207-dim.
mujoco_infer.py and analysis scripts auto-detect from ONNX input shape.

## Physical limits reference

- Robot mass: 2.1 kg, CoM height: 0.26 m
- Single-step kinematic limit (backward): 0.84 m/s
  (hip_pitch forward range +0.524 rad)
- Disney BDX: survives "random disturbance forces" but paper
  doesn't specify exact magnitudes
- Momentum at 1.5 m/s push: 2.1 * 1.5 = 3.15 N·s
- At 2.0 m/s: 4.2 N·s
