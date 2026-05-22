# Open Duck Mini V2 - Training Knowledge Base

Everything we've learned about training walking policies for the Open Duck Mini V2. This is the single source of truth for our understanding of the system. The experiment catalogue (`catalogue.md`) has detailed per-run metrics; this document explains *why* things work the way they do.

Last updated: 2026-05-15 (E11 complete, new best policy).

---

## How RL training works

We simulate 8,192 copies of the duck robot in parallel. Each duck receives random commands ("walk forward", "turn left", "stand still") and tries to follow them. After each action, we give it a score (the **reward**) based on how well it did. Over millions of steps, the neural network learns to produce motor commands that maximise its total reward.

The reward is a weighted sum of terms:
- **Positive**: alive (staying upright), tracking (matching commanded velocity), imitation (matching the duck waddle reference motion)
- **Negative**: penalties for jerky motion, excessive torque, body tilt, vertical velocity

The weights on these terms determine what the duck "cares about." If alive is too dominant relative to tracking, the optimal strategy becomes "stand still and don't fall." If imitation is too low, the duck walks generically instead of waddling.

The policy is a neural network (512-256-128 hidden layers) that takes sensor readings (gyro, accelerometer, joint positions/velocities, commanded velocity, imitation phase) and outputs 14 joint position targets.

### How we measure success

**HEADLINE score** = 0.5 x cold_start_avg + 0.5 x responsiveness_avg

Each sub-test scores how precisely the robot matches the commanded velocity:

```
score = exp(-(achieved_velocity - target_velocity)^2 / scale^2)
```

| Score | Meaning |
|-------|---------|
| 1.0 | Exactly the commanded velocity |
| 0.78 | Off by 25% |
| 0.37 | Off by one scale unit (0.05 m/s for walking, 0.20 rad/s for turning) |
| ~0.00 | Doesn't move or moves in the wrong direction |

**Cold-start tests**: Can the duck walk/turn/strafe when the command is active from the start? (6 scenarios)

**Responsiveness tests**: Can the duck respond to a new command after standing still for 2 seconds? (6 scenarios) This catches a specific failure mode where the duck learns that standing still is safe and won't break out of it.

**Realistic ceiling**: ~0.93-0.96. A waddling biped always has some velocity variance.

---

## Bug fixes that mattered more than tuning

### Angular velocity recording bug (E9)

The reference motion generator computed angular velocity as `axis * angle^2 / dt` instead of `axis * angle / dt`. This made all recorded angular velocities 8-12x too small. The imitation reward was literally penalising the duck for turning at the correct speed.

**Impact**: This single bug fix improved HEADLINE from 0.453 to 0.757. Turn-from-standstill went from 0.013 to 0.950. Worth more than all 14 LHS screening runs combined.

**Lesson**: Before tuning numbers, make sure the data is correct. We spent weeks tuning reward weights while a data bug was silently undermining every experiment.

**Files**: `gait_generator.py` line 180 (fixed), `scripts/fix_angular_vel.py` (patches existing recordings), `scripts/fit_poly.py` (refits polynomials).

### Standing reference motion bug (E17)

The Placo walk engine has no standing mode. At dx=0, dy=0, dtheta=0 it generates a full walking-in-place gait with 22-degree knee swings, alternating foot contacts, and lateral sway. The reference motion polynomial_coefficients.pkl encoded this as the zero-velocity target.

Fix: Auto-enable `--stand` flag in gait_generator.py when all velocity components are zero. This freezes joints at initial pose with both feet on ground. Polynomials refitted.

**Impact**: Correct fix (wrong data is wrong data), but didn't solve the stepping problem on its own. E17 (fixed ref, no gate) and E18 (fixed ref + gate) showed the stepping behaviour is driven by the alive reward, not the reference data. The imitation gate was already zeroing out the walking reference at standstill. Main benefit: backward walking improved in E18 (0.637 vs E14's 0.329), likely because the polynomial surface near zero velocity is now smoother.

**Files**: `gait_generator.py` lines 139 and 290 (auto-stand detection, fixed foot contacts).

### Tracking sigma mismatch (E10 - tested, insufficient alone)

The training reward for velocity tracking uses `sigma=0.25`. The evaluation scoring uses `scale=0.05`. This means a duck walking at 50% of commanded speed gets training reward 0.99 (the optimiser thinks it's perfect) but evaluation score 0.018 (it's failing). The optimiser has no gradient to improve precision.

**Impact**: Fixing sigma to 0.05 alone made things *worse* (HEADLINE 0.641 vs E9's 0.757). The tighter sigma made precision harder to earn, but alive=20 still dominates the reward. The optimiser responded by doubling down on survival rather than improving precision. Training reward went UP while HEADLINE went DOWN - the clearest possible evidence of reward misalignment.

**Lesson**: Aligning training and evaluation metrics is necessary but not sufficient. If other reward terms dominate, the optimiser will ignore the aligned term and maximise the dominant one instead. The alive:tracking ratio (~4:1) must be addressed simultaneously.

---

## What experiments taught us

### LHS screening (14 runs, 50M steps each)

We systematically explored 7 parameters using Latin Hypercube Sampling. Key findings:

- **Imitation/alive ratio** is the single most important variable (Spearman r=0.58, p<0.05). Below 0.4, the duck doesn't walk properly. Disney BDX uses ~1.1 (22/20).
- **tracking_ang_vel** is the strongest predictor of turning quality (r=-0.64). Higher isn't always better though - 4.0 made turning worse by competing with imitation.
- Only 2 of 14 runs achieved stand-then-forward scores above 0.4.
- Full data in `docs/18_reward_tuning_report.md`.

### Biased command sampling (E6)

The random command generator rarely produced pure-turn commands. The duck only saw turning combined with forward walking, so it never learned to turn from standstill.

Fix: 12% pure turns, 10% standstill periods during training. Combined movements improved +32%.

### Disney-style reward weights (Disney Runs 1-4)

We replicated the reward structure from the Disney BDX paper. Key findings:

- **action_rate=-1.5 actively helps turning** by forcing committed, deliberate motions. Loosening to -0.5 made turning 78% worse.
- **imitation >= 15.0 is required** when action_rate=-1.5. The smoothness penalty is so strong that only high imitation gradient pushes the policy to actually walk. imitation=8.0 was catastrophically insufficient.
- **Boosting tracking_ang_vel hurts** at high values. 4.0 competed with imitation rather than complementing it. 2.5 works.

### Training instability (E9 300M, E10 150M)

Extended E9 training from 50M to 300M steps. HEADLINE oscillated between 0.365 and 0.757 instead of converging. Root causes identified:

1. **tracking_sigma too loose** (see bug fixes above) - fixed in E10, insufficient alone
2. **No value function clipping** - fixed in E10 (0.2), insufficient alone
3. **Alive dominance** - alive (20.0 per step) earns ~400/episode, tracking (2.5 per step) earns ~100/episode max. The optimiser gets 4x more signal from survival than precision

E10 tested fixes 1 and 2 simultaneously. Result: oscillation persisted and HEADLINE was *worse* (0.641 vs 0.757). Training reward and HEADLINE became negatively correlated - the optimiser improved survival while walking quality degraded. This confirms **alive dominance is the primary remaining issue**.

### Convergence experiments (E12, E13)

After E11 fixed alive dominance (alive=5, HEADLINE=0.876), oscillation persisted at ~0.14 amplitude. We tried two approaches:

**E12: Fewer PPO epochs** (num_updates_per_batch=2 instead of 4). Reduced oscillation 0.14 to 0.10 but lowered peak HEADLINE from 0.876 to 0.837. Fewer gradient steps per rollout means less overshooting, but also slower learning - the policy may not have peaked at 150M steps. Also tested stand_still=-5.0, which had zero effect because imitation=15 completely overwhelms it (duck still rotates at -0.47 rad/s during standstill).

**E13: Adaptive KL** (LRSchedule.ADAPTIVE_KL, desired_kl=0.01). Reduced oscillation to ~0.09 and the best checkpoint appeared at the final step (convergence). But peak HEADLINE capped at 0.785 - the LR throttling is too aggressive at desired_kl=0.01. Back+turn completely broken (0.000 for both directions).

**Conclusion**: Both convergence fixes trade peak quality for stability. The oscillation is a symptom we can live with - use checkpoint selection to pick the best. The better path forward is fixing specific reward problems (standing behaviour, turning precision) rather than training dynamics.

### Standing behaviour problem (E12, E14, E17-E20)

When commanded to stand still, the duck walks in place. This has been the most persistent defect across all experiments. Every model takes 73+ steps over 10 seconds with 44-65% air time and 17-29mm foot lifts.

**Root cause chain**:
1. The reference motion at dx=0 was a walking-in-place gait from Placo (not a standing pose). Fixed in gait_generator.py - now generates static pose at zero velocity.
2. imitation=15 overwhelmed stand_still=-1 and even -5 (E12). Fixed with imitation gate: `clip(cmd_norm/0.05, 0, 1)` zeros imitation at standstill. Reduced yaw drift from 27 deg/s to 3 deg/s but didn't stop stepping.
3. stand_still=-15 with the gate active (E19): no effect on stepping. The penalty targets joint deviation from default pose, but the policy has learned that rhythmic stepping IS the most stable behaviour.
4. Foot contact reward (stand_contact=+5, E20): rewarding both feet on ground during standstill had no measurable effect. The reward only fires during 10% standstill training time and can't overcome the stepping-is-stable prior.
5. Mode-conditional reward routing (E24): testing whether fundamentally different reward weights for standing vs walking mode can break the stepping behaviour. Final single-policy attempt.

**Deep analysis of standing across all experiments**: Every model - E11 through E20 - takes exactly 73 steps (36-37 per foot) at 3.6 Hz during 10 seconds of standstill. The alive=5 reward creates a strong prior that rhythmic stepping = stability. No penalty or reward at the values tested can overcome this. Research (Disney BDX, Berkeley Cassie) addresses this with either separate policies or clock-based reward gating that explicitly turns off periodic behaviour during standing mode.

---

## Current parameter understanding

These are our current best values based on all experiments to date. They are **assumptions, not facts** - any of them could change if we hit a wall or discover a new interaction. They're listed here as the starting point for future experiments, not as fixed constraints.

| Parameter | Current value | Confidence | Evidence | What could change it |
|-----------|--------------|------------|----------|---------------------|
| alive | 5.0 | High | E11: alive=5 produced best-ever HEADLINE (0.876). Fixed strafe-left asymmetry and backward walking. LHS floor is 3.5. | Pure turning regressed - may need compensating with tracking_ang_vel |
| imitation | 15.0 | High | LHS: imit/alive >= 0.4 mandatory. Disney uses ~22. | Could increase toward Disney's 22 if we need stronger waddle character. Could decrease if it conflicts with backward walking |
| action_rate | -1.5 | Medium-high | Disney value. Tested -0.5 (worse) and -1.0. | Might need loosening if tighter tracking_sigma already provides enough precision signal |
| action_accel | -0.45 | Medium | Stable across Disney runs. Not independently tested. | Untested in isolation - could be too aggressive or too lenient |
| orientation | -5.0 | Medium | Standard across all runs. Never varied. | Never tested at other values. Could be too high (restricting natural body sway) |
| tracking_sigma | 0.05 | Medium | E10 tested: necessary but insufficient alone. 0.25 was too loose (no gradient). 0.05 aligns with eval. | Might need 0.08 or 0.10 if combined with alive reduction makes early training too sparse |
| tracking_lin_vel | 2.5 | Medium | Works at this level. Never tested higher with tight sigma. | With tight sigma, might need increasing to provide stronger gradient |
| tracking_ang_vel | 2.5 | Medium | 4.0 was worse (competed with imitation). 2.5 works. | Might need revisiting if tight sigma changes the balance |
| Biased sampling | 12% turn, 10% still | Medium-high | E6 proved essential for turn-from-standstill | Percentages were chosen intuitively, not optimised |
| clipping_epsilon_value | 0.2 | Medium | E10 tested: didn't prevent oscillation alone, but standard PPO practice - keep it | Standard value, unlikely to need changing |
| num_envs | 8192 | High | Fits in 16GB VRAM with 2GB headroom | Hardware constraint, not a tuning choice |

### Parameters we haven't touched

These exist in the config but we've left them at defaults. They could matter:

| Parameter | Current value | What it does |
|-----------|--------------|-------------|
| learning_rate | 3e-4 | Adam optimiser step size. Constant throughout training. |
| entropy_cost | 0.005 | Encourages exploration. Constant (brax doesn't support annealing). |
| num_updates_per_batch | 4 | PPO epochs per rollout. Tested 2 in E12: reduced oscillation but lowered peak. Keep at 4. |
| discounting | 0.97 | How much the duck values future vs immediate reward. |
| push magnitude | 0.1-1.0 N*s | How hard random pushes are during training. |
| push interval | 5-10 sec | How often pushes happen. |
| stand_still | -15.0 | Penalty for moving when commanded to stand still. Tested -1 (default), -5 (E12), -15 (E19). Even -15 with imitation gated off has zero effect on stepping. The alive reward makes stepping the optimal stability strategy. |
| stand_contact | 5.0 | Reward for both feet on ground during standstill. E20: no measurable effect. Only fires during 10% standstill training time. |
| ang_vel_xy | -0.05 | Penalty for roll/pitch angular velocity. |
| lin_vel_z | -2.0 | Penalty for vertical body velocity. |

---

## Where we are now

### Best walking policy: E14 CP9 (turning/forward/strafe optimised)

E14 remains the best single checkpoint for the capabilities the user prioritises (forward, strafe, turning). E11 has better backward walking but weaker turning.

| Capability | E14 CP9 | E11 CP8 | Status |
|-----------|---------|---------|--------|
| Forward walking | 0.996 | 0.882 | Excellent |
| Backward walking | 0.329 | 0.761 | Slow but acceptable |
| Strafe left/right | 0.932/0.992 | 0.983/1.000 | Good |
| Turn left/right | 0.962/0.977 | 0.305/0.584 | Excellent |
| Stand-then-turn L/R | 0.966/0.990 | 0.351/0.583 | Excellent |
| Standing (steps/10s) | 73 | 73 | Walking in place - NOT FIXED |
| Standing yaw drift | 3 deg/s | 27 deg/s | Reduced but still stepping |
| Push recovery (0.6 N*s) | front: 0%, back/side: 100% | not tested | Weak from front |
| Survival | 100% | 100% | Never falls |
| Gait symmetry | 0.876 | 0.845 | Good |

### HEADLINE progression across experiments

```
Anchor (original weights):     0.327
LHS Run 6 (best screening):   0.532
Disney Run 1 (36M peak):      0.604
E6 (biased sampling):         0.453  (improved combined, not pure turn)
E9 (angular velocity fix):    0.757
E10 (tight sigma + clipping): 0.641  (worse - alive dominance confirmed)
E11 (alive 20 -> 5):          0.876  <-- best HEADLINE
E12 (fewer PPO epochs):       0.837  (more stable, lower peak)
E13 (adaptive KL):            0.785  (most stable, lowest peak)
E14 (imitation gating):       0.832  (best turning, standing yaw fixed)
E15 (imit gate + back bias):  0.782  (backward made worse)
E17 (fixed standing ref):     0.812  (correct data, didn't help standing)
E18 (gate + fixed ref):       0.846  (backward recovered, gate+fix complementary)
E19 (stand_still=-15):        0.854  (stepping unchanged)
E20 (foot contact reward):    0.848  (stepping unchanged)
E24 (mode routing):           RUNNING
```

### Open questions

| Question | Status | Answer |
|----------|--------|--------|
| ~~Can imitation gating fix standing?~~ | ANSWERED (E14) | Reduced yaw drift 89% (27 to 3 deg/s). Did NOT stop stepping (73 steps/10s). |
| ~~Can stand_still penalty fix stepping?~~ | ANSWERED (E19) | No. Tested -15 with gate active. Zero effect. Alive=5 makes stepping the stable strategy. |
| ~~Can foot contact reward fix stepping?~~ | ANSWERED (E20) | No. +5 reward for both feet down during standstill had no effect. |
| Can mode-conditional reward routing fix stepping? | TESTING (E24) | Different reward weights for standing vs walking mode. Final single-policy attempt. |
| Can we combine alive=5 with tight sigma? | Open | E10 tested tight sigma with alive=20 (bad). E11 tested alive=5 with loose sigma. Combination untested. |
| Can tracking_ang_vel=3.5 improve turning? | Open | 4.0 was too high (competed with imitation). 3.5 untested. |
| Should we use separate stand/walk policies? | Depends on E24 | Disney BDX uses separate policies. Berkeley uses clock-gated single policy. If E24 fails, this is the next architecture. |
| Is the backward-vs-turning tradeoff fundamental? | Partially answered | Every experiment shows this oscillation. Research (MoE paper) says conflicting gradients in single networks cause this. May need architectural solution. |

---

## The approach

RL reward tuning is inherently iterative. There is no formula for optimal weights - the Disney BDX paper doesn't explain how they arrived at theirs either.

The difference between productive iteration and aimless tweaking:

**Productive**: Diagnose a specific problem, make a targeted change, measure. Our HEADLINE went 0.33 -> 0.53 -> 0.60 -> 0.76 -> 0.88. Each change taught us something.

**Aimless**: Changing multiple things at once (we learned this the hard way with s01 - changed 3 weights, couldn't interpret results), or changing weights without understanding why the current ones don't work.

We've moved through four phases:
1. **Finding the ballpark** (LHS screening) - what ratio of weights even produces walking?
2. **Fixing bugs** (angular velocity, tracking sigma, standing reference) - making sure the system measures what we think
3. **Reward balance** (alive reduction, imitation gating) - getting the reward signals to align with desired behaviour
4. **Standing fix attempts** (where we are now) - trying to make the duck actually stand still

Phase 2 produced the biggest gains (ang_vel fix: +0.30 HEADLINE). Phase 3 was productive (alive reduction: +0.12, imitation gate recovered turning to 0.96). Phase 4 has been unproductive so far: stand_still=-15 (E19), foot contact reward (E20), and their combination all failed to reduce stepping. The duck takes 73 steps per 10 seconds across every experiment. E24 (mode-conditional reward routing) is the final single-policy attempt. If it fails, the evidence points toward separate policies (Disney approach) or clock-based reward gating (Berkeley approach).

**Key meta-learning**: The standing problem is fundamentally different from the walking/turning problems we solved earlier. Walking and turning are about getting the *right* motion. Standing is about getting *no* motion - and the alive reward creates a strong prior that any motion is better than no motion for stability. Penalty and reward signals during 10% standstill training time cannot overcome a prior learned from 100% of training time.
