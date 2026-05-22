# Multi-Policy Implementation Plan

Two separate neural networks (standing + walking) with runtime switching based on velocity commands and foot contact state. Training runs sequentially (one GPU).

## Current state

- E14 walking policy is best (WALKING_SCORE=0.838, HEADLINE=0.838)
- Standing is broken in every single-policy experiment (36 steps/10s at standstill)
- `standing.py` environment exists but has never been trained by us
- `mujoco_infer.py` supports `--standing` flag but only loads one model at a time
- Observation spaces don't match: walking=77 inputs, standing=65 inputs (missing motor_targets and imitation_phase)
- Push recovery is weak: E14 survives 0.3 all directions, 0.6 side only, fails 0.6 front/back
- Dead code: `reward_stand_contact` imported but unused in joystick.py

## Analysis tooling

### Scripts

| Script | Purpose | When to use |
|--------|---------|-------------|
| `analyse_run.py` | **Primary tool.** Training curves + best checkpoint + full eval + verdict | After any training run |
| `evaluate_policy.py` | Full eval of a single ONNX | When you already know which ONNX to test |
| `compare_policies.py` | Side-by-side comparison of result JSONs | Comparing models across runs |
| `quick_walking_eval.py` | TB hook during walking training | Automatic (called by runner) |
| `quick_standing_eval.py` | TB hook during standing training | Automatic (called by runner) |
| `evaluator_base.py` | Shared base class for all evaluation | Library (not run directly) |

### Workflow

```
1. DURING TRAINING - automatic
   Runner calls quick_walking_eval or quick_standing_eval per checkpoint.
   Watch TB for HEADLINE trends.

2. AFTER TRAINING - one command
   PYTHONPATH=. uv run python analysis/analyse_run.py checkpoints/<run_name>
   
   This does everything:
   - Reads TF events (plateau, regression, trends)
   - Sweeps all checkpoints via quick eval to find best
   - Runs full evaluation on best checkpoint (all scenarios + push + responsiveness)
   - Prints human-readable verdict (GOOD/OK/WEAK/FAIL per metric)
   - Saves JSON to analysis/results/<run_name>_analysis.json

3. COMPARE RUNS
   PYTHONPATH=. uv run python analysis/compare_policies.py results/A.json results/B.json

4. VERIFY IN SIM
   uv run python playground/open_duck_mini_v2/mujoco_infer.py -o <best_onnx>
   Metrics are indicators, not truth. Always check in sim before declaring success.
```

### Headlines

Each training type has its own HEADLINE formula (logged to TB):

- **Walking HEADLINE:** `0.5 * WALKING_SCORE + 0.5 * responsiveness_avg`
  - WALKING_SCORE = mean of cold-start velocity tracking scores (forward, backward, strafe, turn)
  - responsiveness_avg = mean of stand-then-command velocity tracking scores
  
- **Standing HEADLINE:** `0.5 * stillness_score + 0.5 * push_score`
  - stillness_score = mean of Gaussian penalties on step_count, drift, sway
  - push_score = weighted push survival (0.4 mag, 0.6 mag, 0.8 mag)

### Key metrics and thresholds

| Metric | GOOD | OK | FAIL |
|--------|------|-----|------|
| vx/vy tracking score | >= 0.8 | >= 0.5 | < 0.5 |
| Standing step count | < 5/10s | < 20/10s | >= 20/10s |
| Standing drift | < 0.05m | < 0.2m | >= 0.2m |
| CoT | < 30 J/m | < 80 J/m | >= 80 J/m |
| Stride CV | < 0.15 | < 0.3 | >= 0.3 |
| Foot slip | < 0.02 m/s | < 0.05 m/s | >= 0.05 m/s |
| Bang-bang ratio | < 5% | < 15% | >= 15% |
| Responsiveness mean | >= 0.7 | >= 0.4 | < 0.4 |

## Observation space alignment

The two environments currently build different observation vectors. They must match for shared inference code.

**Walking obs (77):** gyro(3) + accel(3) + cmd(7) + joint_pos(10) + joint_vel(10) + last_act(10) + last_last_act(10) + last_last_last_act(10) + motor_targets(10) + contact(2) + imitation_phase(2)

**Standing obs (65):** gyro(3) + accel(3) + cmd(7) + joint_pos(10) + joint_vel(10) + last_act(10) + last_last_act(10) + last_last_last_act(10) + contact(2) + reference_motion(0)

**Inference obs (77):** matches walking env exactly (mujoco_infer.py get_obs)

**Differences:**
1. Walking has `motor_targets` (10) - standing doesn't
2. Walking has `imitation_phase` (2) - standing has `current_reference_motion` (0 when imitation disabled)

**Decision:** Add motor_targets and imitation_phase to standing env obs. Standing doesn't use imitation but the phase values will be zero - the network learns to ignore them. This keeps a single `get_obs` in inference.

## Phases

### Phase 0: Evaluation overhaul - COMPLETE

All sub-phases done. Tools verified against E14, E19, E5.

**What was done:**
- Extracted `PolicyEvaluatorBase` (shared obs, stepping, fall detection)
- Fixed argmax/argmin bug in cost metrics
- Added new metrics: CoT, stride CV, foot slip, bang-bang detection
- Added standing metrics to quick_walking_eval (step_count, drift, sway)
- Created quick_standing_eval.py for standing training TB hook
- Added human-readable verdict summary to evaluate_policy.py
- Rewrote compare_policies.py (was completely broken)
- Consolidated analyse_training + find_best_checkpoint + debug_policy into analyse_run.py
- Fixed runner.py to call the right eval hook based on env type
- Fixed quick eval step counts (200->500 cold start, 100->250 standing)
- Calibrated on E14, E19, E5 - all metrics align with sim observations

**Calibration results:**
- E14: Walking GOOD (forward, strafe, turn), backward WEAK, standing FAIL (36 steps/10s), push 0.3 PASS, 0.6 PARTIAL
- E19: Similar walking, standing FAIL (42 steps/10s), push 0.6 PASS (slightly better than E14)
- E5: Walking OK/WEAK, standing GOOD (1 step/10s, 0.003m drift), responsiveness FAIL (0.335)

### Phase 1: Cleanup and obs alignment

Sequential, no training needed.

**1a. Clean up joystick.py (walking env)**
- Remove unused `reward_stand_contact` import (line 45)
- Remove `stand_still` reward entirely - the standing policy handles standstill now
- Remove the imitation gate (`* jp.clip(...)` on line 670) - walking policy should always use imitation since it only receives non-zero velocity commands
- Remove the 10% zero-command sampling from `sample_command` (line 734: `jp.where(selector < 0.22, jp.zeros(7), normal_cmd)`) - walking policy should never practise standing. Keep the 12% pure-turn sampling
- Increase push magnitude: `magnitude_range=[0.5, 2.0]`, `interval_range=[3.0, 8.0]`

**1b. Update standing.py**
- Add `motor_targets` to obs (10 values) - matches walking env
- Add `imitation_phase` to obs (2 values, always zero) - matches walking env
- Increase push magnitude: `magnitude_range=[0.5, 2.5]`, `interval_range=[3.0, 8.0]`
- Add `motor_targets` tracking to info dict (same pattern as joystick.py)

**1c. Update mujoco_infer.py get_obs**
- Verify the obs construction matches the aligned training obs exactly
- The current inference `get_obs` already includes `motor_targets` and `imitation_phase` so it should match the walking env. For standing inference it will now also match

**1d. Clean up rewards.py**
- Remove the joystick.py import of `reward_stand_contact` (keep the function itself for standing env)

**Test:** Verify both envs produce 77-element obs vectors. Run a short smoke-test training (1M steps) on each env to confirm no crashes.

### Phase 2: Train standing policy

```bash
python -m playground.open_duck_mini_v2.runner \
  --env standing \
  --output_dir checkpoints_standing \
  --num_timesteps 150000000
```

**Monitor during training:** Watch `standing/HEADLINE`, `standing/stillness_score`, `standing/push_score` in TB.

**Analyse after training:**
```bash
PYTHONPATH=. uv run python analysis/analyse_run.py checkpoints_standing
```

**Success criteria:**
- step_count < 10 over 10 seconds at standstill
- air_time_ratio < 0.10
- Survives 1.5 N*s push from all directions
- Survives 2.0 N*s push from back/sides
- Head tracking responds to commands (qualitative in sim)
- Stride CV during any residual stepping < 0.15

**Decision point after training:**

```
IF step_count < 10 AND push_1.5 survival > 80%:
    Standing policy is good. Proceed to Phase 3.

ELIF step_count < 10 BUT push survival is weak (< 50% at 1.0):
    Standing is solved but push recovery isn't.
    Option A: Retrain with higher push magnitude (2.0-3.0)
    Option B: Accept weaker push recovery for now, proceed to Phase 3
    Decision: depends on whether push recovery is a user priority

ELIF step_count > 30 (still stepping significantly):
    The standing env's alive=20 may be creating the same stepping incentive.
    Try: reduce alive to 5.0, increase stand_still to -2.0, add reward_stand_contact
    If that fails: add explicit foot-velocity penalty (penalise foot movement during contact)
    If THAT fails: investigate whether the obs space gives enough information
    to distinguish "should stand" from "should step"

ELIF step_count 10-30 (marginal):
    Try: increase stand_still penalty from -0.3 to -1.0
    Retrain and re-evaluate
```

### Phase 3: Retrain walking policy

Trains from scratch with the cleaned-up joystick.py from Phase 1.

```bash
python -m playground.open_duck_mini_v2.runner \
  --env joystick \
  --output_dir checkpoints_walking_v2 \
  --num_timesteps 150000000
```

**Monitor during training:** Watch `walking/HEADLINE`, `walking/WALKING_SCORE`, `walking/responsiveness_avg` in TB.

**Analyse after training:**
```bash
PYTHONPATH=. uv run python analysis/analyse_run.py checkpoints_walking_v2
# Compare against E14:
PYTHONPATH=. uv run python analysis/compare_policies.py \
    analysis/results/checkpoints_walking_v2_analysis.json \
    analysis/results/E14_consolidated_test.json
```

**Success criteria:**
- WALKING_SCORE >= E14 baseline (0.838)
- Forward velocity >= E14 (~0.10 m/s achieved)
- Strafe >= E14
- Turning >= E14
- Push survival: 1.0 N*s all directions, 1.5 N*s back/sides
- CoT <= E14 (no worse energy efficiency)
- Foot slip rate < 0.01 m/s mean
- Bang-bang fraction < 5%
- Backward: any motion is acceptable (deprioritised)

**Decision point after training:**

```
IF WALKING_SCORE >= 0.85 AND push_1.0 survival >= 80%:
    Walking policy is good. Proceed to Phase 4.

ELIF WALKING_SCORE >= 0.85 BUT push survival weak:
    Push training magnitude may be too high (policy learned caution over speed).
    Option A: Reduce push magnitude to [0.3, 1.5] and retrain
    Option B: Use curriculum - start with [0.1, 1.0] for 75M steps,
              then increase to [0.5, 2.0] for remaining 75M
    Decision: check if gait looks cautious/wide-stance in sim

ELIF WALKING_SCORE < 0.80 (significant regression):
    Removing stand_still or imitation gate may have hurt.
    Diagnose by checking which scenarios regressed (compare_policies.py).
    Option A: Add stand_still back at low weight (-0.3)
    Option B: Add imitation gate back
    Option C: Check if pure-turn sampling (12%) is enough -
              maybe the walking policy needs some standing exposure
    Implement whichever single change addresses the regression, retrain

ELIF WALKING_SCORE 0.80-0.85 (marginal regression):
    Acceptable if push survival improved significantly.
    Run compare_policies.py against E14 to identify which scenarios dropped.
    If only backward regressed further: acceptable, proceed to Phase 4
    If forward/strafe/turning regressed: investigate and fix
```

### Phase 4: Build dual-model inference

Modify `mujoco_infer.py` to load and switch between two ONNX models.

**State machine:**
```
         cmd_vel > threshold
STANDING ---------------------- WALKING
    ^                              |
    |    cmd_vel ~ 0               |
    |    AND both_feet_down        |
    +------------------------------+
```

**Switching rules:**
- Standing to walking: `norm(cmd[:3]) > 0.01` - immediate, no foot contact check needed (walking policy handles any initial pose since it trained with random init)
- Walking to standing: `norm(cmd[:3]) < 0.01` AND `left_contact AND right_contact` - wait for double-support phase before switching

**Double-support detection:** `get_feet_contacts(data)` already exists in `mujoco_infer_base.py` (line 280-283)

**Action blending at transition:** Linear blend over N steps (configurable, start with 5 = 100ms at 50Hz):
```python
for i in range(blend_steps):
    alpha = (i + 1) / blend_steps
    blended_action = (1 - alpha) * old_policy_action + alpha * new_policy_action
```
Both policies queried during blend window

**Action history handoff:** `last_act`, `last_last_act`, `last_last_last_act` carry over between policies unchanged

**Implementation changes to mujoco_infer.py:**
- Constructor takes two ONNX paths: `--walking_onnx` and `--standing_onnx`
- New state: `self.active_policy` (enum: STANDING, WALKING)
- New state: `self.blend_counter`, `self.blend_steps`
- `run()` loop checks commands and contacts each control step, manages transitions
- Old `--standing` flag removed (replaced by dual-model)

**Test:** Run with E14 as walking model and whatever standing model we have. Even a bad standing model tests the switching logic.

### Phase 5: Integration testing in simulator

All tested via the modified `mujoco_infer.py` with keyboard control.

**Test 1 - Standing baseline:** Launch with no key presses. Robot should stand still (< 10 steps/10s). Push it. Should recover

**Test 2 - Walking baseline:** Press forward arrow. Robot walks. Press backward. Robot walks backward. Press Q/E for strafe. Press left/right for turning

**Test 3 - Stand-to-walk transition:** From standing still, press forward. Robot should begin walking within ~200ms. No stumble or fall

**Test 4 - Walk-to-stand transition:** While walking forward, release all keys. Robot should transition to standing when both feet are planted. Should stop stepping within ~500ms of double-support

**Test 5 - Rapid switching:** Tap forward briefly (< 1s), release. Robot should walk briefly, then stop cleanly. Repeat several times

**Test 6 - Push during transition:** Start a walk-to-stand transition, push the robot during the blend window. Should recover

**Test 7 - Push while standing:** Apply pushes from all directions. Target: survive 1.5 N*s from front, 2.0 from back/sides

**Test 8 - Push while walking:** Apply pushes during walking. Target: survive 1.0 N*s from all directions

**Decision points:**

```
IF Tests 1-5 pass, Tests 7-8 meet targets:
    System works. Proceed to programmatic eval and Phase 5b.

IF Test 4 fails (robot stumbles on walk-to-stand):
    Option A: Increase blend_steps from 5 to 10 (200ms blend)
    Option B: Instead of blending, hold last walking action for N steps
              before switching (let the stride complete naturally)
    Option C: Add a "deceleration phase" - before switching to standing,
              send progressively smaller velocity commands to walking policy
    Try A first (cheapest). If still janky, try B. C is last resort.

IF Test 3 fails (robot stumbles on stand-to-walk):
    Option A: Reduce blend_steps (faster transition)
    Option B: Skip blending for stand-to-walk (immediate switch)
    Option C: The standing policy's final pose may be incompatible
              with the walking policy's expected starting state.
              Check joint positions at switch time.

IF Test 6 fails (push during transition):
    This is the hardest case. Options:
    Option A: Accept it - real-world pushes during exact transition moment are rare
    Option B: During blend, if push detected (large velocity spike),
              abort transition and return to previous policy
    Decision: depends on how often this happens in practice

IF push targets not met (Tests 7-8):
    Option A: Retrain with stronger push curriculum
    Option B: Accept current push tolerance and note for future improvement
```

**Programmatic tests:** Extend `evaluate_policy.py` to support dual-model evaluation:
- Load both ONNX files, run with the same state machine as runtime
- Measure transition quality: steps-to-stop after walk-to-stand, steps-to-motion after stand-to-walk
- Standing metrics during standstill phases
- Walking metrics during walking phases
- Push recovery in both modes

### Phase 5b: Re-evaluate past experiments with improved tools

Run analyse_run.py on key past experiments to build a fuller picture:
- E14, E19, E20, E5, E9, anchor/disney runs
- ~89 cached eval JSONs exist but were generated with the old tool version

**Decision point:**
```
IF re-evaluation reveals a past experiment that was actually better than
E14 on the new metrics (e.g. lower CoT, better gait periodicity):
    Consider using that model as the walking policy base instead of E14.
    Compare in sim to verify.

IF re-evaluation confirms E14 is best across all metrics:
    Our previous assessment was correct. Record the new baselines.
```

## Execution order

```
Phase 0 (eval overhaul) - COMPLETE
    |
Phase 1 (cleanup + obs alignment) -> test: both envs produce 77-element obs, smoke-test training
    |
    +-- Phase 2 (train standing) -> GATE: step_count < 10 and push survival
    |       |
    |       +-- Phase 3 (train walking) -> GATE: WALKING_SCORE >= 0.85 and push survival
    |               |
    +-------+-------+-- Phase 4 (dual inference) -> test: switching logic with available models
                        |
                    Phase 5 (integration test) -> GATE: Tests 1-8
                        |
                    Phase 5b (re-evaluate past experiments)
```

Phase 4 can be built while Phase 2/3 trains.

## Risks and mitigations

**Risk: Standing policy learns to step anyway.** The standing env has alive=20 (same "stepping = surviving" incentive). Mitigation: the standing env never sends velocity commands, so there's no walking reward pulling it. If it still steps: reduce alive to 5, increase stand_still to -2.0, add reward_stand_contact. If that fails: add explicit foot-velocity penalty.

**Risk: Transition jank.** The blend window might produce unnatural motion. Mitigation: tune blend_steps (try 3, 5, 10). If blending isn't enough, try holding the last walking action for a few steps before switching. Last resort: add a deceleration phase.

**Risk: Walking policy regresses without standing practice.** Unlikely - we're freeing capacity. But if WALKING_SCORE drops below 0.80, add stand_still back at -0.3 weight. If 0.80-0.85, check which scenarios regressed.

**Risk: Push training too aggressive.** 2.0 N*s might cause cautious gait. Mitigation: check for wide stance and reduced speed. If quality drops, reduce to [0.3, 1.5] or use curriculum (start gentle, increase).

**Risk: Metrics look good but sim looks bad.** Has happened before. After each training phase, verify at least one model in sim. Metrics are indicators, not truth.

**Risk: Old cached eval JSONs are misleading.** The 89 cached results were generated with the old tool version. Don't mix old and new JSON formats. Re-evaluate with new tools in Phase 5b.
