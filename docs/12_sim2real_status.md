# Sim-to-Real Status: Open Duck Mini V2

**Date of assessment:** May 2026 (updated mid-May with CAD fix and model improvements)
**Scope:** Open Duck Mini V2 with Feetech STS3215 servos at 7.4V

> **Update (2026-05-13):** The simulation model has been significantly improved since the initial assessment. A 37mm bilateral asymmetry in the CAD was discovered and fixed, reference motions regenerated from a symmetric model, and tracking_sigma corrected from 0.01 to 0.1. These changes address several of the sim-to-real gap items noted below. The accelerometer bias bug remains unfixed. See `14_cad_symmetry_fix.md` and `catalogue.md` for details.

---

## What "sim-to-real" means here (for non-ML readers)

Training a robot to walk using reinforcement learning happens entirely inside a physics simulator - a computer programme that calculates how a virtual robot would fall, balance, and move. The neural network (policy) learns by trial and error inside this simulation, receiving millions of attempts at walking before it sees a real robot.

"Sim-to-real transfer" is the step where that trained policy is deployed on the physical robot. The core question is: does the robot that behaves well in simulation also behave well in reality? In most projects, the answer is "partly, with caveats", and Open Duck Mini is no exception.

---

## 1. Has this been validated? By how many people?

**Status: validated at proof-of-concept level by approximately 1-2 people.**

The primary evidence of successful transfer is:
- Two pre-trained ONNX policies (`BEST_WALK_ONNX.onnx` and `BEST_WALK_ONNX_2.onnx`) are bundled in the Open Duck Mini repository. These were trained by the project creator (apirrone) and reportedly produce walking behaviour on the real robot.
- A git commit message reads `"CHECKPOINT pretty nice walk on the real robot"`, confirming at least one successful deployment by the author.
- A second checkpoint note reads `"CHECKPOINT pretty nice. turns in place, catches itself, somewhat robust to independent head moves"`.
- The sim2real documentation says the training was validated and provides a working pipeline.

No community members have publicly reported replicating a successful real-robot deployment. GitHub issues contain multiple people asking how to train and deploy, but no reports of having done so successfully themselves. The project is relatively young (documentation marked "not finalized yet") and the Discord server is where success stories - if any - would likely appear, but that is not accessible here.

The project is functionally at the "it works for the author, instructions are published, community is still working through it" stage. This is normal for an early open-source robotics project.

---

## 2. Known failure modes

### 2.1 Policy fails to lift feet ("it can't lift its feet")

The most documented real-robot failure (GitHub issue #5, Open Duck Playground) is that a self-trained policy fails to produce leg lifting on the real robot - the duck shuffles or drags its feet instead of stepping. The project author's response directed the user to Discord rather than providing a code-level fix, suggesting this is a known tuning problem without a simple documented solution.

**Likely causes:**
- Servo torque budget is tight. The STS3215 at 7.4V produces approximately 3 N·m peak torque per joint. The robot weighs approximately 1.6-1.8 kg with a 353 g head. Lifting a leg requires the stance leg joints to hold full body weight while the swing leg joints carry the leg mass. If training allows higher torques than the real servo can sustain, the policy over-commands.
- The motor velocity cap in simulation is 5.24 rad/s, but real servos under load may be significantly slower. A policy that learned to rely on fast servo response will be sluggish on the real robot.

### 2.2 Wrong servo voltage variant

Issue #21 (Open Duck Mini) documents a user running `BEST_WALK_ONNX_2.onnx` with 12V STS3215 servos and experiencing continuous foot trembling. The author confirms: "the 12V servos and the 7.4V ones are different, meaning they require different parameters to be simulated... it's not surprising to me that using `BEST_WALK_ONNX2.onnx` doesn't work with your 12V servos since it was trained for 7.4V servos."

This is a hard failure - the pre-trained policies are strictly coupled to the servo voltage variant used during the BAM identification run. Using the wrong variant produces chaotic, unusable motion.

### 2.3 Instability when receiving a command to move

Issue #43 (Open Duck Mini) documents the policy falling immediately when given a forward movement command during MuJoCo simulation. This is an observation-initialisation problem (the previous-action buffers must be correctly initialised) but it illustrates how sensitive the policy is to the exact observation format. The same problem could manifest on the real robot if any sensor reading is initialised incorrectly or arrives at the wrong time.

### 2.4 Incorrect servo direction (mirroring)

Issue #30 (Open Duck Mini) describes incorrect gait due to leg servo direction not being mirrored correctly between left and right legs. This is a configuration problem during motor setup, but it shows that the pipeline has a fragile per-robot calibration requirement that is easy to get wrong.

### 2.5 Surface limitations

The training terrain options are flat terrain and a mild heightfield (rough terrain). There is no slope training, no stair training, and no soft surface training. The randomised floor friction range is 0.5-1.0 (relatively slippery to normal). Low-friction surfaces (polished floors, smooth tiles) or surfaces outside this range are likely to cause falls.

The feet use a TPU sole specifically to reduce the sim-to-real gap for contact compliance, which is a pragmatic mitigation but does not eliminate surface sensitivity.

### 2.6 Being picked up or disturbed mid-walk

Issue #30 describes behaviour when trying to control the robot by lifting/retreating with a handle - it does not walk properly. Push disturbances are trained (randomised impulses at 5-10 second intervals), but disturbances outside the trained range (being grabbed, placed on an incline, slipping suddenly) are likely to cause falls. The fall termination condition in training is simply "torso tilts past horizontal", so the policy never learns to recover from a near-fall.

---

## 3. How many training runs work on the real robot vs fail?

This is not directly measurable from the repository (no user-reported success/failure counts exist). However, several indirect signals are informative:

- The git log contains at least 30+ experimental commits before the "pretty nice walk" checkpoint, indicating significant iteration was required before a policy transferred well. Many commits show abandoned approaches: "trying stuff", "back", "no randomization ?", "no noise ?", "massive randomization to test", etc.
- The author committed multiple "CHECKPOINT" commits at different stages, suggesting earlier checkpoints did not produce satisfactory real-robot behaviour.
- Training takes approximately 150 million simulation steps (configurable in `runner.py`). On a single GPU (tested on RTX 2080 Ti and RTX 4060 Laptop), each training run takes several hours.
- There are no reported cases of a self-trained policy working first time. Hyperparameter tuning - particularly reward weights, noise levels, and randomisation ranges - appears necessary.

**The honest assessment:** training runs that produce visually good simulation behaviour do not reliably transfer without iteration. The current code represents a configuration that has worked at least once. The number of attempts that did not transfer is unknown but appears to be substantial based on the git history.

---

## 4. The biggest remaining gaps between simulation and reality

### 4.1 Accelerometer bias bug (known open issue)

This is the most clearly documented gap. The physical robot's BNO055 IMU consistently reads approximately +1.3 m/s² on the X-axis even when stationary, due to how the chip is mounted. The training code attempted to replicate this bias but contains a silent bug (PR #24, open as of May 2026):

```python
# In joystick.py (training) - this line does NOTHING:
accelerometer.at[0].set(accelerometer[0] + 1.3)   # JAX arrays are immutable

# In mujoco_infer.py (simulation inference) - this line WORKS:
accelerometer[0] += 1.3   # NumPy arrays are mutable
```

The policy was trained seeing X ≈ 0.0 when stationary. It runs on the real robot seeing X ≈ 1.3 when stationary. These are different observation distributions. The policy has never learned to cope with the value it will actually receive.

This is subtle rather than catastrophic - the robot still walks - but it degrades performance. Any policy improvement from careful reward tuning is partially thrown away at deployment time because one sensor input is consistently wrong. PR #24 proposes removing the bias from simulation inference to at least make sim and real consistent.

### 4.2 Servo friction model accuracy under load

The BAM identification was run on a single servo (ID 24, the left ankle) under step-command tests. The resulting model is applied identically to all 10 leg servos. However:

- Friction parameters were identified without load (not under the weight of the rest of the leg). The actual friction experienced by hip and knee joints under body weight is likely different.
- The stiction parameter (`load_friction_external_stribeck = 0.734`) is very high, indicating the real servos exhibit stick-before-slip behaviour. The simulation's domain randomisation applies only ±10% variation to friction parameters, which may not span the full unit-to-unit variation.
- No re-identification has been done after assembly (individual servo wear, gear lubrication state, and mechanical alignment all affect friction in practice).

### 4.3 No terrain or slope generalisation

Training uses flat terrain and a mild procedurally-generated heightfield. Walking on carpet, door sills, gentle ramps, or uneven outdoor surfaces is entirely out-of-distribution. The project documentation does not claim terrain generalisation as an achieved goal.

### 4.4 Head mass shifting the centre of mass

The head is 353 g, which is approximately 20% of the estimated total mass. The head has 4 DOF and can move substantially. Training samples random head positions during the episode, but the coupling between head pose and body stability is complex. On the real robot, rapid head movements during walking will shift the centre of mass in ways the policy may not fully compensate.

### 4.5 Motor velocity limits are optimistic

The simulation applies a hard motor velocity cap (5.24 rad/s, matching the servo spec sheet). However, the real servo's achievable speed depends on load, temperature, and bus voltage. Under load with a warm battery, real maximum velocity will be lower. The velocity clipping in simulation prevents the policy from demanding faster movements, but the trained gait may still rely on servo responsiveness that degrades as the battery discharges.

### 4.6 Control timing jitter

The simulation runs at exactly 50 Hz (20ms steps). The real robot runs on a Raspberry Pi Zero 2W, a single-core 1 GHz processor. The runtime code explicitly warns when the 20ms budget is exceeded (`"Policy control budget exceeded by X"`). Timing jitter in real execution means observations are sometimes slightly stale (from a previous timestep) while the servo command is sent. The simulation includes an action delay (0-3 steps randomly) and IMU delay (0-3 steps randomly) to approximate this, but the actual jitter pattern on hardware may differ.

---

## 5. Is the BAM servo model accurate enough?

**Answer: it is significantly better than a naive model, but has known limitations.**

BAM captures effects that simple PD simulation completely misses: load-dependent friction, Coulomb friction, stiction (the "stuck until enough force is applied" behaviour), and armature inertia. For the Feetech STS3215, these are critical because it is a cheap plastic-gear servo with significant nonlinearity.

The identification was run at 7.4V nominal. Providing the physical robot also runs at 7.4V with fresh batteries, the BAM model is a reasonably good approximation.

**However:**
- The model was identified from a single servo in isolation. Unit-to-unit variation in cheap servos is significant.
- Identification was done on the ankle joint only. Hip joints carry different loads and may have different effective friction.
- The model does not capture thermal effects - a servo that has been running for 10 minutes has higher friction and lower torque than a cold servo. No thermal model exists in the simulation.
- The 12V variant of the STS3215 has completely different BAM parameters. The pre-trained policy is not portable to 12V hardware.
- As confirmed in issue #20 on Open Duck Mini, the firmware Kp value of 32 (in servo units) corresponds to 17.8 in MuJoCo units. This mapping is correctly implemented in the current model.

The sim2real document itself says: "It's a hard problem, especially for us since we are using cheap servomotors that are hard to model and not overly powerful."

---

## 6. "Dark magic" tuning steps after training

Several runtime adjustments exist that are not obvious from the main documentation:

### 6.1 Per-robot joint offsets (`duck_config.json`)

The runtime accepts a `duck_config.json` file with per-joint angle offsets (`joints_offsets`). The example config has all offsets at 0.0, but this mechanism exists because real builds never achieve exactly the same home position as the simulation. Assembling the servo horns at a slightly wrong angle shifts all subsequent positions by a constant offset. Tuning these offsets per-robot is a required (but undocumented) step.

### 6.2 Kp at runtime differs from simulation

The simulation uses `kp = 17.8` (MuJoCo units, from BAM identification). The runtime default is `kp = 30` in firmware units, not 32 (which is the BAM-identified value). The command-line argument is `-p 30` by default but the code sets 32 as the "default kp" in `rustypot_position_hwi.py`. There is an inconsistency here that may mean deployed policies run with slightly different servo gains than they were trained with. Head joints are explicitly set to a lower Kp of 8 (firmware units) to prevent aggressive head movement.

### 6.3 Optional low-pass filter on actions

The runtime provides an optional `--cutoff_frequency` argument for a low-pass filter on the motor command outputs. Issue #21 shows a user being told to run with `--cutoff_frequency 40`. This filter smooths jitter in the motor commands at the cost of some responsiveness. The training code has a commented-out `LowPassActionFilter` that was disabled - meaning the policy was trained without filtering but may be deployed with it. Applying a filter at deployment that was not present during training is another sim-to-real discrepancy.

### 6.4 Phase frequency offset (`phase_frequency_factor_offset`)

The duck config includes a `phase_frequency_factor_offset` value. This shifts the speed of the imitation reference phase clock at runtime - essentially telling the robot to run its gait cycle faster or slower than it was trained. The D-pad buttons on the Xbox controller let the user tune this in real time. The fact that this manual adjustment exists implies that the "correct" phase rate varies between physical builds.

### 6.5 Pitch bias

The runtime accepts a `--pitch_bias` argument (in degrees). This adds a constant forward/backward lean correction. If the IMU is mounted at a non-zero angle or the robot's centre of mass is not where the simulation expects it, the robot will persistently lean. This parameter corrects it without retraining.

---

## 7. Summary assessment

| Aspect | Status |
|---|---|
| Pipeline validated on real hardware | Yes, by project author. Community replication unconfirmed. |
| Pre-trained policy available | Yes (BEST_WALK_ONNX.onnx, BEST_WALK_ONNX_2.onnx) |
| Self-trained policies transfer reliably | Unknown - significant iteration required before author's policy worked |
| Known blocking bugs | Accelerometer bias bug (PR #24, open) degrades performance |
| Servo model (BAM) adequate for 7.4V hardware | Yes, with caveats (single-servo identification, no thermal model) |
| 12V servo variant supported | No - requires separate identification run |
| Terrain generalisation | Flat and mildly rough only |
| Per-robot calibration required | Yes - joint offsets, pitch bias, phase frequency tuning |
| Runtime filter applied at deploy but not training | Optional, but used in practice (sim-to-real gap) |

**The pipeline is genuine and has produced walking behaviour on at least one physical robot.** It is not a "click and run" system - it requires careful hardware setup, the correct servo variant, per-robot calibration of offsets and bias values, and likely some manual phase-rate tuning. Users who deviate from the exact 7.4V STS3215 hardware will need to redo the BAM identification from scratch. The accelerometer bias bug means all currently-trained policies are running with a sensor mismatch that has not yet been corrected.

For our purposes, the key questions when building our own robot are:

1. **Use 7.4V STS3215 servos** - the pre-trained policy and existing BAM parameters only apply to this variant.
2. **Expect to tune `duck_config.json`** before getting useful walking behaviour.
3. **Be aware of the accelerometer bias bug** - it is a known issue but not yet fixed in the main branch.
4. **Budget time for phase frequency adjustment** using the D-pad during first walks.
5. **Training from scratch requires iteration** - the provided checkpoint is a starting point, not a guaranteed output of a single training run.

---

*Sources: `Open_Duck_Mini/docs/sim2real.md`; `Open_Duck_Playground/playground/open_duck_mini_v2/joystick.py`; `Open_Duck_Playground/playground/common/randomize.py`; `Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py`; `Open_Duck_Mini_Runtime/mini_bdx_runtime/mini_bdx_runtime/rustypot_position_hwi.py`; `Open_Duck_Mini_Runtime/mini_bdx_runtime/mini_bdx_runtime/rl_utils.py`; `Open_Duck_Playground/playground/open_duck_mini_v2/xmls/joints_properties.xml`; `Open_Duck_Mini/experiments/v2/params_m6.json`; GitHub issues: Open_Duck_Playground #5, Open_Duck_Mini #20, #21, #30, #43; GitHub PR: Open_Duck_Playground #24.*
