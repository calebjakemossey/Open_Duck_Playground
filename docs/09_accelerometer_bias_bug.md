# The Accelerometer Bias Bug

**Status:** Open bug. PR #24 on `apirrone/Open_Duck_Playground` proposes a fix (not yet merged as of May 2026). We have not applied either fix locally yet - this is planned for Phase 2 stability experiments (see `catalogue.md`, experiment st01). The current approach is to fix the model and reference motions first, then address this bug as a separate one-factor-at-a-time change.

---

## What is an accelerometer?

An accelerometer is a small sensor that measures forces acting on an object. Think of it like a tiny ball sitting inside a box on a spring. When the box moves - forwards, backwards, tilts, falls - the ball gets pushed around and the spring stretches or compresses. The accelerometer measures how hard the spring is being stretched in each of three directions (X, Y, and Z axes).

For a robot, the accelerometer is built into the IMU (Inertial Measurement Unit), a small chip on the robot's body. It tells the robot things like:

- Am I leaning forward?
- Am I accelerating?
- Which way is gravity pulling on me right now?

The Open Duck Mini uses a BNO055 IMU chip. The accelerometer readings are measured in metres per second squared (m/s²). When the robot is sitting perfectly still on a flat surface, the Z-axis reading will be approximately 9.8 m/s² (Earth's gravitational acceleration pulling downwards). The X-axis reading should be approximately 0 m/s².

The robot's neural network policy receives the accelerometer's three values as part of its "observations" - the complete set of sensor readings it uses every 20ms to decide what the motors should do next.

---

## What is a "bias" in this context?

A bias is a constant, systematic error - a reading that is consistently wrong by the same amount every time.

Imagine you have a kitchen scale with a bowl sitting on it. Before you weigh anything, the scale already reads 200g because of the bowl's weight. Every measurement you take will be 200g too high. That is bias. The scale is not broken - it just has a constant offset.

For an accelerometer, an X-axis bias of +1.3 means the sensor always reports a value that is 1.3 m/s² higher than the true value. The robot might be sitting perfectly still (true X-acceleration = 0), but the sensor reports 1.3. It always reads 1.3 higher than reality.

---

## The bug: what the code intended vs what it actually did

Someone noticed that the physical robot's IMU consistently reported approximately +1.3 m/s² on the X-axis even when the robot was stationary. This is almost certainly a real hardware characteristic of how the BNO055 is mounted and calibrated on this particular robot - a genuine sensor bias.

To compensate for this, the developer added a line to the training environment to simulate this same bias in the physics simulator. The idea was: "the real robot always sees +1.3 on X, so let's add +1.3 during training too, so the policy learns to deal with that offset."

That sounds correct in principle. The problem is that the code in the training environment uses JAX (a numerical computing library), and there is a fundamental difference between how JAX handles arrays versus how NumPy (another library) handles them.

### The broken training code

File: `Open_Duck_Playground/playground/open_duck_mini_v2/joystick.py`, line 502

```python
accelerometer = self.get_accelerometer(data)
# accelerometer[0] += 1.3 # TODO testing
accelerometer.at[0].set(accelerometer[0] + 1.3)   # <-- BUG
```

In JAX, arrays are **immutable** - you cannot change them in place. The `.at[0].set(...)` operation does not modify `accelerometer`. Instead, it creates a brand new array with the modified value and returns it. Because the result is not assigned back to anything, it is immediately discarded. The variable `accelerometer` still holds the original, unmodified values.

The line above does absolutely nothing. The `accelerometer` variable passed into the observation is unchanged.

Notice also the commented-out line above it: `# accelerometer[0] += 1.3 # TODO testing` - that comment reveals this was known to be experimental. The JAX version was written as an attempt to translate that NumPy-style mutation into JAX idiom, but done incorrectly.

The correct JAX code would be:

```python
accelerometer = accelerometer.at[0].set(accelerometer[0] + 1.3)  # assign result back
```

### The working (but wrong) inference code

File: `Open_Duck_Playground/playground/open_duck_mini_v2/mujoco_infer.py`, line 73-74

```python
accelerometer = self.get_accelerometer(data)
accelerometer[0] += 1.3   # <-- this DOES work
```

This file uses NumPy (not JAX). NumPy arrays **are** mutable - in-place modification works exactly as intended. The `+=` operator modifies the array directly. The +1.3 offset is successfully applied.

### The runtime (physical robot) code

File: `Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py`, line 156-159

```python
obs = np.concatenate(
    [
        imu_data["gyro"],
        imu_data["accelero"],   # raw hardware reading, no +1.3 added
        ...
    ]
)
```

The physical robot reads accelerometer data directly from the hardware (via `raw_imu.py`). The IMU hardware itself produces the biased reading (~1.3 m/s² on X when stationary). No software correction is applied in the observation pipeline. The raw hardware value goes straight into the observation vector.

---

## The mismatch: what the policy learned vs what it receives

Here is the situation at each stage:

| Stage | Accelerometer X value the policy sees | +1.3 applied? |
|---|---|---|
| Training (JAX - `joystick.py`) | Raw sim value (e.g. 0.0 when still) | No - JAX bug silently discards it |
| Sim inference (NumPy - `mujoco_infer.py`) | Raw sim value + 1.3 | Yes - NumPy mutation works |
| Real robot (`v2_rl_walk_mujoco.py`) | Raw hardware reading (~1.3 when still) | Hardware produces it naturally |

The policy was trained seeing X ≈ 0.0 when the robot is stationary. But when it is deployed - either in the NumPy-based sim inference tool or on the real robot - it sees X ≈ 1.3 when the robot is stationary.

The neural network was never trained on observations containing that +1.3 offset, so it is receiving input that is systematically outside the distribution it learned from. It is like training a doctor to read blood pressure in mmHg, then showing them readings in kPa without telling them - the numbers look wrong and their diagnoses will be off.

---

## Why this matters for robot behaviour

The neural network policy uses all observations together to decide how to move the motors. The accelerometer reading is one of 71 inputs to the network (it occupies positions 3-5 in the observation vector, right after the gyroscope readings).

When the policy receives a value it was never trained on, it will produce incorrect motor commands. Concretely:

- The robot may lean or pitch incorrectly because it is trying to compensate for acceleration it thinks is happening but is not
- Walking gait stability may degrade - the robot could shuffle, stumble, or fall more easily
- The robot may have a persistent forward/backward lean it cannot correct because it is always fighting the phantom +1.3 signal

The effect is subtle rather than catastrophic - the policy still runs and the robot still approximately walks - but performance is consistently below what was achievable if training and deployment used the same observations. Any policy improvement gained through better training is partially thrown away at deployment time.

---

## How to fix it

There are two valid options. Both produce the same result: the policy sees the same accelerometer value during training as it does at deployment.

### Option A: Fix the training code (recommended by PR #24)

Remove the broken JAX line from `joystick.py`. Leave `mujoco_infer.py` and the real robot code as-is. Accept that the policy will train with whatever raw accelerometer value the simulator produces (X ≈ 0.0 when stationary). Then also remove the +1.3 from `mujoco_infer.py` so sim inference matches training.

**Changes required:**

1. `Open_Duck_Playground/playground/open_duck_mini_v2/joystick.py`, lines 501-502:
   Remove both lines (the commented-out NumPy attempt and the broken JAX line):
   ```python
   # Remove these two lines:
   # accelerometer[0] += 1.3 # TODO testing
   accelerometer.at[0].set(accelerometer[0] + 1.3)
   ```

2. `Open_Duck_Playground/playground/open_duck_mini_v2/mujoco_infer.py`, line 74:
   Remove the working-but-now-mismatched line:
   ```python
   # Remove this line:
   accelerometer[0] += 1.3
   ```

This means the real robot will still see the hardware's natural ~1.3 X-axis bias at runtime, which is now a mismatch. If this matters in practice, the runtime code would also need adjusting (see note below).

**Pros:** Simple. Aligns training with deployment. The real robot hardware already has this bias built in, and the policy trained without it may compensate naturally.

**Cons:** The sim-to-real gap now includes the +1.3 hardware bias. The policy will not have been explicitly trained to handle it.

### Option B: Fix the JAX code to correctly apply the bias during training

Keep the intention (simulate the hardware bias during training) but write the JAX code correctly, and keep `mujoco_infer.py` as-is.

**Change required:**

`Open_Duck_Playground/playground/open_duck_mini_v2/joystick.py`, line 502:
```python
# Current (broken):
accelerometer.at[0].set(accelerometer[0] + 1.3)

# Fixed (assign the result back):
accelerometer = accelerometer.at[0].set(accelerometer[0] + 1.3)
```

**Pros:** Preserves the original intent. The policy trains with data that reflects the real sensor's behaviour. Training and real-robot deployment both see X ≈ 1.3 when stationary.

**Cons:** Requires verifying that 1.3 is the correct value for this hardware. The value was apparently observed empirically but never rigorously confirmed. If the real hardware bias is not exactly 1.3, you have traded one mismatch for a slightly different one.

---

## Relevant file paths and line numbers

| File | Line | Content |
|---|---|---|
| `Open_Duck_Playground/playground/open_duck_mini_v2/joystick.py` | 501-502 | Broken JAX bias code (training) |
| `Open_Duck_Playground/playground/open_duck_mini_v2/mujoco_infer.py` | 73-74 | Working NumPy bias code (sim inference) |
| `Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py` | 156-169 | Real robot observation construction (no explicit +1.3) |
| `Open_Duck_Mini_Runtime/mini_bdx_runtime/mini_bdx_runtime/raw_imu.py` | 123-148 | IMU worker thread - reads hardware, applies only `x_offset` tare (not +1.3) |
| `Open_Duck_Playground/playground/open_duck_mini_v2/base.py` | 256-259 | `get_accelerometer()` - reads from MuJoCo sensor data |
| `Open_Duck_Playground/playground/open_duck_mini_v2/constants.py` | 88 | `ACCELEROMETER_SENSOR = "accelerometer"` |

**GitHub PR:** https://github.com/apirrone/Open_Duck_Playground/pull/24

---

## Summary in one sentence

During training, the attempt to add a +1.3 m/s² X-axis bias to the accelerometer reading silently did nothing (because JAX arrays are immutable and the result was discarded), but the same bias is correctly applied at inference time using NumPy, so the policy was trained on data that differs from what it receives when running - degrading walking performance.
