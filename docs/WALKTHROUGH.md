# Training Your Own Open Duck Mini Policies - A Complete Walkthrough

> **Who this is for**: You (Caleb). You have an RTX 5080 Mobile laptop, zero ML experience, and want to train improved walking policies for your duckling. This document assumes you understand nothing about ML training and explains every step, every command, and every concept from scratch.

---

## Table of Contents

1. [What We're Doing and Why](#1-what-were-doing-and-why)
2. [Your Goals](#2-your-goals)
3. [GPU Requirements](#3-gpu-requirements)
4. [Setting Up Your Laptop for Training](#4-setting-up-your-laptop-for-training)
5. [Understanding the Training Loop](#5-understanding-the-training-loop)
6. [Goal 1: Smoother Walking](#6-goal-1-smoother-walking)
7. [Goal 2: More Stable Walking](#7-goal-2-more-stable-walking)
8. [Reference Motion Generation](#8-reference-motion-generation)
9. [Evaluating and Selecting Checkpoints](#9-evaluating-and-selecting-checkpoints)
10. [Deploying to the Real Robot](#10-deploying-to-the-real-robot)
11. [Troubleshooting](#11-troubleshooting)
12. [Quick Reference](#12-quick-reference)

---

## 1. What We're Doing and Why

You're going to teach a simulated robot to walk better by running experiments on your laptop's GPU. The process works like this:

1. You tweak some settings (reward weights, penalties, filters)
2. You run a training command - your GPU simulates 8,192 robots walking simultaneously for millions of steps
3. The system produces an ONNX file (a portable "brain" file for the neural network)
4. You test that brain in a 3D simulation on your screen
5. If it looks good, you copy it to the Raspberry Pi on the real robot

**What "training" means**: The robot starts with no knowledge. It tries random movements, falls over, tries again slightly differently, and gradually discovers that certain sequences of motor commands keep it upright and moving forward. This process is called Reinforcement Learning (RL). Your GPU runs 8,192 simulated robots simultaneously - a normal CPU would take weeks; the GPU compresses this into hours.

**Is this feasible on your laptop?** Yes. Your RTX 5080 Mobile has 16 GB of VRAM (the GPU's own fast memory). Training needs roughly 14-16 GB at default settings. 8,192 parallel environments fits, peaking at 14,085 MiB of a 16,303 MiB total. A complete training run takes 2-4 hours. A quick test run takes 30-60 minutes.

**Is the sim-to-real gap manageable?** The project author has demonstrated policies trained in simulation walking on the real robot. It's validated at proof-of-concept level. There are some tuning steps needed on the real robot (joint offset calibration, IMU pitch bias), but the pipeline works. The biggest risks are: the accelerometer bias bug (fixable, see Section 7), servo behaviour at temperature extremes (not modelled), and surface variation (only flat surfaces are trained).

**What could go wrong?** Training can fail silently - the reward goes up but the robot learns something weird (spinning in circles, vibrating, falling gracefully). This is normal. You watch the simulation, tweak settings, and try again. Expect 10-30 iterations to get something good. Each iteration is a few hours of GPU time where you can do other things.

---

## 2. Your Goals

### Goal 1: Smoother Walking Motion
Make the existing duckling waddle less jerky without changing the character of the motion. The waddle is intentional - we want a smooth waddle, not a stiff walk.

**Approach**: Increase smoothness penalties in the reward function, enable the low-pass filter, and optionally lower control frequency. All software changes.

### Goal 2: More Stable Walking
Make the robot less likely to fall, better at recovering from bumps, and more solid when standing still.

**Approach**: Enable the torso orientation reward (currently disabled), increase the alive reward, add domain randomisation for perturbations, and fix the accelerometer bias bug.

> **Idle animations (body sway, head bobs)** are deferred. They require a Blender-authored animation clip and the `upstream/episodic` branch, which is experimental. That work is logged separately.

---

## 3. GPU Requirements

### Why you need a GPU at all

Your laptop has a **CPU** (Central Processing Unit) - extremely good at complex sequential tasks. A **GPU** (Graphics Processing Unit) has thousands of simpler cores doing things in parallel. Training RL policies is almost entirely parallel maths. JAX (the ML framework used here) and MJX (the GPU-accelerated physics simulator) are both designed to exploit this parallelism. The key specification is **VRAM** - the GPU's own fast memory. All 8,192 simulated robots need to live in this memory simultaneously.

### GPU compatibility table

| GPU | VRAM | Verdict |
|-----|------|---------|
| RTX 3060 | 12 GB | Marginal - needs reduced environments (1,024-2,048); much slower |
| RTX 3070 | 8 GB | Insufficient for default settings |
| RTX 3080 | 10 GB | Insufficient - less VRAM than the 3060 due to a cost-cutting decision by Nvidia |
| RTX 3080 Ti | 12 GB | Same as RTX 3060 - marginal |
| RTX 3090 | 24 GB | Good |
| RTX 4070 Ti | 12 GB | Marginal |
| RTX 4080 | 16 GB | Good - handles full 8,192 environments |
| **RTX 5080 Mobile** | **16 GB** | **Good - confirmed 8,192 environments fit (14,085 MiB peak / 16,303 MiB total)** |
| RTX 4090 | 24 GB | Excellent - recommended consumer GPU for this task |
| A100 40 GB | 40 GB | Datacenter GPU; roughly 3x faster than RTX 4090 but unnecessary for hobby use |

### RTX 5080 Mobile specifics

- **VRAM**: 16,303 MiB total; 14,085 MiB peak during training at 8,192 environments
- **PTX compilation**: On first run, JAX compiles the simulation into GPU-specific machine code. This takes approximately 7 minutes and looks like the terminal has hung. It only happens once - results are cached in `.tmp/jax_cache/`. Subsequent runs start much faster.
- **CUDA 13.0**: JAX's CUDA 12 builds are backwards-compatible with CUDA 13 drivers. If JAX complains, install a pinned version: `uv pip install jax[cuda12]==0.5.0`

### Cloud and free options

If you need to train on hardware other than your laptop:

| Option | Cost | VRAM | Notes |
|--------|------|------|-------|
| Kaggle free tier | Free | 16 GB (P100) | 30 hr/week; good for short test runs |
| Google Colab free | Free | ~16 GB (T4) | 12 hr sessions; unpredictable GPU assignment |
| Colab Pro | $10/month | L4 or A100 | Best cheap option for sustained iteration |
| Vast.ai RTX 4090 | ~$0.35-0.55/hr | 24 GB | Cheapest on-demand; prices fluctuate |
| RunPod RTX 4090 | ~$0.34/hr | 24 GB | More consistent quality than Vast.ai |

A 3-hour full training run on Vast.ai at $0.40/hr costs $1.20. A realistic beginner total (setup + 30 test and full runs) is $20-40 on cloud, or free on your own GPU.

---

## 4. Setting Up Your Laptop for Training

### 4.1 Prerequisites Check

Open a terminal and run these checks:

```bash
# Check your GPU is visible
nvidia-smi
# You should see: NVIDIA GeForce RTX 5080 ... 16303MiB

# Check Python version (need 3.11+)
python3 --version

# Check if uv is installed (the package manager this project uses)
uv --version
```

If `uv` is not installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Note on operating system**: JAX CUDA wheels are only published for Linux (x86_64). Windows is not natively supported. If you're on Windows, use WSL2 or a cloud notebook. Ubuntu 22.04 or 24.04 is recommended.

### 4.2 Clone and Install the Training Repo

You already have the repo forked and cloned. Set up the dependencies:

```bash
cd ~/Documents/open_duck_mini_research/Open_Duck_Playground

# This one command installs everything: Python, JAX with GPU support,
# MuJoCo, Brax, TensorFlow (for ONNX export), and all other dependencies.
# It creates a virtual environment automatically. First run takes a few minutes.
uv sync
```

**What `uv sync` does**: It reads the `pyproject.toml` file (a recipe listing every software package needed) and installs them all into an isolated environment. You never need to install packages manually. You do not install JAX, MuJoCo, or TensorFlow separately.

### 4.3 Verify It Works

```bash
cd ~/Documents/open_duck_mini_research/Open_Duck_Playground

uv run python -c "import jax; print(jax.devices())"
# Should print something like: [CudaDevice(id=0)]
# If it prints [CpuDevice(id=0)], JAX can't see your GPU - see Troubleshooting
```

### 4.4 Run the Existing Policy in Simulation

Before changing anything, watch the current pre-trained policy walk:

```bash
cd ~/Documents/open_duck_mini_research/Open_Duck_Playground

# This opens a MuJoCo window showing the simulated robot walking.
# Arrow keys control direction. Press 'h' to toggle head control mode.
uv run python playground/open_duck_mini_v2/mujoco_infer.py --onnx_path <path_to_onnx_file>
```

Find an existing ONNX file:
```bash
find ~/Documents/open_duck_mini_research -name "*.onnx"
```

**What you're looking at**: A 3D simulation of the robot. The arrow keys send velocity commands (forward, backward, turn). Watch how the robot moves - notice the jerkiness, the body sway, how it handles starting and stopping. This is your baseline.

---

## 5. Understanding the Training Loop

### 5.1 The Core Concept

Training is a cycle:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   1. EDIT reward weights in joystick.py             │
│          ↓                                          │
│   2. RUN training command (2-4 hours)               │
│          ↓                                          │
│   3. WATCH in simulation (mujoco_infer.py)          │
│          ↓                                          │
│   4. EVALUATE with analysis tools                   │
│          ↓                                          │
│   5. DECIDE: is it better? worse? weird?            │
│          ↓                                          │
│   6. REPEAT from step 1 with adjusted weights       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

This is the entire process. There is no shortcut. Even experts iterate 10-30 times. The skill is in knowing which knob to turn and by how much.

### 5.2 The Training Command

```bash
cd ~/Documents/open_duck_mini_research/Open_Duck_Playground

uv run python playground/open_duck_mini_v2/runner.py \
  --num_timesteps 150000000 \
  --output_dir checkpoints/my_first_run
```

**What each part means**:

- `uv run python` - runs the command inside the virtual environment with all dependencies available
- `playground/open_duck_mini_v2/runner.py` - the main training script
- `--num_timesteps 150000000` - how many total training steps. 150 million is a full run. Use 50,000,000 (50M) for quick tests
- `--output_dir checkpoints/my_first_run` - where to save the results

The default task is `flat_terrain_backlash`. **Backlash** is the mechanical slack in a gear train: when you reverse direction, the gears need to travel a small amount before they engage. This causes the reported joint position to lag slightly behind the actual position. The simulation models this to make training closer to real hardware behaviour. Available tasks:

| Task | What It Is | When to Use |
|------|-----------|-------------|
| `flat_terrain` | Flat floor, no backlash modelling | Quick experiments |
| `flat_terrain_backlash` | Flat floor with gear-slack modelled | **Recommended for deployment** |
| `rough_terrain` | Bumpy floor, no backlash | Robustness testing |
| `rough_terrain_backlash` | Bumpy floor with backlash | Maximum robustness |

### 5.3 What Happens When Training Runs

1. **JIT compilation (~7 minutes on RTX 5080)**: JAX compiles the simulation and neural network into optimised GPU code. Your terminal will appear to hang. This is normal. It only happens once per session (results are cached in `.tmp/jax_cache/` for future runs). The 7-minute overhead is a one-time PTX compilation cost specific to newer Nvidia architectures.

2. **Training begins**: You'll see output like:
   ```
   -----------
   STEP: 16384000 reward: 42.31 reward_std: 8.72
   -----------
   ```
   This prints periodically. The `reward` number is the average score across all evaluation episodes. You want this to trend upward - but see Section 9 for why you should not use this number alone to pick your best checkpoint.

3. **Checkpoints save automatically**: Every time the system evaluates, it saves a checkpoint (a snapshot of the neural network's current state) AND automatically exports an ONNX file. Both go into your `--output_dir`.

4. **Training ends**: After all timesteps are consumed, training stops.

### 5.4 Monitoring Training with TensorBoard

TensorBoard is a web dashboard that plots training metrics as live graphs. Open a second terminal:

```bash
cd ~/Documents/open_duck_mini_research/Open_Duck_Playground
uv run tensorboard --logdir=checkpoints/my_first_run
```

Then open `http://localhost:6006` in your browser.

**What to look at**:

| Metric | What It Means | Good Sign | Bad Sign |
|--------|-------------|-----------|----------|
| `eval/episode_reward` | Overall score | Steadily climbing | Flat or dropping |
| `eval/episode_reward_std` | Consistency of scores | Decreasing over time | Staying high |
| `eval/episode_length` | How long before the robot falls | Approaching max (1000) | Stuck low (<200) |
| `train/entropy_loss` | How exploratory the policy is | Decreasing gradually | Drops to zero too quickly |

**How to read the reward curve**:
- **First 10-30M steps**: Reward climbs steeply. The robot learns to not fall immediately.
- **30-100M steps**: Reward climbs more slowly. The robot refines its walking.
- **100-150M steps**: Reward plateaus. If reward hasn't started climbing by 50M steps, something is wrong with your reward weights.

**Important**: The training reward is not a reliable proxy for walking quality. A high reward score does not guarantee a good-looking walk. Always evaluate with `evaluate_policy.py` (Section 9) rather than picking the checkpoint with the highest training reward.

### 5.5 Testing a Trained Policy in Simulation

After training (or during, using a checkpoint):

```bash
cd ~/Documents/open_duck_mini_research/Open_Duck_Playground

uv run python playground/open_duck_mini_v2/mujoco_infer.py \
  --onnx_path checkpoints/my_first_run/YYYY_MM_DD_HHMMSS_150000000.onnx
```

Arrow keys to steer. Watch for:
- Does it walk smoothly or jerk?
- Does it recover from direction changes?
- Does it stand still cleanly when you release the keys?
- Does it fall when pushed (the simulation applies random pushes)?

### 5.6 Resuming a Training Run

If training is interrupted (laptop sleep, crash, Ctrl+C), resume from the last checkpoint:

```bash
uv run python playground/open_duck_mini_v2/runner.py \
  --num_timesteps 150000000 \
  --output_dir checkpoints/my_first_run \
  --restore_checkpoint_path checkpoints/my_first_run/YYYY_MM_DD_HHMMSS_STEP/
```

The checkpoint directory names include the date and step count. Pick the most recent one.

---

## 6. Goal 1: Smoother Walking

This section walks you through the specific changes to make walking smoother. Each change is independent - you can apply one at a time and test.

### 6.1 Where the Reward Weights Live

All reward weights are in a single file:

```
Open_Duck_Playground/playground/open_duck_mini_v2/joystick.py
```

Inside the `default_config()` function (around line 77), you'll find a block like:

```python
rewards=config_dict.create(
    scales=config_dict.create(
        tracking_lin_vel=2.5,
        tracking_ang_vel=6.0,
        torques=-1e-3,
        action_rate=-0.5,
        stand_still=-0.2,
        alive=20.0,
        imitation=1.0,
    ),
    tracking_sigma=0.1,
),
```

These numbers are the "knobs" you'll turn. Each one controls how much the robot cares about a specific behaviour. Positive numbers reward behaviour. Negative numbers penalise it.

**Note on `tracking_sigma`**: This value controls how tightly the robot must match its target velocity. The correct value is `0.1` (not `0.01`, which was an error in early configs). Industry standard is 0.1-0.25. A value of 0.01 makes the tracking requirement unrealistically strict - the robot can barely deviate from the exact commanded speed - which tends to produce brittle, overfitted behaviour. If you see `tracking_sigma=0.01` in your config, change it to `0.1` before training.

### 6.2 Change 1: Increase the Action Rate Penalty

**What it is**: The `action_rate` penalty scores how much the servo commands change between consecutive timesteps. The penalty is negative, so a larger magnitude means the robot is punished more severely for making sudden, jerky changes. Think of it as penalising hard braking.

**Current value**: `-0.5`
**Disney's value**: `-1.5`
**Recommended starting value**: `-1.0` (halfway between - don't jump straight to Disney's value, your servos are different)

```python
action_rate=-1.0,   # was -0.5, increased for smoother motion
```

**What to expect**: Smoother transitions between movements. The robot may walk slightly slower because it's being penalised for rapid changes. If it becomes too sluggish, back off to -0.75.

### 6.3 Change 2: Add a Jerk Penalty

**What it is**: The action rate penalty stops sudden changes. But the robot can still reverse direction abruptly without a large action rate. A jerk penalty (technically "action acceleration") penalises changes in the rate of change - it is to action rate what "penalising swerving while braking" is to "penalising braking itself".

**The formula**: `||a_t - 2*a_{t-1} + a_{t-2}||^2` - the second derivative (acceleration) of the action sequence. If actions change smoothly, this is small. If they reverse abruptly, this is large.

**How to add it**: Two changes required.

**Step A** - Add the reward function in `playground/open_duck_mini_v2/custom_rewards.py`:

```python
def cost_action_acceleration(act, last_act, last_last_act):
    """Penalises abrupt changes in the rate of change of actions (jerk)."""
    return jp.sum(jp.square(act - 2 * last_act + last_last_act))
```

**Step B** - Wire it into `joystick.py`:
1. Import the new function
2. Add `action_acceleration=-0.3` to the reward scales
3. Call the function in the reward computation section, passing `action`, `state.info["last_act"]`, and `state.info["last_last_act"]`

The `last_last_act` (the action from two steps ago) should already be tracked in `state.info` because the observation includes action history. If not, add it to the info dict in the reset and step functions.

**Recommended starting value**: `-0.3` (Disney uses -0.45, start lower)

### 6.4 Change 3: Enable the Torso Orientation Reward

**What it is**: Inside the imitation reward, there is a sub-component scoring whether the robot's body tilt matches the reference motion. It is currently computed but its weight is set to 0.0 - it has zero effect. Enabling it tells the robot "keep your body angled the way the reference motion says to".

```python
# In the imitation reward sub-weights
"orientation": 1.0,  # was 0.0
```

### 6.5 Change 4: Enable the Low-Pass Filter at Runtime

**What it is**: A low-pass filter smooths the commands sent to the servos. It blends each new command with the previous one. Imagine drawing with a pen attached to a spring - your hand makes sharp movements, but the pen traces a smoother curve because the spring absorbs the jolt. Technically this is an IIR (Infinite Impulse Response) filter - it blends the new value with the previous filtered value using a mixing coefficient `alpha`.

**How to enable it**: When running the policy in `mujoco_infer.py`, use the `--cutoff_frequency` argument:

```bash
uv run python playground/open_duck_mini_v2/mujoco_infer.py \
  --onnx_path your_policy.onnx \
  --cutoff_frequency 10
```

A cutoff of 10 Hz means: keep motion content below 10 Hz (normal walking), remove content above 10 Hz (jitter and vibration). Try values between 8 and 15. Lower = smoother but slower to respond. At 50 Hz control rate, the maximum meaningful frequency is 25 Hz (Nyquist limit), so a 10-15 Hz cutoff is conservative and safe.

**Important caveat**: This filter was not present during training. The policy does not know about it. Applying a filter the policy was never trained with changes what the policy "sees" as its previous action - this can slightly affect walking quality (the Markov assumption: the policy was trained to assume the current observation is sufficient without needing to know about prior filtering). If it causes instability, the proper fix is to include the filter during training too. As a first step though, try it - many people report it helps without issue.

### 6.6 Putting It All Together - Your First Smoothness Run

**Run 1** (quick test, ~45 min): Change just `action_rate` to -1.0. Train for 50M steps. Compare to baseline.

```bash
# Edit joystick.py: action_rate=-1.0, tracking_sigma=0.1
uv run python playground/open_duck_mini_v2/runner.py \
  --num_timesteps 50000000 \
  --output_dir checkpoints/smooth_v1
```

**Run 2** (quick test, ~45 min): If Run 1 looked better, also enable torso orientation. Train 50M steps.

**Run 3** (full run, ~3 hours): If both changes help, add the jerk penalty at -0.3 and train for the full 150M steps.

**Run 4** (refinement): Adjust weights based on what you observed. Too stiff? Reduce penalties. Still jerky? Increase them.

---

## 7. Goal 2: More Stable Walking

### 7.1 Fix the Accelerometer Bias Bug First

**What this bug is**: The robot has an accelerometer (a sensor measuring forces on the body - it can feel gravity and detect tilting). The real hardware sensor has a constant error of +1.3 on its X-axis. Someone tried to add this error into the training simulation so the brain would learn to expect it. But they used JAX incorrectly.

In JAX, arrays are immutable (cannot be changed in place). The code `accelerometer.at[0].set(accelerometer[0] + 1.3)` creates a new array with the change but discards it. The original array is unchanged.

The result: during training, the brain sees X=0.0. On the real robot, it sees X=1.3. The brain is confused.

**How to fix it**: The simplest fix is to remove the bias from the deployment code so training and deployment match. In the runtime script, find and remove the line:

```python
# REMOVE or comment out this line:
# accelerometer[0] += 1.3
```

### 7.2 Increase Push Robustness

Training already includes random pushes (perturbations) to build robustness. To increase them:

```python
push_config=config_dict.create(
    enable=True,
    interval_range=[3.0, 8.0],      # push more frequently (was [5.0, 10.0])
    magnitude_range=[0.2, 1.5],     # push harder (was [0.1, 1.0])
),
```

**Start conservative**. If pushes are too strong, the robot cannot learn to walk at all and just learns to crouch.

### 7.3 Broaden Domain Randomisation

Domain randomisation randomly varies simulation parameters so the brain learns to handle uncertainty. To improve real-world stability:

```python
# Wider friction range (current: 0.5-1.0, try: 0.3-1.2)
friction_range=[0.3, 1.2],

# Wider mass range (current: ±10%, try: ±20%)
mass_scale_range=[0.8, 1.2],
```

### 7.4 Stability Training Run

```bash
# After editing joystick.py with the changes above
uv run python playground/open_duck_mini_v2/runner.py \
  --num_timesteps 150000000 \
  --output_dir checkpoints/stable_v1
```

---

## 8. Reference Motion Generation

The "imitation reward" is how the robot learns the duckling waddle character rather than just "move forward somehow". It compares the robot's movements against a reference motion - a pre-computed sequence of joint angles representing the desired gait. If you want to change the character of the walk (different speed, stride length, waddliness), you regenerate this reference motion.

Reference motions live in:
```
~/Documents/open_duck_mini_research/Open_Duck_reference_motion_generator/
```

### 8.1 Generating New Reference Motions

```bash
cd ~/Documents/open_duck_mini_research/Open_Duck_reference_motion_generator

# Step 1: Generate motion recordings
# -j2 = use 2 CPU threads
# --duck open_duck_mini_v2 = target robot model
# --sweep = generate a sweep of parameter variations
# --output_dir = where to save recordings
uv run scripts/auto_waddle.py -j2 --duck open_duck_mini_v2 --sweep --output_dir recordings

# Step 2: Fit polynomial curves to the recordings
# Polynomials are a compact mathematical representation: instead of storing 50 data
# points per frame, you store 16 coefficients per joint that can reproduce the curve
# with high accuracy. This is what the training loop actually loads.
uv run scripts/fit_poly.py --ref_motion recordings/
```

The output (`polynomial_coefficients.pkl`) goes into `playground/open_duck_mini_v2/data/`. Training will use it automatically.

**When to regenerate**: If the default walk character looks wrong for your goals, or if you want to experiment with slower/faster reference gaits. For smoothness and stability improvements (Goals 1 and 2), the default reference motion is fine.

### 8.2 The `tracking_sigma` Connection

The reference motion and `tracking_sigma` work together. `tracking_sigma` controls how strictly the robot must match the reference at each timestep. With `tracking_sigma=0.1`, the robot has reasonable latitude to deviate slightly from the reference while still scoring well. With `tracking_sigma=0.01` (the incorrect earlier value), the robot is penalised harshly for any deviation - even natural variation in footfall timing - which produces rigid, brittle behaviour.

---

## 9. Evaluating and Selecting Checkpoints

Training saves checkpoints throughout the run. Picking the right one matters. **Do not rely on training reward to select your best checkpoint** - the reward curve reflects many factors including how well the robot avoids falling, not just walking quality. A checkpoint at 120M steps might produce better-looking motion than the final 150M checkpoint even if its reward score is lower.

### 9.1 Evaluate a Specific Policy

```bash
cd ~/Documents/open_duck_mini_research/Open_Duck_Playground

uv run python analysis/evaluate_policy.py --onnx checkpoints/my_run/policy.onnx
```

This runs the policy through a standardised evaluation and produces a walking quality score. Use this to compare checkpoints rather than the TensorBoard reward curve.

### 9.2 Find the Best Checkpoint Across a Run

```bash
uv run python analysis/find_best_checkpoint.py --run_dir checkpoints/my_run/
```

This sweeps all checkpoints in the run directory and ranks them by evaluated walking quality.

### 9.3 Suggested Evaluation Workflow

1. Let training complete (or at least reach 100M steps)
2. Run `find_best_checkpoint.py` to identify the top 2-3 candidates
3. View each candidate with `mujoco_infer.py` - the numbers give you candidates, your eyes make the final call
4. Pick the one that looks best in simulation, not the one with the highest reward score

---

## 10. Deploying to the Real Robot

### 10.1 What You Need

- The ONNX file from your training run
- A built Open Duck Mini robot with calibrated servo offsets
- The Open_Duck_Mini_Runtime installed on the Raspberry Pi

### 10.2 Copying the Policy

```bash
# From your laptop, copy the ONNX file to the Pi
scp checkpoints/smooth_v1/your_policy.onnx pi@<pi-ip-address>:~/policies/
```

### 10.3 Running It

SSH into the Pi and run:

```bash
cd ~/Open_Duck_Mini_Runtime
python scripts/v2_rl_walk_mujoco.py --onnx_path ~/policies/your_policy.onnx
```

### 10.4 Tuning Steps on the Real Robot

These are adjustments needed on physical hardware that training cannot cover:

1. **Per-robot joint offsets**: Each physical robot has slightly different servo zero positions due to assembly tolerances. Servo Kp (proportional gain - how aggressively the servo snaps to its target position) at the firmware level defaults to around 32. Training simulates a Kp of ~17.8, so real servos snap harder than simulated ones. Run `scripts/find_soft_offsets.py` and save results to `duck_config.json`. You can also reduce servo Kp to ~22 in the runtime for more compliant motion. Needs doing once per robot build.

2. **IMU pitch bias**: If the IMU is not perfectly level when mounted, add a `--pitch_bias` argument to compensate. Start at 0 and adjust in 0.05 increments if the robot leans.

3. **Low-pass filter**: Try `--cutoff_frequency 10` if the robot jitters. Start high (15) and reduce until smooth.

4. **Phase frequency**: Use the D-pad during early testing to adjust gait frequency up or down. The default may be too fast or slow for your specific robot's weight and servo response.

### 10.5 What to Watch For

- **Robot falls immediately**: Likely an offset calibration problem or the accelerometer bias bug. Check `duck_config.json` offsets and IMU orientation.
- **Robot vibrates but doesn't walk**: Servo Kp too high, or the policy is outputting high-frequency commands. Enable the low-pass filter.
- **Robot walks but drifts sideways**: IMU pitch or roll bias. Adjust the bias offset.
- **Feet don't lift off the ground**: Try increasing the gait frequency via D-pad, or retrain with a higher foot lift reward in the reference motion.

---

## 11. Troubleshooting

### Training Issues

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| "CUDA out of memory" | 8,192 environments don't fit | Reduce `num_envs` to 4,096 (see below) |
| Training hangs for >10 minutes at start | PTX/JIT compilation on first run | Wait - ~7 min on RTX 5080, cached after first run |
| Reward stays flat | Reward weights too aggressive, or `tracking_sigma=0.01` | Check `tracking_sigma=0.1`; reduce penalties; increase alive reward |
| Robot spins in circles | Tracking reward too low relative to alive reward | Increase `tracking_lin_vel` and `tracking_ang_vel` |
| Robot vibrates in place | Action rate penalty too low, or action scale too high | Increase `action_rate` penalty magnitude; reduce `action_scale` |
| Robot crouches and refuses to move | Penalties too high - the safest strategy is "do nothing" | Reduce smoothness penalties; increase tracking rewards |
| `ModuleNotFoundError` | Dependencies not installed | Run `uv sync` from the repo root |
| TensorBoard shows no data | Wrong log directory | Check `--logdir` path matches `--output_dir` |
| JAX can't see GPU (`CpuDevice`) | CUDA mismatch or JAX build | Try `uv pip install jax[cuda12]==0.5.0` |

### Reducing VRAM Usage

If you hit out-of-memory errors, reduce `num_envs`. In `playground/common/runner.py`, after the PPO config is loaded, add:

```python
self.ppo_training_params["num_envs"] = 4096  # reduced from 8192
```

This halves VRAM usage and roughly doubles training time.

### JAX Memory Pre-allocation

By default JAX claims 75% of your total VRAM the moment it starts. If you're running other programmes on the same GPU, set this before running training:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

---

## 12. Quick Reference

### Commands

```bash
# Install dependencies
cd ~/Documents/open_duck_mini_research/Open_Duck_Playground
uv sync

# Verify GPU is visible to JAX
uv run python -c "import jax; print(jax.devices())"

# Quick test run (50M steps, ~45 min)
uv run python playground/open_duck_mini_v2/runner.py \
  --num_timesteps 50000000 \
  --output_dir checkpoints/test_run

# Full training run (150M steps, ~3 hours)
uv run python playground/open_duck_mini_v2/runner.py \
  --num_timesteps 150000000 \
  --output_dir checkpoints/full_run

# Monitor training (open browser at localhost:6006)
uv run tensorboard --logdir=checkpoints/full_run

# Watch trained policy in simulation
uv run python playground/open_duck_mini_v2/mujoco_infer.py \
  --onnx_path checkpoints/full_run/latest.onnx

# Watch with low-pass smoothing filter
uv run python playground/open_duck_mini_v2/mujoco_infer.py \
  --onnx_path checkpoints/full_run/latest.onnx \
  --cutoff_frequency 10

# Resume interrupted training
uv run python playground/open_duck_mini_v2/runner.py \
  --num_timesteps 150000000 \
  --output_dir checkpoints/full_run \
  --restore_checkpoint_path checkpoints/full_run/CHECKPOINT_DIR/

# Evaluate walking quality of a specific policy
uv run python analysis/evaluate_policy.py --onnx checkpoints/full_run/policy.onnx

# Find best checkpoint across a training run
uv run python analysis/find_best_checkpoint.py --run_dir checkpoints/full_run/

# Generate reference motions
cd ~/Documents/open_duck_mini_research/Open_Duck_reference_motion_generator
uv run scripts/auto_waddle.py -j2 --duck open_duck_mini_v2 --sweep --output_dir recordings
uv run scripts/fit_poly.py --ref_motion recordings/
```

### File Locations

| What | Where |
|------|-------|
| Reward weights | `playground/open_duck_mini_v2/joystick.py` → `default_config()` |
| Reward functions | `playground/open_duck_mini_v2/custom_rewards.py` |
| PPO hyperparameters | Loaded from `mujoco_playground.config.locomotion_params` in `runner.py` |
| Reference motion data | `playground/open_duck_mini_v2/data/polynomial_coefficients.pkl` |
| Trained checkpoints | `--output_dir` (default: `checkpoints/`) |
| ONNX exports | Inside checkpoint dir + `ONNX.onnx` in working directory |
| TensorBoard logs | Same as `--output_dir` |
| JIT cache | `.tmp/jax_cache/` (safe to delete if builds go wrong) |

### Reward Weights Quick Reference

| Weight | Current | For Smoothness | For Stability |
|--------|---------|---------------|--------------|
| `alive` | 20.0 | 20.0 | 25.0 |
| `tracking_lin_vel` | 2.5 | 2.5 | 2.5 |
| `tracking_ang_vel` | 6.0 | 6.0 | 6.0 |
| `tracking_sigma` | **0.1** | **0.1** | **0.1** |
| `imitation` | 1.0 | 1.0 | 1.0 |
| `action_rate` | -0.5 | **-1.0** | -0.75 |
| `action_acceleration` | (none) | **-0.3** | -0.2 |
| `torques` | -0.001 | -0.001 | -0.001 |
| `stand_still` | -0.2 | -0.2 | -0.3 |
| `orientation` (imitation sub-weight) | 0.0 | **1.0** | **1.0** |

### Suggested Run Order

1. **Baseline**: Run the existing pre-trained policy in simulation. Record how it looks.
2. **Config check**: Verify `tracking_sigma=0.1` in your `joystick.py` before any run.
3. **Smoothness v1**: Change `action_rate` to -1.0. Quick 50M step run. Compare.
4. **Smoothness v2**: Add torso orientation (1.0). Quick 50M step run.
5. **Stability v1**: Fix accelerometer bias. Increase push magnitude. Full 150M run.
6. **Combined v1**: All smoothness + stability changes together. Full 150M run.
7. **Evaluate**: Use `find_best_checkpoint.py` + `mujoco_infer.py` to select the best ONNX.
8. **Deploy**: Copy best ONNX to robot, tune Kp and filter, iterate.

### External Links

- Open Duck Mini hardware: https://github.com/apirrone/Open_Duck_Mini
- Open Duck Playground (training): https://github.com/apirrone/Open_Duck_Playground
- Open Duck Mini Runtime (robot deployment): https://github.com/apirrone/Open_Duck_Mini_Runtime
- Reference motion generator: https://github.com/apirrone/Open_Duck_reference_motion_generator
- Community Discord: https://discord.gg/UtJZsgfQGe
- MuJoCo Playground (upstream framework): https://github.com/google-deepmind/mujoco_playground
- Vast.ai (GPU rental): https://vast.ai
- RunPod (GPU rental): https://runpod.io
- Kaggle free GPU: https://www.kaggle.com
