# Open Duck Mini - Complete Ecosystem Overview

> **Audience**: Someone comfortable with software engineering but new to machine learning and robotics AI. Every ML concept is explained simply, with the reasoning behind each design choice. If you already understand something, the detail is there - but you can skip the parenthetical explanations.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [ML and RL - A Ground-Up Explanation](#2-ml-and-rl---a-ground-up-explanation)
3. [The Five Repos - What Each Does](#3-the-five-repos---what-each-does)
4. [How the Whole Pipeline Fits Together](#4-how-the-whole-pipeline-fits-together)
5. [The Training Pipeline in Detail](#5-the-training-pipeline-in-detail)
6. [The Disney BDX Paper - What Open Duck Borrows](#6-the-disney-bdx-paper---what-open-duck-borrows)
7. [What Works, What Doesn't, What's Missing](#7-what-works-what-doesnt-whats-missing)
8. [Improving Motion Quality](#8-improving-motion-quality)
9. [Things to Consider Doing Differently](#9-things-to-consider-doing-differently)
10. [How Would I Actually Get Started?](#10-how-would-i-actually-get-started)
11. [Glossary](#11-glossary)

---

## 1. What Is This Project?

Open Duck Mini is a DIY version of Disney's **BDX droid** - the bipedal Star Wars droids you can see at Galaxy's Edge theme parks. It's a roughly 42 cm tall, 1.7 kg bipedal robot that waddles like a duckling. The waddling is deliberate - this is a character robot, not a humanoid trying to walk like a person.

**The goal**: a sub-$400, fully open-source robot that can walk, be controlled with an Xbox controller, and eventually express personality through head movements, LED eyes, and sounds.

**Who made it**: Antoine Pirrone, an R&D engineer at Pollen Robotics / HuggingFace, part of team Rhoban - a robotics lab at the University of Bordeaux that has won robot football competitions.

**Current state**: The robot walks. A brain (neural network) trained in computer simulation successfully controls the physical robot. You can drive it with an Xbox controller. The robot model has been verified to be bilaterally symmetric - the CAD, MJCF/URDF, and reference motion library all use a corrected symmetric geometry. Head control is experimental. Expression features (eyes, camera, speaker) are planned but not yet built.

---

## 2. ML and RL - A Ground-Up Explanation

This section explains every ML concept used in this project, assuming you know software engineering but have not studied machine learning.

### 2.1 What Problem Are We Solving?

Walking is a control problem. Every 20 milliseconds, something needs to decide: given where each joint currently is and how the body is tilted, what angle should each servo move to next?

Traditional robotics solves this by hand-writing mathematical rules (called "PID controllers" or "trajectory optimisers"). This works but is extremely hard to get right for complex motions - you'd need to write rules for every possible situation the robot might encounter.

This project uses **reinforcement learning (RL)** instead - a computer learns a control strategy entirely by trial and error in simulation. Nobody writes rules; the system discovers them on its own.

### 2.2 What Is a Neural Network (The Policy)?

**Simple version**: A neural network is a mathematical function that takes some numbers in and produces numbers out. It starts out useless (random outputs) and gets improved through training until it does something useful.

**Full version**: A neural network is a chain of matrix multiplications with "squishing" functions in between. It has millions of tunable numbers (called **weights** or **parameters**). Before training, the weights are random, so the output is gibberish. After training, the weights have been adjusted so that the right inputs produce the right outputs.

The specific type used here is an **MLP (Multi-Layer Perceptron)** - the simplest kind of neural network, a chain of layers. The Open Duck policy has three hidden layers, shaped `(512, 256, 128)`:

```
Input (61 sensor readings) 
  -> Layer 1 (512 neurons) 
  -> Layer 2 (256 neurons) 
  -> Layer 3 (128 neurons) 
  -> Output (10 servo commands)
```

In this context, the neural network that controls the robot is called the **policy** - it is the robot's "brain", mapping what it senses to what it does.

The **input** (61 numbers) includes:
- IMU data: a sensor that measures how the body is rotating and what forces it feels (6 numbers)
- Commands: what speed and direction you've told it to go (7 numbers)
- Joint positions: where each leg joint currently is, relative to the default standing pose (14 numbers)
- Joint velocities: how fast each joint is moving (14 numbers)
- Action history: what positions the brain commanded in the last three steps (30 numbers) - helps it learn smooth transitions

The **output** (10 numbers) is a small position correction for each leg joint (between -0.25 and +0.25 radians, added on top of the default standing position). The head joints are not controlled by the brain.

### 2.3 What Is Reinforcement Learning?

**Simple version**: The robot is given a score for everything it does. Actions that led to good scores become more likely in future. Actions that led to bad scores become less likely. Repeat millions of times until the robot learns to walk.

**Full version**: Reinforcement learning is learning by trial and error with a scoring system. Three key concepts:

- **Agent**: the neural network (policy/brain) that decides actions
- **Environment**: the simulated robot and its physics (like a video game engine)
- **Reward**: a number that tells the agent how well it's doing at each moment

The training loop:
1. Reset the simulated robot to standing
2. The policy reads sensor data, outputs joint commands
3. The simulation steps forward (physics happens - gravity pulls, feet push on floor)
4. A **reward function** scores the outcome: did the robot move in the right direction? Is it still upright? Was the motion smooth?
5. Repeat for thousands of steps per episode (one "life" before falling or resetting)
6. Use the accumulated scores to update the policy weights - actions that led to high scores become more likely

This is run **8,192 times in parallel** on GPUs - 8,192 virtual copies of the robot all learning simultaneously. WHY? Speed. Instead of one robot learning for months, 8,192 robots learn simultaneously and the whole thing finishes in 20-25 minutes. Each parallel copy is completely independent, just running alongside the others.

### 2.4 PPO - The Specific RL Algorithm

**Proximal Policy Optimisation (PPO)** is the specific algorithm that updates the weights after each batch of experience. The intuition:

1. Collect a batch of experience: many robots walking for many steps
2. For each action taken, calculate: "was this action better or worse than average?" (this is called the **advantage**)
3. If better than average, make that action slightly more likely in future. If worse, make it slightly less likely
4. The "proximal" part: do not change too much at once. There is a built-in limit (a "clipping ratio") that prevents the policy from suddenly forgetting how to walk after a single bad batch of experience

Key settings used here:
- `gamma = 0.97` - how much the policy cares about future vs immediate rewards. 0.97 means a reward 33 steps in the future is still worth about 37% of an immediate reward. Higher = more long-term thinking; lower = more short-term
- `learning_rate = 3e-4` - how big each weight update step is
- `num_envs = 8192` - parallel simulations
- `num_timesteps = 150-300M` - total training experience (150 to 300 million steps across all environments)

### 2.5 What Is MuJoCo?

**MuJoCo** (Multi-Joint dynamics with Contact) is a physics simulator - think of it as a very accurate physics engine like the one in a video game, but specifically designed for robots. It calculates what happens when you apply forces to rigid bodies connected by joints. It handles gravity, inertia, momentum, contact between the robot's feet and the ground, joint limits, friction, damping, and servo motor dynamics.

The simulation runs at **500 Hz** (500 steps per second, or one step every 2 milliseconds). The policy (brain) runs at **50 Hz** (one decision every 20 milliseconds, with 10 physics steps happening between each brain decision). The physics steps in between are needed to keep the simulation stable and accurate.

**MJX** is a version of MuJoCo that runs on GPUs using Google's JAX framework. WHY? Because running 8,192 copies of a robot simulation simultaneously requires the kind of massively parallel computing that GPUs provide. MJX lets the entire training loop - physics, policy network, and reward calculation - run on the GPU without sending data back and forth to the CPU.

### 2.6 Sim-to-Real Transfer

The hardest part of the whole project. A policy (brain) trained in simulation needs to work on the real physical robot. But the simulation is never perfectly accurate - there is always a gap between how things work in the simulator and how they work in the real world. Techniques used here:

- **Domain randomisation** (WHY: like training a driver in rain, sun, snow, and fog so they can handle anything): during training, randomly vary friction, joint masses, motor gains, centre-of-mass position, and more. The policy learns to handle a whole range of conditions, so the real robot is just "another variation" it has already encountered.

- **Actuator modelling (BAM)**: the Feetech servos were physically tested on a bench to measure their actual friction, stiction (the extra force needed to start moving from rest), damping, and torque. These measurements are baked into the simulator so simulated servos behave like real ones.

- **Observation noise**: random noise is added to sensor readings during training, so the policy is not surprised by the messier real sensors.

- **Action delay**: a random 1-step delay is sometimes added during training, simulating the real communication delay over the serial bus.

### 2.7 The Reward Function - The Heart of It All

The reward function is the single most important design choice in the entire project. It defines *what* the robot learns. If you design it poorly, the robot learns the wrong thing entirely. Open Duck uses seven reward terms:

| Term | Weight | What It Does |
|------|--------|-------------|
| `alive` | +20.0 | Constant reward for not falling over. The largest single term - the policy's primary drive is "stay upright" |
| `tracking_lin_vel` | +2.5 | Rewards matching the commanded forward/lateral velocity |
| `tracking_ang_vel` | +6.0 | Rewards matching the commanded turning rate |
| `imitation` | +1.0 | Rewards matching the reference duckling-walk motion |
| `torques` | -0.001 | Small penalty for using large motor forces. Encourages efficiency |
| `action_rate` | -0.5 | Penalty for changing joint commands abruptly between steps. Encourages smoothness |
| `stand_still` | -0.2 | Penalty for moving when the commanded velocity is zero |

**What are "smoothness penalties"?**

The `action_rate` and (in Disney's version) `action_acceleration` terms deserve explanation because they are crucial for motion quality.

- **Action rate penalty**: imagine a driving instructor who deducts points every time you jerk the steering wheel. The `action_rate` penalty deducts score whenever the policy changes joint commands sharply from one step to the next. Mathematically: it squares the difference between this step's command and last step's command. Large sudden changes produce a large penalty; smooth gradual changes produce almost no penalty.

- **Action acceleration penalty**: this is even stricter. It penalises changes *in the rate of change* - sometimes called "jerk" (the third derivative of position). Imagine the same driving instructor also deducting points when you change gear too abruptly even if each individual gear change was smooth. If the policy was already changing commands in one direction and suddenly reverses direction, even if neither individual change was large, the acceleration penalty catches it. Formula: `(this_step - 2 × last_step + two_steps_ago)²`. Open Duck does NOT currently have this penalty but Disney's system does, and it is a major reason Disney's robot moves more smoothly.

**What is the exponential reward form?**

The velocity tracking terms use `exp(-error)` - an exponential function. WHY? It behaves like "you're getting warmer/colder" - it gives a strong signal when you're close to the target (where small improvements matter most) and a weak signal when you're far away (where the situation is already bad and the exact amount of badness doesn't matter much). A flat penalty would treat "slightly off" and "wildly off" as proportionally different; the exponential form is more like a magnet that gets stronger the closer you get.

**The imitation reward** is the most complex term. It compares the robot's current state to a pre-computed reference motion and scores the match. This is what creates the duckling waddle character - the reference motion was designed with the waddle built in, and the policy learns to reproduce it because the reward tells it to.

**tracking_sigma** controls the width of the exponential "window" in the velocity-tracking reward terms. A value of 0.01 (the original default) is very tight - even a small velocity error produces a near-zero reward, so the signal is useless for large parts of training. The current value is 0.1, which is within the industry-standard range of 0.1-0.25 and gives a useful gradient across a wider range of tracking errors.

### 2.8 Reference Motions - The Walking Template

The policy does not learn to walk from scratch with zero guidance. It has a "template" to imitate: a physically plausible walking gait generated by a traditional motion planner called Placo (a kinematics solver - a tool that figures out what joint angles are needed to put the feet in specific positions).

There are 210 of these templates, covering different speeds and directions (6 forward speeds × 5 sideways speeds × 7 turning rates = 210). During training, the closest template to the current command is selected, and the imitation reward pushes the policy to match it. The grid includes zero values for sideways speed and turning rate so that the robot has reference motions for straight-ahead walking and standing still - an earlier version of the generator had a step-size arithmetic bug that caused zero values to be skipped.

The templates encode the duckling waddle character - lateral body sway, specific foot timing, trunk pitch. The policy faithfully reproduces this because the reward tells it to match the template.

**Why use templates at all?** Without a reference motion, the policy might invent a valid but ugly gait - hopping on one leg, shuffling, or making wild swinging movements. The template constrains the search space to "things that look like a duck walking", which dramatically speeds up training and produces more natural-looking results.

### 2.9 Asymmetric Actor-Critic - Training With More Information Than Deployment Has

**Simple version**: like a student who learns from a textbook with all the answers visible during class, but then has to take the exam without the textbook. The training is easier when you have more information; the deployment only uses what's actually available.

**Full version**: during training, two networks exist:

- The **actor** (the policy/brain that will be deployed): sees only what real sensors can provide - noisy IMU readings, joint positions, commands. This is what runs on the robot.
- The **critic** (a helper that only exists during training): sees a "privileged" version with perfect information - exact body velocity, foot contact forces, true orientation without noise. The critic uses this extra information to better judge whether the actor's choices were good or bad, giving the actor stronger feedback signals.

WHY? The critic is only used during training, never deployed. By letting the critic see extra information, the actor learns faster because it gets more accurate feedback. On deployment, only the actor runs - it has already learned a good policy from the better feedback.

This is called **asymmetric actor-critic** because the two networks (actor and critic) have different, asymmetric amounts of information available to them.

### 2.10 ONNX - From Training to the Real Robot

After training, the policy (neural network) is exported to **ONNX** format - a standard file format for neural networks, like a PDF but for trained brains. The Raspberry Pi Zero 2W on the robot loads this file and runs the network 50 times per second:

```
Read sensors -> Build 61-number input -> Run ONNX model -> Get 10 joint offsets -> Send to servos
```

This inference (running the network) takes a few milliseconds. The Raspberry Pi has no GPU, but the network is small enough to run comfortably on the CPU.

---

## 3. The Five Repos - What Each Does

### 3.1 Open_Duck_Mini (Main Repo)
**Purpose**: Hardware design, CAD, assembly docs, simulation model files

Contains:
- MJCF/URDF robot model files (XML files that describe the robot's physics to MuJoCo)
- Assembly documentation
- Bill of materials
- Servo configuration tools
- Pre-trained ONNX policy files (ready to deploy - you can use these without training)
- Links to the Onshape CAD model (cloud-based 3D CAD)

The simulation model is the bridge between hardware and training. Two variants exist:
- `robot.xml`: simplified model (for quick testing)
- `robot_motors.xml`: model with real servo dynamics identified from physical measurements (for accurate training)

The MJCF and URDF are exported from the Onshape CAD. The original CAD had a 37 mm positional offset in the right lower leg, which produced an asymmetric model - the right foot sat noticeably higher than the left in the MuJoCo world frame. This was traced by comparing world-space body positions in simulation and then inspecting the corresponding Onshape mates. All right-leg mates were rebuilt to achieve bilateral symmetry: 0.000 mm difference across all body pairs. The re-exported model has true mirror-image joint axes (for example, left knee axis `[0,-1,0]`, right knee axis `[0,+1,0]`) and a home keyframe where both feet sit at identical height.

### 3.2 Open_Duck_Playground (Training)
**Purpose**: RL training - where policies are born

This is where you actually train the brain. It contains:
- Environment definition (what the robot observes, what it can do, how it's scored)
- PPO training loop (via the Brax library from Google)
- All seven reward functions
- Reference motion loading and lookup
- Domain randomisation configuration
- Training scripts

Training takes roughly 20-25 minutes on 2x RTX 4090 GPUs for 150 million timesteps.

### 3.3 Open_Duck_Mini_Runtime (Deployment)
**Purpose**: Runs the trained policy on the real robot

A Python application for the Raspberry Pi Zero 2W that:
- Reads the BNO055 IMU (gyroscope + accelerometer connected via I2C - a simple two-wire serial protocol)
- Reads servo positions via the Feetech serial protocol (1 Mbit/s)
- Reads Xbox controller via Bluetooth
- Runs ONNX policy inference at 50 Hz
- Writes servo position commands
- Manages head/antenna joints separately from the RL policy

### 3.4 Open_Duck_reference_motion_generator
**Purpose**: Creates the 210 reference walking gaits

Pipeline:
1. **Placo** (a whole-body kinematics solver - a maths tool that figures out what joint angles produce the desired foot positions) generates physically plausible walking trajectories for a grid of (forward, sideways, turning) velocities
2. Each trajectory is recorded as joint angles over time at 50 Hz
3. One gait cycle is extracted and fitted with degree-15 polynomials (a smooth mathematical curve that approximates the recorded motion compactly)
4. The polynomial coefficients (a small set of numbers that can reproduce the curve) are saved

The `auto_waddle.py` script orchestrates the whole sweep: 6 forward speeds × 5 sideways speeds × 7 turning rates = 210 gaits. The grid is generated using the symmetric URDF, so the reference motions are consistent with the corrected robot geometry. The `auto_gait.json` grid config correctly includes zero values for sideways speed and turning rate.

### 3.5 Open_Duck_Blender (Animation)
**Purpose**: Hand-author custom motions in Blender (3D animation software)

A Blender 4.3+ project with:
- FK/IK rigged armature of the Open Duck (FK = Forward Kinematics: move each joint directly; IK = Inverse Kinematics: move the foot and let the system figure out the joints)
- Data recorder that exports motions in the same format as the motion generator
- Coordinate frame conversion (Blender uses Y-forward; the robot uses X-forward)

Known limitation: foot contacts are hardcoded to "both feet on ground", not computed from the animation. This means Blender-sourced motions have incorrect contact information.

---

## 4. How the Whole Pipeline Fits Together

```
┌─────────────────────────────────────────────────────────────────┐
│                    DESIGN TIME (once)                            │
│                                                                 │
│  Onshape CAD -> 3D Print + Assemble -> Physical Robot           │
│       │                                                         │
│       ▼                                                         │
│  MJCF Model (robot_motors.xml)                                  │
│       │                                                         │
│  BAM Servo Testing -> friction, damping, inertia                │
│       │                  baked into MJCF                        │
│       ▼                                                         │
│  Reference Motion Generator                                     │
│       │                                                         │
│  Placo walk engine -> 240 gaits -> polynomial fitting           │
│       │                                                         │
│       ▼                                                         │
│  polynomial_coefficients.pkl (compact gait library)             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING TIME (hours on GPU)                  │
│                                                                 │
│  Open_Duck_Playground                                           │
│       │                                                         │
│  Load MJCF model -> Spawn 8192 parallel MJX environments       │
│       │                                                         │
│  Load gait library -> reference motion lookup                   │
│       │                                                         │
│  PPO training loop:                                             │
│    for 150-300M timesteps:                                      │
│      - policy reads sensor data (61 numbers)                   │
│      - policy outputs joint offsets (10 numbers)               │
│      - reward function scores the result                       │
│      - PPO updates policy weights                              │
│       │                                                         │
│  Export trained policy -> ONNX file                             │
│       │                                                         │
│  analysis/ evaluation tools:                                    │
│    - find_best_checkpoint.py (sweep all checkpoints by walking) │
│    - evaluate_policy.py (cold-start, push recovery, symmetry)   │
│    - quick_walking_eval.py (live TensorBoard during training)   │
│    - analyse_training.py (read TensorBoard events)             │
│    - debug_policy.py (per-step velocity traces)                │
│    - replay_reference.py (visual reference motion playback)    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT (real robot, real time)            │
│                                                                 │
│  Open_Duck_Mini_Runtime on Raspberry Pi Zero 2W                 │
│       │                                                         │
│  50 Hz loop (50 times per second):                              │
│    1. Read IMU (gyroscope + accelerometer)         ─┐           │
│    2. Read servo positions                          │-> build   │
│    3. Read Xbox controller commands                 │   sensor  │
│    4. Calculate phase from ref motion              ─┘   vector │
│    5. Run ONNX inference -> 10 joint offsets                    │
│    6. Add offsets to standing pose                              │
│    7. Send positions to servos                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. The Training Pipeline in Detail

### 5.1 Observations (What the Policy Sees)

The policy receives a 61-number vector every 20 milliseconds:

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| IMU angular velocity | 3 | How fast the body is rotating (gyroscope) |
| IMU linear acceleration | 3 | Forces on the body (accelerometer) |
| Commands | 3 | Desired (forward speed, sideways speed, turning rate) |
| Phase signal | 2 | sin(phase) and cos(phase) - where in the gait cycle we are. WHY two numbers? Using sin and cos instead of a single angle avoids a discontinuity when the phase wraps from 1.0 back to 0.0 |
| Gait frequency | 1 | How fast the gait cycles |
| Gait switch | 1 | Flag for phase transitions |
| Joint positions | 14 | Current position of each joint minus the default standing pose |
| Joint velocities | 14 | How fast each joint is moving (scaled down by 0.05 to keep numbers in a similar range) |
| Last action | 10 | What the policy commanded last step |
| Second-last action | 10 | Two steps ago |

The critic (helper during training) sees a **privileged** version with perfect data: exact body velocity, foot contact forces, true orientation. This asymmetric actor-critic setup helps training converge faster while keeping the deployed policy working only from real sensor data.

### 5.2 Actions (What the Policy Outputs)

10 joint position offsets (5 per leg: hip yaw, hip roll, hip pitch, knee, ankle). These are small numbers (scaled to -0.25 to +0.25 radians) added to the default standing pose to produce the target joint angle. The head joints (4 degrees of freedom) and antennas (2 degrees of freedom) are not in the action space - they are controlled separately.

### 5.3 The Imitation Reward in Detail

This is the mechanism that creates the duckling waddle character. At each timestep:

1. Look up the current commanded velocity (forward, sideways, turning)
2. Find the nearest pre-computed reference gait from the 240 available
3. Evaluate the polynomial at the current phase to get target joint positions, velocities, foot contacts, and body orientation
4. Score the match across several sub-components:

| Sub-component | Inner Weight | What It Compares |
|--------------|-------------|-----------------|
| Joint positions | 15.0 | Simulated vs reference joint angles |
| Joint velocities | 1.0 | Simulated vs reference joint velocities |
| Foot contacts | 1.0 | Which feet should be on the ground |
| End-effector position | 1.0 | Foot positions in body frame |
| Body linear velocity | 1.0 | Forward/sideways speed matching |
| Body angular velocity | 0.5 | Turning rate matching |
| Body orientation | 0.0 (disabled) | Torso pitch/roll matching - computed but currently turned off |

Each sub-component uses `exp(-error × weight)` - the exponential "getting warmer" form described earlier. The sub-components are multiplied together, so all must be reasonable for the total to be non-zero.

**Important**: the imitation reward is switched off when commanded velocity is near zero. A separate `stand_still` penalty handles the standing case.

### 5.4 Domain Randomisation

Every time the robot falls and the episode resets, these parameters are re-randomised:

| Parameter | Range | Why It's Varied |
|-----------|-------|---------|
| Ground friction | 0.5 to 1.0 | Different floor surfaces |
| All body masses | ±10% of nominal | Manufacturing variation, attachments |
| Motor Kp gains (the stiffness of position control) | ±10% | Servo variation |
| Centre of mass position | ±2 cm per axis | Structural asymmetry |
| Home pose offsets | ±0.05 rad | Servo zero calibration error |
| Joint position noise | σ = 0.005 rad | Encoder noise |
| Joint velocity noise | σ = 0.01 rad/s | Velocity estimation noise |
| Action delay | 0 or 1 step randomly | Communication latency |

WHY domain randomisation? If you train without it, the policy learns to exploit the exact simulation parameters. Put it on a real robot with slightly different friction or slightly heavier legs and it immediately falls. By varying parameters during training, the policy learns behaviour that works across a whole range of conditions - so the real robot is just another point in that range.

### 5.5 Known Bugs and Issues

1. **Accelerometer bias bug**: A bias of +1.3 is added to the X-axis accelerometer reading in the code that runs on the real robot, but this bias was never present during training. The line of code that was supposed to add it during training does not actually work (a JAX behaviour quirk). So the robot's trained brain was never taught to handle this bias, but the real robot always sees it. This is a known discrepancy that degrades performance.

2. **Gravity sensor indexing bug**: `get_gravity()` in `mujoco_infer_base.py` was passing the sensor ID to the MuJoCo data array instead of the sensor's data address. The two happen to agree for the first sensor in the model (ID 0 = address 0), but diverge for any other sensor. This causes the gravity vector used during inference to read from the wrong memory location, silently returning incorrect orientation data.

3. **Torso orientation reward disabled**: The body orientation sub-component of the imitation reward has weight 0.0, so the policy gets no reward for maintaining correct trunk pitch/roll. This may contribute to excessive body sway.

4. **No gait interpolation**: When commanded velocity changes, the reference motion snaps to the nearest pre-computed gait with no blending. This creates sudden jumps in the imitation target.

---

## 6. The Disney BDX Paper - What Open Duck Borrows

The Disney paper ("Design and Control of a Bipedal Robotic Character", RSS 2024) describes the control system for the full-sized BDX droids at Galaxy's Edge (which weigh 15.4 kg - about 10x heavier than Open Duck Mini). Open Duck directly adapts the imitation reward concept, but Disney's full system is significantly more sophisticated.

### 6.1 Disney's Architecture

```
Animator-authored reference states
        │
        ▼
Animation Engine (blend background + triggered + joystick layers)
        │
        ▼
RL Tracking Policy (50 Hz - brain decides 50 times per second)
        │
        ▼
Cubic interpolation to 600 Hz
        │
        ▼
Low-pass filter (37.5 Hz cutoff)
        │
        ▼
Quasi-direct-drive actuators
```

**What is the 600 Hz cubic interpolation?**

The brain decides what to do 50 times per second. The servos can receive commands 600 times per second. Interpolation fills in the gap - instead of having the servos snap to each new command (like a connect-the-dots picture with straight lines), it draws a smooth curve through the 50 decision points, so the servos get smooth gradual guidance 600 times per second.

Think of it this way: imagine you're driving and your GPS updates your steering angle only twice per minute. If you jerked the wheel to each new angle instantly, you'd swerve all over the road. Instead, you smoothly steer towards each new angle between updates. Disney's 600 Hz interpolation does exactly this for the servos.

WHY does this matter? The brain (policy) runs at 50 Hz for computational reasons - that's about as fast as the neural network can run and still leave time for sensor reading. But the servos experience any "step changes" in commands as mechanical jolts. The interpolation absorbs those jolts.

**What is the 37.5 Hz low-pass filter?**

A low-pass filter (think of it as a "smoothing filter" or "shock absorber for signals") removes rapid fluctuations from a signal while letting gradual changes through. The name comes from the fact that it passes "low frequency" (slow-changing) signals and blocks "high frequency" (fast-changing) ones.

Imagine the filter as a shock absorber on a car. When you drive over a sharp pothole (a sudden, brief bump), the shock absorber absorbs most of the jolt - the car body moves much less than the wheel. When you drive up a gradual slope (a slow, sustained change), the shock absorber lets the whole car rise with it. A 37.5 Hz low-pass filter similarly absorbs rapid jitter in the servo commands while letting the robot move smoothly.

WHY use a low-pass filter here? Even after interpolation to 600 Hz, the policy might produce small rapid oscillations in its output - tiny back-and-forth wiggles that would make the joints vibrate. The low-pass filter smooths these out. Disney applies it after interpolation and before sending commands to actuators.

**What is quasi-direct-drive?**

A "quasi-direct-drive" actuator is a motor with very little gear reduction - ideally just 1:1 (the motor shaft IS the joint). Most hobby servos use high gear ratios (50:1 or more) to multiply the motor's torque. More gears = more friction, more "stiction" (the extra force needed to overcome static friction and start moving), and the joint becomes hard to move if pushed externally.

Disney's legs use motors that barely need gearing. WHY does this matter?

1. **Backdrivability**: you can push the joint with your hand and it moves easily. If the robot's foot hits an unexpected bump, the leg can flex to absorb it rather than fighting it rigidly. This makes contact physics smoother and closer to the simulation.

2. **Less sim-to-real gap**: contact forces in simulation can be modelled reasonably accurately. With high-gear-ratio servos, the actual contact dynamics on real hardware are dominated by friction, stiction, and gear backlash - effects that are very hard to simulate accurately.

3. **Smoother motion**: quasi-direct-drive motors can produce very smooth, controlled torque across their full speed range. Geared servos produce significantly less torque at high speeds, so if the policy commands a fast movement, the servo may not be able to follow, causing lag and jitter.

Open Duck uses cheap geared hobby servos (Feetech STS3215) - the opposite extreme. This is one of the most fundamental hardware limitations and is a primary source of motion roughness.

### 6.2 What Open Duck Replicates

- **Imitation reward structure**: exponential matching of joint positions, velocities, contacts
- **50 Hz policy frequency**
- **PPO with domain randomisation**
- **Reference motion-based training**

### 6.3 What Open Duck Does NOT Have

| Disney | Open Duck | Impact |
|--------|-----------|--------|
| 600 Hz cubic interpolation between policy outputs | Raw 50 Hz steps to servos | Jerky motion at servo level |
| 37.5 Hz low-pass filter before actuators | Optional filter exists in code but is disabled | High-frequency jitter passes through |
| Quasi-direct-drive actuators (low gear ratio, backdrivable) | Geared hobby servos (high friction, stiction) | Fundamentally different contact dynamics |
| Multi-layer animation blending engine | Single reference motion lookup | No smooth transitions between gaits |
| Head joint reward weight 100.0 | Head not in policy | No expressive head tracking |
| Action rate penalty -1.5 | Action rate penalty -0.5 | Three times less incentive for smoothness |
| Action acceleration penalty -0.45 | None | No jerk control whatsoever |
| Custom compute hardware | Raspberry Pi Zero 2W | Tight timing budget |

### 6.4 The MuJoCo Playground Paper

This framework (Google DeepMind + UC Berkeley) is what Open Duck Playground is built on. Key facts:
- GPU-accelerated MuJoCo via JAX (MJX)
- Over 120,000 simulation steps per second on a high-end GPU
- Successful zero-shot sim-to-real across 6 robot platforms (meaning: trained in simulation, worked on real hardware first try)
- Open Duck's 150M timesteps takes roughly 20-25 minutes on 2x RTX 4090 GPUs

---

## 7. What Works, What Doesn't, What's Missing

### Works Well
- Basic forward locomotion with duckling character
- Sim-to-real transfer (BAM servo model + domain randomisation)
- Teleoperation via Xbox controller
- Sprint mode (faster gait frequency)
- Stability under moderate perturbation

### Works But Could Be Better
- Motion smoothness (action rate penalty too low, no jerk penalty, no interpolation layer)
- Gait transitions (the reference motion snaps abruptly when commanded velocity changes)
- Standing stability (no dedicated standing policy, stand_still penalty is weak)
- Head control (experimental, not integrated into the RL policy)

### Missing / Not Implemented
- Expression system (LED eyes, camera, speaker, microphone)
- Autonomous behaviours (obstacle avoidance, person following)
- Fall recovery (get-up policy)
- Battery monitoring
- Safety limits at runtime (no joint limits enforced, velocity clamping commented out)
- Proper fall detection and motor shutdown

### Known Bugs
- Accelerometer bias mismatch between training and deployment
- Gravity sensor indexing bug in `mujoco_infer_base.py` (`get_gravity()` uses sensor ID instead of sensor address)
- Torso orientation reward disabled but probably should be enabled
- Foot contacts hardcoded in Blender pipeline
- Sim-to-real documentation marked as "not finalised yet"

---

## 8. Improving Motion Quality

The waddle is intentional and should be preserved - it is the duckling character. The goal is to make the waddle **smoother and more natural**, not to eliminate it. Here are the highest-impact changes, ordered by effort:

### 8.1 Zero-Retraining Fixes (Deploy Immediately)

**Enable the low-pass action filter**: Already implemented in the runtime (`rl_utils.py`), just disabled by default. A 10-15 Hz cutoff IIR filter (a type of low-pass filter - like the shock absorber described earlier) on policy outputs would smooth the 50 Hz step changes into continuous curves. This is the single easiest improvement.

**Reduce servo Kp at runtime**: The current P-gain (a "stiffness" setting for the servo's position controller - higher = snaps more aggressively to target) of 32 makes servos snap aggressively to targets. Dropping to Kp=22 (validated by community members) adds compliance without losing tracking.

### 8.2 Reward Function Changes (Requires Retraining)

**Increase action rate penalty**: From -0.5 to -1.5 (matching Disney's weight). Penalises large changes between consecutive commands.

**Add action acceleration penalty**: `‖a_t - 2×a_{t-1} + a_{t-2}‖²` at weight -0.45. This penalises changes in the rate of change - the "jerk". The infrastructure already exists (action history is in the observations). This is the single highest-impact training change.

**Enable torso orientation reward**: Currently computed but weight is 0.0. Setting it to 1.0 would encourage the policy to maintain consistent trunk pitch/roll rather than swaying erratically.

**Add lateral CoM velocity penalty**: Explicitly penalises unintended body sway outside of what the reference motion prescribes.

**A note on combining penalties**: An experiment combining all smoothness penalties in a single run (action rate, action acceleration, and lateral velocity) over-penalised the robot - it learned to stand still rather than walk, because the safest way to avoid movement penalties is to not move. Add penalties one at a time and evaluate walking quality (not training reward) after each change. The `alive` reward term has weight 20.0, which dominates the training reward signal; a policy can achieve a high training reward simply by staying upright without walking well. Always use the `find_best_checkpoint.py` tool to select checkpoints by actual walking quality rather than raw training reward.

### 8.3 Structural Improvements (More Effort)

**Lipschitz-Constrained Policy (LCP)**: Add a gradient penalty to the training loss. This constrains how rapidly the policy output can change with respect to input changes - enforcing smoothness at the network level rather than just through reward engineering. Think of it as forcing the policy function itself to be smooth, rather than just rewarding smooth behaviour. Validated on comparable-scale robots including the Berkeley Humanoid.

**Gait interpolation**: Replace nearest-neighbour reference motion lookup with interpolation between the 2-3 closest gaits. This eliminates the sudden jump when the robot transitions between speed/direction regimes.

**Lower control frequency**: Try 20 Hz instead of 50 Hz. Research shows lower-frequency policies transfer better on geared servos with backlash. The servo can not meaningfully respond to 50 Hz commands anyway - the Feetech STS3215 has significant mechanical compliance.

**Improve reference motions**: Use the Blender pipeline to hand-craft smoother reference gaits with more natural duckling body dynamics. Fix the hardcoded foot contacts. The quality of the reference motion directly bounds the quality of the learned gait.

### 8.4 Advanced Techniques

**Teacher-student distillation**: Train a capable but potentially jerky "teacher" policy with relaxed smoothness constraints. Then train a "student" policy to mimic the teacher's behaviour while adding strong smoothness penalties. The student learns what to do from the teacher and how to do it smoothly from the penalties.

**Mirror symmetry loss**: Add a loss term that encourages left-right symmetric outputs for symmetric inputs. Symmetric gaits are inherently smoother (the robot does not favour one side).

**Residual policy**: Train a base policy for the waddle, then train a second "residual" policy that adds small corrections on top. The residual can focus purely on smoothness without having to also learn locomotion from scratch.

---

## 9. Things to Consider Doing Differently

### 9.1 Fix the Known Inference Bugs First
Two bugs in the inference path mean the deployed policy sees incorrect sensor data:

- **Accelerometer bias**: a +1.3 bias is applied to the X-axis reading at runtime but was not present during training (issue #24). Either add it to training or remove it from the runtime.
- **Gravity sensor indexing**: `get_gravity()` in `mujoco_infer_base.py` passes the sensor ID to the data array instead of the sensor address. This silently returns orientation data from the wrong memory location. Fix by using `model.sensor_adr[sensor_id]` as the array index.

### 9.2 Do Not Ignore the Torso Orientation
The torso orientation sub-reward is computed but zeroed out. This is a free improvement - just change the weight. It will help the policy maintain a consistent body angle instead of drifting.

### 9.3 The Servo PID Matters More Than You Would Think
The runtime uses P-only control (P:32, I:0, D:0). Adding a small D-term (derivative - which damps oscillations by penalising rapid position changes in the servo itself, independent of the policy) would reduce vibration at the servo level. This is a hardware configuration change, not a training change.

### 9.4 Safety First
The runtime has essentially no safety features. Joint limits are not enforced, velocity clamping is commented out, there is no fall detection, no motor shutdown on anomaly. Before extending the project, add basic safety: joint limit enforcement, current monitoring, fall detection (IMU-based), and a watchdog timer.

### 9.5 Consider the Computation Budget
The Pi Zero 2W is tight for 50 Hz ONNX inference + servo communication + IMU reads. If you are adding features (vision, autonomy), you will need to either move to a more capable board (the commercial kits use RK3576D with NPU - a dedicated neural processing unit) or offload compute.

### 9.6 The Blender Pipeline Is Underused
The Blender animation environment exists but the project primarily uses auto-generated gaits. Hand-crafted reference motions through Blender could produce much more characterful walking - smoother weight transfers, more expressive body language, personality-specific gaits. This is where a robotics project becomes a character animation project.

### 9.7 Training Infrastructure
If you are going to iterate on policies, you will want GPU access. An RTX 5080 Mobile (16 GB) or RTX 4090 (24 GB) handles the full 8,192 environment training. A single run of 150M timesteps takes 2-4 hours on a laptop 5080 or 2-3 hours on a desktop 4090. Cloud GPUs (Vast.ai, RunPod) are $0.34-0.55/hour for an RTX 4090 if you prefer not to heat your laptop. You do not need datacenter hardware.

### 9.8 Head Motions - Three Approaches

The current policy controls only the 10 leg joints. The 4 head joints (neck pitch, head pitch, head yaw, head roll) are managed separately by simple direct commands. Antoine has explored integrating head control into the RL policy across five experimental branches (`head_in_policy` being the most developed). There are three approaches, in order of difficulty:

**Option A - Scripted head, independent of policy (easiest, no retraining)**
Keep the RL policy on legs only. Write a separate state machine in the runtime that layers idle animations (slow head bobbing, random antenna twitches) on top of teleoperation input. The `head_puppet.py` script in the runtime already demonstrates the servo interface and joint limits. The downside: the walking policy cannot compensate when the head moves and shifts the robot's centre of mass.

**Option B - Head position as observation, not action (medium, requires retraining)**
Add the current head joint angles to the policy's observation vector but keep head control external. The walking policy can then "feel" the weight shift when the head moves and compensate. Add head-position randomisation during training so the policy generalises to any head pose.

**Option C - Full policy-controlled head (hardest, most capable)**
The policy outputs 14 joint commands (10 leg + 4 head). Head target positions are passed as command inputs alongside velocity. The `head_in_policy` branch already implements this: separate imitation rewards for head and legs, configurable head joint ranges, and a `cost_head_pos` reward term (commented out at weight -2.0). Disney's paper weights neck tracking at 100x versus 15x for legs because character expression was their priority.

**Recommendation**: Start with Option A to get expressive motions working immediately. Move to Option C once comfortable with the training pipeline, using the `head_in_policy` branch as a starting point.

---

## 10. How Would I Actually Get Started?

This section is specifically for someone who wants to build one and start iterating. In rough order of priority:

### Step 1: Build the Hardware First
Before touching code, build the robot. You need the physical hardware to validate anything. Order of operations:

1. **Order parts**: start with the Bill of Materials in the main repo. Key items that take longest to ship: Feetech STS3215 servos (10 needed for legs), Raspberry Pi Zero 2W, BNO055 IMU breakout board, Xbox controller. Budget roughly $350-400 USD.
2. **3D print**: all PLA parts at 15% infill, the TPU foot pads at 40% infill. The leg sheets are the most fragile - print slowly and carefully.
3. **Configure servos before assembly**: each servo needs its ID assigned and zero position set BEFORE you install it. Use `scripts/configure_motor.py`. This cannot easily be undone after assembly.
4. **Calibrate per-robot offsets**: after assembly, run `scripts/find_soft_offsets.py` to find each joint's software zero offset. Save these in `duck_config.json`.

### Step 2: Get the Pre-Trained Policy Running
Before training anything yourself, get the existing policy working:

1. Flash Raspberry Pi OS Lite (64-bit) to an SD card
2. Install the runtime: `pip install open-duck-mini-runtime` (or clone and install from source)
3. Copy the pre-trained ONNX file from the main repo to the Pi
4. Run `scripts/turn_on.py` to verify servos respond
5. Run `scripts/v2_rl_walk_mujoco.py` and try the Xbox controller

The pre-trained policy is not perfect but should produce visible walking. If it does not, the most common problems are: IMU orientation wrong (check `imu_upside_down` in duck_config.json), servo offset miscalibration, or a servo ID conflict.

### Step 3: Explore in Simulation Before Retraining
Before committing to a full retraining run, use the simulation to understand the system:

1. Clone Open_Duck_Playground and install dependencies (`uv sync`)
2. Run the visualisation script to watch a policy in MuJoCo
3. Modify the reward function weights and observe what changes
4. This is free (no GPU needed for quick experiments, just for full training)

### Step 4: Your First Training Run
When ready to train:

1. You have an RTX 5080 Mobile (16 GB VRAM) which handles the full 8,192 parallel environments. If VRAM is tight, reduce `num_envs` to 4,096 (takes proportionally longer but produces identical quality)
2. The recommended command is: `uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 150000000`
3. Training takes 2-4 hours on a laptop 5080. Checkpoints are saved periodically and auto-exported to ONNX. Start with shorter 50M step runs (~30-60 min) when iterating on reward weights
4. Do not rely on the training reward to pick the best checkpoint. The `alive` term (weight 20.0) dominates, so a policy with a high training reward may simply be good at standing rather than walking. Use `analysis/find_best_checkpoint.py` to sweep all saved checkpoints and rank them by walking quality metrics instead
5. Copy the best ONNX file to the Pi and test

### Step 5: Iteration - What to Try First
In order of ease and expected improvement:

1. **Fix the gravity sensor indexing bug** in `mujoco_infer_base.py` - a one-line fix that corrects silently wrong orientation data
2. **Fix the accelerometer bias bug** (remove the +1.3 from the inference code, or add it to training)
3. **Enable the low-pass filter** in the runtime (`--cutoff_frequency 12` argument) - zero retraining, likely visible improvement
4. **Reduce leg Kp to 22** in the servo configuration - zero retraining, reduces jitter
5. **Increase the action rate penalty** to -1.5 in `joystick.py` and retrain - should noticeably smooth the gait
6. **Add the action acceleration penalty** at -0.45 - biggest single training change for smoothness. Add this alone first, not combined with other new penalties

**Dependencies**: install `playground==0.0.5` and `jax<0.7` explicitly to avoid compatibility issues that cause silent failures during training setup.

### What to Learn Along the Way
- **MuJoCo basics**: the official MuJoCo documentation + tutorials. Understanding how MJCF XML works helps enormously when adjusting the robot model.
- **JAX basics**: the training code is all JAX. Even a superficial understanding of `jax.vmap` (run a function across many inputs simultaneously) and `jax.jit` (compile a function for fast repeated execution) helps.
- **RL fundamentals**: the Spinning Up in Deep RL guide from OpenAI (free online) covers PPO clearly. You do not need deep maths - the conceptual understanding of reward shaping is what matters here.
- **Servo tuning**: understanding PID control (Proportional-Integral-Derivative) for servo position control. There are good YouTube videos on this. The "P" gain is what you adjust most for this robot.

---

## 11. Glossary

| Term | Definition |
|------|-----------|
| **Action** | The output of the policy at each timestep - servo position commands |
| **Action rate** | How much actions change between consecutive timesteps. "Action rate penalty" deducts score for large sudden changes |
| **Action acceleration / jerk** | How much the rate of change itself changes. Penalising this prevents zig-zagging commands even when each individual change is small |
| **Asymmetric actor-critic** | Training setup where the policy (actor) sees realistic sensor observations but the value estimator (critic) sees perfect ground-truth data. Like a student who gets the answers during study but not during the exam |
| **BAM** | Behavioural Actuator Modelling - Rhoban's framework for physically testing servo dynamics and exporting the measurements to the simulator |
| **Backdrivable** | A joint that can be pushed by external force without resistance. Quasi-direct-drive joints are backdrivable; high-gear-ratio servo joints are not |
| **Brax** | Google's JAX-based physics and RL library that provides the PPO implementation |
| **Critic** | The value function network - estimates how much total future reward the policy will accumulate from the current state. Only used during training, discarded at deployment |
| **Domain randomisation** | Varying simulation parameters randomly during training so the policy generalises to real-world variation |
| **Episode** | One "life" of the simulated robot, from reset to termination (fall or time limit) |
| **Exponential reward form** | Using exp(-error) instead of -error. Gives strong gradient signal near the target ("you're getting warmer") and weaker signal far away |
| **Forward Kinematics (FK)** | Calculate where the end of a chain of joints ends up given the joint angles |
| **Gamma (γ)** | Discount factor - how much the policy values future rewards vs immediate ones. 0.97 means a reward 33 steps away is worth about 37% of an immediate reward |
| **IMU** | Inertial Measurement Unit - sensor providing gyroscope (rotation rate) and accelerometer (linear acceleration) data |
| **Imitation reward** | Reward for matching a pre-computed reference motion |
| **Inverse Kinematics (IK)** | Calculate what joint angles are needed to place the end of a chain at a specific position |
| **JAX** | Google's framework for GPU-accelerated numerical computing |
| **Low-pass filter** | A signal processing filter that allows slow changes through while smoothing out rapid fluctuations. Like a shock absorber for signals |
| **LCP** | Lipschitz-Constrained Policy - a training technique that constrains how rapidly the policy output can change, enforcing smoothness at the network level |
| **MJCF** | MuJoCo's XML format for describing robots and environments |
| **MJX** | JAX-accelerated MuJoCo - runs physics on GPU |
| **MLP** | Multi-Layer Perceptron - a simple neural network of stacked linear layers with non-linearities |
| **ONNX** | Open Neural Network Exchange - portable file format for trained neural networks |
| **Observation** | The input to the policy at each timestep - sensor readings and state information |
| **Phase** | A number cycling from 0 to 1 representing where in the gait cycle the robot is (0 = start of step, 1 = end of step) |
| **Placo** | A whole-body inverse kinematics solver from Rhoban that generates reference walking trajectories |
| **Policy** | The trained neural network that maps sensor observations to joint commands - the robot's "brain" |
| **PPO** | Proximal Policy Optimisation - the RL algorithm used to train the policy |
| **Quasi-direct-drive** | Motor actuator with very little gear reduction. Gives high backdrivability, smooth contact dynamics, and consistent torque across speed range. The opposite of high-gear-ratio hobby servos |
| **Reward shaping** | Designing reward function terms to encourage specific behaviours |
| **Sim-to-real** | The process of transferring a simulation-trained policy to a physical robot |
| **Stiction** | Static friction - the extra force needed to start moving a joint from rest. Causes "stick-slip" jerky motion in geared servos |
| **Swish** | An activation function used in the neural network layers: swish(x) = x × sigmoid(x). Smoother than the more common ReLU |
| **Zero-shot** | A trained policy that works on real hardware on the first try, without any additional fine-tuning on real data |

---

## Directory Structure

```
open_duck_mini_research/
├── Open_Duck_Mini/                    # Hardware, CAD, MJCF models, assembly docs
├── Open_Duck_Playground/              # RL training environment (JAX/MJX/PPO)
│   └── analysis/                     # Evaluation and diagnostic tools
│       ├── evaluate_policy.py         # Comprehensive eval: 13 cold-start + 14 stand-then-command + push recovery + gait symmetry
│       ├── quick_walking_eval.py      # Lightweight eval for live TensorBoard during training callbacks
│       ├── find_best_checkpoint.py    # Sweep all checkpoints, rank by walking quality not training reward
│       ├── analyse_training.py        # Read TensorBoard events for training dynamics
│       ├── debug_policy.py            # Per-step velocity traces
│       └── replay_reference.py        # Visual playback of reference motions in MuJoCo viewer
├── Open_Duck_Mini_Runtime/            # Raspberry Pi deployment code
├── Open_Duck_reference_motion_generator/  # Placo gait generation + polynomial fitting
├── Open_Duck_Blender/                 # Blender rigging and animation
└── docs/
    ├── 00_master_overview.md          # This file
    ├── 01_training_pipeline.md        # Deep dive: RL training code
    ├── 02_runtime_deployment.md       # Deep dive: real robot deployment
    ├── 03_reference_motions.md        # Deep dive: gait generation
    ├── 04_hardware_simulation_model.md # Deep dive: MJCF model and hardware
    ├── 05_papers_analysis.md          # Disney BDX + MuJoCo Playground papers
    └── 06_smoothness_improvements.md  # Techniques for better motion quality
```

## Forked Repos

All repos forked under `calebjakemossey`:
- https://github.com/calebjakemossey/Open_Duck_Mini
- https://github.com/calebjakemossey/Open_Duck_Playground
- https://github.com/calebjakemossey/Open_Duck_Mini_Runtime
- https://github.com/calebjakemossey/Open_Duck_reference_motion_generator
- https://github.com/calebjakemossey/Open_Duck_Blender
