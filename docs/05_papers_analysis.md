# Papers Analysis: Disney BDX and MuJoCo Playground

This document analyses the two foundational papers for the Open Duck Mini project, with a focus on what is directly relevant to improving motion smoothness on a low-cost bipedal robot.

---

## Part 1: Disney BDX - "Design and Control of a Bipedal Robotic Character" (arXiv:2501.05204)

### Overview

The paper describes the full engineering stack behind BDX, a 15.4 kg bipedal animatronic robot designed for theme-park entertainment. The core contribution is demonstrating that a character-driven animation workflow and a reinforcement-learning locomotion policy can be unified into a single real-time pipeline that survives physical disturbances while preserving expressive motion intent.

---

### 1. Control Architecture

The architecture has three layers:

**Layer 1 - Animation content:** An animator-authored motion database is the ground truth. All reference states originate here.

**Layer 2 - Runtime composition:** A three-layer blending system produces a target state `x_t` at each timestep:
- Background layer: continuous looped base motion.
- Triggered layer: blends in discrete animations (e.g., a head nod) using facial blend time T_beta = 0.1 s and body blend time T_alpha = 0.35 s.
- Joystick layer: applies velocity commands to modify the blended target via a parametric walk engine.

**Layer 3 - RL policy:** A neural network tracks the composed target state. The policy runs at 50 Hz and outputs joint position setpoints for PD controllers. A microcontroller interpolates these setpoints to 600 Hz using a first-order hold, then applies a 37.5 Hz low-pass filter before sending to actuators.

**Policy structure:**

Three policy types are trained independently:
- Perpetual: `pi(a_t | s_t, g^perp_t)` - standing/idle balance
- Periodic: `pi(a_t | s_t, phi_t, g^peri_t)` - cyclic walking with phase signal
- Episodic: `pi(a_t | s_t, phi_t)` - time-bounded sequences

Policy network: 3 fully connected hidden layers, 512 units each, ELU activations, PPO training.

> **ELU (Exponential Linear Unit)** is an 'activation function' - a mathematical curve applied between neural network layers that gives the network its ability to learn complex patterns. Open Duck uses 'swish' (also called SiLU) instead. Both serve the same purpose; the choice between them has a subtle effect on training dynamics but does not significantly change the behaviour of the final policy.

Policy input `s_t`:
```
s_t = (p^P_t, theta^P_t, v^T_t, omega^T_t, q_t, q_dot_t, a_{t-1}, a_{t-2})
```
where `P` denotes path frame (position/orientation relative to desired trajectory), `T` denotes torso/body frame for velocities, and two previous actions provide temporal context. Note the path frame representation: position and orientation errors are measured relative to the reference trajectory, not the world frame. This helps the policy generalise across spatial positions.

> **The 'path frame' (also called 'local frame')** is a coordinate system that moves along with the desired trajectory. Position and orientation errors are measured relative to where the robot should be, not where it is in the world. This helps the policy generalise: 'I am 2 cm behind my target' means the same thing whether the robot is at the start or end of its course.

---

### 2. Imitation Reward - Exact Mathematical Formulation

This is the core contribution for motion quality. The total reward is:

```
r_t = r^imitation_t + r^regularisation_t + r^survival_t
```

#### Imitation (tracking) component:

Each term tracks a corresponding element of the target state `x_hat_t = (p_hat_t, theta_hat_t, v_hat_t, omega_hat_t, q_hat_t, q_dot_hat_t, c^L_t, c^R_t)`:

| Term | Formula | Weight |
|------|---------|--------|
| Torso xy position | `exp(-200.0 * ||p_{x,y} - p_hat_{x,y}||^2)` | 1.0 |
| Torso orientation | `exp(-20.0 * ||theta ⊟ theta_hat||^2)` | 1.0 |
| Linear velocity xy | `exp(-8.0 * ||v_{x,y} - v_hat_{x,y}||^2)` | 1.0 |
| Linear velocity z | `exp(-8.0 * (v_z - v_hat_z)^2)` | 1.0 |
| Angular velocity xy | `exp(-2.0 * ||omega_{x,y} - omega_hat_{x,y}||^2)` | 0.5 |
| Angular velocity z | `exp(-2.0 * (omega_z - omega_hat_z)^2)` | 0.5 |
| Leg joint positions | `-||q_l - q_hat_l||^2` | 15.0 |
| Neck joint positions | `-||q_n - q_hat_n||^2` | 100.0 |
| Leg joint velocities | `-||q_dot_l - q_dot_hat_l||^2` | 1.0e-3 |
| Neck joint velocities | `-||q_dot_n - q_dot_hat_n||^2` | 1.0 |
| Contact state | `sum_{i in {L,R}} I[c_i == c_hat_i]` | 1.0 |

The `⊟` operator is the quaternion difference (geodesic distance), not element-wise subtraction.

> **Quaternions** are a mathematical way of representing 3D rotations without the problems that simpler methods have (like 'gimbal lock', where you lose a degree of freedom). The 'difference' between two quaternions is the rotation you'd need to apply to go from one orientation to the other. 'Geodesic distance' means measuring the shortest possible rotation between them - the most natural measure of 'how different are these two orientations?'

Key design choices to understand:
- **Exponential form for velocity and orientation terms**: ensures the reward is always positive (bounded 0-1) and is most sensitive near the target, providing strong gradient information when the policy is close to the reference.
- **Squared-error (not exp) for joint positions**: this is negative (a cost), weighted heavily at 15.0 for legs and 100.0 for neck. The neck weight being 6.7x higher than legs reflects the character-expressiveness priority - head motion fidelity matters more for animation than leg aesthetics.
- **Very low weight on joint velocities (1e-3 for legs)**: leg velocity tracking is almost noise. Only velocity smoothness matters, not exact velocity matching.
- **Survival reward**: flat reward of 20.0 per step for not falling, dominating short-episode failures.

#### Regularisation component:

| Term | Formula | Weight |
|------|---------|--------|
| Joint torques | `-||tau||^2` | 1.0e-3 |
| Joint accelerations | `-||q_ddot||^2` | 2.5e-6 |
| Leg action rate | `-||a_l - a_{t-1,l}||^2` | 1.5 |
| Neck action rate | `-||a_n - a_{t-1,n}||^2` | 5.0 |
| Leg action acceleration | `-||a_l - 2*a_{t-1,l} + a_{t-2,l}||^2` | 0.45 |
| Neck action acceleration | `-||a_n - 2*a_{t-1,n} + a_{t-2,n}||^2` | 5.0 |

The action rate and action acceleration penalties are critical for smoothness. They penalise both first and second derivatives of the action sequence, suppressing high-frequency jitter. The neck weights are again 3-10x higher than legs, reinforcing head smoothness.

---

### 3. Animation Engine - How Motions are Composed

Disney uses a parametric walk engine that generates kinematically consistent reference states as a function of velocity commands. This is not mocap replay - it is procedurally generated motion, parameterised by `(v^P_t, omega^P_t)` (planar velocity and yaw rate).

At each control step the walk engine outputs the full target state `x_hat_t`:
- Root position and orientation (in path frame)
- Root linear and angular velocity
- All joint positions and velocities
- Foot contact states (binary, left/right)

The three-layer composition allows show elements (triggered animations) to be blended on top of locomotion without retraining. At runtime the joystick layer modifies the walk engine velocity parameters, and the RL policy then executes whatever target state the animation engine emits.

This is a clean separation of concerns: animators own the reference motion, engineers own the tracking policy.

---

### 4. Sim-to-Real Transfer

Disney's approach combined:

**Actuator modelling via system identification:** First-principles actuator models were fitted to experimental data. The key parameters exported to MuJoCo are: damping, kp (stiffness), frictionloss, armature, and forcerange. Without this step the sim-to-real gap in torque response makes trained policies ineffective.

**Domain randomisation across:**
- Floor friction: varied per episode
- Actuator parameters: coefficients randomised within experimentally-measured ranges
- Link masses: scaled per episode
- Torso centre-of-mass position: jittered
- Initial joint positions: jittered around nominal
- Sensor noise: IMU, encoder noise added to observations
- External disturbances: random forces applied to torso, hips, head, feet during training

**Deployment stack:**
- Policy inference at 50 Hz (onboard PC)
- PD controllers at 600 Hz (microcontroller)
- 37.5 Hz low-pass filter on joint targets before sending to actuators
- State estimation: IMU + encoder fusion

The low-pass filter on joint targets is a key smoothing mechanism. It prevents the 50 Hz policy outputs from creating 50 Hz mechanical vibration in 600 Hz actuators.

---

### 5. Hardware Design Insights

**Why BDX locomotion works well:**

- **Quasi-direct drive legs:** Leg actuators are quasi-direct drive (low gear ratio), giving them high backdrivability and low reflected inertia. This makes contact dynamics smooth and greatly reduces the sim-to-real gap in impact forces.

  > **Backdrivability**: can you push the joint and make it move freely? A backdrivable joint lets external forces (like the ground pushing on the foot) move the joint. A non-backdrivable joint (like a high-gear-ratio servo) resists external forces - the motor locks the joint in place. 'Reflected inertia' is related: the motor's rotor has inertia (resistance to acceleration), and through the gears, this inertia appears multiplied at the output (by the gear ratio squared). A 345:1 gear ratio means the reflected inertia is 345² = ~119,000 times the rotor inertia. This makes the joint feel 'heavy' and causes large force spikes during foot impacts.
- **High torque margin:** 34 N·m peak at the hip/knee, on a 15.4 kg robot with 28 cm nominal leg length. Torque-to-weight is generous, allowing the policy to handle perturbations without saturating.
- **5 DOF per leg:** Hip yaw + roll + pitch, knee, ankle pitch. This is sufficient for full 3D locomotion with separate yaw steering. Many smaller robots omit hip roll or ankle, reducing balance capability.
- **High control bandwidth:** 600 Hz microcontroller loop with interpolation. The policy outputs smooth position setpoints; the microcontroller handles the high-frequency torque loop. This decoupling is what makes 50 Hz RL policies viable on physical hardware.
- **IMU + encoder state estimation at policy frequency:** No vision, no external tracking - fully self-contained. The IMU provides gravity direction and angular velocity; encoders provide joint angles. These are the same signals available on the Open Duck Mini.

---

### 6. Training Details

- 100,000 PPO iterations per policy
- ~2 days per policy on a single NVIDIA RTX 4090
- Preview-quality behaviour visible after ~1,500 iterations (~30 minutes)
- Action scale: outputs are joint position offsets from a nominal pose
- Phase signal for periodic motions: sine/cosine encoding of normalised phase, giving the policy temporal awareness of where it is in the gait cycle

---

## Part 2: MuJoCo Playground - "MuJoCo Playground" (arXiv:2502.08844)

### Overview

MuJoCo Playground is a GPU-accelerated RL training framework built on MJX (MuJoCo XLA), the JAX port of MuJoCo. It enables the entire agent-environment loop to run on-device through JAX's `vmap` and `jit`, eliminating Python/GPU data transfer overhead.

---

### 1. Framework Architecture

Three components:
1. **MJX physics:** JAX-native MuJoCo. Statically shaped computation graphs - all contact pairs are evaluated every step, not just active ones. This is a JAX constraint (static shapes required for JIT compilation) that adds constant overhead proportional to the total number of possible contacts, not active contacts.
2. **Brax PPO / RSL-RL:** Training algorithms. Open Duck Playground uses Brax's PPO implementation.
3. **JAX end-to-end execution:** Policy network, environment step, and reward computation all run as compiled JAX kernels. No Python overhead per step once compiled.

Domain randomisation is applied via `jax.vmap` over a batch of model parameter vectors - each environment in the batch has independently randomised physics parameters.

---

### 2. Training Speed

On a single A100 GPU:
- Go1 joystick flat terrain: ~417,000 PPO steps/second, full training in 5 minutes
- Berkeley Humanoid joystick: ~120,000 steps/second, full training in 15 minutes
- Unitree G1 / Booster T1 (humanoid): under 30 minutes

On the hardware Open Duck uses (2x RTX 4090, from runner.py which trains to 150M timesteps):
- At ~120k steps/second throughput, 150M steps takes roughly 20-25 minutes
- JIT compilation adds 1-3 minutes per training run

The Open Duck Playground runner (`runner.py`) targets 150,000,000 timesteps using Berkeley Humanoid's PPO config - this is the same config used to train humanoid policies in under 15 minutes on comparable hardware.

---

### 3. Sim-to-Real Transfer Results

Quadruped (Unitree Go1), zero-shot transfer:
- Joystick control, handstand, footstand, fall recovery - all transferred without fine-tuning
- Handles uneven terrain and moderate external perturbations

Humanoid (Berkeley Humanoid, G1, Booster T1):
- Stable joystick walking indoors, zero-shot
- Training time: 15-30 minutes per platform on 2x RTX 4090

The framework's transfer quality stems from:
- Accurate actuator modelling (BAM-derived parameters in MJCF)
- Per-episode physics randomisation via vmap
- Observation noise injection (sensor delays, IMU noise)
- Action delay randomisation (0-3 steps)

---

### 4. Deployment Stack

Real-world control: policy inference via ONNX Runtime at 50 Hz, hardware interface at 500-2000 Hz. The Open Duck Mini runtime (`v2_rl_walk_mujoco.py`) exactly mirrors this: ONNX inference at 50 Hz, Feetech STS3215 servos via serial.

---

## Part 3: Other Referenced Work in the Open Duck Ecosystem

The codebase references several additional works:

**AWD (Agile and Versatile Walking)** - `https://github.com/rimim/AWD`
An Isaac Gym-based training framework that the reference motion generator also supports. The Open Duck project maintains compatibility with both Playground and AWD policies.

**Placo** (`https://github.com/Rhoban/placo`) - Rhoban's whole-body controller and inverse kinematics library. The reference motion generator (`Open_Duck_reference_motion_generator`) uses Placo's walk engine to generate kinematically consistent reference trajectories, parameterised by (dx, dy, dtheta) velocity commands.

**BAM (Behaviour-Aware Motor identification)** - `https://github.com/Rhoban/bam`
Used to characterise the Feetech STS3215 servo at 7.4 V. Exports damping, kp, frictionloss, armature, and forcerange to MuJoCo units. The identified parameters are at `https://github.com/Rhoban/bam/tree/main/params/feetech_sts3215_7_4V`. This is the single most critical sim-to-real enabler in the project.

**onshape-to-robot** (`https://github.com/Rhoban/onshape-to-robot`) - Converts the Onshape CAD model to URDF/MJCF, with per-part mass overrides from slicer infill estimates.

**Berkeley Humanoid** - The MuJoCo Playground Berkeley Humanoid joystick task is the direct template for Open Duck Playground (`joystick.py` carries DeepMind + Antoine Pirrone/Steve Nguyen copyright notices and comments like "based on Berkeley Humanoid").

---

## Part 4: What Open Duck Mini CANNOT Replicate

This section identifies the structural gaps between the BDX paper's approach and what is achievable on a ~$400 3D-printed robot with Feetech STS3215 serial bus servos.

### 4.1 Quasi-Direct Drive vs. Geared Servos

**The gap:** BDX uses quasi-direct drive actuators (34 N·m peak, high backdrivability) for the legs. The Open Duck Mini uses Feetech STS3215 smart servos - position-controlled, geared hobby servos with significant gear reduction, high reflected inertia, and low backdrivability.

**Why this matters:**
- High reflected inertia means impacts from foot contact generate large force spikes that are transmitted through the drivetrain. Soft contact dynamics in simulation do not reproduce this.
- Non-backdrivable gears mean the joint resists being moved by external forces, creating a mechanical impedance that is hard to model. The BAM identification captures steady-state behaviour but not transient impact dynamics.
- The policy learns to output smooth position trajectories, but the geared servo must execute them under varying load. At the servo's velocity limits (~5.24 rad/s in the config), the servo simply cannot follow the commanded position, creating tracking error and jitter.

**Consequence for motion smoothness:** Servo velocity limits are hard constraints. The joystick.py config applies `max_motor_velocity = 5.24 rad/s` clamping on motor targets per step, but this is only enforced in sim. On hardware, if the policy commands a position change faster than the servo can achieve, the servo lags and the robot lurches. This is a primary source of jerky motion.

### 4.2 No High-Bandwidth Torque Loop

**The gap:** BDX runs a 600 Hz microcontroller loop that interpolates policy outputs between 50 Hz inference steps, with a 37.5 Hz low-pass filter on joint targets. The Open Duck Mini runs at 50 Hz end-to-end via a serial bus (Rustypot/Feetech protocol). There is no intermediate interpolation layer.

**Consequence:** The Open Duck runtime sends new position targets at 50 Hz directly to servos. Any discontinuity in policy output appears as a 50 Hz step input to the servo's internal PD controller. The BDX low-pass filter, running at 37.5 Hz on a 600 Hz loop, effectively smooths within the 20 ms window between policy steps. Open Duck has a `LowPassActionFilter` class in `rl_utils.py` that can be applied (controlled by `--cutoff_frequency` argument), but it is not enabled by default - this is a significant missed opportunity.

### 4.3 Onboard Compute and Inference Latency

**The gap:** BDX runs policy inference on an onboard PC with unspecified but substantial compute. The Open Duck Mini runs on a Raspberry Pi Zero 2W with 512 MB RAM and a 1 GHz ARM Cortex-A53.

**Consequence:** Policy inference on Pi Zero 2W via ONNX Runtime takes measurably more than the 20 ms budget. The runtime code explicitly checks for budget overruns and prints warnings. If inference takes 15 ms and serial communication takes 8 ms, the robot is already over budget. This forces the trained policy to be small (3 hidden layers but network width is not specified in the Playground config - it inherits Berkeley Humanoid defaults which are typically 256 units, not 512 as in BDX).

**Note:** The Pi Zero 2W has no dedicated NPU or GPU acceleration. ONNX Runtime runs on CPU only.

### 4.4 No Ankle Compliance or Series Elastic Elements

**The gap:** BDX has quasi-direct drive ankles that provide inherent compliance through the motor's low gear ratio. Series elastic actuators, if present, would provide additional shock absorption.

**Consequence:** The Open Duck Mini ankle is a geared servo with no compliance. Ground contact forces produce step inputs to the ankle joint angle, which the servo must resist rigidly. In simulation the contact model is smooth (compliant floor or small timestep integration); on hardware the impact is sharp. The policy never learns to handle this because the sim-to-real gap in impact dynamics is not captured by mass and friction randomisation alone.

### 4.5 Character Expression Pipeline

**The gap:** BDX has a full three-layer animation composition system with animator-authored content, triggered animations, and joystick control. This requires professional animation tooling, a character-specific walk engine, and an independent policy for each motion category.

**What Open Duck does instead:** The reference motion generator uses Placo's walk engine to generate procedurally parameterised walking motions and fits polynomial approximations to them. This produces a `polynomial_coefficients.pkl` file that the `PolyReferenceMotion` class queries at runtime. The polynomial fit is a practical approximation - it trades exact motion fidelity for a compact, differentiable representation that can be queried during GPU-parallelised training.

**Consequence:** The reference motions are analytically smooth (polynomial interpolation guarantees C-infinity smoothness) but they are generic walking gaits, not character-specific animations. There is no equivalent of BDX's triggered animation layer or background character motion. The "expression" features (LEDs, speaker, antennas) are purely cosmetic and not integrated into the motion policy.

### 4.6 Sensor Suite Limitations

**The gap:** BDX likely uses high-quality, low-latency IMU sensors and high-resolution encoders with negligible communication delay. The Open Duck Mini reads IMU data via I2C and joint positions via serial bus at 50 Hz.

**Consequence:** The joystick.py config randomises action delay (0-3 steps) and IMU delay (0-3 steps) during training. At 50 Hz, 3 steps is 60 ms of latency. This is a worst-case assumption that the policy learns to be robust to, but it means the policy is inherently conservative rather than using fresh state estimates. The foot contact sensors (`FeetContacts` in the runtime) add binary contact information, which helps with gait phase estimation but is noisier than force plates.

### 4.7 Weight and Torque-to-Weight Ratio

**The gap:** BDX weighs 15.4 kg with 34 N·m peak hip/knee torque. Rough torque-to-weight per leg: 34 / (15.4/2) = ~4.4 N·m/kg. The Open Duck Mini weighs approximately 0.5-0.8 kg total (42 cm height, 3D-printed PLA). The STS3215 servo has a stall torque of approximately 1.8 N·m (at 7.4 V). Torque-to-weight per leg at the hip: 1.8 / (0.3) = ~6 N·m/kg.

The ratio is actually comparable, but the servo reaches stall torque at very low speeds, and dynamic torque (under high-speed motion) drops significantly. At the 5.24 rad/s velocity limit used in training, a typical geared servo produces less than 50% of stall torque. BDX's quasi-direct drive maintains high torque across its full velocity range.

### 4.8 What Actually Transfers Well

Despite the gaps above, the following elements of the BDX approach transfer directly:

- **Imitation reward structure:** The Open Duck `custom_rewards.py` is a faithful adaptation of the BDX reward table. The weights (15.0 for joint positions, 1e-3 for joint velocities, exponential velocity terms) are copied almost verbatim. This is the correct approach.
- **Phase signal as observation:** The Open Duck policy receives `[cos(phi), sin(phi)]` as part of its observation, exactly as in the BDX periodic policy formulation.
- **Domain randomisation:** The `randomize.py` implementation covers the same parameter categories as BDX: friction, frictionloss, armature, mass, COM position, initial joint position, and KP. This is comprehensive.
- **ONNX export for deployment:** Policies are exported to ONNX and run via ONNX Runtime, a clean sim-to-real deployment boundary.
- **Polynomial reference motion:** Using Placo as the walk engine and fitting polynomials is a sound engineering choice. The polynomial representation is JAX-compatible (vectorised via `vmap` over `jp.polyval`) and provides smooth, analytically differentiable reference trajectories.

---

## Part 5: Implications for Motion Smoothness

The primary sources of roughness on the Open Duck Mini, ranked by impact:

1. **Missing low-pass filter on action output (highest impact):** The `LowPassActionFilter` in `rl_utils.py` exists but is disabled by default. Enabling it with a cutoff frequency of ~15-20 Hz (below the 25 Hz Nyquist of the 50 Hz control loop, aggressive enough to smooth inter-step discontinuities) is likely the single largest improvement available with zero retraining. The BDX approach uses 37.5 Hz on a 600 Hz loop - equivalent to ~3 Hz on a 50 Hz loop after accounting for the interpolation. A 10-15 Hz cutoff on the 50 Hz loop is a reasonable starting point.

   > **Nyquist's theorem** says you cannot accurately represent frequencies above half your sampling rate. At a 50 Hz control rate, the highest meaningful frequency in the output is 25 Hz. Anything above 25 Hz is noise or aliasing artefacts. A 15 Hz low-pass filter therefore keeps all the meaningful motion content while cutting out the noise.

2. **Action rate penalty weight:** The Open Duck config uses `-0.5` for `action_rate`. The BDX equivalent (leg action rate) uses `1.5` - three times higher. Increasing the action rate penalty during training will produce a smoother policy at the cost of some velocity tracking performance. The BDX paper also adds an action acceleration penalty (`1.5` for legs) which the Open Duck config does not currently use.

3. **Servo velocity clamping in sim vs. hardware:** In sim the motor velocity limit is enforced as a soft clip on motor target delta (`max_motor_velocity * dt`). On hardware the servo's own velocity limiter cuts in hard. Reducing `action_scale` (currently 0.25) reduces the magnitude of position steps per inference step, effectively reducing required servo velocity. This trades agility for smoothness.

4. **Phase frequency factor:** The runtime has a `phase_frequency_factor` that can be tuned via controller buttons. Reducing the phase frequency slows the gait reference, which may better match what the servo hardware can actually execute smoothly.

5. **Imitation reward gap - orientation term:** The Open Duck `custom_rewards.py` comments out the torso orientation reward (`torso_orientation_rew`) with "TODO ignore yaw here". The BDX formulation includes this term with weight 1.0. Restoring it (using proper quaternion difference rather than element-wise squared error) would improve torso stability tracking, which directly affects visual smoothness.

6. **Reference motion quality:** The Placo walk engine generates reference trajectories with configurable parameters (CoM height, foot clearance, single-support duration, trunk pitch). Tuning these parameters to better match the physical robot's actual gait (lower CoM height, shorter single-support phase to reduce falling time) would give the imitation reward more achievable targets and reduce policy-hardware gap.
