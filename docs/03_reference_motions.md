# Reference Motion Generation

Sources analysed:
- `/home/lakieb/Documents/open_duck_mini_research/Open_Duck_reference_motion_generator/`
- `/home/lakieb/Documents/open_duck_mini_research/Open_Duck_Blender/`

---

## 1. Motion Generation Pipeline

### What is Placo?

Placo is a whole-body kinematics and walking pattern generation library developed by the Rhoban team at University of Bordeaux. The project depends on `placo==0.6.3`. Placo provides:

- `placo.HumanoidRobot` - loads a URDF and exposes forward/inverse kinematics
- `placo.KinematicsSolver` - a Quadratic Programming (QP) solver for inverse kinematics. Quadratic Programming (QP) is a mathematical optimisation technique. In plain terms: given constraints (joint limits, balance requirements, foot placement targets), it finds the best joint angles that satisfy all the constraints simultaneously. Think of it as a very fast equation solver for "what joint angles put the foot HERE while keeping balance?"
- `placo.WalkTasks` - pre-built task set for bipedal walking (CoM height, foot placement, trunk orientation)
- `placo.FootstepsPlannerRepetitive` - generates a repeating sequence of footsteps given step displacements (dx, dy, dtheta)
- `placo.WalkPatternGenerator` - produces a continuous CoM and foot-trajectory from a footstep plan, using a Zero Moment Point (ZMP) model. ZMP (Zero Moment Point) is the point on the ground where the robot's weight effectively pushes down. If this point stays within the area bounded by the feet (the "support polygon"), the robot will not tip over. Placo uses ZMP calculations to generate walking trajectories that are guaranteed to be physically stable.
- `placo.HumanoidParameters` - struct holding all walking tuning parameters

Placo solves for joint angles at every timestep such that the robot's centre-of-mass follows the planned ZMP-stable trajectory and the feet track their planned swing arcs. This is model-based trajectory generation, not simulation - the output is kinematically consistent but not dynamically validated.

### End-to-end Pipeline

```
auto_gait.json
    |
    v
auto_waddle.py          <- orchestrator; generates tmp preset files and spawns workers
    |
    |-- (parallel) gait_generator.py   <- runs Placo, records frames, saves .json
    |                |
    |                v
    |          PlacoWalkEngine (placo_walk_engine.py)
    |                |
    |                | tick() at DT=0.001s, record at FPS=50
    |                v
    |          recordings/<id>_<x_vel>_<y_vel>_<theta_vel>.json
    |
    v
fit_poly.py             <- reads all recordings, fits degree-15 polynomials per gait
    |
    v
polynomial_coefficients.pkl   <- used at training time by PolyReferenceMotion
```

### Step-by-step Description

1. **Parameter selection** (`auto_waddle.py`): Reads `auto_gait.json` which defines sweep ranges for dx, dy, and dtheta. In sweep mode, the Cartesian product of all values is enumerated (yielding 210 motions for `open_duck_mini_v2`). In random mode, `--num` samples are drawn uniformly.

2. **Preset construction**: Each combination is written into a temporary JSON file in `placo_presets/tmp/`, derived from the `medium.json` preset. The dx/dy/dtheta values are injected into this preset.

3. **Gait recording** (`gait_generator.py`): A `PlacoWalkEngine` is instantiated with the preset parameters. The engine is ticked at 1 ms intervals (`DT=0.001`) for a warmup period (negative initial time), then frames are sampled at 50 Hz. Each frame is the full state vector described in section 3.

4. **Velocity measurement**: After skipping the first 2 seconds to allow the gait to settle, the mean forward, lateral, and yaw velocities are computed and stored in the file metadata. The file is named `<id>_<x_vel>_<y_vel>_<theta_vel>.json`.

5. **Quality filter**: `auto_waddle.py` post-processes the output directory and deletes files whose measured speed does not fall within the expected band for the preset speed tier (slow/medium/fast).

6. **Polynomial fitting** (`fit_poly.py`): One period of motion is extracted from each recording. A degree-15 polynomial is fitted independently to each output dimension as a function of normalised phase t in [0, 1]. Coefficients are stored in a serialised file indexed by `dx_dy_dtheta`.

### Internal Placo Walk Loop

`PlacoWalkEngine.tick(dt)` runs the following per timestep:

- Updates QP tasks from the planned trajectory at the current time
- Calls `robot.update_kinematics()` and `solver.solve(True)`
- If enough time has elapsed and the trajectory can be replanned, replans footsteps and CoM trajectory
- The inner loop is refined by `REFINE=10` sub-steps per dt for numerical stability (`solver.dt = DT/REFINE = 0.0001`)

---

## 2. Polynomial Fitting

### What is Fitted

Each motion recording is reduced to one gait cycle. The extractor (`fit_poly.py`) slices out the following signals from the frame data and concatenates them into a matrix `Y` of shape `[nb_steps_in_period, 39]`:

| Slice | Contents | Dimensions |
|-------|----------|------------|
| `joints_pos` | Joint positions (rad) for all 16 joints | 16 |
| `joints_vel` | Joint velocities (rad/s) for all 16 joints | 16 |
| `foot_contacts` | Binary left/right contact flags | 2 |
| `base_linear_vel` | World-frame linear velocity of root (m/s) | 3 |
| `base_angular_vel` | World-frame angular velocity of root (rad/s) | 3 |

**Total: 39 dimensions**

### Fitting Method

A degree-15 polynomial is fitted to each dimension independently using `numpy.polyfit`. A polynomial is a smooth mathematical curve (like y = ax² + bx + c, but with more terms). "Degree 15" means it has 16 coefficients and can represent quite complex shapes. The reason for using polynomials: instead of storing 50 separate data points (one per animation frame), you store just 16 numbers that can reproduce the entire curve with high accuracy. It is a compact way to encode a smooth motion. Time is normalised to t in [0, 1] over one period. The 16 coefficients per dimension are stored in ascending power order (`np.flip` is applied after `polyfit`).

### RankWarnings During Fitting

Running `fit_poly.py` on a full 210-motion sweep produces `numpy.RankWarning: Polyfit may be poorly conditioned` warnings for many dimensions. These are normal and expected when fitting degree-15 polynomials across a large number of recordings - they do not indicate a problem with the output. The coefficients produced are usable.

### Polynomial Evaluation at Runtime

`PolyReferenceMotion.get_reference_motion(dx, dy, dtheta, i)`:
1. Maps (dx, dy, dtheta) to the nearest index in the data cube using `argmin` on the discrete axis arrays.
2. Computes `t = (i % nb_steps_in_period) / nb_steps_in_period`.
3. Evaluates all 39 polynomials at t using `numpy.polyval`.
4. Returns a 39-element vector - the reference state for that phase.

This means at training time the entire motion library is compressed to a 3D lookup of polynomial coefficient sets. Interpolation between motion files is not performed; the nearest-neighbour gait is selected.

### Why Polynomials?

The polynomial representation compactly encodes one full gait cycle and avoids storing large frame arrays in memory during training. The degree-15 polynomial captures the approximately sinusoidal joint trajectories without overfitting.

---

## 3. Motion File Format

Each recording is a JSON file. Filename convention: `<id>_<x_vel>_<y_vel>_<theta_vel>.json`

### Top-level Fields

| Field | Type | Description |
|-------|------|-------------|
| `LoopMode` | string | Always `"Wrap"` - the animation loops |
| `FPS` | int | Frames per second (50 for MuJoCo Playground, 30 for AWD) |
| `FrameDuration` | float | 1/FPS, rounded to 4 decimal places |
| `EnableCycleOffsetPosition` | bool | `true` - root position advances each cycle |
| `EnableCycleOffsetRotation` | bool | `false` |
| `Joints` | list[str] | Ordered list of 16 joint names |
| `Vel_x` | float | Measured mean forward velocity (m/s) |
| `Vel_y` | float | Measured mean lateral velocity (m/s) |
| `Yaw` | float | Measured mean yaw velocity (rad/s) |
| `Placo` | dict | Complete Placo parameter snapshot (see below) |
| `Frame_offset` | list[dict] | One-element list; byte offsets of each data slice within a frame |
| `Frame_size` | list[dict] | One-element list; size of each slice |
| `Frames` | list[list[float]] | The frame data array (one list per timestep) |
| `MotionWeight` | int | Always `1` |

### Frame Vector Layout (hardware mode, 59 elements total)

| Slice name | Size | Content |
|------------|------|---------|
| `root_pos` | 3 | Root position in world frame [x, y, z] (m) |
| `root_quat` | 4 | Root orientation as quaternion [x, y, z, w] |
| `joints_pos` | 16 | Joint positions (rad), ordered per `Joints` list |
| `left_toe_pos` | 3 | Left foot position in body frame (m) |
| `right_toe_pos` | 3 | Right foot position in body frame (m) |
| `world_linear_vel` | 3 | World-frame linear velocity of root (m/s) |
| `world_angular_vel` | 3 | World-frame angular velocity of root (rad/s) |
| `joints_vel` | 16 | Joint velocities (rad/s) |
| `left_toe_vel` | 3 | Left foot velocity in body frame (m/s) |
| `right_toe_vel` | 3 | Right foot velocity in body frame (m/s) |
| `foot_contacts` | 2 | Binary contact flags [left, right] |

### Joint Order (open_duck_mini_v2)

```
0  left_hip_yaw
1  left_hip_roll
2  left_hip_pitch
3  left_knee
4  left_ankle
5  neck_pitch
6  head_pitch
7  head_yaw
8  head_roll
9  left_antenna
10 right_antenna
11 right_hip_yaw
12 right_hip_roll
13 right_hip_pitch
14 right_knee
15 right_ankle
```

### Placo Metadata Block

The `Placo` field contains the exact parameters used to generate the recording, including measured velocities and the gait period. This enables `fit_poly.py` to extract the correct number of frames per cycle using `period` and the `startend_double_support_ratio`.

---

## 4. Gait Parameters

All parameters live in `placo_defaults.json` (robot-specific defaults) and `placo_presets/{slow,medium,fast}.json` (speed tiers).

### Motion Command Parameters

| Parameter | Default (v2) | Description |
|-----------|-------------|-------------|
| `dx` | 0.0-0.1 | Forward/backward step displacement per half-cycle (m) |
| `dy` | 0.0 | Lateral step displacement per half-cycle (m) |
| `dtheta` | 0.0 | Yaw step rotation per half-cycle (rad) |

In sweep mode for `open_duck_mini_v2`:
- dx: -0.04 to 0.06 in steps of 0.02 (6 values)
- dy: -0.04 to 0.04 in steps of 0.02 (5 values, including 0.0)
- dtheta: -0.3 to 0.3 in steps of 0.1 (7 values, including 0.0)
- **Total: 6 x 5 x 7 = 210 motion files**

The step sizes are chosen so that zero is always an included value. Using an off-centre step (e.g. starting at -0.03 with step 0.02) would never land on 0.0, producing a library with no straight-ahead motion in that axis.

### Biomechanical / Gait Tuning Parameters

Terminology used in this table:
- **CoM** (Centre of Mass) - the average weighted position of all the robot's weight, like its balance point
- **single support** - the phase when only one foot is on the ground (the other is swinging forward)
- **double support** - both feet on the ground simultaneously (the stable "transfer" phase between steps)

| Parameter | Default (v2 medium) | Description |
|-----------|---------------------|-------------|
| `walk_com_height` | 0.205-0.21 m | Target CoM height (determines leg crouch) |
| `walk_foot_height` | 0.04 m | Peak foot clearance during swing (step height) |
| `walk_trunk_pitch` | -4 to 5 deg | Forward lean of the trunk |
| `walk_foot_rise_ratio` | 0.02-0.2 | How high the foot lifts off the ground during a step (fraction of swing phase at which peak height occurs) |
| `single_support_duration` | 0.17-0.18 s | Duration of single-leg support phase (controls cadence/frequency) |
| `double_support_ratio` | 0.18-0.5 | Fraction of step period spent in double support |
| `startend_double_support_ratio` | 1.0-1.5 | Extended double support at start/end of motion |
| `feet_spacing` | 0.16 m | Lateral distance between feet |
| `foot_length` | 0.06 m | Foot length (used for ZMP constraint) |
| `zmp_margin` | 0.0 m | Additional ZMP safety margin |
| `foot_zmp_target_x` | 0.0 m | ZMP target offset along foot x-axis |
| `foot_zmp_target_y` | 0.0 to -0.03 m | ZMP target offset along foot y-axis |
| `walk_max_dx_forward` | 0.08 m | Maximum forward step size clamp |
| `walk_max_dx_backward` | 0.03 m | Maximum backward step size clamp |
| `walk_max_dy` | 0.1 m | Maximum lateral step size clamp |
| `walk_max_dtheta` | 1.0 rad | Maximum turn rate clamp |

### Gait Period

The gait period is derived as:

```
period = 2 * single_support_duration + 2 * double_support_duration()
```

where `double_support_duration = double_support_ratio * single_support_duration`.

For `open_duck_mini_v2` medium preset with `single_support_duration=0.18` and `double_support_ratio=0.5`:
`period = 2 * 0.18 + 2 * (0.5 * 0.18) = 0.36 + 0.18 = 0.54 s`. At 50 Hz this is 27 frames per cycle.

### Planning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `planned_timesteps` | 48 | How many steps ahead the footstep planner looks |
| `replan_timesteps` | 10 | How often the trajectory is replanned (in placo timesteps) |
| `single_support_timesteps` | 10 | Placo internal discretisation of swing phase |

### Speed Tier Definitions (open_duck_mini_v2)

| Tier | Total speed threshold |
|------|----------------------|
| slow | < 0.05 m/s |
| medium | 0.05 - 0.15 m/s |
| fast | > 0.15 m/s |

---

## 5. Integration with RL Training

### How Motions Feed into Training

The motion files are not loaded directly during training. Instead, the polynomial-compressed representation is used via `PolyReferenceMotion`:

```python
# At training initialisation
PRM = PolyReferenceMotion("polynomial_coefficients.pkl")

# At each simulation step
ref = PRM.get_reference_motion(cmd_dx, cmd_dy, cmd_dtheta, step_i)
# ref is a 39-element vector: [joints_pos(16), joints_vel(16), foot_contacts(2),
#                               base_lin_vel(3), base_ang_vel(3)]
```

The closest motion in the library is selected by nearest-neighbour lookup on (dx, dy, dtheta). The phase index `step_i` cycles through `nb_steps_in_period` (one gait cycle).

### Imitation Reward (`custom_rewards_numpy.py`)

The reward function `reward_imitation` compares the simulated robot state to the reference frame using weighted exponential penalties:

| Component | Weight | Metric |
|-----------|--------|--------|
| Trunk orientation | 1.0 | Exp of squared quaternion error |
| Linear velocity XY | 1.0 | Exp of squared velocity error |
| Linear velocity Z | 1.0 | Exp of squared velocity error |
| Angular velocity XY | 0.5 | Exp of squared angular velocity error |
| Angular velocity Z | 0.5 | Exp of squared angular velocity error |
| Joint positions | 15.0 (penalty) | Negative sum of squared position error |
| Joint velocities | 0.001 (penalty) | Negative sum of squared velocity error |
| Foot contacts | 1.0 | Count of matching contact states |

Head, neck, and antenna joints (indices 5-10) are stripped from both the reference and the simulated state before computing the joint reward - only the 10 leg joints (5 per side) are matched.

The imitation reward is gated by `cmd_norm > 0.01` to suppress it when the commanded velocity is zero (preventing the robot from learning to stand still to maximise the zero-motion reward).

The imitation reward is optional (`use_imitation_reward=False` default); it supplements the velocity-tracking reward rather than replacing it.

### Reference Motion Viewer

`replay_sweep.py` contains a `ReferenceMotion` class that loads all recording JSON files and builds the same `data_array[ix][iy][itheta]` lookup structure used by training. It includes a keyboard-controlled interactive viewer allowing manual navigation through the motion library.

### Visual Verification in MuJoCo

Individual motion files can be verified visually using `analysis/replay_reference.py` in the Open_Duck_Playground repository. It loads a motion JSON and plays it in the MuJoCo viewer. One conversion is required: Placo records quaternions in xyzw order, but MuJoCo expects wxyz. `replay_reference.py` handles this reordering before passing frames to the viewer.

---

## 6. Code Structure - File-by-File

### Open_Duck_reference_motion_generator

```
open_duck_reference_motion_generator/
  placo_walk_engine.py         Core Placo wrapper. Loads URDF, sets up QP solver,
                                manages footstep replanning and trajectory execution.
                                Exposes tick(dt) and get_angles().

  gait_generator.py            Single-motion recorder. Runs PlacoWalkEngine for a
                                specified duration, records frames at 50 Hz, computes
                                velocities, writes output JSON. Called by auto_waddle.py.

  gait_playground.py           Interactive Flask web UI. Exposes all Placo parameters
                                as HTTP form controls; visualises in real time via
                                MeshCat. Useful for tuning presets manually.

  robots/
    open_duck_mini_v2/
      placo_defaults.json      Default parameters for open_duck_mini_v2.
      placo_presets/
        medium.json            Medium-speed preset (base for sweep generation).
        fast.json              Fast preset (alternative base).
      auto_gait.json           Sweep ranges: dx, dy, dtheta min/max/step values
                                and speed tier thresholds.
      open_duck_mini_v2.urdf   URDF used by Placo. Exported from a perfectly symmetric
                                OnShape CAD model; joint axes are true left/right mirrors.
                                This symmetry matters: any asymmetry in the URDF would
                                cause the left and right legs to follow subtly different
                                trajectories, producing a gait that drifts laterally.
      assets/                  STL mesh files for the v2 robot.

    open_duck_mini/            Same structure for v1 robot.
    go_bdx/                    Same structure for the larger BDX robot.
                                Notable: uses trunk_mode=true and larger physical
                                parameters (feet_spacing=0.19m, walk_com_height=0.26m).

  templates/
    index.html                 Jinja template for gait_playground Flask UI.

scripts/
  auto_waddle.py               Main orchestrator. Parses auto_gait.json, generates
                                sweep or random presets, launches gait_generator.py
                                as subprocesses (optionally parallel via ThreadPoolExecutor),
                                then post-filters recordings by speed tier.
                                Use -j2 for safe parallelism on a 32 GB laptop; -j8 exhausts
                                RAM. A 210-motion sweep with -j2 takes approximately 18 minutes.

  fit_poly.py                  Polynomial fitting pipeline. Reads all JSON recordings,
                                extracts one period of [joints_pos, joints_vel, foot_contacts,
                                lin_vel, ang_vel], fits degree-15 polynomials per dimension,
                                saves serialised coefficients.

  plot_poly_fit.py             Visualisation tool. Loads polynomial coefficients and
                                compares reconstruction against original recording for all
                                39 signal dimensions. Labels each dimension by name.

  replay_motion.py             Single-file replay. Reads a recording JSON and visualises
                                root pose and toe positions using FramesViewer. Also plots
                                linear and angular velocity traces after playback.

  replay_sweep.py              Multi-file replay. Loads entire recordings directory into
                                a 3D lookup table. Supports keyboard control to navigate
                                the motion library in real time.

pyproject.toml                 Python 3.10.12, uv-managed. Dependencies: placo==0.6.3,
                                flask, framesviewer, matplotlib, pygame, scikit-learn, scipy.
```

---

## 7. Blender Pipeline (Open_Duck_Blender)

### Purpose

The Blender repository provides an alternative, artist-driven method for creating reference motions. Where the Placo pipeline generates motions algorithmically via model-based control, Blender allows hand-crafted animation using a rigged character. The output format is identical to the Placo-generated JSON, making both sources interchangeable for RL training.

### Repository Contents

```
Open_Duck_Blender/
  open-duck-mini.blend           Blender project file. Contains the Duck Mini armature,
                                  FK/IK rig, and a sample walk cycle animation.
  assets/scripts/
    fk_ik_control.py             Blender addon. Manages the FK/IK toggle on both legs.
                                  Switching modes calls copy_fk_to_ik or copy_ik_to_fk
                                  to snap the inactive chain to the active one.
    data_recording.py            Blender addon. Defines DataRecorder class and
                                  StartRecordingOperator. Hooks into Blender's timer
                                  to capture joint state at each animation frame.
    fk_ik_snapping.py            Deprecated predecessor to fk_ik_control.py. Kept as
                                  historical reference.
  assets/*.gif, *.png            Documentation images.
```

### Rig Structure

The armature has two control chains per leg. FK (Forward Kinematics) = you rotate each joint in sequence, like moving your shoulder, then elbow, then wrist, to position your hand. IK (Inverse Kinematics) = you specify where you want the end point (the hand or foot) to be, and the computer calculates what angles all the joints need. For walking animation, IK is typically easier because you think in terms of "put foot here" rather than "rotate hip by X degrees, knee by Y degrees".

- **FK chain**: `hip_yaw_fk`, `hip_roll_fk`, `hip_pitch_fk`, `knee_fk`, `ankle_fk` - direct rotation control
- **IK chain**: `hip_roll_ik`, `hip_pitch_ik`, `knee_ik`, `ankle_ik` + `leg_ik` control bone - IK effector control

`hip_yaw` is FK-only. The head chain (`neck_pitch`, `head_pitch`, `head_yaw`, `head_roll`) is FK-only. Antennas (`antenna.l`, `antenna.r`) are also FK.

### FK/IK Switching

The `fk_ik_controller` bone holds a custom property `fk_ik` (0.0 = FK, 1.0 = IK). When switching:

- FK -> IK: `copy_fk_to_ik` computes the IK effector position from the FK knee chain using relative bone transforms.
- IK -> FK: `copy_ik_to_fk` copies the IK chain matrices into the FK chain bones for `hip_roll`, `hip_pitch`, `knee`, and `ankle`.

### Data Recording Process

`DataRecorder` (in `data_recording.py`) reads the Blender armature pose at each animation frame and constructs a frame vector in the same layout as the Placo recordings:

1. **Root position** - read from `root` bone location, converted from Blender's Z-up coordinate system to the robot frame (`blender_frame_to_robot_frame` applies a [-y, x, z] axis swap)
2. **Root orientation** - read from `root` bone Euler angles, converted to quaternion
3. **Joint positions** - read from the FK bone rotations for all 16 joints; knee offsets of -10 deg and ankle offsets of +10 deg are applied to account for the rest-pose offset baked into the Blender model
4. **Toe positions** - computed as the world-space toe bone position minus the root bone position (body-frame relative)
5. **Velocities** - all velocity terms are finite-differenced from the previous frame at `1/FPS`
6. **Foot contacts** - hard-coded to `[1, 1]` in the current implementation (both feet always marked as contacting)

When the `Start Recording` button is pressed, the animation plays from frame 1. A Blender timer fires at `1/FPS` seconds and calls `update_frame()`. When the final frame is reached, the episode is serialised to JSON in `duck_mini_data_records/<timestamp>.json`.

### Coordinate Convention Notes

- Blender uses Z-up, Y-forward; the robot uses Z-up, X-forward
- The axis mapping `[-y, x, z]` in `blender_frame_to_robot_frame` handles the 90-degree yaw difference
- Roll sign is inverted (`roll_blender_frame = -roll_root_frame`) to match the robot convention

### Known Limitations

- Foot contact flags are not computed from physics - both feet are always marked as contacting. This means recordings from Blender will have incorrect contact data for RL training.
- IK foot orientation is not preserved when switching from FK to IK (noted in README as a known issue).
- Head control is FK only; no IK for head or antennas.
- Replay of Blender recordings requires the `episodic` branch of `Open_Duck_Playground`, not the main branch.
- The RL integration path ("Use recorded data to train RL policies") is listed as TODO in the README.

---

## 8. Key Relationships and Observations

1. **The 210 files** arise from the sweep mode for `open_duck_mini_v2`: 6 dx values x 5 dy values x 7 dtheta values = exactly 210. The sweep ranges are chosen so that 0.0 is always included in each axis, ensuring the library contains a straight-ahead gait and a pure-rotation gait.

2. **The polynomial approach** means the training environment never loads the raw JSON files. Only the serialised coefficients are loaded. The raw recordings are only needed once to run `fit_poly.py`.

3. **No interpolation between motions** - the `vel_to_index` lookup always returns the nearest discrete gait, not a blend. This means there are step-changes at gait boundaries during training when the commanded velocity crosses from one cell to the adjacent one.

4. **Frame rate dependency** - `gait_generator.py` sets `FPS = 50` for MuJoCo Playground and notes 30 fps for AWD (Isaac Gym). The polynomial fitting uses the `fps` stored in the file, so the serialised coefficients file is simulator-specific and the two cannot be swapped.

5. **The Placo ZMP model** assumes flat ground and known contact schedule. The `ignore_feet_contact=False` default means if neither foot contacts for longer than `single_support_duration`, the engine detects a falling state and freezes the trajectory update (the QP still runs, maintaining the last pose).

6. **Head and antenna joints** are present in the motion files (indices 5-10) but are zeroed out in `joint_angles` in the default presets. They are stripped from the joint imitation reward in `custom_rewards_numpy.py`, so the head is effectively unconstrained by the imitation signal.

7. **The go_bdx robot** differs from the duck variants: it uses `trunk_mode=true` (which changes how the trunk orientation task is set up in `WalkTasks`), has a much longer `single_support_duration=0.45 s` (slower, more deliberate gait), and substantially larger physical dimensions.
