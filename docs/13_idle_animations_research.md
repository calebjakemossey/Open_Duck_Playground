# Idle and Personality Animations for Open Duck Mini

Research into how Disney's BDX droids express character whilst standing still, and the practical routes to adding equivalent behaviour to the Open Duck Mini.

---

## Part 1: What the Real BDX Droids Actually Do When Standing

### Observed idle behaviours

Footage and press descriptions of the BDX droids in Galaxy's Edge consistently show the following when the robot is standing without a movement command:

- **Continuous low-frequency body sway** - the torso shifts weight left and right in a slow, duck-like rhythm, about one cycle every 2-3 seconds. This is what makes them read as "alive" even from a distance.
- **Head scanning** - the head pans and tilts unpredictably to look around, mimicking avian curiosity. The neck and head move independently; the neck tilts the whole head assembly up/down whilst the head joint adds finer pitch and yaw.
- **Antenna quiver** - the two antennae on top of the head have small continuous oscillations, separate from the rest of the body.
- **Eye blinking** - intermittent blink animations layered on top of everything else.
- **Weight transfer steps** - occasionally the robot takes a tiny side-step or shifts its foot to redistribute balance, like a person waiting at a bus stop.
- **Reaction to interaction** - when a park guest approaches, the head snaps towards them and the body orients slightly. The operator triggers this via a joystick; the robot does not do it autonomously.
- **Emotional expressions** - distinct episodic clips: happy dance, excited bounce, tantrum. These are triggered explicitly, not generated procedurally.

### What the BDX paper says about standing

The paper (arXiv:2501.05204) describes the standing state explicitly. The key elements are:

**The perpetual standing policy.** The robot has a separate, always-on "perpetual" policy (`pi(a_t | s_t, g^perp_t)`) that maintains balance whilst accepting commands for head height/orientation and torso position/orientation. This is trained independently from the walking policy.

**The background animation layer.** The animation engine runs a continuously looping background sequence during standing. The paper describes it as providing "a basic level of activity" and specifically lists "intermittent eye-blinking and antenna motion." This background layer is always on; triggered animations blend on top of it.

**Head and torso as expressive channels.** The standing policy command input is `g^perp_t = (delta_h_head, delta_theta_head, h_torso, theta_torso)` - offsets to head height and orientation, plus torso height and orientation in the path frame. The operator drives these in real time via joystick to aim the gaze and shift the posture. The policy then executes whatever pose the animation engine specifies.

**Path frame convergence during standing.** When standing, the path frame (the robot's local reference frame) slowly converges towards the midpoint between the two feet. This prevents the robot from "drifting" its reference position and helps maintain natural balance during postural shifts.

**Policy transitions.** When switching from walking to standing, the system waits for the next double-support phase (both feet on the ground) before switching policies. This prevents mid-step policy changes that would cause stumbling. When switching from standing to walking, the walk phase is initialised based on the commanded turning direction.

---

## Part 2: What the Open Duck Mini Currently Has

### The `standing_policy` branch

The `upstream/standing_policy` branch contains a dedicated `standing.py` environment. Key characteristics:

- **Rewards:** `orientation`, `torques`, `action_rate`, `alive`, `stand_still`, and `head_pos`. No imitation reward - `USE_IMITATION_REWARD = False`.
- **The `stand_still` reward** penalises deviation from the default pose and non-zero joint velocities, but *only when velocity commands are near zero* (`cmd_norm < 0.01`). When commands are issued, the penalty vanishes and the robot can move freely.
- **Head commands** are sampled randomly at reset and included in the observation: neck pitch, head pitch, head yaw, head roll. The policy must learn to hold the head at whatever commanded position it receives.
- **No body sway reference.** There is no reference motion and no oscillatory signal driving the torso. The policy learns to stand still and hold the commanded head pose, but has no incentive to produce the characteristic duck-like sway.
- **The `improve_walk_standing_policy` branch** attempted to unify walking and standing into a single policy by sampling a zero velocity command with 10% probability. The commit history shows multiple failed attempts before settling on keeping them separate.

### The `episodic` branch

The `upstream/episodic` branch is the most directly relevant to idle animations. It contains:

- `playground/open_duck_mini_v2/episodic.py` - an environment that trains a policy to imitate a specific Blender-authored animation clip.
- `playground/common/episodic_reference_motion.py` - loads a JSON file produced by the Blender data recorder and serves frames sequentially during training.
- `playground/open_duck_mini_v2/data/animation_data_leg_flexing.json` - a recorded clip of leg flexing (currently the only example).
- `playground/open_duck_mini_v2/data/animation_head_modif_new.json` - a recorded clip of head movement.

The episodic task uses imitation reward (weight 1.0) plus action rate and alive penalties. It feeds the robot a phase signal `[cos(phase), sin(phase)]` so the policy knows where it is in the animation cycle. This is functionally identical to Disney's episodic policy architecture.

**What is missing:** The episodic branch has the machinery but no idle animation clips to train on. It was tested with a leg-flexing clip and a head motion clip, but no body-sway or idle-bobbing reference data exists yet.

### The Blender pipeline

`Open_Duck_Blender/open-duck-mini.blend` is a fully rigged Blender scene with FK and IK controls for legs, and FK controls for the neck/head/antenna joints. The `data_recording.py` addon records joint positions, velocities, foot contact, root position, and root orientation at 50 Hz in the exact JSON format expected by `EpisodicReferenceMotion`.

The joints available for animation are:
- Left and right: `hip_yaw`, `hip_roll`, `hip_pitch`, `knee`, `ankle`
- Head chain: `neck_pitch`, `head_pitch`, `head_yaw`, `head_roll`
- `antenna.r`, `antenna.l`

The root bone (`root`) can also be animated for body sway, so torso weight shifts translate into the recording.

---

## Part 3: Approaches to Creating Idle Animations

There are three distinct approaches, ranging from quickest-to-implement to most expressive.

---

### Approach A: Scripted background motion layered on the existing standing policy

**What:** Write a time-varying script that runs on the robot at runtime, adding sinusoidal offsets to the head and optionally the torso target positions. The existing standing policy handles balance; the script adds cosmetic sway on top.

**How it works in practice:**

1. The existing walking/standing policy runs as normal, outputting motor targets at 50 Hz.
2. A thin wrapper in `mujoco_infer.py` adds time-varying offsets to the commanded head joints before sending to servos. For example:
   ```python
   t = time.time()
   head_yaw_offset = 0.3 * math.sin(0.8 * t)  # slow look-around
   head_pitch_offset = 0.15 * math.sin(0.4 * t + 1.2)  # slight nod
   neck_pitch_offset = 0.1 * math.sin(0.3 * t + 2.5)  # body lean
   ```
3. These offsets are added to whatever the policy outputs for those joints.

**Pros:** Zero retraining required. Can be implemented in an afternoon. Immediate visual effect. Parameters can be tuned live.

**Cons:** The policy was not trained with this offset, so there may be subtle balance effects when the head is pushed to edge positions. The sway pattern is mathematically regular - it will feel slightly mechanical compared to a trained or Blender-authored motion. Does not address body/torso sway (the standing policy has no incentive to shift weight).

**Recommendation for body sway specifically:** The standing policy's `stand_still` reward penalises any deviation from the default pose when commands are zero. This actively suppresses sway. You would need to either disable the `stand_still` penalty, reduce its weight significantly, or zero out its leg-only cost whilst allowing torso motion - the `ignore_head=True` path in `cost_stand_still` is already implemented for this.

---

### Approach B: Author idle clips in Blender and train an episodic policy

**What:** Create 3-5 second idle animation clips in Blender, record them via the data recorder addon, and train episodic policies to imitate each clip. This is exactly what the `episodic` branch was built for - the machinery exists end to end.

**How to create the clips:**

1. Open `open-duck-mini.blend` in Blender 4.3.2+.
2. In Pose mode, use IK controls on the legs and FK on the head to author the following:
   - **Weight-shift sway:** Over 2-3 seconds, slowly shift the root bone 1-2 cm laterally left, then right, whilst keeping the feet flat. The hip roll joints compensate. This is the characteristic duck waddle-in-place.
   - **Head scan:** Pan `head_yaw` from -30 to +30 degrees over 3-4 seconds in a non-linear curve (ease in/out), combined with a slight `head_pitch` dip as if looking at the ground then back up.
   - **Alert/curious:** Quick head snap to one side (0.1 s), hold 1 s, return slowly (0.5 s). Antenna bounce at the snap point.
   - **Bored fidget:** Slow asymmetric weight shift, one knee bends slightly more, body tilts, returns.
3. For each clip, use the Data Recording panel to record at 50 FPS.
4. Train a separate episodic policy per clip using `episodic.py` with `USE_IMITATION_REWARD = True` and the recorded JSON as the reference.
5. At runtime, cycle through policies with a timer or operator trigger.

**Specific Blender animation advice for weight-shift sway:**

The IK control bones for each foot need to stay fixed to the ground plane (Z = 0) whilst the `root` bone is animated. In Blender, this means setting keys on the root bone's X-translation and the hip roll FK joints simultaneously. At the left-weight extreme: root shifts +1.5 cm in X, left hip roll decreases ~5 degrees, right hip roll increases ~8 degrees to maintain foot contact. The knees stay roughly fixed. Animate this as a smooth sine curve over ~2.5 seconds.

**Training details for episodic policies:**

The `episodic.py` config has `USE_IMITATION_REWARD = True` and `imitation=1.0`. Training target is 150M steps (same as walking). On dual RTX 4090, this takes roughly 20-25 minutes per clip. The policy receives the current animation frame as part of its observation, so it knows what the target pose is at each step.

**Pros:** Produces characterful, non-repetitive-feeling motion. Each clip can be designed with specific emotional intent. The episodic policy framework already works (confirmed by the leg-flexing example in the branch). Animations are fully controllable by the operator.

**Cons:** Requires animator time in Blender. Each clip needs a separate training run. Transitions between clips are abrupt unless you add a blend period in the runtime.

**Estimated effort:** 2-3 hours of Blender animation work per clip. 30 minutes of training per clip. 1 day total for 3-4 clips plus integration.

---

### Approach C: Train a "personality standing" policy using a procedural reference motion

**What:** Generate a procedural idle reference motion using the reference motion generator (or a new script), fit it as a polynomial or use it directly, and train a perpetual standing policy that actively tracks that motion rather than the static default pose.

**How it would work:**

1. Write a new script in `Open_Duck_reference_motion_generator` analogous to `gait_generator.py` but for standing. The script uses Placo's IK solver to generate a slow, sinusoidal weight-shift trajectory:
   - Move the centre-of-mass laterally by ±1.5 cm at 0.3 Hz using a smooth sine wave.
   - Allow the trunk to pitch forwards and backwards by ±2 degrees at a separate frequency (0.2 Hz), like breathing.
   - Add a separate, faster neck oscillation at 0.5-0.8 Hz.
   - Record multiple cycles, producing a periodic JSON file.
2. Modify the standing policy to use an imitation reward tracking this reference instead of, or in addition to, the `stand_still` penalty.
3. Train a perpetual policy (`episode_length=1000`, no termination on standing) that tracks the reference.

This matches Disney's approach: their background animation is a "looped playback of a periodic background animation" on top of which the joystick layer applies head and torso commands.

**The key insight from the paper:** Disney's background animation layer is separate from the policy command input. The policy tracks the background animation automatically, and *on top of that* the operator can shift the head and torso via joystick. The policy is trained to handle both simultaneously.

**Pros:** Produces truly organic-feeling motion because the policy is actively balancing whilst producing the sway, not just adding an offset. The motion will look correct under perturbations (a nudge causes the sway to respond physically). Closest to the Disney approach.

**Cons:** Requires writing the procedural idle reference generator. Placo can generate static stands but generating a slow sinusoidal sway sequence requires driving the CoM position explicitly in a time loop, similar to how the walk engine works but without footstep planning. The polynomial fitting step (`fit_poly.py`) would need to be adapted since the standing motion is not parameterised by velocity - it is a fixed periodic pattern.

**Simpler variant:** Skip Placo entirely. Write a Python script that directly generates the JSON frame sequence by analytically computing joint angles for each frame using the robot's geometry (hip spacing, leg lengths). The weight-shift sway is kinematically simple enough to compute without a full IK solver - only hip roll and ankle change significantly.

---

## Part 4: Transition Between Walking and Idle

### Disney's approach

The BDX paper specifies: when transitioning from walking to standing, the policy switch is delayed until the start of the next double-support phase. This prevents the robot from switching policies mid-step when only one foot is on the ground.

### Open Duck Mini current state

The Open Duck Mini runtime (`mujoco_infer.py`) switches between policies by loading a different ONNX model. There is no double-support detection in the current runtime. The walking policy handles the `stand_still` case by penalising leg motion when commands are zero, so the transition is handled within a single policy rather than switching policies.

### Recommended approach for idle transitions

If using Approach A (scripted offsets): the offset script can simply ramp in when velocity commands drop below a threshold and ramp out when movement is commanded. Use a 0.5-second linear ramp on the offset amplitudes.

If using Approach B or C (separate idle policy): the simplest safe transition is to wait until both foot contact sensors register simultaneously before switching. The runtime's `FeetContacts` already provides this signal. Add a 0.1-second debounce to avoid triggering on transient double-support during walking.

---

## Part 5: Practical Recommendation

The fastest path to a visible result that would demonstrate the character concept:

**Step 1 (immediate, no training):** Enable head sway via scripted offsets in `mujoco_infer.py`. Add three sinusoids with different frequencies (0.3 Hz, 0.5 Hz, 0.8 Hz) and phases to `neck_pitch`, `head_yaw`, and `head_pitch`. This gives the duck a "looking around" behaviour immediately. Amplitude: ±0.2 rad for yaw, ±0.1 rad for pitch. This is the same mechanism Disney uses for their background antenna motion.

**Step 2 (1-2 days):** Author a weight-shift sway clip in Blender. Key the root bone laterally ±1.5 cm over 2.5 seconds with a sinusoidal easing curve. Record it and train an episodic policy using the existing `episodic` branch infrastructure. This gives you the characteristic duck sway.

**Step 3 (later):** Author 2-3 distinct "personality" clips (curious head scan, fidget, alert snap) and train episodic policies for each. Add a simple state machine to the runtime that cycles through idle clips when no movement command has been received for >3 seconds.

The body-sway clip (Step 2) is the single most impactful animation because it is what makes the BDX droids read as alive from a distance. The head scanning (Step 1) adds cheaply to that without any training cost.

---

## Part 6: What Would Need to Change in the Codebase

| Change | File | Effort |
|--------|------|--------|
| Scripted head sway offsets | `Open_Duck_Mini_Runtime/v2_rl_walk_mujoco.py` | 1 hour |
| Reduce `stand_still` cost to allow sway | `playground/open_duck_mini_v2/joystick.py` line 83 | 5 minutes |
| Author body-sway Blender clip | `open-duck-mini.blend` | 2-3 hours |
| Train episodic policy on sway clip | `playground/open_duck_mini_v2/episodic.py` | 30 mins training |
| Runtime episodic policy switching | `Open_Duck_Mini_Runtime/v2_rl_walk_mujoco.py` | 2-3 hours |
| Procedural idle reference generator | New script in `Open_Duck_reference_motion_generator` | 1-2 days |

The `episodic` branch from `upstream` contains all the training infrastructure. The `Open_Duck_Blender` repo has the rigging. The gap is simply that nobody has yet created idle animation content and run it through the pipeline.

---

## Sources

- Disney BDX paper: https://arxiv.org/html/2501.05204v1
- Disney BDX project page: https://la.disneyresearch.com/bdx-droids/
- Disney technology article: https://thewaltdisneycompany.com/news/behind-the-bdx-droids/
- BDX AI and personality: https://variety.com/2025/biz/news/disney-imagineering-ai-droids-learning-1236460286/
- Open Duck Mini Blender: `/home/lakieb/Documents/open_duck_mini_research/Open_Duck_Blender/`
- Open Duck Playground episodic branch: `upstream/episodic` in `/home/lakieb/Documents/open_duck_mini_research/Open_Duck_Playground`
- Open Duck Playground standing branch: `upstream/standing_policy` in `/home/lakieb/Documents/open_duck_mini_research/Open_Duck_Playground`
