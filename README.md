# Open Duck Playground

# Installation 

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Training

If you want to use the [imitation reward](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf), you can generate reference motion with [this repo](https://github.com/apirrone/Open_Duck_reference_motion_generator)

Then copy `polynomial_coefficients.pkl` in `playground/<robot>/data/`

You'll also have to set `USE_IMITATION_REWARD=True` in it's `joystick.py` file

Run: 

```bash
uv run playground/<robot>/runner.py 
```

## Tensorboard

```bash
uv run tensorboard --logdir=<yourlogdir>
```

# Inference 

Infer mujoco

(for now this is specific to open_duck_mini_v2)

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <path_to_.onnx>
```

# Documentation

## Project structure : 

```
.
├── pyproject.toml
├── README.md
├── playground
│   ├── common
│   │   ├── export_onnx.py
│   │   ├── onnx_infer.py
│   │   ├── poly_reference_motion.py
│   │   ├── randomize.py
│   │   ├── rewards.py
│   │   └── runner.py
│   ├── open_duck_mini_v2
│   │   ├── base.py
│   │   ├── data
│   │   │   └── polynomial_coefficients.pkl
│   │   ├── joystick.py
│   │   ├── mujoco_infer.py
│   │   ├── constants.py
│   │   ├── runner.py
│   │   └── xmls
│   │       ├── assets
│   │       ├── open_duck_mini_v2_no_head.xml
│   │       ├── open_duck_mini_v2.xml
│   │       ├── scene_mjx_flat_terrain.xml
│   │       ├── scene_mjx_rough_terrain.xml
│   │       └── scene.xml
```

## Adding a new robot

Create a new directory in `playground` named after `<your robot>`. You can copy the `open_duck_mini_v2` directory as a starting point.

You will need to:
- Edit `base.py`: Mainly renaming stuff to match you robot's name
- Edit `constants.py`: specify the names of some important geoms, sensors etc
  - In your `mjcf`, you'll probably have to add some sites, name some bodies/geoms and add the sensors. Look at how we did it for `open_duck_mini_v2`
- Add your `mjcf` assets in `xmls`. 
- Edit `joystick.py` : to choose the rewards you are interested in
  - Note: for now there is still some hard coded values etc. We'll improve things on the way
- Edit `runner.py`



# Notes

Inspired from https://github.com/kscalelabs/mujoco_playground


## Current win

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```

---

## Local Modifications (Caleb's fork)

This fork contains several modifications from the upstream `apirrone/Open_Duck_Playground`:

### Model fixes
- **Symmetric MJCF**: `open_duck_mini_v2.xml` re-exported from fixed OnShape CAD (37mm right-leg asymmetry corrected). Old files backed up in `xmls.backup/`
- **Home keyframe**: `scene_flat_terrain.xml` updated with true-mirrored right leg values (both feet at identical height)
- **Reference motions**: `polynomial_coefficients.pkl` regenerated from symmetric URDF with 210-motion grid (zero-entry bug in auto_gait.json fixed)

### Training fixes
- **tracking_sigma**: Changed from 0.01 to 0.1 in `joystick.py` (industry standard, prevents overshoot trap)
- **get_gravity() bug**: Fixed sensor ID vs address bug in `mujoco_infer_base.py`
- **Dependencies**: Pinned `playground==0.0.5` and `jax<0.7` in `pyproject.toml`

### Analysis tools (`analysis/`)
- `evaluate_policy.py` - comprehensive walking quality evaluator (v4)
- `quick_walking_eval.py` - lightweight eval hooked into training callbacks
- `find_best_checkpoint.py` - checkpoint selection by walking quality (not training reward)
- `analyse_training.py` - TensorBoard event analysis
- `debug_policy.py` - per-step velocity traces
- `replay_reference.py` - visual playback of reference motions
- `backfill_walking_metrics.py` - post-hoc walking metric addition to TF events

### Training hook
- `runner.py` modified to log `walking/HEADLINE` and per-scenario scores to TensorBoard after each checkpoint

See `../docs/` for full documentation.
