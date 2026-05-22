"""Replay a Placo-generated reference motion in MuJoCo viewer.

Use this to visually verify generated reference motions look correct
before training. Helps catch axis/sign convention bugs.

Usage:
    uv run python analysis/replay_reference.py <motion.json>
"""
import argparse, json, sys, time
import numpy as np
import mujoco, mujoco.viewer
from etils import epath
from playground.open_duck_mini_v2 import base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("motion_file")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--rate", type=float, default=1.0, help="Playback rate multiplier")
    args = parser.parse_args()

    with open(args.motion_file) as f:
        motion = json.load(f)

    frames = motion["Frames"]
    fps = motion["FPS"]
    joint_names = motion["Joints"]
    print(f"Loaded {len(frames)} frames at {fps} FPS, joints: {joint_names}")

    # Load model
    model = mujoco.MjModel.from_xml_string(
        epath.Path("playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml").read_text(),
        assets=base.get_assets())
    data = mujoco.MjData(model)
    data.qpos[:] = model.keyframe("home").qpos
    mujoco.mj_forward(model, data)

    # Map motion joint indices to model qpos indices
    joint_qpos_idx = {}
    for jname in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid > 0:
            joint_qpos_idx[jname] = model.jnt_qposadr[jid]
        else:
            print(f"WARNING: joint '{jname}' not in model")

    frame_dt = 1.0 / fps / args.rate

    print("\nStarting viewer. Close window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for frame_idx, frame in enumerate(frames):
                if not viewer.is_running():
                    break
                t0 = time.time()

                # Frame structure:
                # [0:3]   base xyz
                # [3:7]   base quaternion (xyzw from Placo)
                # [7:7+N] joint positions (N = len(joint_names))
                pos_xyz = frame[0:3]
                quat_xyzw = frame[3:7]
                # MuJoCo uses wxyz convention
                data.qpos[0:3] = pos_xyz
                data.qpos[3] = quat_xyzw[3]  # w
                data.qpos[4] = quat_xyzw[0]  # x
                data.qpos[5] = quat_xyzw[1]  # y
                data.qpos[6] = quat_xyzw[2]  # z

                # Set joint positions
                for i, jname in enumerate(joint_names):
                    if jname in joint_qpos_idx:
                        data.qpos[joint_qpos_idx[jname]] = frame[7 + i]

                mujoco.mj_forward(model, data)
                viewer.sync()

                # Frame title
                if frame_idx % 50 == 0:
                    print(f"  frame {frame_idx}/{len(frames)}  base_xy=({pos_xyz[0]:+.3f}, {pos_xyz[1]:+.3f})")

                elapsed = time.time() - t0
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)

            if not args.loop:
                print("Playback complete. Close viewer to exit.")
                # Hold final frame
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
            else:
                print("Looping...")


if __name__ == "__main__":
    main()
