#!/usr/bin/env python3
"""Patch a raw onshape-to-robot MJCF export for Open Duck Mini V2 training.

The onshape-to-robot exporter produces an MJCF that needs four modifications
before it can be used with the MJX training pipeline:

1. Add solver options (iterations=1, ls_iterations=5, eulerdamp=disable) - critical
   for keeping the XLA computation graph small enough for ptxas to compile.
2. Wrap trunk_assembly in a 'base' body with the freejoint and imu site.
3. Name foot TPU collision geoms (training expects 'left_foot_bottom_tpu' etc).
4. Rename right_cache -> knee_and_ankle_assembly_3 (and existing _3 -> _4) for
   symmetric naming with the left leg.

Assumes the OnShape CAD is set up correctly:
- Walking joint mates: `dof_<name>` prefix (Revolute type in OnShape)
- Antenna mates: `fix_<name>` prefix (prevents them from getting separate bodies)
- Foot frame mates: `frame_<name>` prefix

If you see antenna bodies in the export, the script will refuse to proceed -
fix the CAD by renaming antenna mates with `fix_` prefix instead.

Usage:
    python patch_mjcf.py <raw_export.xml> <output.xml>

The script is idempotent - running it twice produces the same result.
Each fix is a no-op if its target structure is not present.

See docs/15_onshape_export_pitfalls.md for the full background and the
recommended OnShape mate setup conventions.
"""
import re
import sys
import tempfile
from pathlib import Path

# Configuration - update these if robot dimensions change significantly
BASE_POS = "0 0 0.22"          # Initial spawn height of base body
IMU_POS = "-0.08 -0.0 0.05"    # IMU mount offset relative to base
FREEJOINT_NAME = "floating_base"

# Servo physics - the working values that produced trained policies.
# OnShape exports a newer set on each run but we pin these for training consistency.
# If you ever want to update, do it deliberately and re-train everything.
STS3215_DAMPING = 0.56
STS3215_FRICTIONLOSS = 0.068
STS3215_ARMATURE = 0.027
STS3215_KP = 13.37          # OnShape default is 17.12; this lower value is for compliance/sim2real
STS3215_FORCERANGE = "-3.23 3.23"


def patch_solver_options(content: str) -> str:
    """Add solver options block after <mujoco> opening tag.

    Without these, MJX generates a computation graph too large for ptxas to compile
    on RTX 5080 (CUDA 13.0). The options limit constraint solver iterations and
    disable Euler damping in the integrator.
    """
    if "<option iterations=" in content:
        return content
    option_block = (
        "\n"
        '  <option iterations="1" ls_iterations="5">\n'
        '    <flag eulerdamp="disable"/>\n'
        "  </option>\n"
    )
    return re.sub(
        r'(<mujoco model="[^"]*">)',
        r"\1" + option_block,
        content,
        count=1,
    )


def patch_base_wrapper(content: str) -> str:
    """Wrap trunk_assembly in a base body with freejoint and imu site.

    The training pipeline expects a 'base' wrapper body that holds the freejoint
    and imu site, with trunk_assembly as its child. This structure matches the
    original working MJCF and keeps the XLA computation graph at the expected size.
    """
    if 'body name="base"' in content:
        return content

    # Remove the existing freejoint from trunk_assembly
    content = re.sub(
        r'(<body name="trunk_assembly"[^>]*>)\s*<freejoint name="[^"]*"/>',
        r"\1",
        content,
    )

    # Insert base wrapper before trunk_assembly
    content = re.sub(
        r'(\s+)(<!-- Link trunk_assembly -->\s*\n\s*<body name="trunk_assembly")',
        rf'\1<body name="base" pos="{BASE_POS}">\n'
        rf'\1  <freejoint name="{FREEJOINT_NAME}"/>\n'
        rf'\1  <site name="imu" pos="{IMU_POS}"/>\n'
        r'\1\2',
        content,
    )

    # Find the closing tag of trunk_assembly and add an extra </body>
    content = re.sub(
        r'(\s+</body>)\s*\n(\s*</worldbody>)',
        r'\1\n\1\n\2',
        content,
        count=1,
    )

    return content


def patch_remove_trunk_imu_site(content: str) -> str:
    """Remove the imu site from trunk_assembly (it's now on the base body)."""
    return re.sub(
        r'\s*<!-- Frame imu -->\s*\n\s*<site group="3" name="imu"[^/]*/>',
        "",
        content,
    )


# Foot TPU sole collision: replace 1025-vertex mesh with bounding-box primitive.
# Measured from foot_bottom_tpu mesh: x_span=8.2mm, y_span=40.7mm, z_span=101.9mm.
# Half-extents used by <geom type="box" size="dx dy dz">.
# Rationale: MJX collision SAT path is O(N^2) over vertex pairs - 1025 verts blows
# up the per-step graph to ~38 GiB on 8192 envs, causing ptxas register spill
# (jaxlib 0.6) or cuBLAS autotuner OOM (jaxlib 0.10). mujoco_playground's own
# locomotion examples (H1, OP3, Cassie) all use primitive foot collision for
# this reason. See MJX docs: mesh-vs-primitive recommended &lt;=200 verts.
FOOT_BOX_HALF_EXTENTS = "0.0041 0.0204 0.0510"  # x y z half-extents in metres


def patch_foot_geom_names(content: str) -> str:
    """Replace foot_bottom_tpu mesh collisions with box primitives.

    The visual foot_bottom_tpu geom keeps the detailed mesh. Only the collision
    geom is swapped to a box of identical bounding-box dimensions. Box keeps the
    same name (left_foot_bottom_tpu / right_foot_bottom_tpu) so training code's
    geoms_colliding(...) lookups continue to work.
    """
    lines = content.split("\n")
    in_foot = None
    for i, line in enumerate(lines):
        if 'name="foot_assembly"' in line:
            in_foot = "left"
        elif 'name="foot_assembly_2"' in line:
            in_foot = "right"
        elif "</body>" in line:
            in_foot = None
        elif (
            in_foot
            and 'class="collision"' in line
            and 'mesh="foot_bottom_tpu"' in line
        ):
            target = f"{in_foot}_foot_bottom_tpu"
            # Extract pos and quat attributes from the original mesh line.
            pos_match = re.search(r'pos="([^"]+)"', line)
            quat_match = re.search(r'quat="([^"]+)"', line)
            pos = pos_match.group(1) if pos_match else "0 0 0"
            quat = quat_match.group(1) if quat_match else "1 0 0 0"
            # Build replacement box geom. Drop mesh + material attrs; keep collision class.
            lines[i] = (
                f'                <geom type="box" class="collision" '
                f'pos="{pos}" quat="{quat}" '
                f'size="{FOOT_BOX_HALF_EXTENTS}" name="{target}"/>'
            )
    return "\n".join(lines)


def patch_servo_values(content: str) -> str:
    """Override sts3215 servo physics with the pinned working values.

    OnShape may export updated BAM-identified values, but we pin to the values
    that produced our trained policies for consistency. To deliberately update,
    change the constants at the top of this file and re-train.
    """
    new_joint = (
        f'<joint damping="{STS3215_DAMPING}" '
        f'frictionloss="{STS3215_FRICTIONLOSS}" '
        f'armature="{STS3215_ARMATURE}"/>'
    )
    new_position = (
        f'<position kp="{STS3215_KP}" kv="0.0" '
        f'forcerange="{STS3215_FORCERANGE}"/>'
    )

    # Replace within the sts3215 default block
    content = re.sub(
        r'(<default class="sts3215">.*?)<joint damping="[^"]*" frictionloss="[^"]*" armature="[^"]*"/>',
        rf'\1{new_joint}',
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r'(<default class="sts3215">.*?)<position kp="[^"]*" kv="[^"]*" forcerange="[^"]*"/>',
        rf'\1{new_position}',
        content,
        flags=re.DOTALL,
    )
    return content


def patch_rename_right_cache(content: str) -> str:
    """Rename right_cache body to knee_and_ankle_assembly_3 for symmetry.

    Order matters: first rename the existing knee_and_ankle_assembly_3 to _4 to
    free up the _3 name, then rename right_cache to _3. The mesh and material
    keep their 'right_cache' names since those refer to STL files and visual
    properties, not the body itself.
    """
    if 'name="right_cache"' not in content:
        return content

    # Step 1: Rename existing knee_and_ankle_assembly_3 -> _4
    content = re.sub(
        r'name="knee_and_ankle_assembly_3"',
        r'name="knee_and_ankle_assembly_4"',
        content,
    )
    content = re.sub(
        r"knee_and_ankle_assembly_3 to ",
        r"knee_and_ankle_assembly_4 to ",
        content,
    )
    content = re.sub(
        r"to knee_and_ankle_assembly_3 ",
        r"to knee_and_ankle_assembly_4 ",
        content,
    )

    # Step 2: Rename right_cache body -> knee_and_ankle_assembly_3
    content = re.sub(
        r'<body name="right_cache"',
        r'<body name="knee_and_ankle_assembly_3"',
        content,
    )
    content = re.sub(
        r"right_cache to ",
        r"knee_and_ankle_assembly_3 to ",
        content,
    )
    content = re.sub(
        r"to right_cache ",
        r"to knee_and_ankle_assembly_3 ",
        content,
    )
    content = re.sub(
        r"<!-- Link right_cache -->",
        r"<!-- Link knee_and_ankle_assembly_3 (was right_cache in export) -->",
        content,
    )

    return content


def patch_remove_flash_geoms(content: str) -> str:
    """Remove flash_light_module and flash_reflector_interface visual geoms.

    These are cosmetic head additions in the OnShape document. Including them
    grows the MJX computation graph just enough to trigger ptxas register-spill
    segfaults on RTX 5080 (CUDA 13.0). The robot doesn't use them; we strip them
    to keep the graph compilable.
    """
    for mesh_name in ["flash_light_module", "flash_reflector_interface"]:
        content = re.sub(
            rf'\s*<!-- Part {mesh_name}[^>]*-->\s*\n\s*<geom[^>]*mesh="{mesh_name}"[^>]*/>',
            f"\n              <!-- Part {mesh_name} (visual only, removed to avoid XLA graph size issue) -->",
            content,
        )
    return content


def check_no_antenna_bodies(content: str) -> None:
    """Fail loudly if antenna bodies are present in the export.

    The preferred approach is to use 'fix_' prefix on antenna mates in OnShape,
    which causes the exporter to merge antennas into head_assembly automatically.
    If we see separate antenna body tags (not just visual mesh references), the
    CAD is misconfigured.
    """
    if re.search(r'<body name="[^"]*antenna[^"]*"', content):
        raise ValueError(
            "Antenna bodies detected in export. This means antenna mates are using "
            "the 'dof_' prefix in OnShape (creating separate revolute joint bodies). "
            "Rename the antenna mates with 'fix_' prefix instead - the exporter will "
            "then merge them into head_assembly as visual meshes only. The OnShape "
            "mate type can stay Revolute; only the name prefix matters."
        )


def verify_patched_model(patched_xml_path: Path) -> None:
    """Load the patched model in a temporary scene wrapper and verify it works.

    Checks:
    - All expected names resolve (bodies, geoms, sites, sensors, joints, actuators)
    - Model loads without errors
    - Home keyframe produces symmetric foot heights (within 1mm)
    """
    try:
        import mujoco  # type: ignore[import-not-found]
    except ImportError:
        print("WARNING: mujoco not installed, skipping verification")
        return

    # Build a minimal scene wrapper for testing
    scene_xml = f"""<mujoco model="scene">
    <include file="{patched_xml_path.name}"/>
    <worldbody>
        <body name="floor">
            <geom name="floor" size="0 0 0.01" type="plane" contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
        </body>
    </worldbody>
    <keyframe>
        <key name="home" qpos="0 0 0.15 1 0 0 0 0.002 0.053 -0.63 1.368 -0.784 0 0 0 0 -0.002 -0.053 0.63 -1.368 0.784"
                          ctrl="0.002 0.053 -0.63 1.368 -0.784 0 0 0 0 -0.002 -0.053 0.63 -1.368 0.784"/>
    </keyframe>
</mujoco>
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # Copy patched file and any sibling assets dir into temp location.
        # Assets can be sibling to output or sibling to input (when output goes elsewhere).
        (tmppath / patched_xml_path.name).write_text(patched_xml_path.read_text())
        for candidate in [patched_xml_path.parent / "assets",
                          patched_xml_path.parent.parent / "robot" / "assets"]:
            if candidate.exists():
                import shutil
                shutil.copytree(candidate, tmppath / "assets")
                break
        else:
            print(f"  WARNING: no assets/ found near {patched_xml_path}, mesh loading may fail")

        scene_path = tmppath / "scene.xml"
        scene_path.write_text(scene_xml)

        print("\n=== Verification ===")
        try:
            model = mujoco.MjModel.from_xml_path(str(scene_path))
        except Exception as e:
            print(f"  FAIL: model failed to load: {e}")
            raise SystemExit(1)

        # Structural checks
        expected = {"nbody": 18, "njnt": 15, "nu": 14, "nq": 21, "nv": 20}
        actual = {
            "nbody": model.nbody, "njnt": model.njnt, "nu": model.nu,
            "nq": model.nq, "nv": model.nv,
        }
        for k, v in expected.items():
            status = "OK" if actual[k] == v else "FAIL"
            print(f"  {k}: {actual[k]} (expected {v}) {status}")
            if actual[k] != v:
                raise SystemExit(1)

        # Solver options
        print(f"  solver iterations: {model.opt.iterations} (expected 1) "
              f"{'OK' if model.opt.iterations == 1 else 'FAIL'}")
        print(f"  ls_iterations: {model.opt.ls_iterations} (expected 5) "
              f"{'OK' if model.opt.ls_iterations == 5 else 'FAIL'}")

        # Critical names resolve
        names_to_check = [
            ("body", "base"), ("body", "trunk_assembly"),
            ("geom", "left_foot_bottom_tpu"), ("geom", "right_foot_bottom_tpu"),
            ("geom", "floor"),
            ("site", "imu"), ("site", "left_foot"), ("site", "right_foot"),
            ("sensor", "gyro"), ("sensor", "accelerometer"), ("sensor", "upvector"),
            ("joint", "left_knee"), ("joint", "right_knee"),
            ("actuator", "left_knee"), ("actuator", "right_knee"),
        ]
        for kind, name in names_to_check:
            try:
                getattr(model, kind)(name).id
                print(f"  {kind} '{name}': OK")
            except Exception as e:
                print(f"  {kind} '{name}': FAIL ({e})")
                raise SystemExit(1)

        # Foot symmetry check at home keyframe
        data = mujoco.MjData(model)
        data.qpos[:] = model.keyframe("home").qpos
        mujoco.mj_forward(model, data)
        lz = data.site_xpos[model.site("left_foot").id][2]
        rz = data.site_xpos[model.site("right_foot").id][2]
        diff_mm = abs(lz - rz) * 1000
        status = "OK" if diff_mm < 1.0 else "FAIL"
        print(f"  foot symmetry: L={lz:.6f}m R={rz:.6f}m diff={diff_mm:.3f}mm {status}")
        if diff_mm >= 1.0:
            raise SystemExit(1)

        print("=== All checks passed ===")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        print(f"Usage: {sys.argv[0]} <input.xml> <output.xml>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"ERROR: input file does not exist: {input_path}")
        sys.exit(1)

    content = input_path.read_text()

    print(f"Patching {input_path} -> {output_path}")

    # Pre-flight check: fail loudly if CAD is misconfigured
    check_no_antenna_bodies(content)
    print("  + CAD check: no antenna bodies (correct fix_ prefix in OnShape)")

    content = patch_solver_options(content)
    print("  + Added solver options (iterations=1, ls_iterations=5, eulerdamp disabled)")

    content = patch_base_wrapper(content)
    print("  + Wrapped trunk_assembly in 'base' body with freejoint and imu site")

    content = patch_remove_trunk_imu_site(content)
    print("  + Removed imu site from trunk_assembly (now on base)")

    content = patch_remove_flash_geoms(content)
    print("  + Removed flash_light_module and flash_reflector_interface visual geoms")

    content = patch_servo_values(content)
    print(f"  + Pinned servo physics (kp={STS3215_KP}, damping={STS3215_DAMPING}, etc.)")

    content = patch_foot_geom_names(content)
    print("  + Named foot TPU collision geoms")

    content = patch_rename_right_cache(content)
    print("  + Renamed right_cache -> knee_and_ankle_assembly_3 (and _3 -> _4)")

    output_path.write_text(content)
    print(f"Wrote: {output_path}")

    verify_patched_model(output_path)


if __name__ == "__main__":
    main()
