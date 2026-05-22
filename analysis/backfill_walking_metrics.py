"""Backfill walking metrics into an existing TF events file for a training run.

Useful when training was run before the walking eval hook was working.
"""
import argparse
import glob
from pathlib import Path
from tensorboardX import SummaryWriter
from analysis.quick_walking_eval import assess_walking


def step_from_name(p):
    return int(Path(p).stem.split("_")[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--min_step", type=int, default=0)
    args = parser.parse_args()

    onnx_files = sorted(glob.glob(str(Path(args.run_dir) / "*.onnx")), key=step_from_name)
    onnx_files = [f for f in onnx_files if step_from_name(f) >= args.min_step]

    print(f"Backfilling {len(onnx_files)} checkpoints into {args.run_dir}")
    writer = SummaryWriter(log_dir=args.run_dir)

    for f in onnx_files:
        step = step_from_name(f)
        try:
            m = assess_walking(f)
            for k, v in m.items():
                writer.add_scalar(k, float(v), step)
            print(f"  step {step:>10,}: HEADLINE={m['walking/HEADLINE']:.3f} "
                  f"cold={m['walking/cold_start_avg']:.3f} resp={m['walking/responsiveness_avg']:.3f}")
        except Exception as e:
            print(f"  step {step:>10,}: failed - {e}")

    writer.flush()
    writer.close()
    print("Done. Refresh TensorBoard to see updates.")


if __name__ == "__main__":
    main()
