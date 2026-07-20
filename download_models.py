"""
download_models.py  —  Download all four OpenVINO IR models via omz_downloader.

Models:
  face-detection-adas-0001          FP32   (replaces deprecated binary-0001)
  head-pose-estimation-adas-0001    FP32
  landmarks-regression-retail-0009  FP32
  gaze-estimation-adas-0002         FP32

Usage:
  python download_models.py [--output_dir intel]
"""

import subprocess
import sys
import shutil
import argparse
from pathlib import Path

MODELS = [
    ("face-detection-adas-0001",         "FP32"),
    ("head-pose-estimation-adas-0001",   "FP32"),
    ("landmarks-regression-retail-0009", "FP32"),
    ("gaze-estimation-adas-0002",        "FP32"),
]


def run_omz(model: str, precision: str, output_dir: str) -> bool:
    cli = shutil.which("omz_downloader")
    if cli:
        cmd = [cli, "--name", model, "--output_dir", output_dir, "--precision", precision]
    else:
        cmd = [sys.executable, "-m", "omz_downloader",
               "--name", model, "--output_dir", output_dir, "--precision", precision]
    print(f"  {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=False)
    return r.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="intel")
    args = p.parse_args()

    failed = []
    for model, precision in MODELS:
        print(f"\n[MODEL] {model}  ({precision})")
        ok = run_omz(model, precision, args.output_dir)
        if ok:
            print(f"  [OK]")
        else:
            failed.append(model)

    print(f"\n{'='*60}")
    if not failed:
        print("[SUCCESS] All 4 models downloaded!")
        print(f"\nRun the pipeline:")
        print(f"  python gaze_estimation.py --input cam --models_dir {args.output_dir}")
    else:
        print(f"[FAILED] {failed}")
        print("\nomz_downloader not found. Install it with:")
        print("  pip install openvino-dev")


if __name__ == "__main__":
    main()