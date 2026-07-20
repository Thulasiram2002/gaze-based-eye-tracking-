"""
demo_synthetic.py  —  Synthetic unit-test / quick demo (no camera needed).

Creates a dummy BGR frame, runs each model wrapper's preprocess step,
and prints shapes — useful to verify model loading before running the
full pipeline against real input.

Usage:
  python demo_synthetic.py --models_dir intel
"""

import numpy as np
import argparse
from pathlib import Path
from openvino.runtime import Core

from gaze_estimation import (
    FaceDetector, HeadPoseEstimator,
    LandmarksDetector, GazeEstimator,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="intel")
    p.add_argument("--device",     default="CPU")
    args = p.parse_args()

    core = Core()
    md   = Path(args.models_dir)
    dev  = args.device

    print("=== Model Load Test ===")

    fd = FaceDetector(
        core,
        str(md / "face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml"),
        dev)
    print(f"[OK] FaceDetector     input shape: {fd.input_layer.shape}")

    hp = HeadPoseEstimator(
        core,
        str(md / "head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"),
        dev)
    print(f"[OK] HeadPoseEstimator input shape: {hp.input_layer.shape}")

    lm = LandmarksDetector(
        core,
        str(md / "landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"),
        dev)
    print(f"[OK] LandmarksDetector input shape: {lm.input_layer.shape}")

    gz = GazeEstimator(
        core,
        str(md / "gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml"),
        dev)
    print(f"[OK] GazeEstimator   left_eye input shape: {gz.left_eye_in.shape}")

    print("\n=== Synthetic Inference (random noise) ===")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Fake a face box covering most of frame
    face_box = (80, 60, 400, 420)
    x1, y1, x2, y2 = face_box
    face_crop = dummy_frame[y1:y2, x1:x2]

    yaw, pitch, roll = hp.infer(face_crop)
    print(f"Head pose  → yaw={yaw:.2f}  pitch={pitch:.2f}  roll={roll:.2f}")

    landmarks = lm.infer(face_crop, face_box)
    print(f"Landmarks  → {landmarks}")

    left_eye, right_eye = landmarks[0], landmarks[1]
    gx, gy, gz_val, _, _ = gz.infer(dummy_frame, left_eye, right_eye,
                                     face_box, yaw, pitch, roll)
    print(f"Gaze vec   → gx={gx:.4f}  gy={gy:.4f}  gz={gz_val:.4f}")
    print("\n[PASS] All models loaded and ran successfully on synthetic input.")


if __name__ == "__main__":
    main()
