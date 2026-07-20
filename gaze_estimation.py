

import cv2
import numpy as np
import argparse
import os
import sys
import math
from pathlib import Path
from openvino.runtime import Core


# ─────────────────────────────────────────────────────────────────────────────
# Model Wrappers
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    face-detection-adas-0001
    Input  : [1, 3, 384, 672]  BGR image
    Output : [1, 1, N, 7]      [image_id, label, conf, x1, y1, x2, y2] (normalized)
    """
    def __init__(self, core: Core, model_xml: str, device: str = "CPU", threshold: float = 0.5):
        self.threshold = threshold
        model = core.read_model(model_xml)
        self.compiled = core.compile_model(model, device)
        self.input_layer  = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)
        _, _, self.h, self.w = self.input_layer.shape

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.w, self.h))
        blob = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        return blob

    def infer(self, frame: np.ndarray):
        blob = self.preprocess(frame)
        result = self.compiled({self.input_layer: blob})[self.output_layer]  # [1,1,N,7]
        detections = result[0][0]
        fh, fw = frame.shape[:2]
        faces = []
        for det in detections:
            _, _, conf, x1, y1, x2, y2 = det
            if conf < self.threshold:
                continue
            # Clip to frame
            x1 = max(0, int(x1 * fw))
            y1 = max(0, int(y1 * fh))
            x2 = min(fw, int(x2 * fw))
            y2 = min(fh, int(y2 * fh))
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append({"box": (x1, y1, x2, y2), "conf": float(conf)})
        return faces


class HeadPoseEstimator:
    """
    head-pose-estimation-adas-0001
    Input  : [1, 3, 60, 60]  face crop (BGR)
    Outputs: angle_y_fc (yaw), angle_p_fc (pitch), angle_r_fc (roll) — each [1,1]
    """
    def __init__(self, core: Core, model_xml: str, device: str = "CPU"):
        model = core.read_model(model_xml)
        self.compiled = core.compile_model(model, device)
        self.input_layer = self.compiled.input(0)
        _, _, self.h, self.w = self.input_layer.shape
        # Map output names
        self.yaw_out   = self.compiled.output("angle_y_fc")
        self.pitch_out = self.compiled.output("angle_p_fc")
        self.roll_out  = self.compiled.output("angle_r_fc")

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_crop, (self.w, self.h))
        blob = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        return blob

    def infer(self, face_crop: np.ndarray):
        blob = self.preprocess(face_crop)
        results = self.compiled({self.input_layer: blob})
        yaw   = float(results[self.yaw_out].flatten()[0])
        pitch = float(results[self.pitch_out].flatten()[0])
        roll  = float(results[self.roll_out].flatten()[0])
        return yaw, pitch, roll


class LandmarksDetector:
    """
    landmarks-regression-retail-0009
    Input  : [1, 3, 48, 48]  face crop (BGR)
    Output : [1, 10, 1, 1]   5 landmarks x,y pairs (normalized 0–1)
             Order: left-eye, right-eye, nose-tip, left-mouth, right-mouth
    """
    def __init__(self, core: Core, model_xml: str, device: str = "CPU"):
        model = core.read_model(model_xml)
        self.compiled = core.compile_model(model, device)
        self.input_layer  = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)
        _, _, self.h, self.w = self.input_layer.shape

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_crop, (self.w, self.h))
        blob = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        return blob

    def infer(self, face_crop: np.ndarray, face_box: tuple):
        blob = self.preprocess(face_crop)
        result = self.compiled({self.input_layer: blob})[self.output_layer]  # [1,10,1,1]
        pts = result.flatten()  # 10 values: x0,y0,x1,y1,...
        x1, y1, x2, y2 = face_box
        fw, fh = x2 - x1, y2 - y1
        landmarks = []
        for i in range(5):
            lx = int(pts[2*i]   * fw) + x1
            ly = int(pts[2*i+1] * fh) + y1
            landmarks.append((lx, ly))
        return landmarks  # [(left_eye), (right_eye), (nose), (left_mouth), (right_mouth)]


class GazeEstimator:
    """
    gaze-estimation-adas-0002
    Inputs:
      left_eye_image  : [1, 3, 60, 60]
      right_eye_image : [1, 3, 60, 60]
      head_pose_angles: [1, 3]  (yaw, pitch, roll in degrees)
    Output:
      gaze_vector     : [1, 3]  (x, y, z) unit-ish gaze direction
    """
    def __init__(self, core: Core, model_xml: str, device: str = "CPU"):
        model = core.read_model(model_xml)
        self.compiled = core.compile_model(model, device)
        self.left_eye_in   = self.compiled.input("left_eye_image")
        self.right_eye_in  = self.compiled.input("right_eye_image")
        self.head_pose_in  = self.compiled.input("head_pose_angles")
        self.output_layer  = self.compiled.output(0)
        _, _, self.h, self.w = self.left_eye_in.shape

    def _crop_eye(self, frame: np.ndarray, center: tuple, face_box: tuple) -> np.ndarray:
        """Crop an eye region around the landmark center."""
        x1, y1, x2, y2 = face_box
        face_w = x2 - x1
        eye_size = max(20, int(face_w * 0.25))
        cx, cy = center
        ex1 = max(0, cx - eye_size)
        ey1 = max(0, cy - eye_size)
        ex2 = min(frame.shape[1], cx + eye_size)
        ey2 = min(frame.shape[0], cy + eye_size)
        crop = frame[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            crop = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        return cv2.resize(crop, (self.w, self.h))

    def preprocess_eye(self, eye_crop: np.ndarray) -> np.ndarray:
        return eye_crop.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def infer(self, frame: np.ndarray, left_center: tuple, right_center: tuple,
              face_box: tuple, yaw: float, pitch: float, roll: float):
        left_crop  = self._crop_eye(frame, left_center,  face_box)
        right_crop = self._crop_eye(frame, right_center, face_box)

        left_blob  = self.preprocess_eye(left_crop)
        right_blob = self.preprocess_eye(right_crop)
        head_pose  = np.array([[yaw, pitch, roll]], dtype=np.float32)

        result = self.compiled({
            self.left_eye_in:  left_blob,
            self.right_eye_in: right_blob,
            self.head_pose_in: head_pose,
        })[self.output_layer]

        gx, gy, gz = result.flatten()[:3]
        return float(gx), float(gy), float(gz), left_crop, right_crop


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_axes(frame, center, yaw, pitch, roll, scale=50):
    """Draw head pose axes (3D rotation) on the frame."""
    cy, sy = math.cos(math.radians(yaw)),   math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)),  math.sin(math.radians(roll))

    # Rotation matrix columns give axis directions projected to 2D
    # X-axis (red) — pointing right in head frame
    xaxis = np.array([cy*cr + sy*sp*sr,  cp*sr, -sy*cr + cy*sp*sr])
    # Y-axis (green) — pointing up in head frame
    yaxis = np.array([-cy*sr + sy*sp*cr, cp*cr,  sy*sr + cy*sp*cr])
    # Z-axis (blue) — pointing out of face
    zaxis = np.array([sy*cp, -sp, cy*cp])

    cx, cy_px = center
    for vec, color in zip([xaxis, yaxis, zaxis],
                          [(0, 0, 230), (0, 230, 0), (230, 0, 0)]):
        end = (int(cx + vec[0]*scale), int(cy_px - vec[1]*scale))
        cv2.arrowedLine(frame, (cx, cy_px), end, color, 2, tipLength=0.3)


def draw_gaze(frame, left_center, right_center, gx, gy, scale=150):
    """Draw gaze arrow from each eye center."""
    for center in [left_center, right_center]:
        cx, cy = center
        ex = int(cx + gx * scale)
        ey = int(cy - gy * scale)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.25)


def draw_landmarks(frame, landmarks):
    colors = [(255, 100, 100), (100, 100, 255), (100, 255, 100),
              (255, 165,   0), (255, 165,   0)]
    names  = ["L-Eye", "R-Eye", "Nose", "L-Mouth", "R-Mouth"]
    for (x, y), color, name in zip(landmarks, colors, names):
        cv2.circle(frame, (x, y), 4, color, -1)
        cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, color, 1, cv2.LINE_AA)


def draw_info_panel(frame, face_idx, yaw, pitch, roll, gx, gy, gz, conf):
    """Draw a semi-transparent info box in the top-left corner."""
    lines = [
        f"Face #{face_idx+1}  conf: {conf:.2f}",
        f"Yaw:   {yaw:+.1f} deg",
        f"Pitch: {pitch:+.1f} deg",
        f"Roll:  {roll:+.1f} deg",
        f"Gaze:  ({gx:+.3f}, {gy:+.3f}, {gz:+.3f})",
    ]
    x0, y0, pad = 10, 10 + face_idx * 110, 6
    box_w, line_h = 260, 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0-pad, y0-pad),
                  (x0+box_w, y0 + len(lines)*line_h + pad),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0, y0 + i*line_h + line_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class GazeEstimationPipeline:
    def __init__(self, args):
        core = Core()
        device = args.device
        models_dir = Path(args.models_dir)

        print("[INFO] Loading models...")
        self.face_det  = FaceDetector(
            core,
            str(models_dir / "face-detection-adas-0001" /
                "FP32" / "face-detection-adas-0001.xml"),
            device, args.threshold)

        self.head_pose = HeadPoseEstimator(
            core,
            str(models_dir / "head-pose-estimation-adas-0001" /
                "FP32" / "head-pose-estimation-adas-0001.xml"),
            device)

        self.landmarks = LandmarksDetector(
            core,
            str(models_dir / "landmarks-regression-retail-0009" /
                "FP32" / "landmarks-regression-retail-0009.xml"),
            device)

        self.gaze = GazeEstimator(
            core,
            str(models_dir / "gaze-estimation-adas-0002" /
                "FP32" / "gaze-estimation-adas-0002.xml"),
            device)
        print("[INFO] All models loaded.")

    def process_frame(self, frame: np.ndarray, show_landmarks: bool = True,
                      show_axes: bool = True) -> np.ndarray:
        out = frame.copy()
        faces = self.face_det.infer(frame)

        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face["box"]
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Step 2 – Head pose
            yaw, pitch, roll = self.head_pose.infer(face_crop)

            # Step 3 – Landmarks
            landmarks = self.landmarks.infer(face_crop, face["box"])
            left_eye_center  = landmarks[0]
            right_eye_center = landmarks[1]

            # Step 4 – Gaze estimation
            gx, gy, gz, _, _ = self.gaze.infer(
                frame, left_eye_center, right_eye_center,
                face["box"], yaw, pitch, roll)

            # ── Draw ──────────────────────────────────────────────────────
            # Face bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 100), 2)

            # Landmarks
            if show_landmarks:
                draw_landmarks(out, landmarks)

            # Head pose axes centered on nose tip
            nose = landmarks[2]
            if show_axes:
                draw_axes(out, nose, yaw, pitch, roll)

            # Gaze arrows
            draw_gaze(out, left_eye_center, right_eye_center, gx, gy)

            # Info panel
            draw_info_panel(out, idx, yaw, pitch, roll, gx, gy, gz, face["conf"])

        # FPS placeholder (caller can overlay actual FPS)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="OpenVINO Gaze Estimation Pipeline")
    p.add_argument("--input",       required=True,
                   help="Path to video/image, or '1' / 'cam' for webcam")
    p.add_argument("--models_dir",  default="intel",
                   help="Root folder containing the four model sub-directories")
    p.add_argument("--device",      default="CPU",
                   help="OpenVINO inference device (CPU, GPU, MYRIAD, AUTO)")
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="Face detection confidence threshold")
    p.add_argument("--no_landmarks", action="store_true",
                   help="Hide facial landmarks")
    p.add_argument("--no_axes",      action="store_true",
                   help="Hide head pose axes")
    p.add_argument("--output",      default=None,
                   help="Optional path to save output video")
    return p


def open_input(source: str):
    if source.lower() in ("1", "cam", "webcam"):
        cap = cv2.VideoCapture(1)
    elif os.path.isfile(source):
        cap = cv2.VideoCapture(source)
    else:
        raise FileNotFoundError(f"Input not found: {source}")
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    return cap


def main():
    args = build_parser().parse_args()
    pipeline = GazeEstimationPipeline(args)

    show_lm   = not args.no_landmarks
    show_axes = not args.no_axes

    # Single image
    if os.path.isfile(args.input) and args.input.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        frame = cv2.imread(args.input)
        result = pipeline.process_frame(frame, show_lm, show_axes)
        out_path = args.output or "gaze_output.jpg"
        cv2.imwrite(out_path, result)
        print(f"[INFO] Saved result to {out_path}")
        cv2.imshow("Gaze Estimation", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Video / webcam
    cap = open_input(args.input)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer  = None

    if args.output:
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_src, (w, h))

    import time
    prev_t = time.time()
    print("[INFO] Running... Press 'q' to quit, 's' to save a snapshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame, show_lm, show_axes)

        # FPS overlay
        now  = time.time()
        fps  = 1.0 / max(now - prev_t, 1e-9)
        prev_t = now
        cv2.putText(result, f"FPS: {fps:.1f}", (result.shape[1]-100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

        if writer:
            writer.write(result)

        cv2.imshow("Gaze Estimation", result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            snap = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snap, result)
            print(f"[INFO] Snapshot saved: {snap}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()