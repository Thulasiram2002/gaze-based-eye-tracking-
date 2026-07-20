"""
gaze_to_screen.py
=================
Full implementation of Falch & Lohan (2024):
  "Webcam-based gaze estimation for computer screen interaction"

Extends the base OpenVINO pipeline with:
  - Coordinate-system algebra  (Sections 3.2 – 3.3)
  - 4-point calibration UI     (Section 3.3.2)
  - Regression for STG matrix  (Section 3.3.3)
  - Per-calibration-point refinement (Section 3.3.4)
  - Structure-from-Motion lateral compensation (Section 3.3.5)

Usage
-----
  python gaze_to_screen.py --input 0 --screen_w_mm 597 --screen_h_mm 336 \
                            --screen_w_px 2560 --screen_h_px 1440 \
                            --models_dir intel

Keyboard shortcuts (after calibration):
  q  – quit
  r  – recalibrate
  s  – save snapshot
  d  – toggle debug overlay
"""

import cv2
import numpy as np
import math
import time
import os
from pathlib import Path
from scipy.optimize import minimize
from collections import deque

# ── Try importing OpenVINO ────────────────────────────────────────────────────
try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("[WARN] OpenVINO not found – running in DEMO mode with synthetic gaze.")


# =============================================================================
# OpenVINO Model Wrappers  (unchanged from original pipeline)
# =============================================================================

class FaceDetector:
    def __init__(self, core, model_xml, device="CPU", threshold=0.5):
        self.threshold = threshold
        model = core.read_model(model_xml)
        self.compiled = core.compile_model(model, device)
        self.input_layer  = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)
        _, _, self.h, self.w = self.input_layer.shape

    def preprocess(self, frame):
        resized = cv2.resize(frame, (self.w, self.h))
        return resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def infer(self, frame):
        blob   = self.preprocess(frame)
        result = self.compiled({self.input_layer: blob})[self.output_layer]
        fh, fw = frame.shape[:2]
        faces  = []
        for det in result[0][0]:
            _, _, conf, x1, y1, x2, y2 = det
            if conf < self.threshold:
                continue
            x1 = max(0, int(x1 * fw)); y1 = max(0, int(y1 * fh))
            x2 = min(fw, int(x2 * fw)); y2 = min(fh, int(y2 * fh))
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append({"box": (x1, y1, x2, y2), "conf": float(conf)})
        return faces


class HeadPoseEstimator:
    def __init__(self, core, model_xml, device="CPU"):
        model = core.read_model(model_xml)
        self.compiled = core.compile_model(model, device)
        self.input_layer = self.compiled.input(0)
        _, _, self.h, self.w = self.input_layer.shape
        self.yaw_out   = self.compiled.output("angle_y_fc")
        self.pitch_out = self.compiled.output("angle_p_fc")
        self.roll_out  = self.compiled.output("angle_r_fc")

    def preprocess(self, face_crop):
        resized = cv2.resize(face_crop, (self.w, self.h))
        return resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def infer(self, face_crop):
        blob    = self.preprocess(face_crop)
        results = self.compiled({self.input_layer: blob})
        return (float(results[self.yaw_out].flatten()[0]),
                float(results[self.pitch_out].flatten()[0]),
                float(results[self.roll_out].flatten()[0]))


class LandmarksDetector:
    def __init__(self, core, model_xml, device="CPU"):
        model = core.read_model(model_xml)
        self.compiled = core.compile_model(model, device)
        self.input_layer  = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)
        _, _, self.h, self.w = self.input_layer.shape

    def preprocess(self, face_crop):
        resized = cv2.resize(face_crop, (self.w, self.h))
        return resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def infer(self, face_crop, face_box):
        blob   = self.preprocess(face_crop)
        result = self.compiled({self.input_layer: blob})[self.output_layer]
        pts    = result.flatten()
        x1, y1, x2, y2 = face_box
        fw, fh = x2 - x1, y2 - y1
        return [(int(pts[2*i] * fw) + x1,
                 int(pts[2*i+1] * fh) + y1) for i in range(5)]


class GazeEstimator:
    def __init__(self, core, model_xml, device="CPU"):
        model = core.read_model(model_xml)
        self.compiled = core.compile_model(model, device)
        self.left_eye_in  = self.compiled.input("left_eye_image")
        self.right_eye_in = self.compiled.input("right_eye_image")
        self.head_pose_in = self.compiled.input("head_pose_angles")
        self.output_layer = self.compiled.output(0)
        _, _, self.h, self.w = self.left_eye_in.shape

    def _crop_eye(self, frame, center, face_box):
        x1, y1, x2, y2 = face_box
        eye_size = max(20, int((x2 - x1) * 0.25))
        cx, cy   = center
        ex1 = max(0, cx - eye_size); ey1 = max(0, cy - eye_size)
        ex2 = min(frame.shape[1], cx + eye_size)
        ey2 = min(frame.shape[0], cy + eye_size)
        crop = frame[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            crop = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        return cv2.resize(crop, (self.w, self.h))

    def preprocess_eye(self, eye_crop):
        return eye_crop.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def infer(self, frame, left_center, right_center, face_box, yaw, pitch, roll):
        left_blob  = self.preprocess_eye(self._crop_eye(frame, left_center,  face_box))
        right_blob = self.preprocess_eye(self._crop_eye(frame, right_center, face_box))
        head_pose  = np.array([[yaw, pitch, roll]], dtype=np.float32)
        result = self.compiled({
            self.left_eye_in:  left_blob,
            self.right_eye_in: right_blob,
            self.head_pose_in: head_pose,
        })[self.output_layer]
        gx, gy, gz = result.flatten()[:3]
        return float(gx), float(gy), float(gz)


# =============================================================================
# Section 3.2 – 3.3.4  :  GazeToScreenProjector
# =============================================================================

class GazeToScreenProjector:
    """
    Implements the coordinate-system algebra and regression from the paper.

    Coordinate frames
    -----------------
    G  – gaze frame  (output of OpenVINO model)
    S  – screen frame (origin = top-left, x right, y down, z out of screen)
    W  – world/camera frame

    Key variables
    -------------
    S_R_G  : 3×3 rotation  G → S  (Eq. 8, fixed for webcam-on-top-of-screen)
    S_t_G  : 3-vector translation from S origin to G origin, in S coords
             ← solved by regression over calibration points (Eqs 9-12)
    """

    def __init__(self, screen_w_mm: float, screen_h_mm: float):
        self.W_mm  = screen_w_mm
        self.H_mm  = screen_h_mm

        # Eq. 8 – webcam x/y-plane parallel to screen x/y-plane
        self.S_R_G = np.array([[-1,  0, 0],
                                [ 0, -1, 0],
                                [ 0,  0, 1]], dtype=np.float64)

        # Gz in gaze frame = S_R_G^T @ [0,0,1]^T
        self.G_z = self.S_R_G.T @ np.array([0., 0., 1.])

        # Learned via calibration
        self.S_t_G          = None   # (3,) translation vector
        self.calib_matrices  = []    # per-calibration-point STG (Section 3.3.4)
        self.calib_pts_mm    = []    # screen positions of calib points (mm)
        self.is_calibrated   = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _lambda(self, g_hat: np.ndarray, G_t_S: np.ndarray) -> float:
        """Eq. 7 – compute scaling factor λ."""
        denom = self.G_z @ g_hat
        if abs(denom) < 1e-9:
            return 0.0
        return float((self.G_z @ G_t_S) / denom)

    def _G_t_S_from_S_t_G(self, S_t_G: np.ndarray) -> np.ndarray:
        """G_t_S = -S_R_G^T @ S_t_G  (inverse transform translation)."""
        return -self.S_R_G.T @ S_t_G

    def _project_one(self, g_hat: np.ndarray, S_t_G: np.ndarray) -> np.ndarray:
        """Eq. 4 – project unit gaze vector onto screen, return S coords (3,)."""
        G_t_S = self._G_t_S_from_S_t_G(S_t_G)
        lam   = self._lambda(g_hat, G_t_S)
        return self.S_R_G @ (lam * g_hat) + S_t_G

    # ── Regression  (Eqs 9-12) ────────────────────────────────────────────────

    def calibrate(self,
                  calib_pts_mm: list,   # [(x_mm, y_mm), …]  screen positions
                  gaze_vectors: list):  # [(gx, gy, gz), …]  unit vectors
        """
        Solve least-squares regression to find S_t_G.
        Each entry in gaze_vectors is the MEDIAN gaze vector collected
        while the user fixated the corresponding calibration point.
        """
        calib_pts_mm = [np.array([p[0], p[1], 0.], dtype=np.float64)
                        for p in calib_pts_mm]
        gaze_vectors = [np.array(g, dtype=np.float64) for g in gaze_vectors]

        def loss(t_G):
            t_G = np.array(t_G)
            total = 0.0
            for g_tilde, g_hat in zip(calib_pts_mm, gaze_vectors):
                s_g = self._project_one(g_hat, t_G)
                total += float(np.sum((g_tilde - s_g) ** 2))
            return total

        # Initial guess: user 800 mm away, centred on screen
        x0 = [self.W_mm / 2, self.H_mm / 2, -800.0]
        result = minimize(loss, x0, method="Nelder-Mead",
                          options={"maxiter": 20000, "xatol": 0.1, "fatol": 0.1})
        self.S_t_G = np.array(result.x)

        # Build per-calibration-point matrices (Section 3.3.4, Eqs 13-15)
        self.calib_matrices = []
        self.calib_pts_mm   = calib_pts_mm
        G_t_S = self._G_t_S_from_S_t_G(self.S_t_G)
        for g_tilde, g_hat in zip(calib_pts_mm, gaze_vectors):
            lam = self._lambda(g_hat, G_t_S)
            # Per-point STG gives zero error at this point
            S_t_p  = g_tilde
            S_t_G_p = S_t_p - self.S_R_G @ (lam * g_hat)
            self.calib_matrices.append(S_t_G_p)

        self.is_calibrated = True
        dist = -self.S_t_G[2]
        print(f"[CALIB] Regression converged. "
              f"Estimated distance: {dist:.1f} mm | "
              f"Position on screen: ({self.S_t_G[0]:.1f}, {self.S_t_G[1]:.1f}) mm")

    # ── Project gaze onto screen (Section 3.3.4 refinement) ──────────────────

    def project(self, gaze_unit: tuple) -> tuple:
        """
        Returns (x_mm, y_mm) on screen, or None if not calibrated.
        Uses initial regression estimate, then refines with nearest
        calibration-point matrix (median of the two estimates).
        """
        if not self.is_calibrated:
            return None

        g_hat = np.array(gaze_unit, dtype=np.float64)

        # Step 1 – initial estimate from regression matrix
        s_g_init = self._project_one(g_hat, self.S_t_G)

        if not self.calib_matrices:
            return (float(s_g_init[0]), float(s_g_init[1]))

        # Step 2 – find nearest calibration point on screen
        dists = [np.linalg.norm(s_g_init[:2] - p[:2])
                 for p in self.calib_pts_mm]
        nearest_idx = int(np.argmin(dists))
        S_t_G_nearest = self.calib_matrices[nearest_idx]

        # Step 3 – refined estimate from that matrix
        s_g_refined = self._project_one(g_hat, S_t_G_nearest)

        # Step 4 – median (paper Section 3.3.4)
        x = (s_g_init[0] + s_g_refined[0]) / 2.0
        y = (s_g_init[1] + s_g_refined[1]) / 2.0
        return (float(x), float(y))

    def mm_to_px(self, x_mm: float, y_mm: float,
                 screen_w_px: int, screen_h_px: int) -> tuple:
        """Convert screen mm coordinates to pixel coordinates."""
        px = int(np.clip(x_mm / self.W_mm * screen_w_px, 0, screen_w_px - 1))
        py = int(np.clip(y_mm / self.H_mm * screen_h_px, 0, screen_h_px - 1))
        return px, py

    def estimated_distance_mm(self) -> float:
        if self.S_t_G is None:
            return 0.0
        return float(-self.S_t_G[2])


# =============================================================================
# Section 3.3.5  :  Structure-from-Motion lateral compensation
# =============================================================================

class SfMHeadTracker:
    """
    Tracks lateral (left/right) head movement using facial landmarks
    between consecutive frames to estimate the essential matrix and
    recover WRG, Wt_G (unit translation in camera/world frame).

    Simplified implementation:
      - Uses optical flow on landmark points instead of full feature matching
      - Estimates lateral shift from centroid displacement
      - Applies correction to the projected gaze point
    """

    def __init__(self):
        self.prev_landmarks  = None
        self.prev_gray       = None
        self.lateral_shift_mm = 0.0        # accumulated x-shift estimate
        self.alpha           = 0.85        # EMA smoothing

        # Calibration: pixels-per-mm in the facial region (rough estimate)
        # Will be updated from face box size and assumed 150mm face width
        self.px_per_mm = 5.0
        self.FACE_WIDTH_MM = 150.0         # approximate human face width

    def update(self, gray_frame: np.ndarray,
               landmarks: list, face_box: tuple) -> float:
        """
        Returns estimated lateral shift in mm (positive = moved right).
        Updates internal state for next frame.
        """
        pts_curr = np.array(landmarks[:4], dtype=np.float32)  # use eye + nose pts

        shift_mm = 0.0
        if self.prev_landmarks is not None and self.prev_gray is not None:
            # Track previous landmark positions with Lucas-Kanade optical flow
            pts_prev = np.array(self.prev_landmarks[:4], dtype=np.float32)
            pts_tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray_frame,
                pts_prev.reshape(-1, 1, 2), None,
                winSize=(21, 21), maxLevel=3)

            if pts_tracked is not None and status is not None:
                good = status.flatten() == 1
                if good.sum() >= 2:
                    dx_px = float(np.median(
                        pts_tracked.reshape(-1, 2)[good, 0] -
                        pts_prev.reshape(-1, 2)[good, 0]))
                    # Estimate px/mm from face box width
                    x1, y1, x2, y2 = face_box
                    face_w_px = x2 - x1
                    if face_w_px > 10:
                        self.px_per_mm = face_w_px / self.FACE_WIDTH_MM
                    shift_mm = dx_px / self.px_per_mm

        # EMA filter
        self.lateral_shift_mm = (self.alpha * self.lateral_shift_mm +
                                 (1 - self.alpha) * shift_mm)

        self.prev_landmarks = landmarks
        self.prev_gray      = gray_frame.copy()
        return float(self.lateral_shift_mm)


# =============================================================================
# Calibration UI
# =============================================================================

CALIB_MARGIN_FRAC = 0.1   # fraction of screen kept as margin for calib points

def generate_calib_points(screen_w_mm, screen_h_mm):
    """
    Four calibration points near screen corners (Section 3.3.2).
    Returns list of (x_mm, y_mm).
    """
    mx = screen_w_mm * CALIB_MARGIN_FRAC
    my = screen_h_mm * CALIB_MARGIN_FRAC
    return [
        (mx,               my),
        (screen_w_mm - mx, my),
        (mx,               screen_h_mm - my),
        (screen_w_mm - mx, screen_h_mm - my),
    ]


class CalibrationManager:
    """
    Manages the 4-point calibration sequence.
    For each point, collects ~2 s of gaze vectors and keeps the median.
    Erroneous vectors (collected during gaze transitions) are filtered
    by discarding the first 0.5 s of each fixation.
    """
    FIXATION_TOTAL_SEC  = 2.5
    FIXATION_SKIP_SEC   = 0.5   # discard saccade / transition period
    MIN_SAMPLES         = 10

    def __init__(self, calib_pts_mm: list):
        self.pts        = calib_pts_mm
        self.n_pts      = len(calib_pts_mm)
        self.current    = 0            # index of active calibration point
        self.phase_start = None
        self.collected  = [[] for _ in range(self.n_pts)]
        self.done       = False
        self.medians    = None         # computed after all points finished

    def reset(self):
        self.current     = 0
        self.phase_start = None
        self.collected   = [[] for _ in range(self.n_pts)]
        self.done        = False
        self.medians     = None

    def current_pt_mm(self):
        return self.pts[self.current]

    def feed(self, gaze_vector: tuple, t: float) -> bool:
        """
        Feed a gaze sample at time t.
        Returns True when all calibration points are complete.
        """
        if self.done:
            return True

        if self.phase_start is None:
            self.phase_start = t

        elapsed = t - self.phase_start

        # Skip saccade period, then collect
        if elapsed >= self.FIXATION_SKIP_SEC:
            self.collected[self.current].append(np.array(gaze_vector))

        if elapsed >= self.FIXATION_TOTAL_SEC:
            # Advance to next point
            self.current    += 1
            self.phase_start = None
            if self.current >= self.n_pts:
                self._compute_medians()
                self.done = True
                return True
        return False

    def _compute_medians(self):
        self.medians = []
        for samples in self.collected:
            arr = np.array(samples) if samples else np.zeros((1, 3))
            med = np.median(arr, axis=0)
            # Normalise to unit vector
            norm = np.linalg.norm(med)
            self.medians.append(med / norm if norm > 1e-9 else med)

    def progress(self, t: float) -> float:
        """Returns fraction [0,1] of current fixation phase elapsed."""
        if self.phase_start is None:
            return 0.0
        return min(1.0, (t - self.phase_start) / self.FIXATION_TOTAL_SEC)

    def samples_collected(self) -> int:
        return len(self.collected[self.current]) if self.current < self.n_pts else 0


# =============================================================================
# Visualization helpers
# =============================================================================

def draw_calibration_overlay(frame, calib: CalibrationManager,
                             screen_w_mm, screen_h_mm, t):
    """Draw the calibration dot, progress ring, and instructions."""
    fh, fw = frame.shape[:2]
    overlay = frame.copy()

    # Dim background
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    if calib.done:
        return

    pt_mm = calib.current_pt_mm()
    px = int(pt_mm[0] / screen_w_mm * fw)
    py = int(pt_mm[1] / screen_h_mm * fh)
    progress = calib.progress(t)

    # Outer progress ring
    axes = (30, 30)
    angle = int(360 * progress)
    cv2.ellipse(frame, (px, py), axes, -90, 0, angle, (0, 220, 120), 4)
    cv2.circle(frame, (px, py), 12, (0, 255, 160), -1)
    cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)

    # Point index label
    label = f"Point {calib.current + 1} / {calib.n_pts}"
    cv2.putText(frame, label, (px + 40, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    # Instruction bar
    inst = "CALIBRATION  |  Look at the dot and keep head still"
    tw, _ = cv2.getTextSize(inst, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0], None
    cv2.putText(frame, inst, (fw // 2 - 280, fh - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1, cv2.LINE_AA)

    n = calib.samples_collected()
    cv2.putText(frame, f"Samples: {n}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)


def draw_gaze_point(frame, gx_px, gy_px, history: deque, debug=False):
    """Draw the projected gaze point and a trailing history."""
    # History trail
    for i, (hx, hy) in enumerate(history):
        alpha = int(80 * (i / max(len(history), 1)))
        cv2.circle(frame, (hx, hy), 5, (0, alpha + 100, alpha + 180), -1)

    # Main gaze circle
    cv2.circle(frame, (gx_px, gy_px), 20, (0, 255, 200),  2)
    cv2.circle(frame, (gx_px, gy_px),  6, (0, 255, 200), -1)

    if debug:
        cv2.putText(frame, f"({gx_px}, {gy_px})",
                    (gx_px + 25, gy_px - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)


def draw_hud(frame, dist_mm, lateral_mm, fps, debug):
    fh, fw = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Distance: {dist_mm:.0f} mm",
        f"Lateral: {lateral_mm:+.1f} mm",
    ]
    if debug:
        lines.append("[d] hide debug")
    else:
        lines.append("[d] show debug")

    x0, y0 = 10, 10
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0, y0 + 22 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1, cv2.LINE_AA)

    # Calibration reminder
    cv2.putText(frame, "[r] recalibrate  [s] snapshot  [q] quit",
                (fw - 360, fh - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)


def draw_gaze_map(frame, history, screen_w_px, screen_h_px):
    """Mini heatmap thumbnail in corner showing gaze history."""
    fh, fw = frame.shape[:2]
    MAP_W, MAP_H = 200, 120
    x0, y0 = fw - MAP_W - 10, fh - MAP_H - 10

    bg = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
    cv2.rectangle(bg, (0, 0), (MAP_W - 1, MAP_H - 1), (40, 40, 60), -1)

    for hx, hy in history:
        mx = int(hx / screen_w_px * MAP_W)
        my = int(hy / screen_h_px * MAP_H)
        if 0 <= mx < MAP_W and 0 <= my < MAP_H:
            cv2.circle(bg, (mx, my), 3, (0, 200, 150), -1)

    frame[y0:y0 + MAP_H, x0:x0 + MAP_W] = \
        cv2.addWeighted(frame[y0:y0 + MAP_H, x0:x0 + MAP_W], 0.3, bg, 0.7, 0)
    cv2.rectangle(frame, (x0, y0), (x0 + MAP_W, y0 + MAP_H), (80, 80, 120), 1)
    cv2.putText(frame, "Gaze map", (x0 + 4, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 200), 1)


# =============================================================================
# Demo / synthetic gaze  (when OpenVINO not available)
# =============================================================================

class SyntheticGazePipeline:
    """Generates a synthetic moving gaze vector for demo purposes."""

    def __init__(self):
        self.t0 = time.time()

    def process_frame(self, frame):
        t  = time.time() - self.t0
        gx = 0.05 * math.sin(t * 0.7)
        gy = 0.03 * math.cos(t * 0.5)
        gz = math.sqrt(max(0.0, 1.0 - gx**2 - gy**2))
        return (gx, gy, gz), frame


# =============================================================================
# Main application
# =============================================================================

# =============================================================================
# ★  CONFIGURATION  — edit these values to match your setup
# =============================================================================

CONFIG = {
    # Webcam index (0 = default camera) or path to a video file
    "input":        0,

    # Folder that contains the four OpenVINO model sub-directories
    "models_dir":   "intel",

    # OpenVINO inference device: "CPU", "GPU", "AUTO"
    "device":       "CPU",

    # Face detection confidence threshold (0–1)
    "threshold":    0.5,

    # Physical screen dimensions in millimetres
    # Measure the actual visible display area (not the bezel)
    "screen_w_mm":  597.0,
    "screen_h_mm":  336.0,

    # Screen resolution in pixels
    "screen_w_px":  2560,
    "screen_h_px":  1440,

    # Set True to enable Structure-from-Motion lateral head compensation
    "use_sfm":      True,

    # Optional path to save output video (None = don't save)
    "output":       None,
}

# =============================================================================


def open_input(source):
    src = 0 if str(source).lower() in ("0", "cam", "webcam") else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    return cap


def load_openvino_models(cfg):
    core     = Core()
    models_d = Path(cfg["models_dir"])
    face_det  = FaceDetector(
        core,
        str(models_d / "face-detection-adas-0001/FP32/face-detection-adas-0001.xml"),
        cfg["device"], cfg["threshold"])
    head_pose = HeadPoseEstimator(
        core,
        str(models_d / "head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"),
        cfg["device"])
    landmarks = LandmarksDetector(
        core,
        str(models_d / "landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"),
        cfg["device"])
    gaze_est  = GazeEstimator(
        core,
        str(models_d / "gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml"),
        cfg["device"])
    return face_det, head_pose, landmarks, gaze_est


def main():
    cfg   = CONFIG
    SW_MM = cfg["screen_w_mm"]
    SH_MM = cfg["screen_h_mm"]
    SW_PX = cfg["screen_w_px"]
    SH_PX = cfg["screen_h_px"]

    # ── Load models ───────────────────────────────────────────────────────────
    use_synthetic = False
    if OPENVINO_AVAILABLE:
        try:
            face_det, head_pose, landmarks_det, gaze_est = \
                load_openvino_models(cfg)
            print("[INFO] OpenVINO models loaded.")
        except Exception as e:
            print(f"[WARN] Could not load models ({e}). Using synthetic demo.")
            use_synthetic = True
    else:
        use_synthetic = True

    if use_synthetic:
        synth = SyntheticGazePipeline()

    # ── Projection + SfM ─────────────────────────────────────────────────────
    projector = GazeToScreenProjector(SW_MM, SH_MM)
    sfm       = SfMHeadTracker() if cfg["use_sfm"] else None

    # ── Calibration ───────────────────────────────────────────────────────────
    calib_pts = generate_calib_points(SW_MM, SH_MM)
    calib     = CalibrationManager(calib_pts)
    in_calib  = True
    print("[INFO] Starting calibration. Look at each dot for ~2.5 s.")

    # ── Video input ───────────────────────────────────────────────────────────
    cap    = open_input(cfg["input"])
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer  = None
    if cfg["output"]:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(cfg["output"], cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_src, (w, h))

    # ── State ─────────────────────────────────────────────────────────────────
    gaze_history = deque(maxlen=40)
    debug_mode   = False
    prev_t       = time.time()
    lateral_mm   = 0.0

    print("[INFO] Running.  Keys: q=quit  r=recalibrate  s=snapshot  d=debug")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-9)
        prev_t = now

        gaze_vec = None

        # ── Inference ─────────────────────────────────────────────────────────
        if use_synthetic:
            gaze_vec, _ = synth.process_frame(frame)
        else:
            faces = face_det.infer(frame)
            if faces:
                face = faces[0]
                x1, y1, x2, y2 = face["box"]
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    yaw, pitch, roll = head_pose.infer(crop)
                    lmks = landmarks_det.infer(crop, face["box"])
                    gaze_vec = gaze_est.infer(
                        frame, lmks[0], lmks[1], face["box"], yaw, pitch, roll)

                    # SfM lateral tracking
                    if sfm is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        lateral_mm = sfm.update(gray, lmks, face["box"])

                    # Draw face box in debug mode
                    if debug_mode:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 80), 1)
                        for lx, ly in lmks:
                            cv2.circle(frame, (lx, ly), 3, (255, 100, 0), -1)

        # ── Calibration phase ─────────────────────────────────────────────────
        if in_calib:
            draw_calibration_overlay(frame, calib, SW_MM, SH_MM, now)

            if gaze_vec is not None:
                finished = calib.feed(gaze_vec, now)
                if finished:
                    print("[INFO] Calibration data collected. Running regression…")
                    projector.calibrate(calib_pts, calib.medians)
                    in_calib = False
                    print("[INFO] Calibration complete!")
        else:
            # ── Projection phase ─────────────────────────────────────────────
            if gaze_vec is not None:
                result_mm = projector.project(gaze_vec)
                if result_mm is not None:
                    xm, ym = result_mm

                    # Apply lateral SfM correction
                    if sfm is not None:
                        xm += lateral_mm

                    gx_px, gy_px = projector.mm_to_px(xm, ym, SW_PX, SH_PX)

                    # Scale to frame for visualization
                    fh, fw = frame.shape[:2]
                    vis_x  = int(np.clip(xm / SW_MM * fw, 0, fw - 1))
                    vis_y  = int(np.clip(ym / SH_MM * fh, 0, fh - 1))
                    gaze_history.append((vis_x, vis_y))

                    draw_gaze_point(frame, vis_x, vis_y, gaze_history, debug_mode)

            dist_mm = projector.estimated_distance_mm()
            draw_hud(frame, dist_mm, lateral_mm, fps, debug_mode)
            draw_gaze_map(frame, gaze_history, fw if not use_synthetic else frame.shape[1],
                          fh if not use_synthetic else frame.shape[0])

        # ── FPS ───────────────────────────────────────────────────────────────
        cv2.putText(frame, f"FPS {fps:.0f}",
                    (frame.shape[1] - 90, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 160), 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        cv2.imshow("Gaze-to-Screen", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            calib.reset()
            gaze_history.clear()
            in_calib = True
            projector.is_calibrated = False
            print("[INFO] Recalibrating…")
        elif key == ord("s"):
            snap = f"snapshot_{int(now)}.jpg"
            cv2.imwrite(snap, frame)
            print(f"[INFO] Saved {snap}")
        elif key == ord("d"):
            debug_mode = not debug_mode

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()