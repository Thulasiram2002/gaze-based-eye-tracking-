"""
gaze_screen_1080p.py
═══════════════════════════════════════════════════════════════════════════════
OpenVINO Gaze-to-Screen  ·  Falch & Lohan (2024)  ·  1080p Edition

Pipeline
────────
  face-detection-adas-0001         → face bounding boxes
  head-pose-estimation-adas-0001   → yaw / pitch / roll
  landmarks-regression-retail-0009 → 5 facial landmarks (eye centres)
  gaze-estimation-adas-0002        → 3-D unit gaze vector (gx, gy, gz)

Screen projection  (paper Sections 3.2 – 3.3.4)
────────────────────────────────────────────────
  • S_R_G  rotation matrix  G → S  (Eq. 8)
  • Regression over 9 calibration points to find S_t_G  (Eqs 9-12)
  • Per-calibration-point refinement + median blend  (Eqs 13-15)

SfM lateral compensation  (paper Section 3.3.5)
───────────────────────────────────────────────
  • Lucas-Kanade optical flow on facial landmarks

Calibration UI  (Tkinter fullscreen)
─────────────────────────────────────
  • 9-point grid  (3 × 3)
  • 2 s stabilisation + 80-frame collection per point
  • 1.5σ outlier rejection
  • Live purple preview dot after 3+ points
  • Saves  calibration.pkl

═══════════════════════════════════════════════════════════════════════════════
CONFIGURATION — edit the block below, then just run:
    python gaze_screen_1080p.py
═══════════════════════════════════════════════════════════════════════════════
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import math, time, json, pickle, threading, os
from pathlib import Path
from datetime import datetime
from collections import deque
from scipy.optimize import minimize

# ── Tkinter ───────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, messagebox

# ── OpenVINO (optional — falls back to synthetic demo) ───────────────────────
try:
    from openvino.runtime import Core
    OPENVINO_OK = True
except ImportError:
    OPENVINO_OK = False
    print("[WARN] OpenVINO not installed – running in DEMO mode.")


# ═════════════════════════════════════════════════════════════════════════════
#  ★  CONFIGURATION  (edit here)
# ═════════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Camera ──────────────────────────────────────────────────────────────
    "camera_id":        0,         # webcam index

    # ── OpenVINO ────────────────────────────────────────────────────────────
    "models_dir":       "intel",   # folder containing the 4 model sub-dirs
    "device":           "CPU",     # "CPU" | "GPU" | "AUTO"
    "det_threshold":    0.5,

    # ── Physical screen (1080p monitor) ─────────────────────────────────────
    # Measure the VISIBLE display area with a ruler (not the bezel)
    "screen_w_mm":      527.0,     # ~23.5" 16:9  → adjust to your monitor
    "screen_h_mm":      296.0,
    "screen_w_px":      1920,
    "screen_h_px":      1080,

    # ── Calibration ─────────────────────────────────────────────────────────
    "stabilise_sec":    2.0,       # wait before collecting
    "collect_frames":   150,        # frames per calibration point
    "outlier_sigma":    1.2,       # σ-threshold for outlier rejection

    # ── SfM lateral compensation ─────────────────────────────────────────────
    "use_sfm":          True,

    # ── Gaze cursor EMA smoothing (0 = no smooth, 1 = frozen) ────────────────
    "ema_alpha":        0.20,
}
# ══════════════════════════9═══════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#  PART 1 · OpenVINO model wrappers
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    def __init__(self, core, xml, device="CPU", threshold=0.5):
        self.thr = threshold
        m = core.read_model(xml)
        self.net = core.compile_model(m, device)
        self.inp = self.net.input(0)
        self.out = self.net.output(0)
        _, _, self.h, self.w = self.inp.shape

    def infer(self, frame):
        blob = cv2.resize(frame, (self.w, self.h))
        blob = blob.transpose(2,0,1)[np.newaxis].astype(np.float32)
        dets = self.net({self.inp: blob})[self.out][0][0]
        fh, fw = frame.shape[:2]
        out = []
        for d in dets:
            _, _, c, x1, y1, x2, y2 = d
            if c < self.thr: continue
            x1=max(0,int(x1*fw)); y1=max(0,int(y1*fh))
            x2=min(fw,int(x2*fw)); y2=min(fh,int(y2*fh))
            if x2>x1 and y2>y1:
                out.append({"box":(x1,y1,x2,y2),"conf":float(c)})
        return out


class HeadPoseEstimator:
    def __init__(self, core, xml, device="CPU"):
        m = core.read_model(xml)
        self.net = core.compile_model(m, device)
        self.inp = self.net.input(0)
        self.yo = self.net.output("angle_y_fc")
        self.po = self.net.output("angle_p_fc")
        self.ro = self.net.output("angle_r_fc")
        _, _, self.h, self.w = self.inp.shape

    def infer(self, crop):
        b = cv2.resize(crop,(self.w,self.h)).transpose(2,0,1)[np.newaxis].astype(np.float32)
        r = self.net({self.inp:b})
        return float(r[self.yo].flat[0]), float(r[self.po].flat[0]), float(r[self.ro].flat[0])


class LandmarksDetector:
    def __init__(self, core, xml, device="CPU"):
        m = core.read_model(xml)
        self.net = core.compile_model(m, device)
        self.inp = self.net.input(0)
        self.out = self.net.output(0)
        _, _, self.h, self.w = self.inp.shape

    def infer(self, crop, box):
        b = cv2.resize(crop,(self.w,self.h)).transpose(2,0,1)[np.newaxis].astype(np.float32)
        pts = self.net({self.inp:b})[self.out].flatten()
        x1,y1,x2,y2 = box
        fw,fh = x2-x1, y2-y1
        return [(int(pts[2*i]*fw)+x1, int(pts[2*i+1]*fh)+y1) for i in range(5)]


class GazeEstimator:
    def __init__(self, core, xml, device="CPU"):
        m = core.read_model(xml)
        self.net = core.compile_model(m, device)
        self.li  = self.net.input("left_eye_image")
        self.ri  = self.net.input("right_eye_image")
        self.hi  = self.net.input("head_pose_angles")
        self.out = self.net.output(0)
        _, _, self.h, self.w = self.li.shape

    def _eye(self, frame, c, box):
        x1,y1,x2,y2=box; sz=max(20,int((x2-x1)*0.25))
        cx,cy=c
        crop=frame[max(0,cy-sz):min(frame.shape[0],cy+sz),
                   max(0,cx-sz):min(frame.shape[1],cx+sz)]
        if crop.size==0: crop=np.zeros((self.h,self.w,3),dtype=np.uint8)
        return cv2.resize(crop,(self.w,self.h))

    def infer(self, frame, lc, rc, box, yaw, pitch, roll):
        def b(c): return c.transpose(2,0,1)[np.newaxis].astype(np.float32)
        hp = np.array([[yaw,pitch,roll]],dtype=np.float32)
        r  = self.net({self.li:b(self._eye(frame,lc,box)),
                       self.ri:b(self._eye(frame,rc,box)),
                       self.hi:hp})[self.out]
        gx,gy,gz = r.flatten()[:3]
        return float(gx),float(gy),float(gz)


# ─────────────────────────────────────────────────────────────────────────────
#  PART 2 · Threaded inference engine
# ─────────────────────────────────────────────────────────────────────────────

class GazeEngine:
    """Thread-safe inference engine.  engine.predict() → dict | None"""

    def __init__(self):
        self._lock    = threading.Lock()
        self._result  = None
        self._frame   = None
        self._running = False
        self._thread  = None
        self._cap     = None
        self._models  = None

    def load(self, cfg):
        md = Path(cfg["models_dir"])
        if not (md/"face-detection-adas-0001").exists() and \
               (md/"intel"/"face-detection-adas-0001").exists():
            md = md/"intel"
        core = Core()
        self._models = {
            "face": FaceDetector(core,
                str(md/"face-detection-adas-0001/FP32/face-detection-adas-0001.xml"),
                cfg["device"], cfg["det_threshold"]),
            "pose": HeadPoseEstimator(core,
                str(md/"head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"),
                cfg["device"]),
            "lmk":  LandmarksDetector(core,
                str(md/"landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"),
                cfg["device"]),
            "gaze": GazeEstimator(core,
                str(md/"gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml"),
                cfg["device"]),
        }
        self._cap = cv2.VideoCapture(cfg["camera_id"])
        print(f"[Engine] OpenVINO models loaded from {md}")

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01); continue
            r = self._infer(frame)
            with self._lock:
                self._result = r
                self._frame  = frame.copy()

    def _infer(self, frame):
        m = self._models
        faces = m["face"].infer(frame)
        if not faces: return None
        face = faces[0]; box = face["box"]
        x1,y1,x2,y2 = box
        crop = frame[y1:y2,x1:x2]
        if crop.size == 0: return None
        yaw,pitch,roll = m["pose"].infer(crop)
        lms            = m["lmk"].infer(crop, box)
        gx,gy,gz       = m["gaze"].infer(frame,lms[0],lms[1],box,yaw,pitch,roll)
        return {"gaze3d":(gx,gy,gz),"landmarks":lms,
                "head_pose":(yaw,pitch,roll),"face_box":box,"conf":face["conf"]}

    def predict(self):
        with self._lock: return self._result

    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


class SyntheticEngine:
    """Demo engine – no camera needed."""
    def __init__(self):
        self._t0 = time.time()
    def load(self, cfg): pass
    def start(self): pass
    def stop(self): pass
    def predict(self):
        t  = time.time()-self._t0
        gx = 0.06*math.sin(t*0.6)
        gy = 0.04*math.cos(t*0.4)
        gz = math.sqrt(max(0.0, 1-gx**2-gy**2))
        return {"gaze3d":(gx,gy,gz),"landmarks":None,"head_pose":(0,0,0),
                "face_box":(0,0,0,0),"conf":1.0}
    def get_frame(self): return None


# ─────────────────────────────────────────────────────────────────────────────
#  PART 3 · Paper math: GazeToScreenProjector  (Sections 3.2 – 3.3.4)
# ─────────────────────────────────────────────────────────────────────────────

class GazeToScreenProjector:
    """
    Projects the unit gaze vector onto the physical screen plane.

    Coordinate frames
    -----------------
    G  – gaze  (output of OpenVINO)
    S  – screen (origin = top-left, x→right, y↓, z out of screen)

    Key equation (Eq. 4):
        ˢg = S_R_G · λ · ᴳĝ + S_t_G

    λ (Eq. 7):
        λ = (ᴳz · ᴳt_S) / (ᴳz · ᴳĝ)

    S_t_G (Eqs 9-12):
        Solved by Nelder-Mead regression over 9 calibration points.
    """

    def __init__(self, w_mm: float, h_mm: float):
        self.W = w_mm; self.H = h_mm
        # Eq. 8: webcam x/y-plane ∥ screen x/y-plane
        self.S_R_G = np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=np.float64)
        self.G_z   = self.S_R_G.T @ np.array([0.,0.,1.])
        self.S_t_G       = None
        self.calib_mats  = []
        self.calib_pts   = []
        self.calibrated  = False

    # ── internal ──────────────────────────────────────────────────────────────
    def _G_t_S(self, S_t_G):
        return -self.S_R_G.T @ S_t_G

    def _lam(self, g_hat, G_t_S):
        d = self.G_z @ g_hat
        return float((self.G_z @ G_t_S) / d) if abs(d) > 1e-9 else 0.

    def _project1(self, g_hat, S_t_G):
        G_t_S = self._G_t_S(S_t_G)
        lam   = self._lam(g_hat, G_t_S)
        return self.S_R_G @ (lam * g_hat) + S_t_G

    # ── calibration (Eqs 9-12) ────────────────────────────────────────────────
    def calibrate(self, pts_mm: list, gazes: list):
        """
        pts_mm : [(x_mm, y_mm), …]  – screen positions of calibration dots
        gazes  : [(gx,gy,gz), …]    – median gaze vectors for each dot
        """
        pts  = [np.array([p[0],p[1],0.]) for p in pts_mm]
        gvec = [np.array(g)/np.linalg.norm(g) for g in gazes]

        def loss(t):
            return sum(np.sum((p - self._project1(g, np.array(t)))**2)
                       for p,g in zip(pts, gvec))

        res = minimize(loss, [self.W/2, self.H/2, -600.],
                       method="Nelder-Mead",
                       options={"maxiter":60000,"xatol":0.05,"fatol":0.05})
        self.S_t_G = np.array(res.x)

        # Per-calibration-point matrices (Eqs 13-15)
        self.calib_pts  = pts
        self.calib_mats = []
        G_t_S = self._G_t_S(self.S_t_G)
        for p, g in zip(pts, gvec):
            lam = self._lam(g, G_t_S)
            self.calib_mats.append(p - self.S_R_G @ (lam * g))
        self.calibrated = True
        print(f"[Proj] S_t_G = {np.round(self.S_t_G,1)}  "
              f"(distance ≈ {-self.S_t_G[2]:.0f} mm)")

    # ── predict (Section 3.3.4 median refinement) ─────────────────────────────
    def project(self, gx, gy, gz) -> tuple:
        """Returns (x_mm, y_mm) on screen, or None."""
        if not self.calibrated: return None
        g = np.array([gx,gy,gz]); g /= np.linalg.norm(g)+1e-9

        s0 = self._project1(g, self.S_t_G)
        if not self.calib_mats:
            return float(s0[0]), float(s0[1])

        dists = [np.linalg.norm(s0[:2]-p[:2]) for p in self.calib_pts]
        sr    = self._project1(g, self.calib_mats[int(np.argmin(dists))])
        return (s0[0]+sr[0])/2., (s0[1]+sr[1])/2.

    def mm_to_px(self, x_mm, y_mm, w_px, h_px):
        px = int(np.clip(x_mm/self.W*w_px, 0, w_px-1))
        py = int(np.clip(y_mm/self.H*h_px, 0, h_px-1))
        return px, py

    def distance_mm(self):
        return float(-self.S_t_G[2]) if self.S_t_G is not None else 0.


# ─────────────────────────────────────────────────────────────────────────────
#  PART 4 · SfM lateral head tracker  (Section 3.3.5)
# ─────────────────────────────────────────────────────────────────────────────

class SfMTracker:
    FACE_W_MM = 150.0

    def __init__(self):
        self._prev_lmk   = None
        self._prev_gray  = None
        self._shift_mm   = 0.0
        self._alpha      = 0.88

    def update(self, gray, landmarks, face_box) -> float:
        pts_now = np.array(landmarks[:4], dtype=np.float32)
        shift   = 0.0
        if self._prev_lmk is not None:
            prev = np.array(self._prev_lmk[:4], dtype=np.float32)
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, prev.reshape(-1,1,2), None,
                winSize=(21,21), maxLevel=3)
            if tracked is not None and status is not None:
                good = status.flatten() == 1
                if good.sum() >= 2:
                    dx = float(np.median(
                        tracked.reshape(-1,2)[good,0] -
                        prev.reshape(-1,2)[good,0]))
                    x1,y1,x2,y2 = face_box
                    fw = max(x2-x1, 1)
                    px_mm = fw / self.FACE_W_MM
                    shift = dx / px_mm
        self._shift_mm = self._alpha*self._shift_mm + (1-self._alpha)*shift
        self._prev_lmk  = landmarks
        self._prev_gray = gray.copy()
        return float(self._shift_mm)


# ─────────────────────────────────────────────────────────────────────────────
#  PART 5 · Calibration data store
# ─────────────────────────────────────────────────────────────────────────────

def robust_mean(samples, sigma=1.5):
    arr = np.array(samples)
    med = np.median(arr, axis=0)
    std = arr.std(axis=0)
    mask = np.all(np.abs(arr-med) < sigma*(std+1e-8), axis=1)
    clean = arr[mask] if mask.sum() >= 3 else arr
    return clean.mean(axis=0), len(clean)/len(arr)


class CalibData:
    def __init__(self):
        self.pts_mm    = []   # (x_mm, y_mm)
        self.gazes     = []   # (gx, gy, gz) median per point
        self.qualities = []   # fraction of inlier samples

    def add(self, x_mm, y_mm, gaze_mean, quality):
        self.pts_mm.append((x_mm, y_mm))
        self.gazes.append(gaze_mean)
        self.qualities.append(quality)

    def save(self, projector, path="calibration.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"pts_mm":self.pts_mm,"gazes":self.gazes,
                         "projector":projector}, f)
        print(f"[✓] Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 6 · Tkinter Calibration UI
# ─────────────────────────────────────────────────────────────────────────────

# 9-point grid (normalised 0-1)
GRID_POS = [
    (0.10,0.10),(0.50,0.10),(0.90,0.10),
    (0.10,0.50),(0.50,0.50),(0.90,0.50),
    (0.10,0.90),(0.50,0.90),(0.90,0.90),
]
GRID_LABELS = [
    "Top-Left","Top-Centre","Top-Right",
    "Mid-Left","Centre","Mid-Right",
    "Bot-Left","Bot-Centre","Bot-Right",
]

# ── Colour palette ─────────────────────────────────────────────────────────────
BG        = "#04040f"
ACCENT    = "#00e5cc"
ACCENT2   = "#7c3aed"
WARN      = "#f59e0b"
TEXT_DIM  = "#2d4a5a"
TEXT_MID  = "#7a9aaa"
TEXT_LT   = "#c8dde8"
DOT_DONE  = "#00e5cc"
DOT_LIVE  = "#7c3aed"


class CalibrationWindow(tk.Toplevel):
    """Fullscreen 9-point calibration with polished Tkinter UI."""

    def __init__(self, parent, engine, projector: GazeToScreenProjector,
                 cfg: dict, on_done):
        super().__init__(parent)
        self.engine    = engine
        self.proj      = projector
        self.cfg       = cfg
        self.on_done   = on_done
        self.data      = CalibData()

        self.current   = 0
        self.phase     = "idle"   # idle | wait | collect
        self._stab_t   = 0.
        self._coll_pct = 0.
        self._live_xy  = None
        self._pulse_r  = 16.
        self._pulse_d  = 1
        self._partial  = False    # True once ≥3 pts → live preview enabled
        self._job      = None

        self.attributes("-fullscreen", True)
        self.configure(bg=BG)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._click)
        self.bind("<Escape>", lambda e: self._cancel())
        self.after(400, self._redraw)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _sw(self): return self.winfo_screenwidth()
    def _sh(self): return self.winfo_screenheight()
    def _sxy(self, nx, ny): return int(nx*self._sw()), int(ny*self._sh())

    # ── main draw ─────────────────────────────────────────────────────────────
    def _redraw(self):
        c  = self.canvas
        sw = self._sw(); sh = self._sh()
        c.delete("all")

        # ── background grid ───────────────────────────────────────────────────
        for i in range(0, sw, 80):
            c.create_line(i,0,i,sh, fill="#070718", width=1)
        for j in range(0, sh, 80):
            c.create_line(0,j,sw,j, fill="#070718", width=1)

        # Corner labels
        for txt, ax, ay in [("GAZE CALIBRATION",sw//2,28),]:
            c.create_text(ax, ay, text=txt, fill=ACCENT,
                          font=("Courier New",13,"bold"), anchor="center")

        # ── top progress bar ───────────────────────────────────────────────────
        bar_w = int(sw * self.current/len(GRID_POS))
        c.create_rectangle(0,48, sw,54, fill="#0d1a26", outline="")
        c.create_rectangle(0,48, bar_w,54, fill=ACCENT, outline="")

        # ── status text ───────────────────────────────────────────────────────
        if self.phase == "wait":
            pct = 1 - max(0,self._stab_t) / self.cfg["stabilise_sec"]
            msg = f"STABILISING  {int(pct*100)}%  —  hold still"
            col = WARN
        elif self.phase == "collect":
            msg = f"COLLECTING  {int(self._coll_pct*100)}%"
            col = "#ff9900"
        elif self.current < len(GRID_POS):
            msg = (f"Point {self.current+1} / {len(GRID_POS)}  ·  "
                   f"{GRID_LABELS[self.current].upper()}  ·  click the dot")
            col = TEXT_MID
        else:
            msg = ""
            col = TEXT_DIM
        c.create_text(sw//2, 70, text=msg, fill=col,
                      font=("Courier New",10), anchor="center")

        # ── completed dots ────────────────────────────────────────────────────
        for i, ((nx,ny), q) in enumerate(zip(GRID_POS[:self.current],
                                              self.data.qualities)):
            gx, gy = self._sxy(nx, ny)
            col = DOT_DONE if q > 0.65 else WARN
            c.create_oval(gx-8,gy-8,gx+8,gy+8,
                          fill="#001a14", outline=col, width=2)
            c.create_text(gx, gy, text="✓", fill=col,
                          font=("Courier New",7,"bold"))
            c.create_text(gx, gy+18, text=f"{int(q*100)}%",
                          fill=TEXT_DIM, font=("Courier New",7))

        # ── upcoming dots (dim) ───────────────────────────────────────────────
        for nx, ny in GRID_POS[self.current+1:]:
            fx, fy = self._sxy(nx, ny)
            c.create_oval(fx-4,fy-4,fx+4,fy+4,
                          fill="#0a1018", outline="#152030", width=1)

        # ── active calibration dot ─────────────────────────────────────────────
        if self.current < len(GRID_POS):
            nx, ny = GRID_POS[self.current]
            x, y   = self._sxy(nx, ny)

            # Outer guide rings
            for r, col in [(54,TEXT_DIM),(42,"#0d2030"),(32,"#0a2020")]:
                c.create_oval(x-r,y-r,x+r,y+r,
                              outline=col, fill="", width=1)

            # Crosshair
            c.create_line(x-40,y,x+40,y, fill=TEXT_DIM, width=1, dash=(3,5))
            c.create_line(x,y-40,x,y+40, fill=TEXT_DIM, width=1, dash=(3,5))

            # Progress arc
            if self.phase == "wait":
                pct_arc = 1 - max(0,self._stab_t)/self.cfg["stabilise_sec"]
                ext = int(360*pct_arc)
                c.create_arc(x-52,y-52,x+52,y+52,
                             start=90, extent=-ext,
                             outline=WARN, width=2, style="arc")
            elif self.phase == "collect":
                ext = int(360*self._coll_pct)
                c.create_arc(x-52,y-52,x+52,y+52,
                             start=90, extent=-ext,
                             outline="#ff9900", width=2, style="arc")

            # Pulsing ring
            pr = self._pulse_r
            rcol = WARN if self.phase=="wait" else \
                   "#ff9900" if self.phase=="collect" else ACCENT
            c.create_oval(x-pr,y-pr,x+pr,y+pr,
                          outline=rcol, width=2, fill="", tags="pulse")

            # Centre dot
            cdot = rcol if self.phase != "idle" else ACCENT
            c.create_oval(x-6,y-6,x+6,y+6,
                          fill=cdot, outline="#ffffff", width=1)

            # Label below
            c.create_text(x, y+62, text=GRID_LABELS[self.current],
                          fill=ACCENT, font=("Courier New",9,"bold"))

        # ── live gaze dot (purple) ────────────────────────────────────────────
        if self._live_xy is not None:
            lx, ly = self._live_xy
            c.create_oval(lx-10,ly-10,lx+10,ly+10,
                          fill=DOT_LIVE, outline="", tags="live")
            c.create_text(sw-14, sh-14,
                          text="● live gaze preview",
                          fill=DOT_LIVE, font=("Courier New",8),
                          anchor="se")

        # ── bottom hint ───────────────────────────────────────────────────────
        c.create_text(sw//2, sh-14,
                      text=f"{self.current}/{len(GRID_POS)} points  ·  ESC = cancel",
                      fill=TEXT_DIM, font=("Courier New",8), anchor="center")

        # pulse animation
        if self._job: self.after_cancel(self._job)
        self._job = self.after(32, self._pulse_tick)

    def _pulse_tick(self):
        self._pulse_r += self._pulse_d * 0.6
        if self._pulse_r > 24: self._pulse_d = -1
        if self._pulse_r < 11: self._pulse_d =  1
        if self.current < len(GRID_POS):
            nx, ny = GRID_POS[self.current]
            x, y   = self._sxy(nx, ny)
            pr     = self._pulse_r
            self.canvas.coords("pulse", x-pr,y-pr,x+pr,y+pr)
        self._job = self.after(32, self._pulse_tick)

    # ── click ─────────────────────────────────────────────────────────────────
    def _click(self, ev):
        if self.phase != "idle" or self.current >= len(GRID_POS):
            return
        nx, ny = GRID_POS[self.current]
        tx, ty = self._sxy(nx, ny)
        if ((ev.x-tx)**2+(ev.y-ty)**2)**0.5 > 90:
            t = self.canvas.create_text(ev.x, ev.y-24,
                text="click the dot!", fill="#ff4444",
                font=("Courier New",9))
            self.after(700, lambda: self.canvas.delete(t))
            return
        if self._job: self.after_cancel(self._job); self._job=None
        self.phase = "wait"
        self._redraw()
        threading.Thread(target=self._run_point,
                         args=(nx, ny, tx, ty), daemon=True).start()

    # ── background data collection ────────────────────────────────────────────
    def _run_point(self, nx, ny, canvas_x, canvas_y):
        # Convert canvas → physical mm
        sw_mm = self.cfg["screen_w_mm"]; sh_mm = self.cfg["screen_h_mm"]
        x_mm  = nx * sw_mm
        y_mm  = ny * sh_mm

        # Phase 1: stabilise
        deadline = time.time() + self.cfg["stabilise_sec"]
        while time.time() < deadline:
            self._stab_t = deadline - time.time()
            self._refresh_live()
            self.after(0, self._redraw)
            time.sleep(0.033)

        # Phase 2: collect
        self.phase = "collect"; self._coll_pct = 0.
        samples = []
        n = self.cfg["collect_frames"]
        for i in range(n):
            r = self.engine.predict()
            if r: samples.append(list(r["gaze3d"]))
            self._coll_pct = (i+1)/n
            self._refresh_live()
            self.after(0, self._redraw)
            time.sleep(0.033)

        if len(samples) < 5:
            self.after(0, self._no_face); return

        mean, quality = robust_mean(samples, self.cfg["outlier_sigma"])
        self.data.add(x_mm, y_mm, mean.tolist(), quality)

        # Partial fit after 3 points → enable live preview
        if len(self.data.pts_mm) >= 3 and not self._partial:
            try:
                self.proj.calibrate(self.data.pts_mm, self.data.gazes)
                self._partial = True
            except Exception: pass
        elif self._partial:
            try:
                self.proj.calibrate(self.data.pts_mm, self.data.gazes)
            except Exception: pass

        self.after(0, self._advance)

    def _refresh_live(self):
        if not self._partial: self._live_xy = None; return
        r = self.engine.predict()
        if r:
            res = self.proj.project(*r["gaze3d"])
            if res:
                xm, ym = res
                cx = int(xm/self.cfg["screen_w_mm"]*self._sw())
                cy = int(ym/self.cfg["screen_h_mm"]*self._sh())
                self._live_xy = (cx, cy)

    def _no_face(self):
        self.phase = "idle"; self._redraw()
        t = self.canvas.create_text(
            self._sw()//2, 100,
            text="⚠  No face detected — make sure you are visible to the camera",
            fill="#ef4444", font=("Courier New",11))
        self.after(2500, lambda: self.canvas.delete(t))

    def _advance(self):
        self.phase   = "idle"
        self.current += 1
        if self.current >= len(GRID_POS):
            self._finish()
        else:
            self._redraw()

    def _finish(self):
        if self._job: self.after_cancel(self._job); self._job=None
        sw, sh = self._sw(), self._sh()
        c = self.canvas; c.delete("all")
        c.create_text(sw//2, sh//2-30,
                      text="✓  ALL 9 POINTS COMPLETE",
                      fill=ACCENT, font=("Courier New",22,"bold"),
                      anchor="center")
        c.create_text(sw//2, sh//2+24,
                      text="Fitting model…",
                      fill=TEXT_MID, font=("Courier New",11),
                      anchor="center")
        self.after(1400, lambda: [self.on_done(self.data), self.destroy()])

    def _cancel(self):
        if self._job: self.after_cancel(self._job)
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
#  PART 7 · Results window
# ─────────────────────────────────────────────────────────────────────────────

class ResultsWindow(tk.Toplevel):
    def __init__(self, parent, data: CalibData, proj: GazeToScreenProjector,
                 engine, cfg, launch_cursor_fn):
        super().__init__(parent)
        self.title("Calibration Complete")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._parent = parent
        self._proj   = proj
        self._engine = engine
        self._cfg    = cfg
        self._launch = launch_cursor_fn
        self._data   = data

        # Final fit + save
        proj.calibrate(data.pts_mm, data.gazes)
        data.save(proj)

        self._build(data, proj)
        w, h = 700, 680
        x = (self.winfo_screenwidth()-w)//2
        y = (self.winfo_screenheight()-h)//2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build(self, data, proj):
        # Title
        tk.Label(self, text="CALIBRATION COMPLETE",
                 font=("Courier New",14,"bold"),
                 fg=ACCENT, bg=BG).pack(pady=(22,4))
        tk.Label(self, text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                 font=("Courier New",8), fg=TEXT_DIM, bg=BG).pack()

        # Distance badge
        dist = proj.distance_mm()
        tk.Label(self,
                 text=f"  Estimated viewing distance:  {dist:.0f} mm  ",
                 font=("Courier New",11,"bold"),
                 fg=BG, bg=ACCENT).pack(pady=10)

        # Separator
        tk.Frame(self, bg="#1a2d3a", height=1).pack(fill="x", padx=24, pady=6)

        # Results table
        cols = ("Point","x mm","y mm","gx","gy","gz","Quality")
        tv   = ttk.Treeview(self, columns=cols, show="headings", height=9)
        sty  = ttk.Style(self); sty.theme_use("clam")
        sty.configure("Treeview", background="#050a10",
                      foreground=TEXT_LT, fieldbackground="#050a10",
                      rowheight=28, font=("Courier New",9))
        sty.configure("Treeview.Heading", background="#0c1a24",
                      foreground=ACCENT, font=("Courier New",9,"bold"))
        ws = [110,70,70,80,80,80,70]
        for col, w in zip(cols, ws):
            tv.heading(col, text=col); tv.column(col, width=w, anchor="center")
        for i, ((xm,ym), g, q) in enumerate(
                zip(data.pts_mm, data.gazes, data.qualities)):
            tag = "ok" if q>0.65 else "warn"
            tv.insert("","end",
                      values=(GRID_LABELS[i], f"{xm:.0f}", f"{ym:.0f}",
                              f"{g[0]:+.4f}", f"{g[1]:+.4f}", f"{g[2]:+.4f}",
                              f"{q*100:.0f}%"),
                      tags=(tag,))
        tv.tag_configure("ok",   foreground=ACCENT)
        tv.tag_configure("warn", foreground=WARN)
        tv.pack(padx=16, fill="x")

        tk.Label(self,
                 text="Quality = fraction of frames kept after 1.5σ outlier rejection",
                 font=("Courier New",8), fg=TEXT_DIM, bg=BG).pack(pady=(4,0))
        tk.Frame(self, bg="#1a2d3a", height=1).pack(fill="x", padx=24, pady=10)
        tk.Label(self, text="Saved  →  calibration.pkl",
                 font=("Courier New",8), fg=TEXT_DIM, bg=BG).pack()

        # Code snippet
        snip = tk.Frame(self, bg="#07080f", padx=16, pady=10)
        snip.pack(padx=24, pady=8, fill="x")
        tk.Label(snip,
                 text=("import pickle\n"
                       "d = pickle.load(open('calibration.pkl','rb'))\n"
                       "proj = d['projector']\n"
                       "# in your loop:\n"
                       "x_mm, y_mm = proj.project(gx, gy, gz)"),
                 font=("Courier New",9), fg=ACCENT, bg="#07080f",
                 justify="left").pack(anchor="w")

        # Buttons
        bf = tk.Frame(self, bg=BG); bf.pack(pady=14)
        def btn(t, cmd, hi=False):
            return tk.Button(bf, text=t, command=cmd,
                             font=("Courier New",10,"bold" if hi else "normal"),
                             fg=BG if hi else TEXT_LT,
                             bg=ACCENT if hi else "#0e1a24",
                             activebackground="#00ccaa" if hi else "#162030",
                             relief="flat", bd=0,
                             padx=16, pady=9, cursor="hand2")
        btn("▶  Live Cursor", self._go_live, hi=True).pack(side="left", padx=6)
        btn("↺  Recalibrate", self._recal).pack(side="left", padx=6)
        btn("✗  Close", self.destroy).pack(side="left", padx=6)

    def _go_live(self):
        self.destroy()
        self._launch(self._proj, self._engine, self._cfg)

    def _recal(self):
        self.destroy()
        self._parent.run_calibration()


# ─────────────────────────────────────────────────────────────────────────────
#  PART 8 · Live gaze cursor overlay
# ─────────────────────────────────────────────────────────────────────────────

class GazeCursor(tk.Toplevel):
    def __init__(self, parent, proj: GazeToScreenProjector,
                 engine, cfg: dict):
        super().__init__(parent)
        self.proj   = proj
        self.engine = engine
        self.cfg    = cfg
        self._sfm   = SfMTracker() if cfg["use_sfm"] else None
        self._sx    = float(cfg["screen_w_px"]//2)
        self._sy    = float(cfg["screen_h_px"]//2)
        self._alpha = cfg["ema_alpha"]
        self._on    = True
        self._trail = deque(maxlen=30)
        self._dist  = proj.distance_mm()

        self.attributes("-fullscreen", True)
        self.attributes("-topmost", True)
        self.configure(bg="#000000")
        self.overrideredirect(True)
        try: self.attributes("-transparentcolor","#000000")
        except tk.TclError: pass

        self.canvas = tk.Canvas(self, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        sw = self.winfo_screenwidth(); sh = self.winfo_screenheight()
        self.canvas.create_text(
            sw//2, 32,
            text="LIVE GAZE  ·  ESC to exit  ·  d = debug window",
            fill=DOT_LIVE, font=("Courier New",11,"bold"), tags="hint")

        self.bind("<Escape>", lambda e: self._close())
        self.bind("d", lambda e: self._toggle_debug())
        self._debug_on = False
        self._tick()

    def _tick(self):
        if not self._on: return
        r = self.engine.predict()
        if r:
            gx, gy, gz = r["gaze3d"]

            # SfM correction
            lat_mm = 0.
            if self._sfm and r["landmarks"] and r["face_box"]:
                frame = self.engine.get_frame()
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    lat_mm = self._sfm.update(gray, r["landmarks"], r["face_box"])

            res = self.proj.project(gx, gy, gz)
            if res:
                xm, ym = res
                xm += lat_mm
                # map mm → screen px → canvas px
                cfg = self.cfg
                sw_mm = cfg["screen_w_mm"]; sh_mm = cfg["screen_h_mm"]
                sw_px = cfg["screen_w_px"]; sh_px = cfg["screen_h_px"]
                cw = self.winfo_screenwidth(); ch = self.winfo_screenheight()
                tx = int(np.clip(xm/sw_mm*sw_px, 0, sw_px-1))
                ty = int(np.clip(ym/sh_mm*sh_px, 0, sh_px-1))
                # scale to canvas
                cx = int(tx/sw_px*cw)
                cy = int(ty/sh_px*ch)
                a  = self._alpha
                self._sx = self._sx*(1-a) + cx*a
                self._sy = self._sy*(1-a) + cy*a
                sx = int(self._sx); sy = int(self._sy)
                self._trail.append((sx, sy))

                self._draw_cursor(sx, sy, tx, ty)

            # Debug window
            if self._debug_on:
                frame = self.engine.get_frame()
                if frame is not None:
                    self._draw_debug(frame, r)
                    cv2.imshow("Gaze Debug", frame)
                    cv2.waitKey(1)

        self.after(28, self._tick)

    def _draw_cursor(self, sx, sy, px, py):
        c = self.canvas; c.delete("cursor")

        # Trail
        for i, (tx, ty) in enumerate(self._trail):
            alpha_val = int(120 * i / max(len(self._trail),1))
            col = f"#{alpha_val//2:02x}{alpha_val:02x}{alpha_val+80:02x}"
            try:
                c.create_oval(tx-3,ty-3,tx+3,ty+3,
                              fill=col, outline="", tags="cursor")
            except Exception: pass

        # Crosshair
        c.create_line(sx-28,sy,sx+28,sy,
                      fill=DOT_LIVE, width=1, dash=(3,4), tags="cursor")
        c.create_line(sx,sy-28,sx,sy+28,
                      fill=DOT_LIVE, width=1, dash=(3,4), tags="cursor")
        # Outer ring
        c.create_oval(sx-18,sy-18,sx+18,sy+18,
                      outline=DOT_LIVE, width=2, fill="", tags="cursor")
        # Inner dot
        c.create_oval(sx-5,sy-5,sx+5,sy+5,
                      fill=DOT_LIVE, outline="#ffffff", width=1, tags="cursor")
        # Coords label
        c.create_text(sx+26, sy,
                      text=f"({px},{py})  d≈{self._dist:.0f}mm",
                      fill=DOT_LIVE, font=("Courier New",8),
                      anchor="w", tags="cursor")

    def _draw_debug(self, frame, r):
        x1,y1,x2,y2 = r["face_box"]
        yaw,pitch,roll = r["head_pose"]
        lms = r["landmarks"]
        gx,gy,gz = r["gaze3d"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,100),2)
        if lms:
            for (lx,ly),col in zip(lms,
                [(255,100,100),(100,100,255),(100,255,100),(255,165,0),(255,165,0)]):
                cv2.circle(frame,(lx,ly),4,col,-1)
            # head-pose axes from nose
            cx,cy_p = lms[2]
            for ypr, arr, c in [(yaw,pitch,roll),]:
                _head_axes(frame,(cx,cy_p),yaw,pitch,roll)
            # gaze arrows
            for ctr in lms[:2]:
                ex=int(ctr[0]+gx*140); ey=int(ctr[1]-gy*140)
                cv2.arrowedLine(frame,ctr,(ex,ey),(0,255,255),2,tipLength=0.25)
        cv2.putText(frame,f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}",
                    (8,22),cv2.FONT_HERSHEY_SIMPLEX,0.45,(220,220,220),1)
        cv2.putText(frame,f"g=({gx:+.3f},{gy:+.3f},{gz:+.3f})",
                    (8,40),cv2.FONT_HERSHEY_SIMPLEX,0.45,(220,220,220),1)

    def _toggle_debug(self):
        self._debug_on = not self._debug_on
        if not self._debug_on:
            cv2.destroyAllWindows()

    def _close(self):
        self._on = False
        cv2.destroyAllWindows()
        self.destroy()


def _head_axes(frame, center, yaw, pitch, roll, scale=50):
    cy,sy = math.cos(math.radians(yaw)),  math.sin(math.radians(yaw))
    cp,sp = math.cos(math.radians(pitch)),math.sin(math.radians(pitch))
    cr,sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    axes = [
        (np.array([cy*cr+sy*sp*sr, cp*sr,-sy*cr+cy*sp*sr]),(0,0,230)),
        (np.array([-cy*sr+sy*sp*cr,cp*cr, sy*sr+cy*sp*cr]),(0,230,0)),
        (np.array([sy*cp,-sp,cy*cp]),                        (230,0,0)),
    ]
    cx,cy_px = center
    for vec,col in axes:
        end=(int(cx+vec[0]*scale),int(cy_px-vec[1]*scale))
        cv2.arrowedLine(frame,(cx,cy_px),end,col,2,tipLength=0.3)


# ─────────────────────────────────────────────────────────────────────────────
#  PART 9 · Application root
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.title("Gaze Estimation")

        # Init engine
        print("[*] Initialising gaze engine…")
        if OPENVINO_OK:
            self._engine = GazeEngine()
            try:
                self._engine.load(CFG)
                self._engine.start()
                print("[*] Camera stream started.")
            except Exception as e:
                print(f"[WARN] Model load failed ({e}). Using synthetic demo.")
                self._engine = SyntheticEngine()
                self._engine.load(CFG)
                self._engine.start()
        else:
            self._engine = SyntheticEngine()
            self._engine.load(CFG)
            self._engine.start()

        self._proj = GazeToScreenProjector(CFG["screen_w_mm"], CFG["screen_h_mm"])
        self.run_calibration()

    def run_calibration(self):
        self._proj = GazeToScreenProjector(CFG["screen_w_mm"], CFG["screen_h_mm"])
        CalibrationWindow(self, self._engine, self._proj, CFG,
                          on_done=self._calib_done)

    def _calib_done(self, data: CalibData):
        ResultsWindow(self, data, self._proj, self._engine, CFG,
                      launch_cursor_fn=self._launch_cursor)

    def _launch_cursor(self, proj, engine, cfg):
        GazeCursor(self, proj, engine, cfg)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()