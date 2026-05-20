"""
ball_detection.py

Multi-camera tennis ball detection + physics-based trajectory prediction.
Standalone Python replacement for ball_detection.cpp (ZED SDK) for single-camera testing.

Camera backends (selected via `camera.backend` in the YAML config):
    realsense → Intel RealSense D435 / D455           (config/d455.yaml)
    zed       → Stereolabs ZED Mini / ZED 2 / ZED X   (config/zedmini.yaml)
    webcam    → Any V4L2 / UVC webcam (RGB-only)      (config/webcam.yaml)
                e.g. HIKROBOT MV-CS016 industrial GS, FLIR Blackfly UVC mode,
                     Logitech BRIO 4K, Razer Kiyo Pro, Arducam IMX477

Detection:   MOG2 motion mask + HSV colour segmentation  (ported from ball_detection.cpp)
Depth:       visual (fx·R/r_px), fused with sensor depth when available
Tracking:    6-state physics EKF — gravity + quadratic drag  (ported from ball_ekf.cpp)
Prediction:  physics rollout with ground bounce  (from ball_ekf.cpp::predict)

Physics constants (ball_localization/src/ball_ekf.cpp):
    GRAVITY     = -9.79528 m/s²   (change to your venue)
    AIR_DENSITY = 1.225 kg/m³
    BALL_MASS   = 0.0575 kg
    BALL_RADIUS = 0.0335 m
    C_d         = 0.47 (sphere, tunable via --coeff-drag)
    Restitution : x=0.75, y=0.75, z=0.75 (tunable via --rest-*)

Coordinate frame: camera body (X-forward, Y-left, Z-up).  Gravity → −Z.

Requirements:
    conda activate catchball
    # plus, depending on backend:
    #   realsense → pyrealsense2 (already in catchball)
    #   zed       → ZED SDK + pyzed.sl  (run /usr/local/zed/get_python_api.py)
    #   webcam    → opencv only (already in catchball)

Usage:
    python ball_detection.py                                 # default = config/d455.yaml
    python ball_detection.py --config config/zedmini.yaml    # ZED Mini
    python ball_detection.py --config config/webcam.yaml     # generic webcam
    python ball_detection.py --no-viz                         # headless
    python ball_detection.py --show-mask                      # show HSV+motion mask
    python ball_detection.py --width 848 --height 480         # ask backend for this resolution
    python ball_detection.py --no-motion                      # skip MOG2 (pure HSV)
    python ball_detection.py --h-low 10 --h-high 35           # HSV hue overrides
    python ball_detection.py --coeff-drag 0.47                # tune drag
"""

import argparse
import json
import os
import threading
import time
import numpy as np
import cv2

import cameras as _cameras_pkg   # noqa: F401  (registers backend factory)


# ══════════════════════════════════════════════════════════════════════════════
#  Physics constants  (ball_localization/src/ball_ekf.cpp)
# ══════════════════════════════════════════════════════════════════════════════
GRAVITY      = -9.79528   # m/s² — Atlanta; change to 9.8 for generic use
AIR_DENSITY  = 1.225      # kg/m³
BALL_MASS    = 0.0575     # kg
BALL_RADIUS  = 0.0335     # m — tennis ball physical radius

DEFAULT_COEFF_DRAG   = 0.47   # sphere drag coefficient C_d
DEFAULT_COEFF_REST_X = 0.75   # horizontal (X) restitution after bounce
DEFAULT_COEFF_REST_Y = 0.75   # horizontal (Y) restitution after bounce
DEFAULT_COEFF_REST_Z = 0.75   # vertical   (Z) restitution after bounce

# ── HSV colour range (defaults from ball_detection/config/settings.yaml) ──────
HSV_H_LOW  = 10    # hue lower  (10–35: yellow-orange, 25–80: fluorescent green)
HSV_H_HIGH = 35    # hue upper
HSV_S_MIN  = 170   # saturation minimum
HSV_V_MIN  = 170   # value minimum

# ── Pixel-level detection thresholds ─────────────────────────────────────────
MIN_RADIUS_PX   = 3
MAX_RADIUS_PX   = 200
MIN_CIRCULARITY = 0.55
BG_RESIZE       = 0.4    # MOG2 downscale fraction (from settings.yaml: background_resize)

# ── Depth params (from catch_ball/camera_ball_color.py) ───────────────────────
DEPTH_MIN      = 0.15    # m — discard readings closer than this
DEPTH_MAX      = 8.0     # m — discard readings farther than this
DEPTH_SAMPLE_R = 5       # px — median patch radius for sensor depth
VIS_WEIGHT     = 0.5     # depth fusion weight (0=sensor-only, 1=visual-only)

# ── EKF process noise (ball_filter_base.cpp defaults) ────────────────────────
Q_POS = 0.05     # m² / s  (position axes)
Q_VEL = 0.025    # (m/s)² / s  (velocity axes)

# ── Measurement covariance polynomial (settings.yaml calibrated values) ──────
#   cov(d) = k2·d² + k1·d + k0    (d = depth in metres)
Z_K2,  Z_K1,  Z_K0  = 0.00104,  0.000444, 0.01   # depth/forward axis
XY_K2, XY_K1, XY_K0 = 0.000296, 0.00156,  0.01   # lateral axes

# ── Trajectory rollout ────────────────────────────────────────────────────────
TRAJ_PREDICT_SEC = 1.0    # seconds ahead to predict
DT_ROLLOUT       = 0.01   # Euler integration step (s)

COAST_FRAMES = 10   # hold last detection this many frames after miss

# ── AprilTag ground calibration ───────────────────────────────────────────────
TAG_EMA          = 0.3    # EMA weight applied to each new tag measurement (per frame)
TAG_STALE_FRAMES = 30     # frames without tag before showing RED indicator

_ARUCO_FAMILIES = {
    "tag16h5":  cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9":  cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Coordinate transforms (inline — no external transform package needed)
# ══════════════════════════════════════════════════════════════════════════════

def optical_to_body(p_opt):
    """RealSense optical (Z-fwd, X-right, Y-down) → body (X-fwd, Y-left, Z-up)."""
    return np.array([float(p_opt[2]), -float(p_opt[0]), -float(p_opt[1])])


def body_to_optical(p_body):
    """Body (X-fwd, Y-left, Z-up) → RealSense optical (Z-fwd, X-right, Y-down)."""
    return np.array([-float(p_body[1]), -float(p_body[2]), float(p_body[0])])


def body_to_world(p_body, height, pitch_rad):
    """Camera body (X-fwd, Y-left, Z-up) → world (X-fwd, Y-left, Z-up, ground=Z=0).

    height    : camera centre height above ground (m)
    pitch_rad : camera pitch in radians; negative = looking down toward court
    """
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    return np.array([
        c * p_body[0] + s * p_body[2],
        float(p_body[1]),
        -s * p_body[0] + c * p_body[2] + height,
    ])


def world_to_body(p_world, height, pitch_rad):
    """World (ground=Z=0) → camera body frame.  Inverse of body_to_world."""
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    dx = float(p_world[0])
    dz = float(p_world[2]) - height
    return np.array([
        c * dx - s * dz,
        float(p_world[1]),
        s * dx + c * dz,
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  Physics EKF  (ported from ball_ekf.cpp, simplified to 6-state)
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsEKF:
    """
    6-state EKF for tennis ball tracking.

    State  x = [px, py, pz, vx, vy, vz]  (camera body frame, Z-up).
    Gravity acts in −Z direction.

    Process model (ball_ekf.cpp Ekf::predict):
        v_sq   = vx² + vy² + vz²
        a_drag = C_d · ½ · ρ · π·r² · v_sq / m
        ax = −sign(vx) · a_drag · |vx| / √v_sq
        ay = −sign(vy) · a_drag · |vy| / √v_sq
        az =  g  − sign(vz) · a_drag · |vz| / √v_sq

    Measurement model:  z = H·x + noise,  H = [I₃ | 0₃]
    """

    def __init__(self, coeff_drag=DEFAULT_COEFF_DRAG):
        # Drag pre-factor:  C_d · ½ · ρ · π·r² / m
        self._drag_k = (coeff_drag * 0.5 * AIR_DENSITY *
                        np.pi * BALL_RADIUS**2 / BALL_MASS)
        self.reset()

    def reset(self):
        self.x           = np.zeros(6)
        self.P           = np.eye(6) * 1e3
        self.initialized = False
        self._last_t     = None

    # ── private helpers ───────────────────────────────────────────────────────

    def _accel(self, state):
        """Return (ax, ay, az) for given state."""
        vx, vy, vz = state[3], state[4], state[5]
        v_sq = vx*vx + vy*vy + vz*vz
        if v_sq > 0 and v_sq < 40**2:
            v_norm = np.sqrt(v_sq)
            a_drag = self._drag_k * v_sq
            ax = -np.sign(vx) * a_drag * abs(vx) / v_norm
            ay = -np.sign(vy) * a_drag * abs(vy) / v_norm
            az = GRAVITY - np.sign(vz) * a_drag * abs(vz) / v_norm
        else:
            ax = ay = 0.0
            az = GRAVITY
        return ax, ay, az

    def _state_jacobian(self, state, dt):
        """Linearised state-transition Jacobian F (6×6)."""
        F = np.eye(6)
        F[0, 3] = dt   # px += vx·dt
        F[1, 4] = dt
        F[2, 5] = dt
        # Numerical ∂a/∂v → fills F[3:6, 3:6]
        a0 = np.array(self._accel(state))
        eps = 1e-5
        for i in range(3, 6):
            s2 = state.copy(); s2[i] += eps
            da = (np.array(self._accel(s2)) - a0) / eps
            F[3:6, i] += da * dt
        return F

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, z_body, R_meas, timestamp):
        """
        Fuse one 3-D position measurement.

        z_body  : (3,) — ball position in camera body frame
        R_meas  : (3,3) — measurement noise covariance
        timestamp: float — seconds since epoch
        """
        z = np.asarray(z_body, dtype=float)

        if not self.initialized:
            self.x[:3] = z
            self.P     = np.eye(6) * 1e2
            self.P[:3, :3] = R_meas
            self.initialized = True
            self._last_t = timestamp
            return

        dt = timestamp - self._last_t
        self._last_t = timestamp
        if dt <= 1e-9 or dt > 1.0:
            return

        # ── Predict step ──────────────────────────────────────────────────
        ax, ay, az = self._accel(self.x)
        a = np.array([ax, ay, az])
        x_p = self.x.copy()
        x_p[:3] += self.x[3:] * dt + 0.5 * a * dt**2
        x_p[3:] += a * dt
        if x_p[2] < 0:                        # floor contact: clamp pos AND vel
            x_p[2] = 0.0
            if x_p[5] < 0:
                x_p[5] = 0.0

        F   = self._state_jacobian(self.x, dt)
        Q   = np.diag([Q_POS]*3 + [Q_VEL]*3) * dt
        P_p = F @ self.P @ F.T + Q

        self.x, self.P = x_p, P_p

        # ── Correct step ──────────────────────────────────────────────────
        H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.solve(S.T, np.eye(3)).T
        inn = z - self.x[:3]

        # Mahalanobis gate (5σ): reject wild outliers
        if float(inn @ np.linalg.solve(S, inn)) < 25.0:
            self.x += K @ inn
            self.P  = (np.eye(6) - K @ H) @ self.P

        if self.x[2] < 0:                    # same contact constraint post-correct
            self.x[2] = 0.0
            if self.x[5] < 0:
                self.x[5] = 0.0

    def rollout(self, t_ahead=TRAJ_PREDICT_SEC, dt=DT_ROLLOUT,
                cx=DEFAULT_COEFF_REST_X,
                cy=DEFAULT_COEFF_REST_Y,
                cz=DEFAULT_COEFF_REST_Z):
        """
        Euler-integrate physics forward.
        Returns list of (t_rel, pos_body) pairs.

        Bounce model (ball_ekf.cpp lines 329-347):
            - Detect floor crossing via quadratic solve
            - Apply restitution to each velocity component
        """
        if not self.initialized:
            return []

        state = self.x.copy()
        traj  = []
        t     = 0.0

        while t < t_ahead:
            traj.append((t, state[:3].copy()))
            ax, ay, az = self._accel(state)
            a  = np.array([ax, ay, az])
            z0 = state[2]
            vz = state[5]
            z1 = z0 + vz * dt + 0.5 * az * dt**2

            if z0 > 0 and z1 < 0:
                # Quadratic solve for exact floor-hit time
                A   = 0.5 * az
                B   = vz
                C   = z0
                disc = B*B - 4*A*C
                if disc >= 0:
                    t1 = (-B + np.sqrt(disc)) / (2*A)
                    t2 = (-B - np.sqrt(disc)) / (2*A)
                    fd = max(t1, t2)
                    if 0 < fd < dt:
                        # Propagate to floor
                        a2 = np.array(self._accel(state))
                        state[:3] += state[3:] * fd + 0.5 * a2 * fd**2
                        state[3:] += a2 * fd
                        state[2]   = abs(state[2])
                        # Apply restitution
                        state[3]  *= cx
                        state[4]  *= cy
                        state[5]   = cz * abs(state[5])
                        t += fd
                        # Continue remaining step after bounce
                        rem = dt - fd
                        a3  = np.array(self._accel(state))
                        state[:3] += state[3:] * rem + 0.5 * a3 * rem**2
                        state[3:] += a3 * rem
                        state[2]   = abs(state[2])
                        t += rem
                        continue

            state[:3] += state[3:] * dt + 0.5 * a * dt**2
            state[3:] += a * dt
            state[2]   = max(state[2], 0.0)
            t += dt

        return traj


# ══════════════════════════════════════════════════════════════════════════════
#  Detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_tennis_ball(frame_bgr, hsv_low, hsv_high, back_sub=None,
                       min_r=MIN_RADIUS_PX, min_circ=MIN_CIRCULARITY):
    """
    Detect tennis ball in a BGR frame.

    Stage 1 (optional): MOG2 background subtraction on downscaled frame to find
    moving regions  — ported from ball_detection.cpp::findMovingCandidates.
    Stage 2: HSV colour mask + contour circularity within those regions.

    Returns (cx, cy, r_px, mask) or (None, None, None, mask).
    """
    h, w = frame_bgr.shape[:2]

    # ── Stage 1: motion mask ──────────────────────────────────────────────────
    motion_mask = None
    if back_sub is not None:
        small  = cv2.resize(frame_bgr, (0, 0), fx=BG_RESIZE, fy=BG_RESIZE,
                            interpolation=cv2.INTER_AREA)
        fgmask = back_sub.apply(small)
        # Expand to full resolution
        motion_mask = cv2.resize(fgmask, (w, h), interpolation=cv2.INTER_NEAREST)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kern, iterations=2)

    # ── Stage 2: HSV colour mask ──────────────────────────────────────────────
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    if motion_mask is not None:
        mask = cv2.bitwise_and(mask, motion_mask)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None   # (score, cx, cy, r)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * min_r**2:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circ = 4 * np.pi * area / (peri**2)
        if circ < min_circ:
            continue
        (cx_f, cy_f), r = cv2.minEnclosingCircle(cnt)
        if not (min_r <= r <= MAX_RADIUS_PX):
            continue
        score = area * circ
        if best is None or score > best[0]:
            best = (score, int(cx_f), int(cy_f), r)

    if best is None:
        return None, None, None, mask
    _, cx, cy, r = best
    return cx, cy, r, mask


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _meas_covariance(depth_m):
    """
    Polynomial covariance model from settings.yaml.
    cov(d) = k2·d² + k1·d + k0
    Returns 3×3 measurement noise matrix in body frame (X-forward=depth, Y,Z lateral).
    """
    d = depth_m
    cov_x  = Z_K2  * d*d + Z_K1  * d + Z_K0   # forward axis (= depth direction)
    cov_yz = XY_K2 * d*d + XY_K1 * d + XY_K0  # lateral axes
    return np.diag([cov_x, cov_yz, cov_yz])


def _body_to_pixel(p_body, intrin):
    """Project body-frame point to image pixel.  Returns (u, v) or None."""
    p_opt = body_to_optical(p_body)
    z = p_opt[2]
    if z <= 0.01:
        return None
    u = intrin.fx * p_opt[0] / z + intrin.ppx
    v = intrin.fy * p_opt[1] / z + intrin.ppy
    if not (0 <= u < intrin.width and 0 <= v < intrin.height):
        return None
    return (int(u + 0.5), int(v + 0.5))


class _FPS:
    def __init__(self, window=30):
        self._t = []
        self._w = window

    def tick(self):
        now = time.perf_counter()
        self._t.append(now)
        if len(self._t) > self._w:
            self._t.pop(0)

    @property
    def fps(self):
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


# ══════════════════════════════════════════════════════════════════════════════
#  Config loader
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config", "d455.yaml")


def _load_config(path):
    """Read d455.yaml and return {argparse_dest: value} for parser.set_defaults().

    YAML uses positive boolean names (viz, motion, traj); this function inverts
    them to match the argparse store_true dest names (no_viz, no_motion, no_traj).
    Missing keys are silently skipped — code defaults remain in effect.
    """
    try:
        import yaml
    except ImportError:
        print("[WARN] PyYAML not installed; config file ignored.  pip install pyyaml")
        return {}
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[WARN] Cannot read config {path}: {e}")
        return {}

    out = {}

    def _get(key, cast, dest):
        if key in cfg:
            out[dest] = cast(cfg[key])

    # Resolution
    _get("width",  int,   "width")
    _get("height", int,   "height")

    # Ground / world frame
    _get("camera_height", float, "camera_height")
    _get("camera_pitch",  float, "camera_pitch")

    # Detection backend
    _get("detector", str, "detector")

    # HSV
    _get("h_low",  int, "h_low")
    _get("h_high", int, "h_high")
    _get("s_min",  int, "s_min")
    _get("v_min",  int, "v_min")

    # Motion filter — YAML key is positive; argparse dest is negated
    if "motion" in cfg:
        out["no_motion"] = not bool(cfg["motion"])

    # YOLO
    _get("yolo_model", str,   "model")
    _get("yolo_imgsz", int,   "imgsz")
    _get("yolo_conf",  float, "conf")

    # Physics
    _get("coeff_drag", float, "coeff_drag")
    _get("rest_x",     float, "rest_x")
    _get("rest_y",     float, "rest_y")
    _get("rest_z",     float, "rest_z")

    # AprilTag ground calibration
    _get("tag_family",  str,   "tag_family")
    _get("tag_size_m",  float, "tag_size_m")
    # tag_ids accepts a list [1,2,3] or a single int
    if "tag_ids" in cfg:
        raw = cfg["tag_ids"]
        out["tag_ids"] = list(raw) if isinstance(raw, list) else [int(raw)]
    elif "tag_id" in cfg:
        out["tag_ids"] = [int(cfg["tag_id"])]

    # HSV detection thresholds
    _get("mog2_threshold", int,   "mog2_threshold")
    _get("min_radius_px",  int,   "min_radius")
    _get("circularity",    float, "circularity")

    # Visualisation — YAML positive, argparse negated
    if "viz" in cfg:
        out["no_viz"] = not bool(cfg["viz"])
    _get("show_mask", bool, "show_mask")
    if "traj" in cfg:
        out["no_traj"] = not bool(cfg["traj"])

    # Recording: false→None, true→"" (auto name), "path.mp4"→"path.mp4"
    if "record" in cfg:
        rec = cfg["record"]
        if rec is False or rec is None:
            out["record"] = None
        elif rec is True:
            out["record"] = ""
        else:
            out["record"] = str(rec)

    # 3-D web viewer (viz3d.py)
    if "viz3d" in cfg:
        out["viz3d"] = bool(cfg["viz3d"])
    _get("viz3d_port", int, "viz3d_port")
    _get("ctrl_port",  int, "ctrl_port")

    # 2-D web viewer (webview.py)
    if "webview" in cfg:
        out["webview"] = bool(cfg["webview"])
    _get("webview_port", int, "webview_port")
    _get("mjpeg_port",   int, "mjpeg_port")

    if "record_raw" in cfg:
        out["record_raw"] = bool(cfg["record_raw"])

    # Camera backend section — pass through verbatim for cameras.from_config()
    if "camera" in cfg and isinstance(cfg["camera"], dict):
        out["_camera_section"] = dict(cfg["camera"])
        if "backend" in cfg["camera"]:
            out["camera_backend"] = str(cfg["camera"]["backend"])
        if "id" in cfg["camera"]:
            out["cam_id"] = str(cfg["camera"]["id"])

    # Multi-camera fusion broadcasting
    _get("cam_id",       str, "cam_id")
    _get("fusion_host",  str, "fusion_host")
    _get("fusion_port",  int, "fusion_port")
    _get("port_offset",  int, "port_offset")

    # Trajectory recording: false→"", true→"logs/", "path.json"→"path.json"
    if "save_traj" in cfg:
        st = cfg["save_traj"]
        if st is False or st is None:
            out["save_traj"] = ""
        elif st is True:
            out["save_traj"] = "logs/"
        else:
            out["save_traj"] = str(st)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Pass 1: find --config path before building the full parser ────────────
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=_DEFAULT_CONFIG)
    _pre_args, _ = _pre.parse_known_args()
    _cfg = _load_config(_pre_args.config)
    if _cfg:
        print(f"[INFO] Config loaded: {_pre_args.config}")

    # ── Pass 2: full parser — config values become defaults, CLI overrides ────
    parser = argparse.ArgumentParser(
        description="RealSense D455 tennis ball detection + physics EKF trajectory prediction")
    parser.add_argument("--config",       default=_DEFAULT_CONFIG,
                        help="YAML config file (default: config/d455.yaml)")
    parser.add_argument("--camera-backend", default="realsense",
                        choices=["realsense", "zed", "webcam"],
                        help="Camera backend (overrides config camera.backend)")
    parser.add_argument("--no-viz",       action="store_true", help="disable OpenCV window")
    parser.add_argument("--show-mask",    action="store_true", help="overlay HSV+motion mask")
    parser.add_argument("--no-motion",    action="store_true", help="disable MOG2 motion filter")
    parser.add_argument("--no-traj",      action="store_true", help="disable trajectory prediction overlay")
    parser.add_argument("--width",        type=int, default=1280)
    parser.add_argument("--height",       type=int, default=720)
    parser.add_argument("--h-low",        type=int, default=HSV_H_LOW,  help="HSV hue lower bound")
    parser.add_argument("--h-high",       type=int, default=HSV_H_HIGH, help="HSV hue upper bound")
    parser.add_argument("--s-min",        type=int, default=HSV_S_MIN,  help="HSV saturation min")
    parser.add_argument("--v-min",        type=int, default=HSV_V_MIN,  help="HSV value min")
    parser.add_argument("--coeff-drag",   type=float, default=DEFAULT_COEFF_DRAG)
    parser.add_argument("--rest-x",       type=float, default=DEFAULT_COEFF_REST_X)
    parser.add_argument("--rest-y",       type=float, default=DEFAULT_COEFF_REST_Y)
    parser.add_argument("--rest-z",       type=float, default=DEFAULT_COEFF_REST_Z)
    parser.add_argument("--detector",     choices=["hsv", "yolo", "both"], default="hsv",
                        help="detection backend: hsv | yolo | both")
    parser.add_argument("--model",        default="models/yolov8n.pt",
                        help="YOLO model path (auto-downloaded on first use)")
    parser.add_argument("--imgsz",        type=int, default=480, help="YOLO inference size")
    parser.add_argument("--conf",         type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--record",       metavar="FILE", nargs="?", const="",
                        help="record annotated video; omit FILE for auto timestamp name")
    parser.add_argument("--record-raw",   action="store_true",
                        help="record raw (unannotated) colour frame instead of annotated output")
    parser.add_argument("--camera-height", type=float, default=0.0,
                        help="camera centre height above ground (m); cold-start value when tag not yet visible")
    parser.add_argument("--camera-pitch",  type=float, default=0.0,
                        help="camera pitch in degrees; negative = looking down (cold-start value)")
    parser.add_argument("--tag-family",   default="tag36h11",
                        help="ArUco/AprilTag family  (tag36h11 | tag25h9 | tag16h5)")
    parser.add_argument("--tag-ids",      default=[0], type=lambda s: [int(x) for x in s.split(",")],
                        help="comma-separated AprilTag IDs to accept, e.g. --tag-ids 1,2,3")
    parser.add_argument("--tag-size-m",   type=float, default=0.0,
                        help="Physical tag size — black-square outer edge in metres; "
                             "0 = disable tag-based calibration")
    parser.add_argument("--no-tag",          action="store_true",
                        help="disable per-frame AprilTag ground calibration")
    parser.add_argument("--mog2-threshold", type=int,   default=50,
                        help="MOG2 varThreshold (higher = less sensitive, default 50)")
    parser.add_argument("--min-radius",     type=int,   default=MIN_RADIUS_PX,
                        help="minimum ball radius in pixels (default 3)")
    parser.add_argument("--circularity",    type=float, default=MIN_CIRCULARITY,
                        help="minimum contour circularity 0–1 (default 0.55)")
    parser.add_argument("--viz3d",      action="store_true",
                        help="broadcast state via UDP for viz3d.py 3-D viewer")
    parser.add_argument("--viz3d-port", type=int, default=5565,
                        help="UDP port for viz3d state broadcast (default 5565)")
    parser.add_argument("--ctrl-port",  type=int, default=5566,
                        help="UDP port to receive rest-control from viz3d.py (default 5566)")
    parser.add_argument("--webview",      action="store_true",
                        help="stream annotated frames to webview.py 2-D web viewer")
    parser.add_argument("--webview-port", type=int, default=5567,
                        help="UDP port for state stream to webview.py (default 5567)")
    parser.add_argument("--mjpeg-port",   type=int, default=5568,
                        help="port for internal full-res MJPEG server (default 5568, 0=disable)")
    parser.add_argument("--save-traj",    type=str, default="",
                        help="save EKF trajectory to JSON for rest calibration. "
                             "Pass a directory to auto-name: --save-traj logs/ "
                             "→ logs/traj_YYYYMMDD_HHMMSS.json")
    # ── Multi-camera fusion broadcasting ─────────────────────────────────────
    parser.add_argument("--cam-id",       default="cam1",
                        help="this instance's ID, broadcast in fusion packets (default cam1)")
    parser.add_argument("--fusion-host",  default="127.0.0.1",
                        help="UDP host for fusion.py (default 127.0.0.1)")
    parser.add_argument("--fusion-port",  type=int, default=0,
                        help="UDP port for fusion.py (0 = disable broadcasting)")
    parser.add_argument("--port-offset",  type=int, default=0,
                        help="offset added to all UDP/HTTP ports — use a different value "
                             "per instance when running multiple detectors on the same host")
    parser.set_defaults(**_cfg)   # config file values override code defaults
    args = parser.parse_args()    # CLI args override everything

    # Apply port offset so multiple detectors on one host don't collide
    if args.port_offset:
        for _p in ("viz3d_port", "ctrl_port", "webview_port", "mjpeg_port"):
            v = getattr(args, _p, 0)
            if v > 0:
                setattr(args, _p, v + args.port_offset)

    viz     = not args.no_viz
    hsv_low  = np.array([args.h_low,  args.s_min, args.v_min], dtype=np.uint8)

    # ── World-frame transform ─────────────────────────────────────────────────
    # _pose is the single source of truth for camera→ground geometry.
    # It starts with yaml cold-start values and is updated every frame by
    # per-frame AprilTag detection (EMA-smoothed).  All closures read it via
    # the dict reference — no nonlocal needed.
    cam_height    = args.camera_height
    cam_pitch_rad = np.radians(args.camera_pitch)
    tag_size_m    = 0.0 if args.no_tag else args.tag_size_m
    use_world     = cam_height > 0.01 or tag_size_m > 0.01

    _pose = {
        "height":    cam_height,     # EMA-smoothed height (cold-start fallback only)
        "pitch_rad": cam_pitch_rad,  # EMA-smoothed pitch  (cold-start fallback only)
        "age":       9999,           # frames since last successful tag detection
        "corners":   None,           # last detected tag corners (int32 Nx2) for overlay
        "rvec":      None,           # last tag rvec from solvePnP (for drawFrameAxes)
        "tvec":      None,           # last tag tvec
        # Full rotation: tag world → camera optical frame (from solvePnP).
        # When not None these replace the pitch-only EMA model and handle
        # roll, yaw, and pitch simultaneously.
        "R_cw":      None,           # 3×3 ndarray, R from solvePnP
        "tvec_flat": None,           # tvec as 1-D (3,) ndarray
        "tag_id":    -1,             # ID of the tag currently providing calibration (-1 = none)
    }

    hsv_high = np.array([args.h_high, 255, 255], dtype=np.uint8)

    # Mutable restitution coefficients — modified live by keyboard in main thread,
    # read by detection thread in rollout().  Dict is safe under the GIL.
    _rest = {"x": args.rest_x, "y": args.rest_y, "z": args.rest_z}

    # Optional UDP socket for viz3d.py state broadcast
    import socket as _socket_mod
    _udp_sock = None
    if args.viz3d:
        _udp_sock = _socket_mod.socket(_socket_mod.AF_INET, _socket_mod.SOCK_DGRAM)
        print(f"[INFO] viz3d UDP → localhost:{args.viz3d_port}  "
              f"(open http://localhost:5001 after starting viz3d.py)")

    _webview_sock = None
    if args.webview:
        _webview_sock = _socket_mod.socket(_socket_mod.AF_INET, _socket_mod.SOCK_DGRAM)
        print(f"[INFO] webview UDP → localhost:{args.webview_port}  "
              f"(open http://localhost:5002 after starting webview.py)")

    # Multi-camera fusion: this instance broadcasts per-frame world-frame measurements
    # to a fusion.py listener that runs a single combined EKF over N detectors.
    _fusion_sock = None
    if args.fusion_port > 0:
        _fusion_sock = _socket_mod.socket(_socket_mod.AF_INET, _socket_mod.SOCK_DGRAM)
        print(f"[INFO] fusion UDP → {args.fusion_host}:{args.fusion_port}  "
              f"cam_id={args.cam_id}  (start fusion.py on this port)")

    # Control listener — receives rest / det updates from viz3d.py and webview.py
    if args.viz3d or args.webview:
        def _ctrl_listener():
            sock = _socket_mod.socket(_socket_mod.AF_INET, _socket_mod.SOCK_DGRAM)
            sock.bind(("127.0.0.1", args.ctrl_port))
            sock.settimeout(1.0)
            while True:
                try:
                    data, _ = sock.recvfrom(4096)
                    pkt = json.loads(data)
                    if "rest" in pkt and len(pkt["rest"]) == 3:
                        _rest["x"] = round(float(pkt["rest"][0]), 2)
                        _rest["y"] = round(float(pkt["rest"][1]), 2)
                        _rest["z"] = round(float(pkt["rest"][2]), 2)
                    if "det" in pkt:
                        _disp_det[0] = pkt["det"] or None
                except _socket_mod.timeout:
                    pass
                except Exception:
                    pass

        threading.Thread(target=_ctrl_listener, daemon=True).start()
        print(f"[INFO] ctrl listener on localhost:{args.ctrl_port}  "
              f"(rest + detector selection from web sliders)")

    rec_path = None
    if args.record is not None:
        _r = args.record
        if not _r or _r.endswith("/") or _r.endswith(os.sep) or os.path.isdir(_r):
            _rdir = _r if _r else "recordings/"
            os.makedirs(_rdir, exist_ok=True)
            rec_path = os.path.join(_rdir, time.strftime("ball_%Y%m%d_%H%M%S.mp4"))
        else:
            _rdir = os.path.dirname(_r)
            if _rdir:
                os.makedirs(_rdir, exist_ok=True)
            rec_path = _r
        print(f"[INFO] Recording to: {rec_path}")

    print(f"[INFO] Detector: {args.detector.upper()}")
    if args.detector in ("hsv", "both"):
        print(f"[INFO] HSV range: H=[{args.h_low},{args.h_high}]  S>={args.s_min}  V>={args.v_min}")
        print(f"[INFO] MOG2 motion filter: {'OFF' if args.no_motion else 'ON'}")
    if args.detector in ("yolo", "both"):
        print(f"[INFO] YOLO model: {args.model}  imgsz={args.imgsz}  conf={args.conf}")
    print(f"[INFO] Physics: drag={args.coeff_drag:.3f}  "
          f"rest=({args.rest_x:.2f}, {args.rest_y:.2f}, {args.rest_z:.2f})")
    if use_world:
        print(f"[INFO] World frame ON  cold-start: height={cam_height:.2f}m  "
              f"pitch={args.camera_pitch:.1f}°  → Z=0=ground, bounce prediction active")
        if tag_size_m > 0.01:
            print(f"[INFO] AprilTag calibration: family={args.tag_family}  "
                  f"ids={args.tag_ids}  size={tag_size_m:.3f}m  (updates height+pitch per frame)")
        else:
            print("[INFO] AprilTag calibration: disabled (--tag-size-m not set)")
    else:
        print("[INFO] World frame OFF → EKF in camera body frame, bounce prediction disabled")
        print("[INFO]   Set --camera-height or --tag-size-m to enable world frame")

    # ── Camera backend (realsense | zed | webcam — see config[camera][backend])
    from cameras import from_config as _cam_from_config
    # Merge yaml `camera:` section with CLI override; CLI --camera-backend always wins.
    _cam_section = dict(getattr(args, "_camera_section", None) or {})
    _cam_section["backend"] = args.camera_backend
    _cam_cfg = {
        "width":  args.width,
        "height": args.height,
        "camera": _cam_section,
    }
    cam = _cam_from_config(_cam_cfg)
    cam.start()
    intr = cam.intrinsics
    fx   = intr.fx   # used for visual depth estimate
    has_depth = cam.has_depth

    print(f"[INFO] Camera intr  fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"ppx={intr.ppx:.1f} ppy={intr.ppy:.1f}  depth={'YES' if has_depth else 'NO (visual only)'}")
    print(f"[INFO] Max visual range ≈ {fx * BALL_RADIUS / args.min_radius:.1f} m "
          f"(fx={fx:.0f}, R={BALL_RADIUS}m, min_r={args.min_radius}px)")

    # Save camera intrinsics sidecar alongside raw recordings for offline replay
    if rec_path and getattr(args, "record_raw", False):
        _intr_path = os.path.splitext(rec_path)[0] + "_intrinsics.json"
        with open(_intr_path, "w") as _f:
            json.dump({
                "fx": intr.fx, "fy": intr.fy,
                "ppx": intr.ppx, "ppy": intr.ppy,
                "width": intr.width, "height": intr.height,
                "dist": list(intr.coeffs[:5]),
                "tag_family": args.tag_family, "tag_ids": list(args.tag_ids),
                "tag_size_m": tag_size_m,
                "coeff_drag": args.coeff_drag,
                "rest_x": args.rest_x, "rest_y": args.rest_y, "rest_z": args.rest_z,
                "h_low": args.h_low, "h_high": args.h_high,
                "s_min": args.s_min, "v_min": args.v_min,
                "min_radius_px": args.min_radius, "circularity": args.circularity,
            }, _f, indent=2)
        print(f"[INFO] Intrinsics saved → {_intr_path}  (load with replay.py)")

    # ── Shared state ──────────────────────────────────────────────────────────
    buf_lock    = threading.Lock()
    buf_frames  = None
    buf_updated = threading.Event()
    stop_flag   = threading.Event()

    disp_lock   = threading.Lock()
    disp_frame  = [None]
    court_frame = [None]
    _disp_det   = [None]   # which detector label to show in MJPEG (None = composite)
    video_writer = [None]  # cv2.VideoWriter; kept here so finally can release it safely

    # Trajectory log for rest-coefficient calibration (--save-traj)
    _traj_log    = []       # list of frame dicts; appended by detection_worker
    _traj_prev_z = [None]  # previous frame's Z for bounce detection

    # ── Internal full-resolution MJPEG server ─────────────────────────────────
    if args.webview and getattr(args, "mjpeg_port", 0) > 0:
        import socketserver as _mjpeg_ss
        from http.server import BaseHTTPRequestHandler as _MJPEG_BHR, HTTPServer as _MJPEG_HS

        class _MJPEGHandler(_MJPEG_BHR):
            def log_message(self, *_): pass
            def do_GET(self):
                p = self.path.split("?")[0].rstrip("/")
                if p in ("", "/main"):
                    buf_ref = disp_frame
                elif p == "/court":
                    buf_ref = court_frame
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=--frame")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    while not stop_flag.is_set():
                        with disp_lock:
                            f = buf_ref[0]
                        if f is not None:
                            ok, enc = cv2.imencode(
                                ".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
                            if ok:
                                body = enc.tobytes()
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                                self.wfile.write(body)
                                self.wfile.write(b"\r\n")
                                self.wfile.flush()
                        time.sleep(1 / 25)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        class _MJPEGServer(_mjpeg_ss.ThreadingMixIn, _MJPEG_HS):
            daemon_threads = True

        _mjpeg_srv = _MJPEGServer(("0.0.0.0", args.mjpeg_port), _MJPEGHandler)
        threading.Thread(target=_mjpeg_srv.serve_forever, daemon=True).start()
        print(f"[INFO] MJPEG  → http://localhost:{args.mjpeg_port}/main  "
              f"(full resolution, used by webview.py)")

    # ── Detection thread ──────────────────────────────────────────────────────
    def detection_worker():
        import os
        try:
            param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO) - 1)
            os.sched_setscheduler(0, os.SCHED_FIFO, param)
        except (PermissionError, OSError):
            try:
                os.nice(-10)
            except PermissionError:
                pass

        # ── Build per-detector closures ───────────────────────────────────────
        SPORTS_BALL_CLS = 32

        # YOLO detect function (shared if both modes use it)
        _yolo_fn = None
        if args.detector in ("yolo", "both"):
            from ultralytics import YOLO as _YOLO
            import torch as _torch
            _device = "cuda:0" if _torch.cuda.is_available() else "cpu"
            _half   = _device != "cpu"
            print(f"[INFO] Loading YOLO {args.model} on {_device} ...")
            _model = _YOLO(args.model)
            _dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
            for _ in range(3):
                _model(_dummy, verbose=False, device=_device, half=_half)
            print("[INFO] YOLO ready.")
            def _yolo_fn(frame):
                orig_h, orig_w = frame.shape[:2]
                small = cv2.resize(frame, (args.imgsz, args.imgsz))
                sx, sy = orig_w / args.imgsz, orig_h / args.imgsz
                results = _model.track(small, conf=args.conf, persist=True,
                                       verbose=False, device=_device, half=_half)
                best_box = None; best_c = 0.0
                for res in results:
                    for box in res.boxes:
                        if int(box.cls[0]) == SPORTS_BALL_CLS:
                            c = float(box.conf[0])
                            if c > best_c:
                                best_c, best_box = c, box
                if best_box is None:
                    return None, None, None, None
                x1, y1, x2, y2 = [float(v) for v in best_box.xyxy[0]]
                cx = int((x1 + x2) / 2 * sx)
                cy = int((y1 + y2) / 2 * sy)
                r  = float(max((x2 - x1) * sx, (y2 - y1) * sy) / 2)
                return cx, cy, r, None

        # HSV detect function
        _hsv_fn = None
        if args.detector in ("hsv", "both"):
            _back_sub = (None if args.no_motion else
                         cv2.createBackgroundSubtractorMOG2(
                             history=100, varThreshold=args.mog2_threshold,
                             detectShadows=False))
            def _hsv_fn(frame):
                return detect_tennis_ball(frame, hsv_low, hsv_high, _back_sub,
                                          min_r=args.min_radius,
                                          min_circ=args.circularity)

        # Active detector list: order = [YOLO, HSV] for "both"
        _detectors = []   # list of (label, detect_fn)
        if args.detector in ("yolo", "both"):
            _detectors.append(("YOLO", _yolo_fn))
        if args.detector in ("hsv", "both"):
            _detectors.append(("HSV",  _hsv_fn))

        # Per-detector state: last position, miss counter, EKF, FPS, cached rollout
        _st = {
            label: {"cx": None, "cy": None, "r": None, "miss": 0,
                    "ekf": PhysicsEKF(coeff_drag=args.coeff_drag),
                    "fps": _FPS(),
                    "traj_pts": []}   # cached rollout — computed once per frame
            for label, _ in _detectors
        }

        # Court-view world-frame state (written by detection loop, read by renderer)
        _court = {
            "ball_hist": [],   # [(x,y,z)] last 120 EKF positions in world frame
            "traj":      [],   # [(x,y,z)] predicted trajectory in world frame
            "bounces":   [],   # [(x,y,z)] predicted bounce points in world frame
        }

        # Per-detector ball history for UDP/web selector (label → list of (x,y,z))
        _hist = {label: [] for label, _ in _detectors}

        # ── ArUco / AprilTag ground calibration ───────────────────────────────
        _tag_active = tag_size_m > 0.01
        if _tag_active:
            _aruco_dict   = cv2.aruco.getPredefinedDictionary(
                _ARUCO_FAMILIES.get(args.tag_family, cv2.aruco.DICT_APRILTAG_36h11))
            _aruco_params = cv2.aruco.DetectorParameters()
            try:                                    # OpenCV 4.7+ OOP API
                _aruco_obj = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)
                def _detect_markers(img):
                    return _aruco_obj.detectMarkers(img)
            except AttributeError:                  # OpenCV ≤ 4.6 legacy API
                def _detect_markers(img):
                    return cv2.aruco.detectMarkers(img, _aruco_dict,
                                                   parameters=_aruco_params)
            # 3-D tag corners in tag frame (Z=0=ground, Z-up)
            _th = tag_size_m / 2.0   # tag half-size (avoid colliding with YOLO _half)
            _tag_obj = np.array([
                [-_th,  _th, 0.0],
                [ _th,  _th, 0.0],
                [ _th, -_th, 0.0],
                [-_th, -_th, 0.0],
            ], dtype=np.float32)
            _cam_mat  = np.array([
                [intr.fx, 0,       intr.ppx],
                [0,       intr.fy, intr.ppy],
                [0,       0,       1       ],
            ], dtype=np.float32)
            _dist     = np.array(intr.coeffs[:5], dtype=np.float32)
            print(f"[INFO] ArUco detector ready: {args.tag_family} ids={args.tag_ids} "
                  f"size={tag_size_m:.3f}m")

        # ── Save restitution to yaml ──────────────────────────────────────────
        def _save_yaml():
            import re
            try:
                with open(args.config) as _f:
                    _txt = _f.read()
                _txt = re.sub(r"(rest_x:\s*)[\d.]+", f"rest_x: {_rest['x']:.2f}", _txt)
                _txt = re.sub(r"(rest_y:\s*)[\d.]+", f"rest_y: {_rest['y']:.2f}", _txt)
                _txt = re.sub(r"(rest_z:\s*)[\d.]+", f"rest_z: {_rest['z']:.2f}", _txt)
                with open(args.config, "w") as _f:
                    _f.write(_txt)
                print(f"\n[INFO] Saved rest ({_rest['x']:.2f}/{_rest['y']:.2f}/{_rest['z']:.2f})"
                      f" → {args.config}")
            except Exception as _e:
                print(f"\n[WARN] yaml save failed: {_e}")

        # ── 2-D court view renderer ───────────────────────────────────────────
        def _render_court_view():
            """Two side-by-side panels: top-down (XY) and side (XZ), world frame."""
            PANEL = 500
            SCALE = 80          # px per metre → ±3.1 m visible range
            MID   = PANEL // 2  # world origin in pixels

            def _xy(wx, wy):    # world XY → top-down pixel (Y up on screen)
                return (int(MID + wx * SCALE), int(MID - wy * SCALE))

            def _xz(wx, wz):    # world XZ → side pixel (Z up on screen)
                return (int(MID + wx * SCALE), int(MID - wz * SCALE))

            def _in(pt):        # pixel inside panel?
                return 0 <= pt[0] < PANEL and 0 <= pt[1] < PANEL

            top  = np.full((PANEL, PANEL, 3), (28, 28, 28), dtype=np.uint8)
            side = np.full((PANEL, PANEL, 3), (28, 28, 28), dtype=np.uint8)

            # Grid lines every 0.5 m
            for _i in range(-6, 7):
                _m = _i * 0.5
                _gc = (55, 55, 55) if _i % 2 else (70, 70, 70)
                _px = int(MID + _m * SCALE); _py = int(MID - _m * SCALE)
                cv2.line(top,  (_px, 0), (_px, PANEL), _gc, 1)
                cv2.line(top,  (0, _py), (PANEL, _py), _gc, 1)
                cv2.line(side, (_px, 0), (_px, PANEL), _gc, 1)
                cv2.line(side, (0, _py), (PANEL, _py), _gc, 1)

            # Ground line (Z=0) in side panel
            cv2.line(side, (0, MID), (PANEL, MID), (50, 100, 50), 2)

            # Axis arrows
            cv2.arrowedLine(top,  _xy(0,0), _xy(1.2,0),   (80,80,220), 2, tipLength=0.08)
            cv2.arrowedLine(top,  _xy(0,0), _xy(0,1.2),   (80,220,80), 2, tipLength=0.08)
            cv2.arrowedLine(side, _xz(0,0), _xz(1.2,0),   (80,80,220), 2, tipLength=0.08)
            cv2.arrowedLine(side, _xz(0,0), _xz(0,1.2),   (80,220,80), 2, tipLength=0.08)
            cv2.putText(top,  "X", _xy(1.25,0), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(80,80,220),1)
            cv2.putText(top,  "Y", _xy(0,1.3),  cv2.FONT_HERSHEY_SIMPLEX, 0.4,(80,220,80),1)
            cv2.putText(side, "X", _xz(1.25,0), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(80,80,220),1)
            cv2.putText(side, "Z", _xz(0,1.3),  cv2.FONT_HERSHEY_SIMPLEX, 0.4,(80,220,80),1)

            # Title
            cv2.putText(top,  "TOP (X-Y)", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,160,160), 1)
            cv2.putText(side, "SIDE (X-Z)", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,160,160), 1)

            # Tag at origin
            for _p, _fn in ((top, _xy), (side, _xz)):
                cv2.drawMarker(_p, _fn(0,0), (0,220,220),
                               cv2.MARKER_SQUARE, 14, 2)

            # Camera position + look-direction arrow
            if _pose["R_cw"] is not None:
                _t_cam = -(_pose["R_cw"].T @ _pose["tvec_flat"])
                _cx, _cy, _cz = _t_cam
                _look = _pose["R_cw"].T @ np.array([0.0, 0.0, 1.0])
                for _p, _fn, _a1, _a2 in (
                    (top,  _xy, _cx, _cy),
                    (side, _xz, _cx, _cz),
                ):
                    _lx = _look[0]; _la = _look[1] if _fn is _xy else _look[2]
                    _cp  = _fn(_a1, _a2)
                    _tip = _fn(_a1 + _lx * 0.8, _a2 + _la * 0.8)
                    cv2.circle(_p, _cp, 7, (255, 140, 50), -1)
                    if _in(_cp) and _in(_tip):
                        cv2.arrowedLine(_p, _cp, _tip, (255,140,50), 2, tipLength=0.25)
                cv2.putText(top, "CAM",
                            (_xy(_cx, _cy)[0]+9, _xy(_cx, _cy)[1]+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,140,50), 1)

            # Ball history trail
            for _bx, _by, _bz in _court["ball_hist"]:
                _pt = _xy(_bx, _by); _ps = _xz(_bx, _bz)
                if _in(_pt): cv2.circle(top,  _pt, 2, (50,160,255), -1)
                if _in(_ps): cv2.circle(side, _ps, 2, (50,160,255), -1)

            # Predicted trajectory
            _pp_top = _pp_side = None
            for _bx, _by, _bz in _court["traj"]:
                _pt = _xy(_bx, _by); _ps = _xz(_bx, _bz)
                if _pp_top  and _in(_pt): cv2.line(top,  _pp_top,  _pt, (0,60,255), 1)
                if _pp_side and _in(_ps): cv2.line(side, _pp_side, _ps, (0,60,255), 1)
                _pp_top = _pt; _pp_side = _ps

            # Bounce points
            for _bx, _by, _bz in _court["bounces"]:
                _pt = _xy(_bx, _by); _ps = _xz(_bx, _bz)
                if _in(_pt): cv2.drawMarker(top,  _pt, (0,255,255), cv2.MARKER_CROSS,12,2)
                if _in(_ps): cv2.drawMarker(side, _ps, (0,255,255), cv2.MARKER_CROSS,12,2)

            # Rest values footer
            _rf = f"rest  X={_rest['x']:.2f}  Y={_rest['y']:.2f}  Z={_rest['z']:.2f}"
            _kf = "z/Z:rest_z   x/X:rest_x   y/Y:rest_y   s:save"
            cv2.putText(top,  _rf, (6, PANEL-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,180,120), 1)
            cv2.putText(side, _kf, (6, PANEL-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120,120,120), 1)

            _div = np.full((PANEL, 4, 3), (100,100,100), dtype=np.uint8)
            return np.hstack([top, _div, side])

        # ── World ↔ body transforms using full R matrix when available ──────────
        def _world_active():
            """True when a world frame (Z=0=ground) transform is available."""
            return _pose["R_cw"] is not None or _pose["height"] > 0.01

        def _to_world(p_body):
            """Camera body frame → world frame (Z=0=ground).
            Uses full R matrix from solvePnP when available (handles roll/yaw/pitch);
            falls back to EMA pitch-only model when R_cw is not yet set.
            """
            if _pose["R_cw"] is not None:
                p_opt = body_to_optical(p_body)
                return _pose["R_cw"].T @ (p_opt - _pose["tvec_flat"])
            return body_to_world(p_body, _pose["height"], _pose["pitch_rad"])

        def _to_body(p_world):
            """World frame → camera body frame.  Inverse of _to_world."""
            if _pose["R_cw"] is not None:
                p_opt = _pose["R_cw"] @ np.asarray(p_world) + _pose["tvec_flat"]
                return optical_to_body(p_opt)
            return world_to_body(p_world, _pose["height"], _pose["pitch_rad"])

        # ── Depth + EKF closure  (identical pipeline for every detector) ──────
        def _process(cx_new, cy_new, r_new, st, t_now):
            """Update st with one new detection result. Returns (detected, coasting, pos_ekf, depth_fused).
            pos_ekf is in world frame when use_world=True, camera body frame otherwise.
            """
            if cx_new is not None:
                st["miss"] = 0
                st["cx"], st["cy"], st["r"] = cx_new, cy_new, r_new
            else:
                st["miss"] += 1
                if st["miss"] > COAST_FRAMES:
                    st["ekf"].reset()

            if st["cx"] is None or st["miss"] > COAST_FRAMES:
                return False, False, None, 0.0

            detected = (cx_new is not None)
            coasting = not detected

            # Visual depth: depth = fx · R / r_px  (always available — needs only fx + ball radius)
            depth_vis = (fx * BALL_RADIUS / st["r"]) if st["r"] and st["r"] > 0 else 0.0

            # Sensor depth: only if backend supplied an aligned depth frame.
            # depth_arr is in metres (float32) and on the SAME pixel grid as colour
            # (camera backends do alignment internally — see cameras/realsense.py
            # rs.align(rs.stream.color), and ZED's depth is left-cam-aligned).
            depth_sensor = 0.0
            if depth_arr is not None:
                ih, iw = depth_arr.shape[:2]
                dx = int(np.clip(st["cx"], 0, iw - 1))
                dy = int(np.clip(st["cy"], 0, ih - 1))
                r  = DEPTH_SAMPLE_R
                patch   = depth_arr[max(0, dy-r):min(ih, dy+r+1),
                                    max(0, dx-r):min(iw, dx+r+1)]
                valid_d = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
                d_surf  = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0
                depth_sensor = d_surf + BALL_RADIUS if d_surf > 0 else 0.0

            vis_ok    = DEPTH_MIN < depth_vis    < DEPTH_MAX
            sensor_ok = DEPTH_MIN < depth_sensor < DEPTH_MAX
            if vis_ok and sensor_ok:
                ratio = depth_vis / depth_sensor
                depth_fused = (VIS_WEIGHT * depth_vis + (1 - VIS_WEIGHT) * depth_sensor
                               if 0.5 < ratio < 2.0 else depth_vis)
            elif vis_ok:
                depth_fused = depth_vis
            elif sensor_ok:
                depth_fused = depth_sensor
            else:
                depth_fused = 0.0

            if depth_fused > 0 and detected:
                p_opt  = intr.deproject(st["cx"], st["cy"], depth_fused)
                p_body = optical_to_body(p_opt)
                # Convert to world frame (Z=0=ground) using full R matrix when available
                p_ekf = _to_world(p_body) if _world_active() else p_body
                st["ekf"].update(p_ekf, _meas_covariance(depth_fused), t_now)

                # ── Multi-camera fusion broadcast ──────────────────────────────
                # Send the RAW per-frame measurement (NOT EKF-smoothed) so the
                # fusion process can run its own EKF across all N sources.
                # World frame is required (so all cameras agree on the origin).
                if _fusion_sock is not None and _world_active():
                    # Isotropic cov estimate from depth-axis variance polynomial
                    _cov_var = float(_meas_covariance(depth_fused)[0, 0])
                    _pkt = {
                        "t":       time.time(),
                        "cam_id":  args.cam_id,
                        "pos":     [round(float(v), 4) for v in p_ekf],
                        "cov":     round(_cov_var, 6),
                        "depth":   round(float(depth_fused), 3),
                        "tag_id":  int(_pose["tag_id"]),
                        "tag_age": int(_pose["age"]),
                    }
                    try:
                        _fusion_sock.sendto(json.dumps(_pkt).encode(),
                                            (args.fusion_host, args.fusion_port))
                    except Exception:
                        pass

            pos_ekf = st["ekf"].x[:3].copy() if st["ekf"].initialized else None
            return detected, coasting, pos_ekf, depth_fused

        # ── Annotation closure (per-panel) ────────────────────────────────────
        def _annotate(frame, st, mask, label, detected, coasting, pos_ekf, depth_fused, hist=None):
            vis  = frame.copy()
            cx, cy, r, miss = st["cx"], st["cy"], st["r"], st["miss"]

            # Header strip
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (30, 30, 30), -1)
            status = "BALL" if detected else ("COAST" if coasting else "—")
            cv2.putText(vis, f"{label}  [{status}]  {st['fps'].fps:.0f} fps",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ── Tag ground-calibration indicator ──────────────────────────────
            _h, _pr = _pose["height"], _pose["pitch_rad"]
            _age    = _pose["age"]
            if _tag_active:
                # Decompose full R into ZYX Euler angles when available
                _R = _pose["R_cw"]
                if _R is not None:
                    Rt = _R.T   # camera axes in world frame
                    _yaw_d   = np.degrees(np.arctan2(Rt[1, 0], Rt[0, 0]))
                    _pitch_d = np.degrees(np.arcsin(np.clip(-Rt[2, 0], -1.0, 1.0)))
                    _roll_d  = np.degrees(np.arctan2(Rt[2, 1], Rt[2, 2]))
                    _rpy = f"R={_roll_d:+.1f}°  P={_pitch_d:+.1f}°  Y={_yaw_d:+.1f}°"
                else:
                    _rpy = f"P={np.degrees(_pr):+.1f}° (EMA)"
                if _age == 0:
                    _tag_col = (0, 230, 0)
                    _tag_lbl = f"TAG OK   H={_h:.2f}m  {_rpy}"
                elif _age < TAG_STALE_FRAMES:
                    _tag_col = (0, 165, 255)
                    _tag_lbl = f"TAG [{_age}f]  H={_h:.2f}m  {_rpy}"
                else:
                    _tag_col = (0, 0, 220)
                    _tag_lbl = f"NO TAG   H={_h:.2f}m  {_rpy}"
                cv2.putText(vis, _tag_lbl,
                            (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _tag_col, 1)
                # Draw tag outline + coordinate axes when recently seen
                if _age < 10 and _pose["corners"] is not None:
                    cv2.polylines(vis,
                                  [_pose["corners"].reshape(-1, 1, 2)],
                                  True, _tag_col, 2)
                    if _pose["rvec"] is not None:
                        cv2.drawFrameAxes(vis, _cam_mat, _dist,
                                          _pose["rvec"], _pose["tvec"],
                                          tag_size_m * 0.5)

            if cx is not None and miss <= COAST_FRAMES:
                col = (0, 255, 0) if detected else (0, 165, 255)
                cv2.circle(vis, (cx, cy), max(int(r), 3), col, 2)
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
                btag = "ball" if detected else f"coast {miss}/{COAST_FRAMES}"
                cv2.putText(vis, f"{btag} r={r:.0f}px",
                            (cx - int(r), max(cy - int(r) - 6, 68)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                if pos_ekf is not None:
                    z_suffix = " agl" if _world_active() else ""
                    cv2.putText(vis,
                                f"({pos_ekf[0]:+.2f},{pos_ekf[1]:+.2f},{pos_ekf[2]:+.2f})m{z_suffix}",
                                (cx - int(r), max(cy - int(r) - 20, 84)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
            else:
                cv2.putText(vis, "No ball",
                            (20, vis.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            # Historical trail — past 3-D positions projected onto image
            if hist and _world_active():
                n = len(hist)
                prev_px = None
                for i, h_pt in enumerate(hist):
                    t = i / max(n - 1, 1)          # 0 = oldest, 1 = newest
                    pt_body = _to_body(h_pt)
                    px = _body_to_pixel(pt_body, intr)
                    if px is None:
                        prev_px = None
                        continue
                    # BGR fade: dark orange → bright orange
                    col = (0, int(30 + 135 * t), int(80 + 175 * t))
                    cv2.circle(vis, px, max(2, int(2 + 2 * t)), col, -1)
                    if prev_px is not None:
                        cv2.line(vis, prev_px, px, col, 1)
                    prev_px = px

            # Trajectory overlay (use pre-computed rollout from st["traj_pts"])
            traj_pts = st["traj_pts"]
            if not args.no_traj and traj_pts:
                prev_px  = None
                last_z   = traj_pts[0][1][2]
                for _, pt in traj_pts:
                    pt_body = _to_body(pt) if _world_active() else pt
                    px = _body_to_pixel(pt_body, intr)
                    if px is None:
                        prev_px = None; continue
                    is_bounce = (last_z > 0.01 and pt[2] <= 0.01)
                    cv2.circle(vis, px, 3, (0, 255, 255) if is_bounce else (0, 60, 255), -1)
                    if prev_px is not None:
                        cv2.line(vis, prev_px, px, (0, 60, 255), 1)
                    prev_px = px; last_z = pt[2]

            # ── EKF debug status bar (bottom strip) ───────────────────────────
            _BAR_H = 68
            _fy    = vis.shape[0] - _BAR_H
            cv2.rectangle(vis, (0, _fy), (vis.shape[1], vis.shape[0]), (18, 18, 18), -1)

            if pos_ekf is not None:
                _px, _py, _pz = pos_ekf
                _vx, _vy, _vz = st["ekf"].x[3:6]
                # Z colour: green ≈ ground, yellow = small offset, red = large offset
                _zc = ((0, 210, 0)   if abs(_pz) < 0.05 else
                       (0, 200, 255) if abs(_pz) < 0.20 else
                       (0, 60,  220))
                _zs = ("Z OK" if abs(_pz) < 0.05 else
                       "Z ~"  if abs(_pz) < 0.20 else "Z !")
                # Row 1: position
                cv2.putText(vis,
                            f"POS  X={_px:+.3f}  Y={_py:+.3f}  Z=",
                            (8, _fy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
                _tw = cv2.getTextSize(f"POS  X={_px:+.3f}  Y={_py:+.3f}  Z=",
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
                cv2.putText(vis, f"{_pz:+.3f} m",
                            (8 + _tw, _fy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _zc, 2)
                # Row 2: velocity + depth
                cv2.putText(vis,
                            f"VEL  Vx={_vx:+.2f}  Vy={_vy:+.2f}  Vz={_vz:+.2f} m/s"
                            f"    depth={depth_fused:.2f} m",
                            (8, _fy + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (160, 160, 160), 1)
                # Row 3: restitution coefficients
                cv2.putText(vis,
                            f"rest  X={_rest['x']:.2f}  Y={_rest['y']:.2f}  Z={_rest['z']:.2f}"
                            f"    [x/X  y/Y  z/Z: ±0.05  |  s: save yaml]",
                            (8, _fy + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 140, 110), 1)
                # Right-side Z indicator
                cv2.putText(vis, _zs,
                            (vis.shape[1] - 90, _fy + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, _zc, 2)
            else:
                cv2.putText(vis, "EKF: waiting for first detection…",
                            (8, _fy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)
                cv2.putText(vis,
                            f"rest  X={_rest['x']:.2f}  Y={_rest['y']:.2f}  Z={_rest['z']:.2f}"
                            f"    [x/X  y/Y  z/Z: ±0.05  |  s: save yaml]",
                            (8, _fy + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 140, 110), 1)

            # HSV mask side-by-side (only in single HSV mode with --show-mask)
            if args.show_mask and mask is not None and args.detector == "hsv":
                vis = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

            return vis

        _court_last_t = 0.0   # throttle court_view rendering to 10 fps

        # ── Main loop ─────────────────────────────────────────────────────────
        while not stop_flag.is_set():
            if not buf_updated.wait(timeout=1.0):
                continue
            buf_updated.clear()

            with buf_lock:
                frames = buf_frames

            # frames is a Frame dataclass from cameras.base
            color     = frames.color
            depth_arr = frames.depth   # may be None for RGB-only cameras
            t_now     = frames.t

            # ── Per-frame AprilTag ground calibration ──────────────────────────
            _pose["age"] = min(_pose["age"] + 1, 9999)
            if _tag_active:
                try:
                    corners, ids, _ = _detect_markers(color)
                except Exception:
                    corners, ids = [], None
                if ids is not None:
                    # Among all accepted tag IDs visible this frame, pick the
                    # largest (= closest to camera = best solvePnP accuracy)
                    _best_area, _best_i = -1.0, -1
                    for _i, _tid in enumerate(ids.flatten()):
                        if int(_tid) not in args.tag_ids:
                            continue
                        _area = cv2.contourArea(corners[_i][0])
                        if _area > _best_area:
                            _best_area, _best_i = _area, _i

                    if _best_i >= 0:
                        _img_pts = corners[_best_i][0].astype(np.float32)
                        _ok, _rvec, _tvec = cv2.solvePnP(
                            _tag_obj, _img_pts, _cam_mat, _dist,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
                        if _ok:
                            _R, _ = cv2.Rodrigues(_rvec)
                            _t_cam = -(_R.T @ _tvec.flatten())
                            _new_h = float(_t_cam[2])
                            _look  = _R.T @ np.array([0.0, 0.0, 1.0])
                            _new_p = float(np.arcsin(np.clip(_look[2], -1.0, 1.0)))
                            if 0.05 < _new_h < 3.0:
                                a = TAG_EMA
                                _pose["height"]    = a * _new_h + (1 - a) * _pose["height"]
                                _pose["pitch_rad"] = a * _new_p + (1 - a) * _pose["pitch_rad"]
                                _pose["age"]       = 0
                                _pose["corners"]   = corners[_best_i][0].astype(np.int32)
                                _pose["rvec"]      = _rvec
                                _pose["tvec"]      = _tvec
                                _pose["R_cw"]      = _R
                                _pose["tvec_flat"] = _tvec.flatten()
                                _pose["tag_id"]    = int(ids.flatten()[_best_i])

            panels      = []
            term_parts  = []

            for label, det_fn in _detectors:
                st = _st[label]
                cx, cy, r_px, mask = det_fn(color)
                detected, coasting, pos_ekf, depth_fused = _process(cx, cy, r_px, st, t_now)
                st["fps"].tick()

                # Pre-compute rollout once per detector per frame (reused by
                # _annotate overlay and court/UDP state — avoids double rollout)
                if st["ekf"].initialized and not args.no_traj:
                    st["traj_pts"] = st["ekf"].rollout(
                        cx=_rest["x"], cy=_rest["y"], cz=_rest["z"])
                else:
                    st["traj_pts"] = []

                # terminal line segment
                z_tag = "agl" if _world_active() else "body"
                p_str  = "—" if pos_ekf is None else (
                    f"({pos_ekf[0]:+.2f},{pos_ekf[1]:+.2f},{pos_ekf[2]:+.2f}){z_tag}")
                status = "BALL " if detected else ("COAST" if coasting else "     ")
                term_parts.append(
                    f"[{label}:{status}] {p_str} {depth_fused:.2f}m {st['fps'].fps:.0f}fps")

                if viz or args.show_mask or rec_path is not None:
                    panels.append(
                        _annotate(color, st, mask, label,
                                  detected, coasting, pos_ekf, depth_fused,
                                  _hist[label]))

            print(f"\r{'  ||  '.join(term_parts)}", end="", flush=True)

            # ── Update court-view state from the first active detector ─────────
            _first_label = _detectors[0][0]
            _first_st    = _st[_first_label]
            if _world_active() and _first_st["ekf"].initialized:
                _ep = _first_st["ekf"].x[:3].copy()
                _court["ball_hist"].append((_ep[0], _ep[1], _ep[2]))
                if len(_court["ball_hist"]) > 120:
                    _court["ball_hist"].pop(0)
                # Reuse cached rollout — already computed above
                _traj_raw = _first_st["traj_pts"]
                _court["traj"]    = [tuple(p) for _, p in _traj_raw]
                _court["bounces"] = []
                _last_z = _traj_raw[0][1][2] if _traj_raw else 1.0
                for _, _pt in _traj_raw:
                    if _last_z > 0.01 and _pt[2] <= 0.01:
                        _court["bounces"].append(tuple(_pt))
                    _last_z = _pt[2]
            elif not _world_active():
                _court["ball_hist"].clear()
                _court["traj"].clear()
                _court["bounces"].clear()

            # Trajectory log for calibrate_rest.py (--save-traj)
            if args.save_traj and _world_active():
                _se0 = _st[_detectors[0][0]]["ekf"]
                if _se0.initialized:
                    _pos_w = [round(float(v), 4) for v in _se0.x[:3]]
                    _vel_w = [round(float(v), 4) for v in _se0.x[3:]]
                    _pz    = _traj_prev_z[0]
                    _bounce = (_pz is not None and _pz > 0.06 and _pos_w[2] <= 0.06)
                    _traj_prev_z[0] = _pos_w[2]
                    _traj_log.append({
                        "t":      round(time.time(), 4),
                        "pos":    _pos_w,
                        "vel":    _vel_w,
                        "bounce": _bounce,
                    })

            # Per-detector ball history (for UDP multi-detector selector)
            for _lbl, _ in _detectors:
                _se = _st[_lbl]["ekf"]
                if _world_active() and _se.initialized:
                    _ep2 = _se.x[:3]
                    _hist[_lbl].append((round(float(_ep2[0]),4),
                                        round(float(_ep2[1]),4),
                                        round(float(_ep2[2]),4)))
                    if len(_hist[_lbl]) > 120:
                        _hist[_lbl].pop(0)
                elif not _world_active():
                    _hist[_lbl].clear()

            if (viz or args.show_mask or rec_path is not None
                    or _webview_sock is not None) and panels:
                if len(panels) == 2:
                    # Both mode: scale each panel to 50% and place side-by-side
                    h, w = color.shape[:2]
                    p0 = cv2.resize(panels[0], (w // 2, h // 2))
                    p1 = cv2.resize(panels[1], (w // 2, h // 2))
                    out = np.hstack([p0, p1])
                    # centre divider
                    mid = out.shape[1] // 2
                    out[:, mid-1:mid+1] = (200, 200, 200)
                else:
                    out = panels[0]

                if rec_path is not None:
                    _rec_frame = color if getattr(args, "record_raw", False) else out
                    if video_writer[0] is None:
                        fh, fw = _rec_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer[0] = cv2.VideoWriter(rec_path, fourcc, 30.0, (fw, fh))
                    video_writer[0].write(_rec_frame)

                # Update shared display buffers (for OpenCV window + MJPEG server)
                if viz or args.show_mask or args.webview:
                    _now_t = time.perf_counter()
                    # Build label→panel map for detector selection
                    _panel_map = {lbl: panels[i]
                                  for i, (lbl, _) in enumerate(_detectors)
                                  if i < len(panels)}
                    _sel = _disp_det[0]
                    _mjpeg_frame = _panel_map.get(_sel, out)
                    with disp_lock:
                        disp_frame[0] = _mjpeg_frame
                        if _now_t - _court_last_t >= 0.10:   # 10 fps cap
                            court_frame[0] = _render_court_view()
                            _court_last_t  = _now_t

            # ── UDP state broadcast (viz3d + webview) ─────────────────────────
            if _udp_sock is not None or _webview_sock is not None:
                _pkt = {
                    "t":        time.time(),   # heartbeat — receivers check freshness
                    "detectors": [lbl for lbl, _ in _detectors],
                    "cam":      ([round(float(v),4) for v in
                                  -(_pose["R_cw"].T @ _pose["tvec_flat"])]
                                 if _pose["R_cw"] is not None else None),
                    "cam_look": ([round(float(v),4) for v in
                                  _pose["R_cw"].T @ np.array([0.,0.,1.])]
                                 if _pose["R_cw"] is not None else None),
                    "tag_age":  int(_pose["age"]),
                    "rest":     [round(_rest["x"],2), round(_rest["y"],2), round(_rest["z"],2)],
                }
                for _lbl, _ in _detectors:
                    _se  = _st[_lbl]["ekf"]
                    _tr  = _st[_lbl]["traj_pts"]
                    _bn  = []
                    _lz2 = _tr[0][1][2] if _tr else 1.0
                    for _, _pp in _tr:
                        if _lz2 > 0.01 and _pp[2] <= 0.01:
                            _bn.append([round(float(v),4) for v in _pp])
                        _lz2 = _pp[2]
                    _pkt[_lbl] = {
                        "ball":      ([round(float(v),4) for v in _se.x[:3]]
                                      if _se.initialized else None),
                        "vel":       ([round(float(v),4) for v in _se.x[3:]]
                                      if _se.initialized else None),
                        "traj":      [[round(float(v),4) for v in p]
                                      for _, p in _tr[::2]],
                        "bounces":   _bn,
                        "ball_hist": list(_hist[_lbl][::3]),
                    }
                _pkt_bytes = json.dumps(_pkt).encode()
                if _udp_sock is not None:
                    try:
                        _udp_sock.sendto(_pkt_bytes, ("127.0.0.1", args.viz3d_port))
                    except Exception:
                        pass
                if _webview_sock is not None:
                    try:
                        _webview_sock.sendto(
                            b'\x00' + _pkt_bytes, ("127.0.0.1", args.webview_port))
                    except Exception:
                        pass

        if video_writer[0] is not None:
            video_writer[0].release()
            video_writer[0] = None
            print(f"\n[INFO] Video saved: {rec_path}")

    det_thread = threading.Thread(target=detection_worker, daemon=True)
    det_thread.start()

    # ── Save restitution helper (called from main thread) ────────────────────
    def _save_yaml_main():
        import re
        try:
            with open(args.config) as _f:
                _txt = _f.read()
            _txt = re.sub(r"(rest_x:\s*)[\d.]+", f"rest_x: {_rest['x']:.2f}", _txt)
            _txt = re.sub(r"(rest_y:\s*)[\d.]+", f"rest_y: {_rest['y']:.2f}", _txt)
            _txt = re.sub(r"(rest_z:\s*)[\d.]+", f"rest_z: {_rest['z']:.2f}", _txt)
            with open(args.config, "w") as _f:
                _f.write(_txt)
            print(f"\n[INFO] Saved rest "
                  f"({_rest['x']:.2f}/{_rest['y']:.2f}/{_rest['z']:.2f})"
                  f" → {args.config}")
        except Exception as _e:
            print(f"\n[WARN] yaml save failed: {_e}")

    # ── Main thread: camera grab + display ───────────────────────────────────
    print(f"[INFO] Running. Press {'q' if viz else 'Ctrl+C'} to quit.")
    print("[INFO] Keys: q=quit  x/X=rest_x±0.05  y/Y=rest_y±0.05  z/Z=rest_z±0.05  s=save")
    try:
        while True:
            frames = cam.grab()
            if frames is None:
                continue
            with buf_lock:
                buf_frames = frames
            buf_updated.set()

            if viz or args.show_mask:
                with disp_lock:
                    frame  = disp_frame[0]
                    cframe = court_frame[0]
                if frame is not None:
                    cv2.imshow("ball_detection_d455", frame)
                if cframe is not None:
                    cv2.imshow("court_view", cframe)

                key = cv2.waitKey(1) & 0xFF
                if   key == ord("q"): break
                elif key == ord("x"): _rest["x"] = min(1.5, round(_rest["x"] + 0.05, 2))
                elif key == ord("X"): _rest["x"] = max(0.0, round(_rest["x"] - 0.05, 2))
                elif key == ord("y"): _rest["y"] = min(1.5, round(_rest["y"] + 0.05, 2))
                elif key == ord("Y"): _rest["y"] = max(0.0, round(_rest["y"] - 0.05, 2))
                elif key == ord("z"): _rest["z"] = min(1.5, round(_rest["z"] + 0.05, 2))
                elif key == ord("Z"): _rest["z"] = max(0.0, round(_rest["z"] - 0.05, 2))
                elif key == ord("s"): _save_yaml_main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_flag.set()
        det_thread.join(timeout=2)
        # Release video writer here (not just inside detection_worker) so that
        # Ctrl+C always flushes and closes the MP4 container properly.
        if video_writer[0] is not None:
            video_writer[0].release()
            video_writer[0] = None
            if rec_path:
                print(f"\n[INFO] Video saved: {rec_path}")
        try:
            cam.stop()
        except Exception:
            pass
        if viz or args.show_mask:
            cv2.destroyAllWindows()

        if args.save_traj and _traj_log:
            import datetime as _dt
            _n_bounces = sum(1 for f in _traj_log if f["bounce"])
            # If path is a directory (or ends with /), auto-generate filename
            _sp = args.save_traj
            if _sp.endswith("/") or _sp.endswith(os.sep) or os.path.isdir(_sp):
                os.makedirs(_sp, exist_ok=True)
                _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                _sp = os.path.join(_sp, f"traj_{_ts}.json")
            else:
                _dir = os.path.dirname(_sp)
                if _dir:
                    os.makedirs(_dir, exist_ok=True)
            _meta = {
                "coeff_drag": args.coeff_drag,
                "detector":   args.detector,
                "n_frames":   len(_traj_log),
                "n_bounces":  _n_bounces,
            }
            with open(_sp, "w") as _f:
                json.dump({"meta": _meta, "frames": _traj_log},
                          _f, separators=(",", ":"))
            print(f"[INFO] Trajectory saved → {_sp}  "
                  f"({len(_traj_log)} frames, {_n_bounces} bounces detected)")

        print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
