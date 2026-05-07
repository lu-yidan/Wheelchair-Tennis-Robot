"""
ball_detection_d455.py

Intel RealSense D455 tennis ball detection + physics-based trajectory prediction.
Standalone Python replacement for ball_detection.cpp (ZED SDK) for single-camera testing.

Detection:   MOG2 motion mask + HSV colour segmentation  (ported from ball_detection.cpp)
Depth:       visual (fx·R/r_px) fused with D455 sensor depth  (from catch_ball)
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
    conda activate catchball          # pyrealsense2, opencv-python, numpy, scipy

Usage:
    python ball_detection_d455.py                         # 1280×720, viz on
    python ball_detection_d455.py --no-viz                 # headless
    python ball_detection_d455.py --show-mask              # show HSV+motion mask
    python ball_detection_d455.py --width 848 --height 480 # 60fps mode
    python ball_detection_d455.py --no-motion              # skip MOG2 (pure HSV)
    python ball_detection_d455.py --h-low 10 --h-high 35   # HSV hue from settings.yaml
    python ball_detection_d455.py --coeff-drag 0.47        # tune drag
"""

import argparse
import os
import threading
import time
import numpy as np
import cv2
import pyrealsense2 as rs


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
        x_p[2]   = max(x_p[2], 0.0)          # clamp to floor

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

        self.x[2] = max(self.x[2], 0.0)

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

def detect_tennis_ball(frame_bgr, hsv_low, hsv_high, back_sub=None):
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
        if area < np.pi * MIN_RADIUS_PX**2:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circ = 4 * np.pi * area / (peri**2)
        if circ < MIN_CIRCULARITY:
            continue
        (cx_f, cy_f), r = cv2.minEnclosingCircle(cnt)
        if not (MIN_RADIUS_PX <= r <= MAX_RADIUS_PX):
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
    parser.add_argument("--camera-height", type=float, default=0.0,
                        help="camera centre height above ground (m); enables world-frame EKF")
    parser.add_argument("--camera-pitch",  type=float, default=0.0,
                        help="camera pitch in degrees; negative = looking down (e.g. -15)")
    parser.set_defaults(**_cfg)   # config file values override code defaults
    args = parser.parse_args()    # CLI args override everything

    viz     = not args.no_viz
    hsv_low  = np.array([args.h_low,  args.s_min, args.v_min], dtype=np.uint8)

    # ── World-frame transform ─────────────────────────────────────────────────
    # When camera-height is given, the EKF runs in world frame (Z=0 = ground)
    # so bounce prediction and the Z display are physically meaningful.
    # Without it, the EKF runs in camera body frame (old behaviour, no bounce).
    cam_height    = args.camera_height
    cam_pitch_rad = np.radians(args.camera_pitch)
    use_world     = cam_height > 0.01   # treat <1 cm as "not set"
    hsv_high = np.array([args.h_high, 255,        255       ], dtype=np.uint8)

    rec_path = None
    if args.record is not None:
        rec_path = args.record if args.record else time.strftime("ball_%Y%m%d_%H%M%S.mp4")
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
        print(f"[INFO] World frame ON: camera height={cam_height:.2f} m  "
              f"pitch={args.camera_pitch:.1f}°  → Z=0 = ground, bounce prediction active")
    else:
        print("[INFO] World frame OFF (--camera-height not set) → "
              "EKF in camera body frame, bounce prediction disabled")

    # ── RealSense pipeline ────────────────────────────────────────────────────
    pipeline   = rs.pipeline()
    _FPS_TRIES = [(60, 60), (30, 30), (15, 15)]

    def _hw_reset():
        """Hardware-reset the first RealSense device and wait for re-enumeration."""
        nonlocal pipeline
        devs = rs.context().query_devices()
        if len(devs) == 0:
            raise RuntimeError("No RealSense device found.")
        print("[INFO] Hardware reset...")
        devs[0].hardware_reset()
        time.sleep(8)
        pipeline = rs.pipeline()

    def _start_pipeline():
        nonlocal pipeline
        # Pre-emptive reset: clears stale /dev/videoX nodes from previous sessions
        try:
            _hw_reset()
        except RuntimeError as e:
            raise RuntimeError(f"No RealSense device found on startup: {e}")
        last_err = None
        for c_fps, d_fps in _FPS_TRIES:
            rs_cfg = rs.config()
            rs_cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, c_fps)
            rs_cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16,  d_fps)
            for attempt in range(2):
                try:
                    print(f"[INFO] Starting RealSense ({c_fps}/{d_fps} Hz attempt {attempt+1})...")
                    profile = pipeline.start(rs_cfg)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "resolve" in msg or "couldn't" in msg:
                        # profile/resolution not supported → try next fps
                        last_err = e
                        print(f"[WARN] Profile not supported: {e}")
                        break
                    elif "no such file" in msg or "cannot open" in msg or "map_device" in msg:
                        # stale /dev/videoX node → hardware reset and retry
                        print(f"[WARN] Stale device node ({e}); resetting...")
                        _hw_reset()
                        continue
                    raise
                try:
                    pipeline.wait_for_frames(timeout_ms=5000)
                    print(f"[INFO] RealSense OK ({c_fps}/{d_fps} Hz)")
                    return profile
                except RuntimeError:
                    print("[WARN] Frame timeout — hardware reset...")
                    pipeline.stop()
                    _hw_reset()
        raise RuntimeError(f"RealSense failed. Tried {_FPS_TRIES}. Last: {last_err!r}")

    profile = _start_pipeline()

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intrin  = color_profile.get_intrinsics()
    depth_intrin  = depth_profile.get_intrinsics()
    c2d_extr      = color_profile.get_extrinsics_to(depth_profile)
    depth_scale   = profile.get_device().first_depth_sensor().get_depth_scale()
    dh, dw        = depth_intrin.height, depth_intrin.width

    fx = color_intrin.fx   # used for visual depth estimate

    print(f"[INFO] Color  fx={color_intrin.fx:.1f} fy={color_intrin.fy:.1f} "
          f"ppx={color_intrin.ppx:.1f} ppy={color_intrin.ppy:.1f}")
    t_cd = c2d_extr.translation
    print(f"[INFO] Color→Depth  tx={t_cd[0]*1000:.1f}mm "
          f"ty={t_cd[1]*1000:.1f}mm tz={t_cd[2]*1000:.1f}mm")
    print(f"[INFO] Max visual range ≈ {fx * BALL_RADIUS / MIN_RADIUS_PX:.1f} m "
          f"(fx={fx:.0f}, R={BALL_RADIUS}m, min_r={MIN_RADIUS_PX}px)")

    # ── Shared state ──────────────────────────────────────────────────────────
    buf_lock    = threading.Lock()
    buf_frames  = None
    buf_updated = threading.Event()
    stop_flag   = threading.Event()

    disp_lock  = threading.Lock()
    disp_frame = [None]

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
                x1, y1, x2, y2 = best_box.xyxy[0]
                cx = int((x1 + x2) / 2 * sx)
                cy = int((y1 + y2) / 2 * sy)
                r  = max((x2 - x1) * sx, (y2 - y1) * sy) / 2
                return cx, cy, r, None

        # HSV detect function
        _hsv_fn = None
        if args.detector in ("hsv", "both"):
            _back_sub = (None if args.no_motion else
                         cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50,
                                                            detectShadows=False))
            def _hsv_fn(frame):
                return detect_tennis_ball(frame, hsv_low, hsv_high, _back_sub)

        # Active detector list: order = [YOLO, HSV] for "both"
        _detectors = []   # list of (label, detect_fn)
        if args.detector in ("yolo", "both"):
            _detectors.append(("YOLO", _yolo_fn))
        if args.detector in ("hsv", "both"):
            _detectors.append(("HSV",  _hsv_fn))

        # Per-detector state: last position, miss counter, EKF, FPS
        _st = {
            label: {"cx": None, "cy": None, "r": None, "miss": 0,
                    "ekf": PhysicsEKF(coeff_drag=args.coeff_drag),
                    "fps": _FPS()}
            for label, _ in _detectors
        }

        video_writer = [None]

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

            # Visual depth: depth = fx · R / r_px
            depth_vis = (fx * BALL_RADIUS / st["r"]) if st["r"] and st["r"] > 0 else 0.0

            # Sensor depth: 3-step Color→Depth mapping
            ndcx = (st["cx"] - color_intrin.ppx) / color_intrin.fx
            ndcy = (st["cy"] - color_intrin.ppy) / color_intrin.fy
            dx0  = int(np.clip(ndcx * depth_intrin.fx + depth_intrin.ppx + 0.5, 0, dw-1))
            dy0  = int(np.clip(ndcy * depth_intrin.fy + depth_intrin.ppy + 0.5, 0, dh-1))
            raw0 = depth_arr[dy0, dx0]
            d_coarse = raw0 * depth_scale if raw0 > 0 else 1.0
            tx = c2d_extr.translation[0]; ty = c2d_extr.translation[1]
            dx = int(np.clip(ndcx * depth_intrin.fx + depth_intrin.ppx
                             + tx / d_coarse * depth_intrin.fx + 0.5, 0, dw-1))
            dy = int(np.clip(ndcy * depth_intrin.fy + depth_intrin.ppy
                             + ty / d_coarse * depth_intrin.fy + 0.5, 0, dh-1))
            r  = DEPTH_SAMPLE_R
            patch   = (depth_arr[max(0, dy-r):min(dh, dy+r+1),
                                  max(0, dx-r):min(dw, dx+r+1)]
                       .astype(np.float32) * depth_scale)
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
                p_opt  = rs.rs2_deproject_pixel_to_point(
                    color_intrin, [st["cx"], st["cy"]], depth_fused)
                p_body = optical_to_body(p_opt)
                # Convert to world frame (Z=0=ground) if camera geometry is known
                p_ekf  = body_to_world(p_body, cam_height, cam_pitch_rad) if use_world else p_body
                st["ekf"].update(p_ekf, _meas_covariance(depth_fused), t_now)

            pos_ekf = st["ekf"].x[:3].copy() if st["ekf"].initialized else None
            return detected, coasting, pos_ekf, depth_fused

        # ── Annotation closure (per-panel) ────────────────────────────────────
        def _annotate(frame, st, mask, label, detected, coasting, pos_ekf, depth_fused):
            vis  = frame.copy()
            cx, cy, r, miss = st["cx"], st["cy"], st["r"], st["miss"]

            # Header strip
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (30, 30, 30), -1)
            status = "BALL" if detected else ("COAST" if coasting else "—")
            cv2.putText(vis, f"{label}  [{status}]  {st['fps'].fps:.0f} fps",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if cx is not None and miss <= COAST_FRAMES:
                col = (0, 255, 0) if detected else (0, 165, 255)
                cv2.circle(vis, (cx, cy), max(int(r), 3), col, 2)
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
                tag = "ball" if detected else f"coast {miss}/{COAST_FRAMES}"
                cv2.putText(vis, f"{tag} r={r:.0f}px",
                            (cx - int(r), max(cy - int(r) - 6, 46)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                if pos_ekf is not None:
                    # Z label: world frame shows height above ground, body frame shows raw Z
                    z_suffix = " agl" if use_world else ""
                    cv2.putText(vis,
                                f"({pos_ekf[0]:+.2f},{pos_ekf[1]:+.2f},{pos_ekf[2]:+.2f})m{z_suffix}",
                                (cx - int(r), max(cy - int(r) - 20, 60)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
                cv2.putText(vis, f"d={depth_fused:.2f}m",
                            (8, vis.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                cv2.putText(vis, "No ball",
                            (20, vis.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            # Trajectory overlay
            if not args.no_traj and st["ekf"].initialized:
                traj_pts = st["ekf"].rollout(cx=args.rest_x, cy=args.rest_y, cz=args.rest_z)
                prev_px  = None
                last_z   = traj_pts[0][1][2] if traj_pts else 1.0
                for _, pt in traj_pts:
                    # EKF rollout is in world frame; project back to image via body frame
                    pt_body = world_to_body(pt, cam_height, cam_pitch_rad) if use_world else pt
                    px = _body_to_pixel(pt_body, color_intrin)
                    if px is None:
                        prev_px = None; continue
                    is_bounce = (last_z > 0.01 and pt[2] <= 0.01)
                    cv2.circle(vis, px, 3, (0, 255, 255) if is_bounce else (0, 60, 255), -1)
                    if prev_px is not None:
                        cv2.line(vis, prev_px, px, (0, 60, 255), 1)
                    prev_px = px; last_z = pt[2]

            # HSV mask side-by-side (only in single HSV mode with --show-mask)
            if args.show_mask and mask is not None and args.detector == "hsv":
                vis = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

            return vis

        # ── Main loop ─────────────────────────────────────────────────────────
        while not stop_flag.is_set():
            if not buf_updated.wait(timeout=1.0):
                continue
            buf_updated.clear()

            with buf_lock:
                frames = buf_frames

            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue

            color     = np.asanyarray(cf.get_data()).copy()
            depth_arr = np.asanyarray(df.get_data()).copy()
            t_now     = time.perf_counter()

            panels      = []
            term_parts  = []

            for label, det_fn in _detectors:
                st = _st[label]
                cx, cy, r_px, mask = det_fn(color)
                detected, coasting, pos_ekf, depth_fused = _process(cx, cy, r_px, st, t_now)
                st["fps"].tick()

                # terminal line segment
                z_tag = "agl" if use_world else "body"
                p_str  = "—" if pos_ekf is None else (
                    f"({pos_ekf[0]:+.2f},{pos_ekf[1]:+.2f},{pos_ekf[2]:+.2f}){z_tag}")
                status = "BALL " if detected else ("COAST" if coasting else "     ")
                term_parts.append(
                    f"[{label}:{status}] {p_str} {depth_fused:.2f}m {st['fps'].fps:.0f}fps")

                if viz or args.show_mask or rec_path is not None:
                    panels.append(
                        _annotate(color, st, mask, label,
                                  detected, coasting, pos_ekf, depth_fused))

            print(f"\r{'  ||  '.join(term_parts)}", end="", flush=True)

            if (viz or args.show_mask or rec_path is not None) and panels:
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
                    if video_writer[0] is None:
                        fh, fw = out.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer[0] = cv2.VideoWriter(rec_path, fourcc, 30.0, (fw, fh))
                    video_writer[0].write(out)

                if viz or args.show_mask:
                    with disp_lock:
                        disp_frame[0] = out

        if video_writer[0] is not None:
            video_writer[0].release()
            print(f"\n[INFO] Video saved: {rec_path}")

    det_thread = threading.Thread(target=detection_worker, daemon=True)
    det_thread.start()

    # ── Main thread: camera grab + display ───────────────────────────────────
    print(f"[INFO] Running. Press {'q' if viz else 'Ctrl+C'} to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            with buf_lock:
                buf_frames = frames
            buf_updated.set()

            if viz or args.show_mask:
                with disp_lock:
                    frame = disp_frame[0]
                if frame is not None:
                    cv2.imshow("ball_detection_d455", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_flag.set()
        det_thread.join(timeout=2)
        pipeline.stop()
        if viz or args.show_mask:
            cv2.destroyAllWindows()
        print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
