"""
replay.py — offline replay of raw recordings through the detection pipeline.

Reads camera intrinsics from a sidecar *_intrinsics.json saved automatically
when ball_detection_d455.py records with record_raw: true.
Falls back to D455 1280×720 factory defaults when no sidecar is found.

Usage:
    python replay.py recordings/ball_20260511_143022.mp4
    python replay.py recordings/ball_20260511_143022.mp4 --out recordings/annotated.mp4
    python replay.py recordings/ball_20260511_143022.mp4 --save-traj logs/
    python replay.py recordings/ball_20260511_143022.mp4 --show --speed 0.5
"""

import argparse
import datetime
import json
import os
import sys
import types

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball_detection_d455 import (
    PhysicsEKF, detect_tennis_ball, _body_to_pixel, _meas_covariance, _FPS,
    BALL_RADIUS, COAST_FRAMES,
    optical_to_body, body_to_optical, body_to_world, world_to_body,
)

# ── D455 factory defaults for 1280×720 ───────────────────────────────────────
_D455_DEFAULTS = {
    "fx": 635.7, "fy": 635.7, "ppx": 640.0, "ppy": 360.0,
    "width": 1280, "height": 720,
    "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
    "tag_family": "tag36h11", "tag_ids": [0, 1, 2, 3], "tag_size_m": 0.15,
    "coeff_drag": 0.47, "rest_x": 0.75, "rest_y": 0.75, "rest_z": 0.65,
    "h_low": 25, "h_high": 80, "s_min": 100, "v_min": 100,
    "min_radius_px": 3, "circularity": 0.55,
}

TAG_STALE  = 30
TAG_EMA    = 0.3


def _load_intrinsics(video_path, config_path):
    """Load sidecar *_intrinsics.json, then overlay d455.yaml config values."""
    stem = os.path.splitext(video_path)[0]
    sidecar = stem + "_intrinsics.json"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            data = json.load(f)
        intr = {**_D455_DEFAULTS, **data}
        print(f"[replay] Intrinsics  : {sidecar}")
    else:
        intr = dict(_D455_DEFAULTS)
        print(f"[replay] Intrinsics  : sidecar not found — using D455 1280×720 defaults")

    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            for k in ("coeff_drag", "rest_x", "rest_y", "rest_z",
                      "h_low", "h_high", "s_min", "v_min",
                      "tag_family", "tag_size_m", "min_radius_px", "circularity"):
                if k in cfg:
                    intr[k] = cfg[k]
            if "tag_ids" in cfg:
                intr["tag_ids"] = list(cfg["tag_ids"])
            print(f"[replay] Config      : {config_path}")
        except Exception as e:
            print(f"[replay] Config warning: {e}")

    return intr


def _build_aruco(tag_family):
    """Return (detector_or_None, dict, params) for the detected OpenCV version."""
    # "tag36h11" → DICT_APRILTAG_36h11  (strip leading "tag" prefix)
    fam  = tag_family.lower().replace("-", "_")
    bare = fam[3:] if fam.startswith("tag") else fam   # "36h11"
    candidates = [
        f"DICT_APRILTAG_{bare.upper()}",   # DICT_APRILTAG_36H11
        f"DICT_APRILTAG_{bare}",           # DICT_APRILTAG_36h11
        f"DICT_{fam.upper()}",             # DICT_TAG36H11
        f"DICT_{fam}",                     # DICT_tag36h11
    ]
    fam_id = None
    for name in candidates:
        fam_id = getattr(cv2.aruco, name, None)
        if fam_id is not None:
            break
    if fam_id is None:
        raise ValueError(
            f"Unknown ArUco family '{tag_family}'. "
            f"Tried: {candidates}")
    try:
        d   = cv2.aruco.getPredefinedDictionary(fam_id)
        p   = cv2.aruco.DetectorParameters()
        det = cv2.aruco.ArucoDetector(d, p)
        return det, d, p
    except AttributeError:
        d   = cv2.aruco.Dictionary_get(fam_id)
        p   = cv2.aruco.DetectorParameters_create()
        return None, d, p


def main():
    ap = argparse.ArgumentParser(
        description="Replay raw video through ball-detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("video",       help="raw input video (recordings/ball_*.mp4)")
    ap.add_argument("--out",       default="",
                    help="annotated output video; default: recordings/<stem>_annotated.mp4")
    ap.add_argument("--save-traj", default="",
                    help="save trajectory JSON; directory for auto-name (e.g. logs/)")
    ap.add_argument("--show",      action="store_true",
                    help="show OpenCV preview window while processing")
    ap.add_argument("--speed",     type=float, default=1.0,
                    help="preview playback speed multiplier (default 1.0)")
    ap.add_argument("--config",    default="config/d455.yaml",
                    help="d455.yaml config to override detection params (default: config/d455.yaml)")
    args = ap.parse_args()

    # ── Intrinsics ─────────────────────────────────────────────────────────────
    intr = _load_intrinsics(args.video, args.config)

    intrin = types.SimpleNamespace(
        fx=float(intr["fx"]), fy=float(intr["fy"]),
        ppx=float(intr["ppx"]), ppy=float(intr["ppy"]),
        width=int(intr["width"]), height=int(intr["height"]),
    )
    cam_mat = np.array([
        [intr["fx"], 0,          intr["ppx"]],
        [0,          intr["fy"], intr["ppy"]],
        [0,          0,          1          ],
    ], dtype=np.float32)
    dist_coeffs = np.array(intr["dist"], dtype=np.float32)

    hsv_low  = np.array([intr["h_low"],  intr["s_min"], intr["v_min"]], dtype=np.uint8)
    hsv_high = np.array([intr["h_high"], 255,           255          ], dtype=np.uint8)
    min_r    = float(intr["min_radius_px"])
    min_circ = float(intr["circularity"])

    # ── Open input video ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[replay] ERROR: cannot open {args.video}")
        sys.exit(1)

    vid_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[replay] Input       : {args.video}")
    print(f"[replay]               {vid_w}×{vid_h}  {vid_fps:.1f} fps  ~{n_frames} frames")

    # ── Output paths ───────────────────────────────────────────────────────────
    _ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _stem = os.path.splitext(os.path.basename(args.video))[0]

    out_path = args.out or os.path.join(
        "recordings", f"{_stem}_annotated.mp4")
    _odir = os.path.dirname(out_path)
    if _odir:
        os.makedirs(_odir, exist_ok=True)

    traj_path = args.save_traj
    if traj_path:
        if traj_path.endswith("/") or traj_path.endswith(os.sep) or os.path.isdir(traj_path):
            os.makedirs(traj_path, exist_ok=True)
            traj_path = os.path.join(traj_path, f"traj_{_ts}.json")
        else:
            _d = os.path.dirname(traj_path)
            if _d:
                os.makedirs(_d, exist_ok=True)

    # ── Video writer ───────────────────────────────────────────────────────────
    vw = cv2.VideoWriter(out_path,
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         vid_fps, (vid_w, vid_h))
    if not vw.isOpened():
        print(f"[replay] ERROR: cannot open output {out_path}")
        sys.exit(1)
    print(f"[replay] Output      : {out_path}")
    if traj_path:
        print(f"[replay] Trajectory  : {traj_path}")

    # ── AprilTag setup ─────────────────────────────────────────────────────────
    tag_size_m  = float(intr["tag_size_m"])
    _tag_active = tag_size_m > 0.01
    _aruco_det  = _aruco_dict = _aruco_params = None
    _tag_obj    = None
    _tag_ids    = set()

    if _tag_active:
        _aruco_det, _aruco_dict, _aruco_params = _build_aruco(intr["tag_family"])
        _tag_ids = set(intr["tag_ids"])
        _th = tag_size_m / 2.0
        _tag_obj = np.array([
            [-_th,  _th, 0], [ _th,  _th, 0],
            [ _th, -_th, 0], [-_th, -_th, 0],
        ], dtype=np.float32)

    # ── Detection / EKF state ──────────────────────────────────────────────────
    ekf  = PhysicsEKF(coeff_drag=float(intr["coeff_drag"]))
    rest = (float(intr["rest_x"]), float(intr["rest_y"]), float(intr["rest_z"]))

    _pose = {
        "height": 0.0, "pitch_rad": 0.0,
        "R_cw": None, "tvec_flat": None,
        "rvec": None, "tvec": None, "corners": None,
        "age": 9999,
    }
    _prev_world = False   # track world-frame transitions to reset EKF

    _det = {"cx": None, "cy": None, "r": None, "miss": COAST_FRAMES + 1}
    _hist      = []   # recent world-frame positions for trail
    _traj_log  = []

    fps_ctr   = _FPS()
    frame_idx = 0
    delay_ms  = max(1, int(1000 / (vid_fps * max(args.speed, 0.01)))) if args.show else 1

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _world_active():
        return _pose["R_cw"] is not None or _pose["height"] > 0.01

    def _to_world(p_body):
        if _pose["R_cw"] is not None:
            return _pose["R_cw"].T @ (body_to_optical(p_body) - _pose["tvec_flat"])
        return body_to_world(p_body, _pose["height"], _pose["pitch_rad"])

    def _to_body(p_world):
        if _pose["R_cw"] is not None:
            return optical_to_body(_pose["R_cw"] @ np.asarray(p_world) + _pose["tvec_flat"])
        return world_to_body(p_world, _pose["height"], _pose["pitch_rad"])

    print("[replay] Processing… (Ctrl+C or Q to stop)")

    # ── Main loop ──────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            t_now = frame_idx / vid_fps

            vis = frame.copy()

            # ── AprilTag ───────────────────────────────────────────────────────
            if _tag_active:
                if _aruco_det is not None:
                    corners_list, ids, _ = _aruco_det.detectMarkers(frame)
                else:
                    corners_list, ids, _ = cv2.aruco.detectMarkers(
                        frame, _aruco_dict, parameters=_aruco_params)

                if ids is not None:
                    best_area, best_i = 0.0, -1
                    for i, tid in enumerate(ids.ravel()):
                        if tid in _tag_ids:
                            area = cv2.contourArea(corners_list[i].reshape(4, 2))
                            if area > best_area:
                                best_area, best_i = area, i
                    if best_i >= 0:
                        img_pts = corners_list[best_i].reshape(4, 2).astype(np.float32)
                        ok, rvec, tvec = cv2.solvePnP(
                            _tag_obj, img_pts, cam_mat, dist_coeffs,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
                        if ok:
                            R, _ = cv2.Rodrigues(rvec)
                            t_cam = -R.T @ tvec.ravel()
                            look  = R.T @ np.array([0.0, 0.0, 1.0])
                            new_h = float(t_cam[2])
                            new_p = float(np.arcsin(np.clip(look[2], -1.0, 1.0)))
                            if _pose["age"] > 5:
                                _pose["height"]    = new_h
                                _pose["pitch_rad"] = new_p
                            else:
                                _pose["height"]    = (1-TAG_EMA)*_pose["height"]    + TAG_EMA*new_h
                                _pose["pitch_rad"] = (1-TAG_EMA)*_pose["pitch_rad"] + TAG_EMA*new_p
                            _pose["R_cw"]       = R
                            _pose["tvec_flat"]  = tvec.ravel().copy()
                            _pose["rvec"]       = rvec
                            _pose["tvec"]       = tvec
                            _pose["corners"]    = corners_list[best_i].reshape(4, 1, 2).astype(np.int32)
                            _pose["age"]        = 0

                _pose["age"] = min(_pose["age"] + 1, 9999)

                # Reset EKF when world frame first becomes available
                if _world_active() and not _prev_world:
                    ekf.reset()
                _prev_world = _world_active()

                # Draw tag overlay
                _age = _pose["age"]
                _tc  = ((0, 230, 0) if _age == 0 else
                        (0, 165, 255) if _age < TAG_STALE else (0, 0, 220))
                _tl  = (f"TAG OK   H={_pose['height']:.2f}m" if _age == 0 else
                        f"TAG [{_age}f]  H={_pose['height']:.2f}m" if _age < TAG_STALE else
                        f"NO TAG   H={_pose['height']:.2f}m")
                cv2.putText(vis, _tl, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _tc, 1)
                if _age < 10 and _pose["corners"] is not None:
                    cv2.polylines(vis, [_pose["corners"]], True, _tc, 2)
                    if _pose["rvec"] is not None:
                        cv2.drawFrameAxes(vis, cam_mat, dist_coeffs,
                                          _pose["rvec"], _pose["tvec"], tag_size_m * 0.5)

            # ── Ball detection (HSV, no MOG2) ──────────────────────────────────
            cx, cy, r_px, _ = detect_tennis_ball(
                frame, hsv_low, hsv_high,
                back_sub=None, min_r=min_r, min_circ=min_circ)

            detected  = cx is not None
            if detected:
                _det["miss"] = 0
                _det["cx"] = cx; _det["cy"] = cy; _det["r"] = r_px
            else:
                _det["miss"] += 1
                if _det["miss"] > COAST_FRAMES:
                    ekf.reset()
                    _hist.clear()
            coasting = not detected and _det["miss"] <= COAST_FRAMES

            # ── Visual depth + EKF update ──────────────────────────────────────
            pos_ekf = None
            if (detected or coasting) and _det["r"] and _det["r"] > 0:
                _r = _det["r"]
                depth_vis = intrin.fx * BALL_RADIUS / _r
                z_opt     = depth_vis
                x_opt     = (_det["cx"] - intrin.ppx) / intrin.fx * z_opt
                y_opt     = (_det["cy"] - intrin.ppy) / intrin.fy * z_opt
                p_body    = optical_to_body(np.array([x_opt, y_opt, z_opt]))
                p_meas    = _to_world(p_body) if _world_active() else p_body

                if detected:
                    ekf.update(p_meas, _meas_covariance(depth_vis), t_now)

                if ekf.initialized:
                    pos_ekf = ekf.x[:3].copy()
                    if _world_active():
                        _hist.append(pos_ekf.copy())
                        if len(_hist) > 60:
                            _hist.pop(0)

            # Trajectory log entry
            if traj_path and _world_active() and ekf.initialized:
                _pz_prev = _traj_log[-1]["pos"][2] if _traj_log else None
                _pw      = ekf.x[:3].tolist()
                _vw      = ekf.x[3:6].tolist()
                _bounce  = _pz_prev is not None and _pz_prev > 0.06 and _pw[2] <= 0.06
                _traj_log.append({
                    "t":      round(t_now, 4),
                    "pos":    [round(v, 5) for v in _pw],
                    "vel":    [round(v, 5) for v in _vw],
                    "bounce": _bounce,
                })

            # ── Annotation ────────────────────────────────────────────────────
            fps_ctr.tick()
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (30, 30, 30), -1)
            stat = "BALL" if detected else ("COAST" if coasting else "—")
            cv2.putText(vis,
                        f"REPLAY  [{stat}]  {fps_ctr.fps:.0f} fps  "
                        f"frame {frame_idx}/{n_frames}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if _det["cx"] is not None and _det["miss"] <= COAST_FRAMES:
                _cc  = (_det["cx"], _det["cy"])
                _cr  = max(int(_det["r"]) if _det["r"] else 5, 3)
                _col = (0, 255, 0) if detected else (0, 165, 255)
                cv2.circle(vis, _cc, _cr, _col, 2)
                cv2.circle(vis, _cc, 3, (0, 0, 255), -1)
                if pos_ekf is not None:
                    suf = " agl" if _world_active() else ""
                    cv2.putText(vis,
                                f"({pos_ekf[0]:+.2f},{pos_ekf[1]:+.2f},{pos_ekf[2]:+.2f})m{suf}",
                                (_cc[0]-_cr, max(_cc[1]-_cr-20, 68)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, _col, 1)

            # Historical trail
            if _hist and _world_active():
                n_h    = len(_hist)
                prev_p = None
                for i, hp in enumerate(_hist):
                    alpha = i / max(n_h - 1, 1)
                    px    = _body_to_pixel(_to_body(hp), intrin)
                    if px is None:
                        prev_p = None; continue
                    c = (0, int(30 + 135*alpha), int(80 + 175*alpha))
                    cv2.circle(vis, px, max(2, int(2 + 2*alpha)), c, -1)
                    if prev_p:
                        cv2.line(vis, prev_p, px, c, 1)
                    prev_p = px

            # Predicted trajectory
            if ekf.initialized and _world_active():
                traj_pts = ekf.rollout(cx=rest[0], cy=rest[1], cz=rest[2])
                prev_p   = None
                for _, pt in traj_pts:
                    px = _body_to_pixel(_to_body(pt), intrin)
                    if px is None:
                        prev_p = None; continue
                    is_bounce = pt[2] <= 0.01
                    cv2.circle(vis, px, 3, (0, 255, 255) if is_bounce else (0, 60, 255), -1)
                    if prev_p:
                        cv2.line(vis, prev_p, px, (0, 60, 255), 1)
                    prev_p = px

            vw.write(vis)

            if args.show:
                cv2.imshow("replay", vis)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord("q")):
                    break

            if frame_idx % 150 == 0:
                pct = frame_idx / max(n_frames, 1) * 100
                print(f"[replay] {frame_idx}/{n_frames}  ({pct:.0f}%)", end="\r", flush=True)

    except KeyboardInterrupt:
        print()

    finally:
        cap.release()
        vw.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"\n[replay] Annotated   → {out_path}  ({frame_idx} frames)")

    if traj_path and _traj_log:
        n_b  = sum(1 for fr in _traj_log if fr["bounce"])
        meta = {
            "coeff_drag": float(intr["coeff_drag"]),
            "detector":   "hsv_replay",
            "n_frames":   len(_traj_log),
            "n_bounces":  n_b,
        }
        with open(traj_path, "w") as f:
            json.dump({"meta": meta, "frames": _traj_log}, f, separators=(",", ":"))
        print(f"[replay] Trajectory  → {traj_path}  ({len(_traj_log)} frames, {n_b} bounces)")


if __name__ == "__main__":
    main()
