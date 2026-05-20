"""
fusion.py — multi-camera EKF fusion for ball_detection.

Listens for UDP packets from N `ball_detection.py` instances (each broadcasting
its per-frame world-frame ball measurement) and runs a single physics EKF that
fuses all sources.  Output: terminal status, optional forward to viz3d/webview
(same wire format as ball_detection so existing viewers Just Work™).

Wire format (json, ≤300 B per packet):
    {
      "t":       1719600000.123,   # wall clock seconds (time.time())
      "cam_id":  "aoni",            # source identifier (string)
      "pos":     [x, y, z],         # world-frame position (m)
      "cov":     0.0123,            # isotropic measurement variance (m²)
      "depth":   5.23,              # depth used to derive pos (m)
      "tag_id":  0,                 # AprilTag ID currently providing calibration
      "tag_age": 0                  # frames since last successful tag detection
    }

Assumptions:
  - All cameras share the SAME world frame.  Phase 1 = all cameras look at the
    SAME AprilTag.  Phase 2 (not implemented here) = per-tag world-pose map.
  - Single physical ball (no multi-ball tracking).
  - All sources on the same machine OR clocks NTP-synced (max-age-ms drops stale).

Usage:
    # terminal 1 — fusion
    python fusion.py --port 5570

    # terminal 2 — detector A
    python ball_detection.py --config config/webcam.yaml \\
        --cam-id aoni --fusion-port 5570

    # terminal 3 — detector B
    python ball_detection.py --config config/razer.yaml  \\
        --cam-id razer --fusion-port 5570 --port-offset 100
"""

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

# Reuse the physics EKF from the detection module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball_detection import (
    PhysicsEKF,
    DEFAULT_COEFF_DRAG,
    DEFAULT_COEFF_REST_X, DEFAULT_COEFF_REST_Y, DEFAULT_COEFF_REST_Z,
    TRAJ_PREDICT_SEC, DT_ROLLOUT,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port",       type=int, default=5570,
                    help="UDP port to listen on (default 5570)")
    ap.add_argument("--host",       default="0.0.0.0",
                    help="bind host (default 0.0.0.0 = accept LAN packets)")
    ap.add_argument("--coeff-drag", type=float, default=DEFAULT_COEFF_DRAG)
    ap.add_argument("--rest-x",     type=float, default=DEFAULT_COEFF_REST_X)
    ap.add_argument("--rest-y",     type=float, default=DEFAULT_COEFF_REST_Y)
    ap.add_argument("--rest-z",     type=float, default=DEFAULT_COEFF_REST_Z)
    ap.add_argument("--max-age-ms", type=int, default=200,
                    help="drop packets older than this (clock skew protection, default 200)")
    ap.add_argument("--print-hz",   type=float, default=20.0,
                    help="terminal status refresh rate (default 20)")
    # Optional forwarding to existing viewers (same ports as ball_detection)
    ap.add_argument("--viz3d-port", type=int, default=0,
                    help="if >0, forward fused state to viz3d.py on this port")
    ap.add_argument("--webview-port", type=int, default=0,
                    help="if >0, forward fused state to webview.py on this port")
    args = ap.parse_args()

    ekf = PhysicsEKF(coeff_drag=args.coeff_drag)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.05)   # 50 ms — wakes up to refresh terminal even when idle
    print(f"[fusion] listening on {args.host}:{args.port}  "
          f"drag={args.coeff_drag} rest=({args.rest_x},{args.rest_y},{args.rest_z})")

    fwd_viz3d = fwd_web = None
    if args.viz3d_port > 0:
        fwd_viz3d = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[fusion] forward viz3d → 127.0.0.1:{args.viz3d_port}")
    if args.webview_port > 0:
        fwd_web = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[fusion] forward webview → 127.0.0.1:{args.webview_port}")

    # Per-source rolling stats
    stats = {}     # cam_id → {"n_total", "t_first", "t_last", "last_pos", "last_depth", "last_tag"}
    last_print_t = 0.0
    print_period = 1.0 / max(args.print_hz, 1.0)
    last_traj_t  = 0.0
    cached_traj  = []   # cached EKF rollout for forwarding

    print("[fusion] waiting for packets…")
    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
                pkt = json.loads(data)
            except socket.timeout:
                pkt = None
            except Exception as e:
                print(f"\n[fusion] bad packet from {addr}: {e}")
                pkt = None

            now = time.time()

            if pkt is not None:
                _consume(pkt, now, ekf, stats, args.max_age_ms)

            # Throttled terminal status + viewer forwarding
            if now - last_print_t >= print_period:
                _print_status(stats, ekf, now)
                last_print_t = now

            # Rollout cache (10 Hz) for forwarders
            if (fwd_viz3d or fwd_web) and ekf.initialized and now - last_traj_t >= 0.1:
                cached_traj = ekf.rollout(cx=args.rest_x, cy=args.rest_y, cz=args.rest_z)
                last_traj_t = now
                _forward(cached_traj, ekf, stats, args, fwd_viz3d, fwd_web)

    except KeyboardInterrupt:
        print("\n[fusion] stopped.")


# ─────────────────────────────────────────────────────────────────────────────
#  Internals
# ─────────────────────────────────────────────────────────────────────────────

def _consume(pkt, now, ekf, stats, max_age_ms):
    try:
        t_pkt  = float(pkt["t"])
        cam_id = str(pkt["cam_id"])
        pos    = np.asarray(pkt["pos"], dtype=float)
        cov    = max(float(pkt.get("cov", 0.05)), 1e-6)
    except (KeyError, TypeError, ValueError):
        return

    age_ms = (now - t_pkt) * 1000.0
    if abs(age_ms) > max_age_ms:
        # Clock skew or buffer pile-up; ignore quietly to avoid log flood
        return

    # Per-source bookkeeping
    s = stats.setdefault(cam_id, {
        "n_total": 0, "t_first": now, "t_last": now,
        "last_pos": None, "last_depth": 0.0, "last_tag": -1, "last_tag_age": 9999,
        "ema_dt": None,
    })
    if s["last_pos"] is not None:
        # Exponential moving average of inter-arrival period (proxy for fps)
        dt = now - s["t_last"]
        s["ema_dt"] = dt if s["ema_dt"] is None else (0.1 * dt + 0.9 * s["ema_dt"])
    s["n_total"] += 1
    s["t_last"]  = now
    s["last_pos"] = pos
    s["last_depth"] = float(pkt.get("depth", 0.0))
    s["last_tag"] = int(pkt.get("tag_id", -1))
    s["last_tag_age"] = int(pkt.get("tag_age", 9999))

    # EKF update — isotropic 3×3 measurement noise from the cov scalar.
    # (Phase 2 idea: receive a full 3×3 in world frame for proper anisotropic fusion.)
    R_meas = np.eye(3) * cov
    ekf.update(pos, R_meas, t_pkt)


def _print_status(stats, ekf, now):
    parts = []
    for cam_id in sorted(stats.keys()):
        s = stats[cam_id]
        age = now - s["t_last"]
        if age > 1.0:
            tag = "STALE"
        elif s["ema_dt"] and s["ema_dt"] > 0:
            tag = f"{1.0 / s['ema_dt']:.0f}fps"
        else:
            tag = "  ?fps"
        depth = s["last_depth"]
        parts.append(f"[{cam_id} {tag} d={depth:.1f}m tag={s['last_tag']}{'!' if s['last_tag_age']>30 else ''}]")

    if ekf.initialized:
        x, y, z   = ekf.x[:3]
        vx, vy, vz = ekf.x[3:]
        spd = float(np.linalg.norm([vx, vy, vz]))
        head = f"FUSED ({x:+.2f},{y:+.2f},{z:+.2f}) v={spd:5.2f}m/s"
    else:
        head = "FUSED (waiting for first detection)            "

    line = f"\r{head}  {' '.join(parts) if parts else '(no sources yet)'}"
    # Pad to clear leftover chars from longer previous lines
    sys.stdout.write(line + " " * 10)
    sys.stdout.flush()


def _forward(traj_pts, ekf, stats, args, sock_viz3d, sock_web):
    """Emit a viz3d/webview-compatible state packet from the fused EKF."""
    if not ekf.initialized:
        return
    px, py, pz = ekf.x[:3]
    vx, vy, vz = ekf.x[3:]
    # Trajectory + bounce detection (same logic as ball_detection.py UDP send)
    traj = [[round(float(v), 4) for v in p] for _, p in traj_pts[::2]]
    bounces = []
    last_z = traj_pts[0][1][2] if traj_pts else 1.0
    for _, p in traj_pts:
        if last_z > 0.01 and p[2] <= 0.01:
            bounces.append([round(float(v), 4) for v in p])
        last_z = p[2]

    pkt = {
        "t": time.time(),
        "detectors": ["FUSED"],
        "cam": None, "cam_look": None, "tag_age": 0,
        "rest": [round(args.rest_x, 2), round(args.rest_y, 2), round(args.rest_z, 2)],
        "FUSED": {
            "ball":      [round(float(v), 4) for v in ekf.x[:3]],
            "vel":       [round(float(v), 4) for v in ekf.x[3:]],
            "traj":      traj,
            "bounces":   bounces,
            "ball_hist": [[round(float(v), 4) for v in ekf.x[:3]]],
        },
    }
    data = json.dumps(pkt).encode()
    if sock_viz3d is not None:
        try: sock_viz3d.sendto(data, ("127.0.0.1", args.viz3d_port))
        except Exception: pass
    if sock_web is not None:
        try: sock_web.sendto(b"\x00" + data, ("127.0.0.1", args.webview_port))
        except Exception: pass


if __name__ == "__main__":
    main()
