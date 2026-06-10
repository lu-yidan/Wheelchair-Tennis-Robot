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

try:
    import zmq as _zmq
    _ZMQ_OK = True
except ImportError:
    _ZMQ_OK = False

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
    ap.add_argument("--require-tag", action="store_true", default=True,
                    help="drop packets where the source has no AprilTag in view "
                         "(prevents mixing world frames; default ON)")
    ap.add_argument("--no-require-tag", dest="require_tag", action="store_false",
                    help="accept packets even without a tag (for debugging only)")
    ap.add_argument("--max-tag-age", type=int, default=30,
                    help="drop packets when source's tag_age > this many frames "
                         "(stale calibration; default 30 = ~1s @30fps)")
    ap.add_argument("--print-hz",   type=float, default=20.0,
                    help="terminal status refresh rate (default 20)")
    # Optional forwarding to existing viewers (same ports as ball_detection)
    ap.add_argument("--viz3d-port", type=int, default=0,
                    help="if >0, forward fused state to viz3d.py on this port")
    ap.add_argument("--webview-port", type=int, default=0,
                    help="if >0, forward fused state to webview.py on this port")
    # Trajectory publisher for external subscribers (ZMQ PUB)
    ap.add_argument("--traj-pub-port", type=int, default=0,
                    help="if >0, publish trajectory on this ZMQ PUB port at 30 Hz "
                         "(e.g. --traj-pub-port 5580; subscribers: tcp://<host>:5580)")
    ap.add_argument("--traj-pub-hz",  type=float, default=30.0,
                    help="trajectory publish rate in Hz (default 30)")
    ap.add_argument("--traj-pub-sec", type=float, default=2.0,
                    help="seconds ahead to predict in published trajectory (default 2.0)")
    args = ap.parse_args()

    ekf = PhysicsEKF(coeff_drag=args.coeff_drag)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.010)  # 10 ms — allows 30 Hz timers to fire even when idle
    print(f"[fusion] listening on {args.host}:{args.port}  "
          f"drag={args.coeff_drag} rest=({args.rest_x},{args.rest_y},{args.rest_z})")

    fwd_viz3d = fwd_web = None
    if args.viz3d_port > 0:
        fwd_viz3d = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[fusion] forward viz3d → 127.0.0.1:{args.viz3d_port}")
    if args.webview_port > 0:
        fwd_web = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[fusion] forward webview → 127.0.0.1:{args.webview_port}")

    # ZMQ trajectory publisher
    traj_pub = None
    if args.traj_pub_port > 0:
        if not _ZMQ_OK:
            print("[fusion] WARNING: --traj-pub-port set but pyzmq not installed; "
                  "run: pip install pyzmq", file=sys.stderr)
        else:
            _zmq_ctx = _zmq.Context.instance()
            traj_pub = _zmq_ctx.socket(_zmq.PUB)
            traj_pub.setsockopt(_zmq.SNDHWM, 2)   # drop old frames, never block
            traj_pub.bind(f"tcp://*:{args.traj_pub_port}")
            print(f"[fusion] traj ZMQ PUB → tcp://*:{args.traj_pub_port}  "
                  f"{args.traj_pub_hz:.0f} Hz  {args.traj_pub_sec:.1f}s ahead")

    # Per-source rolling stats
    stats = {}     # cam_id → {"n_total", "t_first", "t_last", "last_pos", "last_depth", "last_tag"}
    last_print_t      = 0.0
    print_period      = 1.0 / max(args.print_hz, 1.0)
    last_traj_t       = 0.0
    last_cam_hb_t     = 0.0   # cam-position heartbeat when EKF not yet initialized
    last_pub_t        = 0.0   # ZMQ traj publisher timer
    pub_period        = 1.0 / max(args.traj_pub_hz, 1.0)
    last_accepted_t   = 0.0   # wall time of last accepted EKF measurement
    cached_traj       = []   # cached EKF rollout for forwarding

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
                accepted = _consume(pkt, now, ekf, stats, args.max_age_ms,
                                    args.require_tag, args.max_tag_age)
                if accepted:
                    last_accepted_t = now

            # Throttled terminal status + viewer forwarding
            if now - last_print_t >= print_period:
                _print_status(stats, ekf, now)
                last_print_t = now

            # Rollout cache (10 Hz) for forwarders
            if (fwd_viz3d or fwd_web) and ekf.initialized and now - last_traj_t >= 0.1:
                cached_traj = ekf.rollout(cx=args.rest_x, cy=args.rest_y, cz=args.rest_z)
                last_traj_t = now
                _forward(cached_traj, ekf, stats, args, fwd_viz3d, fwd_web)

            # ── ZMQ trajectory publisher (30 Hz) ──────────────────────────────
            if traj_pub is not None and ekf.initialized and now - last_pub_t >= pub_period:
                last_pub_t = now
                _publish_traj(traj_pub, ekf, now, last_accepted_t,
                              args.traj_pub_sec, args.rest_x, args.rest_y, args.rest_z)

            # Cam-position heartbeat (5 Hz) when EKF not yet initialized
            # Lets camera triangles appear in monitor.html before any ball is seen.
            if fwd_web and not ekf.initialized and now - last_cam_hb_t >= 0.2:
                last_cam_hb_t = now
                _cams = {cid: {"pos": s["last_cam_pos"], "look": s["last_cam_look"]}
                         for cid, s in stats.items() if s.get("last_cam_pos") is not None}
                if _cams:
                    try:
                        fwd_web.sendto(
                            b"\x00" + json.dumps({"t": now, "detectors": [], "cams": _cams}).encode(),
                            ("127.0.0.1", args.webview_port))
                    except Exception:
                        pass

    except KeyboardInterrupt:
        print("\n[fusion] stopped.")


# ─────────────────────────────────────────────────────────────────────────────
#  Internals
# ─────────────────────────────────────────────────────────────────────────────

def _consume(pkt, now, ekf, stats, max_age_ms, require_tag, max_tag_age):
    """Returns True if a measurement was accepted into the EKF, False otherwise."""
    try:
        t_pkt   = float(pkt["t"])
        cam_id  = str(pkt["cam_id"])
        tag_id  = int(pkt.get("tag_id", -1))
        tag_age = int(pkt.get("tag_age", 9999))
    except (KeyError, TypeError, ValueError):
        return False

    age_ms = (now - t_pkt) * 1000.0
    if abs(age_ms) > max_age_ms:
        return False

    s = stats.setdefault(cam_id, {
        "n_total": 0, "n_dropped": 0, "t_first": now, "t_last": now,
        "last_pos": None, "last_depth": 0.0, "last_tag": -1, "last_tag_age": 9999,
        "ema_dt": None,
        "last_cam_pos": None, "last_cam_look": None,
    })
    s["last_tag"]     = tag_id
    s["last_tag_age"] = tag_age
    if "cam_pos" in pkt:
        s["last_cam_pos"]  = pkt["cam_pos"]
        s["last_cam_look"] = pkt.get("cam_look")

    # Cam-only heartbeat (no ball measurement) — update pose but skip EKF
    pos_raw = pkt.get("pos")
    if pos_raw is None:
        return False

    try:
        pos = np.asarray(pos_raw, dtype=float)
        cov = max(float(pkt.get("cov", 0.05)), 1e-6)
    except (TypeError, ValueError):
        return False

    # Per-source fps bookkeeping (only for ball packets, not heartbeats)
    if s["last_pos"] is not None:
        dt = now - s["t_last"]
        s["ema_dt"] = dt if s["ema_dt"] is None else (0.1 * dt + 0.9 * s["ema_dt"])
    s["n_total"] += 1
    s["t_last"]   = now
    s["last_pos"]  = pos
    s["last_depth"] = float(pkt.get("depth", 0.0))

    if require_tag and (tag_id < 0 or tag_age > max_tag_age):
        s["n_dropped"] += 1
        return False

    R_meas = np.eye(3) * cov
    ekf.update(pos, R_meas, t_pkt)
    return True


def _publish_traj(pub_sock, ekf, now, last_accepted_t, t_ahead, cx, cy, cz):
    """
    Publish fused trajectory over ZMQ PUB.

    Message format (JSON):
      {
        "stamp":    <float>  Unix time of this publish (seconds)
        "t_obs":    <float>  Unix time of last accepted camera measurement
        "detected": <bool>   True = fresh measurement ≤200 ms ago; False = EKF coasting
        "pos":      [x,y,z]  current EKF position (m), world frame (AprilTag origin, Z-up)
        "vel":      [vx,vy,vz]  current EKF velocity (m/s)
        "traj":     [[x,y,z,t], ...]   predicted waypoints;
                    t = seconds from 'stamp' when ball reaches that point
      }

    Coordinate frame:
      Origin = AprilTag centre (0,0,0).  Z+ = up.  Units: metres, seconds.
    """
    traj_pts = ekf.rollout(t_ahead=t_ahead, cx=cx, cy=cy, cz=cz)
    detected = (now - last_accepted_t) < 0.2   # fresh if observed within 200 ms
    msg = {
        "stamp":    round(now, 4),
        "t_obs":    round(last_accepted_t, 4),
        "detected": detected,
        "pos":  [round(float(v), 4) for v in ekf.x[:3]],
        "vel":  [round(float(v), 4) for v in ekf.x[3:]],
        "traj": [[round(float(p[0]), 4), round(float(p[1]), 4),
                  round(float(p[2]), 4), round(float(t),    4)]
                 for t, p in traj_pts],
    }
    try:
        pub_sock.send(json.dumps(msg).encode(), _zmq.NOBLOCK)
    except Exception:
        pass


def _print_status(stats, ekf, now):
    parts = []
    for cam_id in sorted(stats.keys()):
        s = stats[cam_id]
        age = now - s["t_last"]
        if age > 1.0:
            fps_str = "STALE"
        elif s["ema_dt"] and s["ema_dt"] > 0:
            fps_str = f"{1.0 / s['ema_dt']:.0f}fps"
        else:
            fps_str = "  ?fps"
        depth = s["last_depth"]
        # Show drop ratio when meaningful — quickly surfaces "Razer can't see tag" type bugs
        drop_str = ""
        if s["n_total"] > 0 and s["n_dropped"] > 0:
            drop_pct = 100.0 * s["n_dropped"] / s["n_total"]
            drop_str = f" DROP {drop_pct:.0f}%"
        tag_str = f"tag={s['last_tag']}{'!' if s['last_tag_age']>30 else ''}"
        parts.append(f"[{cam_id} {fps_str} d={depth:.1f}m {tag_str}{drop_str}]")

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

    # Collect per-source camera positions for the map view
    cams = {}
    for cid, s in stats.items():
        if s.get("last_cam_pos") is not None:
            cams[cid] = {
                "pos":  s["last_cam_pos"],
                "look": s["last_cam_look"],
            }

    pkt = {
        "t": time.time(),
        "detectors": ["FUSED"],
        "cam": None, "cam_look": None, "tag_age": 0,
        "rest": [round(args.rest_x, 2), round(args.rest_y, 2), round(args.rest_z, 2)],
        "cams": cams,
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
