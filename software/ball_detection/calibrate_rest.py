"""
calibrate_rest.py — fit restitution coefficients from recorded ball bounces.

Workflow:
  1. Record a trajectory with ball_detection_d455.py:
       python ball_detection_d455.py --save-traj traj.json
     Bounce the ball ≥5 times clearly in camera view, then Ctrl+C.

  2. Inspect the overview plot to identify clean segments:
       python calibrate_rest.py traj.json --plot

  3. Restrict to clean segments and fit:
       python calibrate_rest.py traj.json --segments "2.5-8.0,14.0-21.5" --plot

  Output: optimal rest_x/y/z + RMSE, ready to paste into config/d455.yaml.

Physics model (ball_localization/src/ball_ekf.cpp):
  drag_k = C_d · ½ρ · πR² / m
  ax = −sign(vx) · drag_k · v² · |vx|/|v|
  ay = −sign(vy) · drag_k · v² · |vy|/|v|
  az =  g  − sign(vz) · drag_k · v² · |vz|/|v|

Bounce model:
  vx_post =  vx_pre · rest_x
  vy_post =  vy_pre · rest_y
  vz_post = |vz_pre| · rest_z   (direction reversed)
"""

import argparse
import json
import sys

import numpy as np
from scipy.optimize import differential_evolution

# ── Physics constants (must match ball_detection_d455.py) ─────────────────────
GRAVITY     = -9.79528   # m/s²
AIR_DENSITY =  1.225     # kg/m³
BALL_MASS   =  0.0575    # kg
BALL_RADIUS =  0.0335    # m

_DT_SUB = 0.004          # integration sub-step (4 ms)


def _drag_k(coeff_drag: float) -> float:
    return coeff_drag * 0.5 * AIR_DENSITY * np.pi * BALL_RADIUS**2 / BALL_MASS


def _accel(state: np.ndarray, dk: float) -> np.ndarray:
    vx, vy, vz = state[3], state[4], state[5]
    v_sq = vx*vx + vy*vy + vz*vz
    if 1e-6 < v_sq < 40**2:
        v   = np.sqrt(v_sq)
        a_d = dk * v_sq
        return np.array([
            -np.sign(vx) * a_d * abs(vx) / v,
            -np.sign(vy) * a_d * abs(vy) / v,
            GRAVITY - np.sign(vz) * a_d * abs(vz) / v,
        ])
    return np.array([0.0, 0.0, GRAVITY])


def _simulate(state0: np.ndarray, times_rel: list, dk: float) -> list:
    """
    Euler-integrate physics from state0 and return positions at each time
    in times_rel (seconds relative to t=0, must be non-decreasing).
    """
    s = state0.copy()
    results = []
    t_cur = 0.0
    for t_next in times_rel:
        dt = t_next - t_cur
        if dt > 0:
            n_sub = max(1, int(dt / _DT_SUB))
            dt_s  = dt / n_sub
            for _ in range(n_sub):
                s[3:] += _accel(s, dk) * dt_s
                s[:3] += s[3:] * dt_s
        t_cur = t_next
        results.append(s[:3].copy())
    return results


# ── Data loading and segment filtering ────────────────────────────────────────

def _parse_segments(s: str) -> list:
    """Parse "1.5-8.2,12.0-19.5" → [(1.5, 8.2), (12.0, 19.5)]."""
    segs = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        lo, hi = part.split("-", 1)
        segs.append((float(lo), float(hi)))
    return segs


def _load(path: str, segments=None):
    with open(path) as f:
        data = json.load(f)

    meta   = data.get("meta", {})
    frames = data["frames"]

    if segments:
        frames = [fr for fr in frames
                  if any(t0 <= fr["t"] <= t1 for t0, t1 in segments)]
        frames.sort(key=lambda fr: fr["t"])

    return meta, frames


# ── Bounce event extraction ────────────────────────────────────────────────────

def _extract_bounces(frames: list, min_post: int = 8) -> list:
    """
    Scan for bounce events (Z descends through ~0) and return a list of dicts:
      pre_pos, pre_vel   — EKF state 2 frames before the bounce
      t_bounce           — wall-clock time of the bounce frame
      times_post         — list of elapsed seconds after bounce for each post frame
      pos_post           — list of np.array([x,y,z]) observed positions
    """
    events = []
    n = len(frames)

    for i in range(2, n - min_post):
        pz_prev = frames[i - 1]["pos"][2]
        pz_curr = frames[i]["pos"][2]

        # Bounce: Z was clearly above ground, now at/below ground
        is_bounce = (frames[i].get("bounce", False)
                     or (pz_prev > 0.06 and pz_curr <= 0.06))
        if not is_bounce:
            continue

        # Pre-bounce: use the frame 2 steps before the bounce for a cleaner
        # velocity estimate (avoids EKF transient right at the bounce point)
        pre = frames[i - 2]

        # Collect post-bounce frames, stopping at:
        #   • MAX_POST_FRAMES frames
        #   • a large time gap (camera dropout)
        #   • the second bounce (Z drops back near 0 after rising)
        MAX_POST = 35
        post_frames = []
        rose = False
        t_bounce = frames[i]["t"]

        for j in range(i, min(i + MAX_POST + 1, n)):
            fr = frames[j]
            pz = fr["pos"][2]

            # Check for camera dropout (gap > 120 ms ≈ 4 missed frames at 30 fps)
            if post_frames:
                if fr["t"] - post_frames[-1]["t"] > 0.12:
                    break

            if pz > 0.10:
                rose = True
            if rose and pz < 0.03:   # second bounce — stop here
                break

            post_frames.append(fr)

        if len(post_frames) < min_post:
            continue

        # Require the ball to actually rise after the bounce
        if not any(fr["pos"][2] > 0.08 for fr in post_frames[:20]):
            continue

        events.append({
            "pre_pos":   np.array(pre["pos"],  dtype=float),
            "pre_vel":   np.array(pre["vel"],  dtype=float),
            "t_bounce":  t_bounce,
            "times_post": [fr["t"] - t_bounce for fr in post_frames],
            "pos_post":   [np.array(fr["pos"], dtype=float) for fr in post_frames],
        })

    return events


# ── Optimisation ──────────────────────────────────────────────────────────────

def _objective(rest: np.ndarray, events: list, dk: float) -> float:
    rx, ry, rz = rest
    total_sq = 0.0
    n_pts    = 0

    for ev in events:
        # Initial state at the bounce: position ≈ (x_pre, y_pre, 0),
        # velocity with restitution applied
        state0 = np.array([
            ev["pre_pos"][0],
            ev["pre_pos"][1],
            0.0,
            ev["pre_vel"][0] * rx,
            ev["pre_vel"][1] * ry,
            abs(ev["pre_vel"][2]) * rz,
        ])

        preds = _simulate(state0, ev["times_post"], dk)

        for pred, actual in zip(preds, ev["pos_post"]):
            diff    = pred - actual
            total_sq += float(diff @ diff)
            n_pts   += 1

    return total_sq / max(n_pts, 1)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_overview(frames: list, events: list):
    import matplotlib.pyplot as plt

    ts = [fr["t"] for fr in frames]
    t0 = ts[0]
    ts = [t - t0 for t in ts]

    zs = [fr["pos"][2] for fr in frames]
    vs = [float(np.linalg.norm(fr["vel"])) for fr in frames]
    bounce_ts = [ev["t_bounce"] - t0 for ev in events]

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    fig.suptitle("Trajectory overview  —  use --segments to exclude bad regions",
                 fontsize=11)

    ax0 = axes[0]
    ax0.plot(ts, zs, lw=0.9, color="steelblue", label="Z (m)")
    ax0.axhline(0, color="#888", lw=0.5, ls="--")
    for bt in bounce_ts:
        ax0.axvline(bt, color="orange", alpha=0.7, lw=0.9, label="_")
    ax0.set_ylabel("Height Z (m)")
    ax0.legend(["Z", "bounce detected"], loc="upper right", fontsize=8)

    ax1 = axes[1]
    ax1.plot(ts, vs, lw=0.9, color="firebrick", label="|vel| (m/s)")
    ax1.set_ylabel("|vel| (m/s)")
    ax1.set_xlabel("time (s)")

    # Mark first/last frame of any user-selected segments (can't know here,
    # but axes x-ticks give visual reference)
    fig.tight_layout()
    print(f"[calibrate] Overview plot: t=0 corresponds to wall time {frames[0]['t']:.2f}")
    print(f"[calibrate] Total duration: {ts[-1]:.1f} s")


def _plot_bounces(events: list, rest: np.ndarray, dk: float):
    import matplotlib.pyplot as plt

    n    = len(events)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    fig.suptitle(
        f"Per-bounce Z fit  rest=({rest[0]:.3f}, {rest[1]:.3f}, {rest[2]:.3f})",
        fontsize=11)
    axes = np.array(axes).ravel()

    for i, ev in enumerate(events):
        state0 = np.array([
            ev["pre_pos"][0], ev["pre_pos"][1], 0.0,
            ev["pre_vel"][0] * rest[0],
            ev["pre_vel"][1] * rest[1],
            abs(ev["pre_vel"][2]) * rest[2],
        ])
        preds  = np.array(_simulate(state0, ev["times_post"], dk))
        actual = np.array(ev["pos_post"])
        t_post = ev["times_post"]

        ax = axes[i]
        ax.plot(t_post, actual[:, 2], "o-", ms=3, lw=0.9,
                color="steelblue", label="measured Z")
        ax.plot(t_post, preds[:, 2],  "--", lw=1.2,
                color="orange",    label="predicted Z")
        ax.set_title(f"Bounce {i+1}  (t={ev['t_bounce']:.1f}s)", fontsize=9)
        ax.set_xlabel("t after bounce (s)", fontsize=8)
        ax.set_ylabel("Z (m)", fontsize=8)
        ax.axhline(0, color="#bbb", lw=0.5)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit restitution coefficients (rest_x/y/z) from recorded ball bounces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("traj_json",
                        help="JSON file from  ball_detection_d455.py --save-traj")
    parser.add_argument("--segments", default="",
                        metavar="T0-T1[,T2-T3,...]",
                        help='wall-clock time ranges to include, e.g. "2.5-8.0,14.0-21.5". '
                             'Run with --plot first to identify good segments.')
    parser.add_argument("--plot", action="store_true",
                        help="show Z(t) overview and per-bounce fit plots")
    parser.add_argument("--min-bounces", type=int, default=3,
                        help="minimum valid bounce events required (default 3)")
    args = parser.parse_args()

    segments = _parse_segments(args.segments) if args.segments else None
    meta, frames = _load(args.traj_json, segments)

    if not frames:
        print("[calibrate] ERROR: no frames loaded — check --segments range vs actual timestamps")
        sys.exit(1)

    coeff_drag = float(meta.get("coeff_drag", 0.47))
    dk         = _drag_k(coeff_drag)
    t0_wall    = frames[0]["t"]

    print(f"[calibrate] File      : {args.traj_json}")
    print(f"[calibrate] Frames    : {len(frames)}  "
          f"t=[{t0_wall:.1f}, {frames[-1]['t']:.1f}]s  "
          f"(duration {frames[-1]['t'] - t0_wall:.1f} s)")
    print(f"[calibrate] Drag coeff: {coeff_drag}")
    if segments:
        print(f"[calibrate] Segments  : {args.segments}")

    events = _extract_bounces(frames)
    print(f"[calibrate] Bounces   : {len(events)} valid events detected")

    if args.plot:
        try:
            _plot_overview(frames, events)
        except ImportError:
            print("[calibrate] matplotlib not available — skipping overview plot")

    if len(events) < args.min_bounces:
        print(f"\n[calibrate] Need ≥{args.min_bounces} valid bounces.")
        print("  Tips:")
        print("  • Use --segments to cut out regions where you were holding/rolling the ball")
        print("  • Bounce the ball clearly in the camera field of view")
        print("  • Ensure AprilTag is visible so world frame is active")
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except ImportError:
                pass
        sys.exit(1)

    print(f"\n[calibrate] Optimising over {len(events)} bounce events …  "
          f"(may take 15–40 s)")

    bounds = [(0.05, 1.50)] * 3
    result = differential_evolution(
        _objective, bounds,
        args=(events, dk),
        seed=42, tol=1e-5, maxiter=1000,
        popsize=15, mutation=(0.5, 1.2), recombination=0.8,
        disp=False,
    )

    rx, ry, rz = result.x
    rmse_cm = np.sqrt(result.fun) * 100.0   # convert m → cm

    print()
    print("─" * 46)
    print(f"  rest_x : {rx:.4f}")
    print(f"  rest_y : {ry:.4f}")
    print(f"  rest_z : {rz:.4f}")
    print(f"  RMSE   : {rmse_cm:.1f} cm  (position error, post-bounce)")
    print(f"  bounces: {len(events)}")
    print("─" * 46)
    print()
    print("Paste into config/d455.yaml:")
    print(f"  rest_x: {rx:.2f}")
    print(f"  rest_y: {ry:.2f}")
    print(f"  rest_z: {rz:.2f}")

    if args.plot:
        try:
            _plot_bounces(events, result.x, dk)
            import matplotlib.pyplot as plt
            plt.show()
        except ImportError:
            print("[calibrate] matplotlib not available — skipping bounce plots")


if __name__ == "__main__":
    main()
