"""
RealSense backend (Intel D435 / D455 / ...).

Depth is auto-aligned to the colour image via rs.align(rs.stream.color),
so downstream code can treat depth as a metres-valued float32 array on the
exact same pixel grid as colour — no separate depth_intrin / c2d_extr needed.
"""

import time
import numpy as np
import pyrealsense2 as rs

from .base import Camera, Frame, Intrinsics


class RealSenseCamera(Camera):
    def __init__(self, cfg: dict):
        ccfg = cfg.get("camera") or {}
        self._width  = int(cfg.get("width",  1280))
        self._height = int(cfg.get("height", 720))
        self._fps_tries = list(ccfg.get("fps_tries", [(60, 60), (30, 30), (15, 15)]))
        # Allow yaml to express "fps: 60" → both color and depth at 60
        if "fps" in ccfg:
            f = int(ccfg["fps"])
            self._fps_tries = [(f, f)] + self._fps_tries
        # rs.align is cheap but optional
        self._align_to_color = bool(ccfg.get("align_depth_to_color", True))
        self._enable_depth   = bool(ccfg.get("enable_depth", True))

        self._pipeline = None
        self._align    = None
        self._intr     = None          # set in start()
        self._depth_scale = 1.0

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        self._pipeline = rs.pipeline()
        last_err = None
        for c_fps, d_fps in self._normalize_tries():
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, self._width, self._height,
                              rs.format.bgr8, c_fps)
            if self._enable_depth:
                cfg.enable_stream(rs.stream.depth, self._width, self._height,
                                  rs.format.z16, d_fps)
            for attempt in range(2):
                try:
                    print(f"[INFO] RealSense start ({c_fps}/{d_fps} Hz, attempt {attempt+1})")
                    profile = self._pipeline.start(cfg)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "resolve" in msg or "couldn't" in msg:
                        last_err = e
                        print(f"[WARN] Profile not supported: {e}")
                        break
                    if "no such file" in msg or "cannot open" in msg or "map_device" in msg:
                        print(f"[WARN] Stale device node ({e}); resetting...")
                        self._hw_reset()
                        continue
                    raise
                try:
                    self._pipeline.wait_for_frames(timeout_ms=5000)
                    print(f"[INFO] RealSense OK ({c_fps}/{d_fps} Hz)")
                    self._post_start(profile)
                    return
                except RuntimeError:
                    print("[WARN] First frame timeout — hardware reset...")
                    self._pipeline.stop()
                    self._hw_reset()
        raise RuntimeError(f"RealSense failed. Tried {self._fps_tries}. Last: {last_err!r}")

    def stop(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    def grab(self):
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError:
            return None
        if self._align is not None:
            frames = self._align.process(frames)
        cf = frames.get_color_frame()
        if not cf:
            return None
        color = np.asanyarray(cf.get_data()).copy()
        depth = None
        if self._enable_depth:
            df = frames.get_depth_frame()
            if df:
                depth = np.asanyarray(df.get_data()).astype(np.float32) * self._depth_scale
        return Frame(color=color, depth=depth, t=time.perf_counter())

    @property
    def intrinsics(self):
        return self._intr

    @property
    def has_depth(self):
        return self._enable_depth

    # ── internals ─────────────────────────────────────────────────────────────

    def _normalize_tries(self):
        """Allow either [(c,d),(c,d)] or flat [60,30,15] in yaml."""
        out = []
        for t in self._fps_tries:
            if isinstance(t, (list, tuple)) and len(t) == 2:
                out.append((int(t[0]), int(t[1])))
            else:
                out.append((int(t), int(t)))
        return out

    def _hw_reset(self):
        devs = rs.context().query_devices()
        if len(devs) == 0:
            raise RuntimeError("No RealSense device found.")
        print("[INFO] RealSense hardware reset...")
        devs[0].hardware_reset()
        time.sleep(8)
        self._pipeline = rs.pipeline()

    def _post_start(self, profile):
        cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ci = cp.get_intrinsics()
        self._intr = Intrinsics(
            fx=ci.fx, fy=ci.fy, ppx=ci.ppx, ppy=ci.ppy,
            width=ci.width, height=ci.height,
            coeffs=list(ci.coeffs[:5]),
        )
        if self._enable_depth:
            self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            if self._align_to_color:
                self._align = rs.align(rs.stream.color)
        print(f"[INFO] RealSense intr: fx={ci.fx:.1f} fy={ci.fy:.1f} "
              f"ppx={ci.ppx:.1f} ppy={ci.ppy:.1f}  depth_scale={self._depth_scale:.5f}")
