"""
ZED backend (Stereolabs ZED Mini / ZED 2 / ZED X / ZED 2i).

Requires:
    ZED SDK installed system-wide  (https://www.stereolabs.com/developers/release)
    pyzed.sl  →  python3 /usr/local/zed/get_python_api.py  (run inside the env)

Depth is intrinsically aligned to the LEFT colour image — same intrinsics,
same pixel grid — so downstream code is identical to the aligned RealSense path.

YAML keys (under `camera:`):
    backend:      zed
    resolution:   HD720 | HD1080 | HD2K | VGA      (default HD720)
    fps:          15 | 30 | 60 | 100               (must match resolution)
    depth_mode:   NEURAL | ULTRA | QUALITY | PERFORMANCE | NONE   (default PERFORMANCE)
    serial:       <int>     (optional — pin to a specific camera)
    svo:          <path>    (optional — replay an .svo recording)
    flip:         false     (set true if camera is mounted upside-down)
"""

import time
import numpy as np

try:
    import pyzed.sl as sl
    _HAS_ZED = True
except ImportError:
    sl = None
    _HAS_ZED = False

from .base import Camera, Frame, Intrinsics


_RES_MAP = {
    "HD2K":   "HD2K",
    "HD1080": "HD1080",
    "HD1200": "HD1200",   # ZED X only
    "HD720":  "HD720",
    "SVGA":   "SVGA",     # ZED X only
    "VGA":    "VGA",      # ZED 1/2/Mini
    "AUTO":   "AUTO",
}

_DEPTH_MAP = {
    "NEURAL":      "NEURAL",
    "ULTRA":       "ULTRA",
    "QUALITY":     "QUALITY",
    "PERFORMANCE": "PERFORMANCE",
    "NONE":        "NONE",
}


class ZedCamera(Camera):
    def __init__(self, cfg: dict):
        if not _HAS_ZED:
            raise ImportError(
                "pyzed.sl not installed.  Install ZED SDK then run:\n"
                "  conda activate <env> && python /usr/local/zed/get_python_api.py")
        ccfg = cfg.get("camera") or {}
        self._res_name   = _RES_MAP.get(str(ccfg.get("resolution", "HD720")).upper(), "HD720")
        self._fps        = int(ccfg.get("fps", 60))
        self._depth_name = _DEPTH_MAP.get(str(ccfg.get("depth_mode", "PERFORMANCE")).upper(),
                                          "PERFORMANCE")
        self._serial     = int(ccfg.get("serial", 0))
        self._svo        = str(ccfg.get("svo", "") or "")
        self._flip       = bool(ccfg.get("flip", False))
        self._min_depth  = float(ccfg.get("min_depth", 0.15))
        self._max_depth  = float(ccfg.get("max_depth", 20.0))

        self._cam      = None
        self._runtime  = None
        self._mat_c    = None
        self._mat_d    = None
        self._intr     = None

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        self._cam = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self._res_name)
        init.camera_fps        = self._fps
        init.depth_mode        = getattr(sl.DEPTH_MODE, self._depth_name)
        init.coordinate_units  = sl.UNIT.METER
        init.depth_minimum_distance = self._min_depth
        init.depth_maximum_distance = self._max_depth
        if self._flip:
            init.camera_image_flip = sl.FLIP_MODE.ON
        if self._svo:
            init.set_from_svo_file(self._svo)
            print(f"[INFO] ZED replaying SVO: {self._svo}")
        elif self._serial:
            init.set_from_serial_number(self._serial)

        err = self._cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")

        self._runtime = sl.RuntimeParameters()

        cinfo = self._cam.get_camera_information()
        # SDK 4.x: camera_configuration.calibration_parameters.left_cam
        try:
            cal = cinfo.camera_configuration.calibration_parameters.left_cam
        except AttributeError:                           # SDK 3.x fallback
            cal = cinfo.calibration_parameters.left_cam
        try:
            iw = cal.image_size.width;  ih = cal.image_size.height
        except AttributeError:
            iw = self._cam.get_camera_information().camera_resolution.width
            ih = self._cam.get_camera_information().camera_resolution.height
        disto_raw = getattr(cal, "disto", None)
        disto = list(disto_raw) if disto_raw is not None and len(disto_raw) > 0 else []
        self._intr = Intrinsics(
            fx=float(cal.fx), fy=float(cal.fy),
            ppx=float(cal.cx), ppy=float(cal.cy),
            width=int(iw), height=int(ih),
            coeffs=(disto[:5] + [0.0]*5)[:5],
        )

        self._mat_c = sl.Mat(self._intr.width, self._intr.height,
                             sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
        self._mat_d = sl.Mat(self._intr.width, self._intr.height,
                             sl.MAT_TYPE.F32_C1, sl.MEM.CPU)

        print(f"[INFO] ZED OK: {self._res_name} @ {self._fps} fps  "
              f"depth={self._depth_name}  intr fx={cal.fx:.1f} ppx={cal.cx:.1f}")

    def stop(self):
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    def grab(self):
        err = self._cam.grab(self._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            return None
        self._cam.retrieve_image(self._mat_c, sl.VIEW.LEFT)
        bgra = self._mat_c.get_data()                # H×W×4
        bgr  = np.ascontiguousarray(bgra[:, :, :3])  # drop alpha
        depth = None
        if self._depth_name != "NONE":
            self._cam.retrieve_measure(self._mat_d, sl.MEASURE.DEPTH)
            d = self._mat_d.get_data().copy()        # float32 metres
            # NaN / inf → 0 so downstream code treats them as "no reading"
            d[~np.isfinite(d)] = 0.0
            depth = d
        return Frame(color=bgr, depth=depth, t=time.perf_counter())

    @property
    def intrinsics(self):
        return self._intr

    @property
    def has_depth(self):
        return self._depth_name != "NONE"
