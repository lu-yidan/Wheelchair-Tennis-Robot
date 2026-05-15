"""
Generic V4L2 / UVC webcam backend (HIKROBOT, FLIR Blackfly via Spinnaker UVC,
Logitech BRIO, Razer Kiyo, Arducam IMX477, etc.).

No depth — pipeline must use visual depth (`fx · R / r_px`).
Intrinsics MUST be provided in the YAML; without them PnP / depth are wrong.
Calibrate once with cv2.calibrateCamera + a chessboard.

YAML keys (under `camera:`):
    backend:   webcam
    device:    0                 # /dev/video0  or  "rtsp://..."
    fourcc:    MJPG              # MJPG → high fps; YUYV often capped at 30
    fps:       60
    intrinsics:
      fx:  1280
      fy:  1280
      ppx: 640
      ppy: 360
      dist: [0, 0, 0, 0, 0]      # [k1,k2,p1,p2,k3]

The shared top-level `width:` / `height:` keys are honoured so the same yaml
can be diffed against d455.yaml / zedmini.yaml.
"""

import time
import cv2
import numpy as np

from .base import Camera, Frame, Intrinsics


class WebcamCamera(Camera):
    def __init__(self, cfg: dict):
        ccfg = cfg.get("camera") or {}
        dev  = ccfg.get("device", 0)
        # Accept "0" / 0 / "/dev/video0" / rtsp URL
        try:
            self._device = int(dev)
        except (TypeError, ValueError):
            self._device = str(dev)
        self._fourcc = str(ccfg.get("fourcc", "MJPG")).upper()
        self._width  = int(cfg.get("width",  1280))
        self._height = int(cfg.get("height", 720))
        self._fps    = int(ccfg.get("fps", 60))

        ic = ccfg.get("intrinsics") or {}
        if "fx" not in ic:
            print("[WARN] webcam: no intrinsics in YAML — using rough guess "
                  "(fx=fy=width). Visual depth & AprilTag PnP will be inaccurate. "
                  "Calibrate with a chessboard and put the result in config/<your>.yaml.")
        self._intr = Intrinsics(
            fx=float(ic.get("fx", self._width)),
            fy=float(ic.get("fy", self._width)),
            ppx=float(ic.get("ppx", self._width / 2)),
            ppy=float(ic.get("ppy", self._height / 2)),
            width=self._width,
            height=self._height,
            coeffs=list((ic.get("dist") or [0.0]*5))[:5] + [0.0]*5,
        )
        self._intr.coeffs = self._intr.coeffs[:5]

        self._cap = None

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        # CAP_V4L2 first (lets us actually set FOURCC); fall back to default.
        if isinstance(self._device, int):
            self._cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self._device)
        else:
            self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Webcam open failed: {self._device}")

        # Order matters on V4L2: FOURCC → resolution → fps
        try:
            self._cap.set(cv2.CAP_PROP_FOURCC,
                          cv2.VideoWriter_fourcc(*self._fourcc))
        except Exception:
            pass
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        # Small buffer so we don't accumulate latency
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        af = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Webcam: {aw}×{ah} @ {af:.0f} fps  "
              f"FOURCC={self._fourcc}  device={self._device}")
        if (aw, ah) != (self._width, self._height):
            print(f"[WARN] Driver returned {aw}×{ah} (asked {self._width}×{self._height}). "
                  "Intrinsics in YAML are for the requested resolution — recalibrate "
                  "or update fx/fy/ppx/ppy if you keep this resolution.")
        # Refresh intrinsics width/height to what the driver actually produced,
        # but DO NOT auto-rescale fx/fy — that requires a real recalibration.
        self._intr.width  = aw
        self._intr.height = ah

    def stop(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def grab(self):
        ok, bgr = self._cap.read()
        if not ok:
            return None
        return Frame(color=bgr, depth=None, t=time.perf_counter())

    @property
    def intrinsics(self):
        return self._intr

    @property
    def has_depth(self):
        return False
