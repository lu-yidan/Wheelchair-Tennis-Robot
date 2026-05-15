"""
Camera abstraction for ball_detection.

A backend yields Frames containing:
  - color : H×W×3 BGR uint8         (always present)
  - depth : H×W   float32 metres    (None for RGB-only cameras)
            *aligned to the colour image* — same intrinsics, same pixel grid
  - t     : capture timestamp from time.perf_counter()

Backends:
  realsense  → cameras/realsense.py   (Intel RealSense via pyrealsense2)
  zed        → cameras/zed.py         (Stereolabs ZED via pyzed.sl)
  webcam     → cameras/webcam.py      (any V4L2 / USB UVC device via OpenCV)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class Intrinsics:
    """Pinhole intrinsics for the colour image (and aligned depth, if any)."""
    fx:    float
    fy:    float
    ppx:   float
    ppy:   float
    width: int
    height: int
    coeffs: List[float] = field(default_factory=lambda: [0.0]*5)   # [k1,k2,p1,p2,k3]

    def deproject(self, u, v, depth_m):
        """Pixel (u,v) at depth d → 3-D point in optical frame (Z-fwd, X-right, Y-down)."""
        x = (u - self.ppx) * depth_m / self.fx
        y = (v - self.ppy) * depth_m / self.fy
        return np.array([x, y, depth_m], dtype=np.float64)


@dataclass
class Frame:
    color: np.ndarray              # H×W×3 BGR uint8
    depth: Optional[np.ndarray]    # H×W float32 metres, aligned to color (or None)
    t:     float                   # perf_counter seconds


class Camera(ABC):
    """Pluggable camera backend.  Subclasses implement start/grab/stop."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def grab(self) -> Optional[Frame]:
        """Block until next frame.  Return None on transient timeout."""

    @property
    @abstractmethod
    def intrinsics(self) -> Intrinsics: ...

    @property
    def has_depth(self) -> bool:
        """True if grab() returns frames with non-None .depth."""
        return False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def from_config(cfg: dict) -> Camera:
    """
    Factory.  Reads cfg['camera']['backend'] and instantiates the matching class.
    The full cfg dict is passed through so backends can read the shared
    `width`/`height` keys plus their own `camera:` sub-section.
    """
    cam_cfg = cfg.get("camera") or {}
    backend = str(cam_cfg.get("backend", "realsense")).lower()
    if backend == "realsense":
        from .realsense import RealSenseCamera
        return RealSenseCamera(cfg)
    if backend == "zed":
        from .zed import ZedCamera
        return ZedCamera(cfg)
    if backend == "webcam":
        from .webcam import WebcamCamera
        return WebcamCamera(cfg)
    raise ValueError(f"Unknown camera backend: {backend!r} "
                     f"(expected: realsense | zed | webcam)")
