"""ball_detection camera backends — see base.py for the Camera interface."""

from .base import Camera, Frame, Intrinsics, from_config

__all__ = ["Camera", "Frame", "Intrinsics", "from_config"]
