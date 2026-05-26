from __future__ import annotations

import sys

from .base import AudioBackend, AudioStream


def get_backend() -> AudioBackend:
    if sys.platform == "win32":
        from .windows import WindowsWasapiBackend
        return WindowsWasapiBackend()
    if sys.platform == "linux":
        from .linux import LinuxPulseBackend
        return LinuxPulseBackend()
    raise NotImplementedError(f"audio backend for {sys.platform} not implemented")


__all__ = ["get_backend", "AudioBackend", "AudioStream"]
