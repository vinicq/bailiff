from __future__ import annotations

import logging

import numpy as np

from .base import AudioBackend, AudioStream

logger = logging.getLogger("bailiff.audio.backends.linux")


class _SoundDeviceStream(AudioStream):
    def __init__(self, sd_stream, sample_rate: int, channels: int, frames_per_buffer: int):
        self._stream = sd_stream
        self._rate = sample_rate
        self._channels = channels
        self._frames = frames_per_buffer

    @property
    def sample_rate(self) -> int:
        return self._rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frames_per_buffer(self) -> int:
        return self._frames

    def read(self, n_samples: int):
        data, _overflowed = self._stream.read(n_samples)
        arr = np.asarray(data, dtype=np.float32)
        return arr.tobytes()

    def close(self) -> None:
        try:
            self._stream.stop()
        except Exception as exc:
            logger.debug("sd stop failed: %s", exc)
        try:
            self._stream.close()
        except Exception as exc:
            logger.debug("sd close failed: %s", exc)


class LinuxPulseBackend(AudioBackend):
    def __init__(self):
        self._sd = None

    def _sounddevice(self):
        if self._sd is None:
            import sounddevice as sd
            self._sd = sd
            logger.info("sounddevice initialized")
        return self._sd

    def _open(self, device_index: int, channels: int, rate: int, chunk: int) -> _SoundDeviceStream:
        sd = self._sounddevice()
        stream = sd.InputStream(
            samplerate=rate,
            channels=channels,
            dtype="float32",
            blocksize=chunk,
            device=device_index,
        )
        stream.start()
        return _SoundDeviceStream(stream, rate, channels, chunk)

    def open_microphone(self, rate: int, chunk: int) -> AudioStream:
        sd = self._sounddevice()
        default_input = sd.default.device[0] if sd.default.device else None
        info = sd.query_devices(default_input, kind="input")
        index = info.get("index", default_input)
        channels = max(1, int(info.get("max_input_channels", 1)))
        logger.info("Default microphone: %s (index=%s)", info.get("name"), index)
        return self._open(index, channels, rate, chunk)

    def open_system_loopback(self, rate: int, chunk: int) -> AudioStream | None:
        sd = self._sounddevice()
        try:
            devices = sd.query_devices()
        except Exception as exc:
            logger.warning("query_devices failed: %s", exc)
            return None

        monitor = None
        for idx, dev in enumerate(devices):
            name = dev.get("name", "")
            if ".monitor" in name and int(dev.get("max_input_channels", 0)) > 0:
                monitor = (idx, dev)
                break

        if monitor is None:
            logger.warning("No PulseAudio monitor source found; system loopback unavailable")
            return None

        idx, dev = monitor
        native_rate = int(dev.get("default_samplerate", rate))
        channels = max(1, int(dev.get("max_input_channels", 1)))
        native_chunk = int(chunk * native_rate / rate) if rate else chunk
        logger.info("PulseAudio monitor: %s (index=%d, native_rate=%d)", dev.get("name"), idx, native_rate)
        return self._open(idx, channels, native_rate, native_chunk)

    def cleanup(self) -> None:
        self._sd = None
        logger.info("sounddevice cleanup complete")
