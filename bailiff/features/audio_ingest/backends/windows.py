from __future__ import annotations

import logging

from .base import AudioBackend, AudioStream

logger = logging.getLogger("bailiff.audio.backends.windows")


class _WasapiStream(AudioStream):
    def __init__(self, pa_stream, sample_rate: int, channels: int, frames_per_buffer: int):
        self._stream = pa_stream
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
        return self._stream.read(n_samples, exception_on_overflow=False)

    def close(self) -> None:
        try:
            self._stream.stop_stream()
        except Exception as exc:
            logger.debug("stop_stream failed: %s", exc)
        try:
            self._stream.close()
        except Exception as exc:
            logger.debug("close failed: %s", exc)


class WindowsWasapiBackend(AudioBackend):
    def __init__(self):
        self._pa = None

    def _pyaudio(self):
        if self._pa is None:
            import pyaudiowpatch as pyaudio
            self._pa = pyaudio.PyAudio()
            logger.info("PyAudio (WASAPI) initialized")
        return self._pa

    def _open(self, device_info: dict, rate: int, chunk: int) -> _WasapiStream:
        import pyaudiowpatch as pyaudio
        channels = max(1, int(device_info.get("maxInputChannels", 1)))
        stream = self._pyaudio().open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(device_info["index"]),
            frames_per_buffer=chunk,
        )
        logger.debug("Opened stream: device=%s rate=%d chunk=%d channels=%d",
                     device_info["name"], rate, chunk, channels)
        return _WasapiStream(stream, rate, channels, chunk)

    def open_microphone(self, rate: int, chunk: int) -> AudioStream:
        pa = self._pyaudio()
        info = pa.get_default_input_device_info()
        logger.info("Default microphone: %s (index=%d)", info["name"], info["index"])
        return self._open(info, rate, chunk)

    def open_system_loopback(self, rate: int, chunk: int) -> AudioStream | None:
        pa = self._pyaudio()
        try:
            device = pa.get_default_wasapi_loopback()
        except Exception as exc:
            logger.warning("No WASAPI loopback device available: %s", exc)
            return None

        native_rate = int(device["defaultSampleRate"])
        native_chunk = int(chunk * native_rate / rate) if rate else chunk
        logger.info("Loopback device: %s (index=%d, native_rate=%d, native_chunk=%d)",
                    device["name"], device["index"], native_rate, native_chunk)
        return self._open(device, native_rate, native_chunk)

    def cleanup(self) -> None:
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception as exc:
                logger.debug("PyAudio terminate failed: %s", exc)
            self._pa = None
            logger.info("PyAudio (WASAPI) terminated")
