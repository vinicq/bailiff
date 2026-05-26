import sys
import types

import pytest


def _install_fake_pyaudiowpatch(monkeypatch):
    fake = types.ModuleType("pyaudiowpatch")
    fake.paFloat32 = 1

    class _FakePA:
        def get_default_input_device_info(self):
            return {"name": "fake-mic", "index": 0, "maxInputChannels": 1}

        def get_default_wasapi_loopback(self):
            return {"name": "fake-loop", "index": 1, "maxInputChannels": 2, "defaultSampleRate": 48000}

        def open(self, **kwargs):
            return object()

        def terminate(self):
            pass

    fake.PyAudio = _FakePA
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", fake)


def _install_fake_sounddevice(monkeypatch):
    fake = types.ModuleType("sounddevice")

    class _Default:
        device = (0, 0)

    fake.default = _Default()

    def query_devices(idx=None, kind=None):
        if idx is None:
            return [{"name": "alsa_output.monitor", "index": 1, "max_input_channels": 2, "default_samplerate": 48000}]
        return {"name": "default-mic", "index": 0, "max_input_channels": 1, "default_samplerate": 48000}

    fake.query_devices = query_devices

    class _InputStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

        def read(self, n):
            import numpy as np
            return np.zeros(n, dtype="float32"), False

    fake.InputStream = _InputStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


def test_get_backend_returns_windows_on_win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    _install_fake_pyaudiowpatch(monkeypatch)

    from bailiff.features.audio_ingest.backends import get_backend
    from bailiff.features.audio_ingest.backends.windows import WindowsWasapiBackend

    backend = get_backend()
    assert isinstance(backend, WindowsWasapiBackend)


def test_get_backend_returns_linux_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _install_fake_sounddevice(monkeypatch)

    from bailiff.features.audio_ingest.backends import get_backend
    from bailiff.features.audio_ingest.backends.linux import LinuxPulseBackend

    backend = get_backend()
    assert isinstance(backend, LinuxPulseBackend)


def test_get_backend_raises_for_unknown_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    from bailiff.features.audio_ingest.backends import get_backend
    with pytest.raises(NotImplementedError):
        get_backend()


def test_audio_stream_protocol_shape():
    from bailiff.features.audio_ingest.backends.base import AudioStream

    class FakeStream(AudioStream):
        def __init__(self):
            self._rate = 16000
            self._channels = 1
            self._frames = 512

        @property
        def sample_rate(self):
            return self._rate

        @property
        def channels(self):
            return self._channels

        @property
        def frames_per_buffer(self):
            return self._frames

        def read(self, n_samples):
            return b"\x00" * (n_samples * 4)

        def close(self):
            pass

    s = FakeStream()
    assert s.sample_rate == 16000
    assert s.channels == 1
    assert s.frames_per_buffer == 512
    assert isinstance(s.read(8), (bytes, bytearray))
    s.close()
