import sys
import types

import pytest


def test_vad_prefers_pip_package_when_available(monkeypatch):
    sentinel_model = object()

    fake = types.ModuleType("silero_vad")

    def load_silero_vad():
        return sentinel_model

    fake.load_silero_vad = load_silero_vad
    monkeypatch.setitem(sys.modules, "silero_vad", fake)

    import torch
    def _torch_hub_should_not_be_called(*a, **kw):
        raise AssertionError("torch.hub.load should not be called when pip package available")
    monkeypatch.setattr(torch.hub, "load", _torch_hub_should_not_be_called)

    import importlib
    import bailiff.features.audio_ingest.vad as vad_module
    importlib.reload(vad_module)

    model = vad_module._load_silero_model()
    assert model is sentinel_model


def test_vad_falls_back_to_torch_hub_when_pip_package_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "silero_vad", raising=False)

    blocker = types.ModuleType("silero_vad_blocker")

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "silero_vad" or name.startswith("silero_vad."):
            raise ImportError("simulated missing silero_vad")
        return real_import(name, globals, locals, fromlist, level)

    import builtins
    monkeypatch.setattr(builtins, "__import__", fake_import)

    sentinel_model = object()
    call_log = {"called": False}

    def fake_torch_hub_load(*args, **kwargs):
        call_log["called"] = True
        return sentinel_model, ("get_speech_timestamps", "_", "read_audio", "_", "_")

    import torch
    monkeypatch.setattr(torch.hub, "load", fake_torch_hub_load)

    import importlib
    import bailiff.features.audio_ingest.vad as vad_module
    importlib.reload(vad_module)

    model = vad_module._load_silero_model()
    assert call_log["called"] is True
    assert model is sentinel_model
