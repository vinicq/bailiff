import io

import numpy as np
import pytest

from bailiff.features.transcription.backends import (
    HttpOpenAIBackend,
    LocalFasterWhisperBackend,
    get_backend,
)
from bailiff.features.transcription.backends.http_openai import _audio_to_wav_bytes


def test_get_backend_returns_local_for_faster_whisper(monkeypatch):
    from bailiff.core.config import settings
    monkeypatch.setattr(settings.transcription, "backend", "faster-whisper")
    backend = get_backend()
    assert isinstance(backend, LocalFasterWhisperBackend)


def test_get_backend_returns_http_for_openai_http(monkeypatch):
    from bailiff.core.config import settings
    monkeypatch.setattr(settings.transcription, "backend", "openai-http")
    monkeypatch.setattr(settings.transcription, "base_url", "http://example.test/v1")
    backend = get_backend()
    assert isinstance(backend, HttpOpenAIBackend)
    assert backend.base_url == "http://example.test/v1"


def test_http_backend_falls_back_to_models_llm_base_url(monkeypatch):
    from bailiff.core.config import settings
    monkeypatch.setattr(settings.transcription, "backend", "openai-http")
    monkeypatch.setattr(settings.transcription, "base_url", None)
    monkeypatch.setattr(settings.models, "llm_base_url", "http://ollama.test/v1")
    backend = get_backend()
    assert backend.base_url == "http://ollama.test/v1"


def test_get_backend_raises_for_unknown(monkeypatch):
    from bailiff.core.config import settings
    monkeypatch.setattr(settings.transcription, "backend", "not-a-real-backend")
    with pytest.raises(ValueError, match="unknown transcription backend"):
        get_backend()


def test_audio_to_wav_bytes_produces_valid_wav():
    audio = np.zeros(16000, dtype=np.float32)
    wav = _audio_to_wav_bytes(audio, sample_rate=16000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_audio_to_wav_bytes_clips_and_converts_float(monkeypatch):
    audio = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
    wav = _audio_to_wav_bytes(audio, sample_rate=16000)
    assert len(wav) > 44


def test_http_backend_picks_default_model_when_language_has_no_override():
    backend = HttpOpenAIBackend(
        base_url="http://x/v1", api_key="k",
        model="default-model",
        model_by_language={"pt": "pt-model"},
    )
    assert backend._pick_model("en") == "default-model"
    assert backend._pick_model(None) == "default-model"


def test_http_backend_picks_language_specific_model():
    backend = HttpOpenAIBackend(
        base_url="http://x/v1", api_key="k",
        model="default-model",
        model_by_language={"pt": "pt-model", "en": "en-model"},
    )
    assert backend._pick_model("pt") == "pt-model"
    assert backend._pick_model("en") == "en-model"


def test_http_backend_transcribe_posts_with_correct_kwargs(mocker):
    backend = HttpOpenAIBackend(
        base_url="http://x/v1", api_key="k",
        model="sendmeaiohyeah/whisper-large-v2",
        sample_rate=16000,
    )
    backend.load()

    fake_response = mocker.MagicMock()
    fake_response.text = "olá mundo"
    spy = mocker.patch.object(backend._client.audio.transcriptions, "create", return_value=fake_response)

    audio = np.zeros(8000, dtype=np.float32)
    result = backend.transcribe(audio, language="pt")

    assert result == "olá mundo"
    call_kwargs = spy.call_args.kwargs
    assert call_kwargs["model"] == "sendmeaiohyeah/whisper-large-v2"
    assert call_kwargs["language"] == "pt"
    assert call_kwargs["file"].name == "audio.wav"


def test_http_backend_omits_language_when_none(mocker):
    backend = HttpOpenAIBackend(
        base_url="http://x/v1", api_key="k",
        model="m",
        sample_rate=16000,
    )
    backend.load()
    fake_response = mocker.MagicMock()
    fake_response.text = ""
    spy = mocker.patch.object(backend._client.audio.transcriptions, "create", return_value=fake_response)

    backend.transcribe(np.zeros(8000, dtype=np.float32), language=None)

    call_kwargs = spy.call_args.kwargs
    assert "language" not in call_kwargs


def test_http_backend_returns_empty_string_when_response_text_is_none(mocker):
    backend = HttpOpenAIBackend(base_url="http://x/v1", api_key="k", model="m")
    backend.load()
    fake_response = mocker.MagicMock()
    fake_response.text = None
    mocker.patch.object(backend._client.audio.transcriptions, "create", return_value=fake_response)

    assert backend.transcribe(np.zeros(800, dtype=np.float32)) == ""


def test_local_backend_requires_load_before_transcribe():
    backend = LocalFasterWhisperBackend(model="tiny", device="cpu", compute_type="int8")
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.transcribe(np.zeros(800, dtype=np.float32))


def test_http_backend_requires_load_before_transcribe():
    backend = HttpOpenAIBackend(base_url="http://x/v1", api_key="k", model="m")
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.transcribe(np.zeros(800, dtype=np.float32))


def test_whisper_engine_delegates_to_injected_backend(mocker):
    from bailiff.features.transcription.engine import WhisperEngine

    fake_backend = mocker.MagicMock()
    fake_backend.transcribe.return_value = "fake-text"
    engine = WhisperEngine(backend=fake_backend)
    engine.load()
    result = engine.transcribe(np.zeros(800, dtype=np.float32), language="pt")

    fake_backend.load.assert_called_once()
    fake_backend.transcribe.assert_called_once()
    assert fake_backend.transcribe.call_args.kwargs == {"language": "pt"}
    assert result == "fake-text"
