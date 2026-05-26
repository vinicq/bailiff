from bailiff.features.transcription.backends.base import TranscriptionBackend
from bailiff.features.transcription.backends.http_openai import HttpOpenAIBackend
from bailiff.features.transcription.backends.local_whisper import LocalFasterWhisperBackend


def get_backend() -> TranscriptionBackend:
    from bailiff.core.config import settings

    cfg = settings.transcription
    if cfg.backend == "faster-whisper":
        return LocalFasterWhisperBackend(
            model=cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
            language=cfg.language,
        )
    if cfg.backend == "openai-http":
        base_url = cfg.base_url or settings.models.llm_base_url
        api_key_secret = cfg.api_key or settings.models.llm_api_key
        api_key = api_key_secret.get_secret_value() if api_key_secret else "ollama"
        return HttpOpenAIBackend(
            base_url=base_url,
            api_key=api_key,
            model=cfg.model_size,
            model_by_language=cfg.model_by_language or {},
            language=cfg.language,
            sample_rate=settings.audio.sample_rate,
        )
    raise ValueError(f"unknown transcription backend: {cfg.backend!r}")


__all__ = [
    "TranscriptionBackend",
    "LocalFasterWhisperBackend",
    "HttpOpenAIBackend",
    "get_backend",
]
