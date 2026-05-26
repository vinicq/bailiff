from __future__ import annotations

import io
import logging

import numpy as np

from bailiff.features.transcription.backends.base import TranscriptionBackend

logger = logging.getLogger("bailiff.transcription.backends.http")


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    from scipy.io import wavfile

    if audio.dtype != np.int16:
        clipped = np.clip(audio, -1.0, 1.0)
        audio = (clipped * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio)
    return buf.getvalue()


class HttpOpenAIBackend(TranscriptionBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        model_by_language: dict[str, str] | None = None,
        language: str | None = None,
        sample_rate: int = 16000,
        timeout: float = 60.0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.default_model = model
        self.model_by_language = model_by_language or {}
        self.language = language
        self.sample_rate = sample_rate
        self.timeout = timeout
        self._client = None

    def load(self) -> None:
        import httpx
        import openai

        self._client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(self.timeout, connect=5.0),
            max_retries=2,
        )
        logger.info(
            "HTTP transcription backend ready: base_url=%s default_model=%s lang_routes=%s",
            self.base_url,
            self.default_model,
            sorted(self.model_by_language.keys()) or "none",
        )

    def _pick_model(self, language: str | None) -> str:
        if language and language in self.model_by_language:
            return self.model_by_language[language]
        return self.default_model

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        if self._client is None:
            raise RuntimeError("backend not loaded; call load() first")
        effective_language = language if language is not None else self.language
        wav_bytes = _audio_to_wav_bytes(audio, self.sample_rate)
        buf = io.BytesIO(wav_bytes)
        buf.name = "audio.wav"
        kwargs: dict = {"file": buf, "model": self._pick_model(effective_language)}
        if effective_language:
            kwargs["language"] = effective_language
        response = self._client.audio.transcriptions.create(**kwargs)
        return (response.text or "").strip()
