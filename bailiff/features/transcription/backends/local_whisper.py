from __future__ import annotations

import logging

import numpy as np

from bailiff.features.transcription.backends.base import TranscriptionBackend

logger = logging.getLogger("bailiff.transcription.backends.local")


class LocalFasterWhisperBackend(TranscriptionBackend):
    def __init__(self, model: str, device: str, compute_type: str, language: str | None = None):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        logger.info("faster-whisper loaded: %s", self.model_name)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        if self._model is None:
            raise RuntimeError("backend not loaded; call load() first")
        lang = language if language is not None else self.language
        segments, _ = self._model.transcribe(
            audio,
            beam_size=5,
            language=lang,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text for seg in segments).strip()
