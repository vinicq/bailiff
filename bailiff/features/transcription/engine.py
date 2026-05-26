import logging

import numpy as np

from bailiff.features.transcription.backends import TranscriptionBackend, get_backend

logger = logging.getLogger("bailiff.transcription.engine")


class WhisperEngine:
    """
    Thin dispatcher over a TranscriptionBackend.

    Default backend is selected from settings.transcription.backend; tests can inject
    a fake backend.
    """

    def __init__(self, backend: TranscriptionBackend | None = None):
        self.backend = backend or get_backend()

    def load(self) -> None:
        self.backend.load()

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        text = self.backend.transcribe(audio, language=language)
        if text:
            logger.info("Transcription: %s", text)
        return text
