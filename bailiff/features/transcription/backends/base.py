from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TranscriptionBackend(ABC):
    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        ...
