from __future__ import annotations

from abc import ABC, abstractmethod


class AudioStream(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frames_per_buffer(self) -> int: ...

    @abstractmethod
    def read(self, n_samples: int):
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class AudioBackend(ABC):
    @abstractmethod
    def open_microphone(self, rate: int, chunk: int) -> AudioStream:
        ...

    @abstractmethod
    def open_system_loopback(self, rate: int, chunk: int) -> AudioStream | None:
        ...

    @abstractmethod
    def cleanup(self) -> None:
        ...
