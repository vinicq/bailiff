import logging

from .backends import AudioBackend, AudioStream, get_backend

logger = logging.getLogger("bailiff.audio.capture")


class AudioCaptureManager:
    def __init__(self, backend: AudioBackend | None = None):
        self.backend = backend or get_backend()
        logger.info("Capture manager using backend %s", type(self.backend).__name__)

    def open_microphone(self, sample_rate: int, chunk_size: int) -> AudioStream:
        stream = self.backend.open_microphone(sample_rate, chunk_size)
        logger.info("Mic stream opened: channels=%d, rate=%d, chunk=%d",
                    stream.channels, stream.sample_rate, stream.frames_per_buffer)
        return stream

    def open_system_loopback(self, sample_rate: int, chunk_size: int) -> AudioStream | None:
        stream = self.backend.open_system_loopback(sample_rate, chunk_size)
        if stream is None:
            return None
        logger.info("Loopback stream opened: channels=%d, rate=%d, chunk=%d",
                    stream.channels, stream.sample_rate, stream.frames_per_buffer)
        return stream

    def terminate(self):
        self.backend.cleanup()
