from __future__ import annotations

import logging
import queue
import threading
import time
from multiprocessing import Process
from multiprocessing import Queue as ProcessQueue
from typing import Callable

import numpy as np
from scipy.signal import resample

from bailiff.core.config import AudioConfig
from bailiff.core.events import AudioChunk
from bailiff.core.logging import setup_logging
from bailiff.features.audio_ingest.capture import AudioCaptureManager
from bailiff.features.audio_ingest.preprocessor import AudioPreprocessor
from bailiff.features.audio_ingest.vad import VADEngine

logger = logging.getLogger("bailiff.audio.service")

WORKER_NAME = "audio-ingest"


class AudioIngestService:
    def __init__(self,
                 output_queue: ProcessQueue,
                 config: AudioConfig,
                 vad_factory: Callable[[], VADEngine] | None = None,
                 device_provider: AudioCaptureManager | None = None,
                 control_queue: ProcessQueue | None = None):
        self.output_queue = output_queue
        self.config = config
        self.vad_factory = vad_factory or (
            lambda: VADEngine(threshold=config.vad_threshold, sample_rate=config.sample_rate)
        )
        self.device_provider = device_provider or AudioCaptureManager()
        self.control_queue = control_queue

        self._stop_event = threading.Event()
        self._mic_queue = queue.Queue()
        self._sys_queue = queue.Queue()

    def _capture_worker(self, stream, target_queue: queue.Queue,
                        target_rate: int | None = None,
                        name: str = "unknown"):
        worker_log = logging.getLogger(f"bailiff.audio.worker.{name}")

        if stream is None:
            worker_log.warning("Stream is None, exiting")
            return

        channels = stream.channels
        source_rate = stream.sample_rate
        read_size = stream.frames_per_buffer

        needs_resample = bool(target_rate and source_rate != target_rate)
        effective_rate = target_rate if target_rate else source_rate
        preprocessor = AudioPreprocessor(sample_rate=effective_rate)

        worker_log.info("Started: channels=%d, read_size=%d, source_rate=%d, target_rate=%s, resample=%s",
                        channels, read_size, source_rate, target_rate, needs_resample)

        chunk_count = 0
        try:
            while not self._stop_event.is_set():
                try:
                    raw_data = stream.read(read_size)
                    data = np.frombuffer(raw_data, dtype=np.float32)

                    if channels > 1:
                        data = data.reshape(-1, channels).mean(axis=1)

                    if needs_resample:
                        data = resample(data, self.config.chunk_size).astype(np.float32)

                    data = preprocessor.process(data)

                    chunk_count += 1
                    if chunk_count % 100 == 1:
                        level = np.abs(data).max()
                        worker_log.debug("chunk #%d, shape=%s, level=%.4f, qsize=%d",
                                         chunk_count, data.shape, level, target_queue.qsize())

                    target_queue.put(data)
                except Exception as e:
                    if not self._stop_event.is_set():
                        worker_log.error("Capture error: %s", e, exc_info=True)
        finally:
            worker_log.info("Stopping after %d chunks", chunk_count)
            stream.close()

    def _control_listener(self):
        if self.control_queue is None:
            return
        try:
            while not self._stop_event.is_set():
                try:
                    msg = self.control_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if msg is None:
                    logger.info("Poison pill received on control queue")
                    self._stop_event.set()
                    return
        except Exception as exc:
            logger.debug("Control listener exiting: %s", exc)

    def _process_audio_stream(self, vad_engine):
        buffer = []
        silence_chunk = np.zeros(self.config.chunk_size, dtype=np.float32)
        process_count = 0
        speech_count = 0

        logger.info("Audio processing started: chunk_size=%d, sample_rate=%d, silence_limit=%.2fs",
                    self.config.chunk_size, self.config.sample_rate, self.config.silence_limit)

        while not self._stop_event.is_set():
            try:
                mic_data = self._mic_queue.get(timeout=1.0)
            except queue.Empty:
                logger.debug("Mic queue empty (timeout)")
                continue

            try:
                sys_data = self._sys_queue.get(timeout=1.0)
            except queue.Empty:
                sys_data = silence_chunk

            mixed_data = (mic_data + sys_data) / 2.0

            process_count += 1
            mic_level = np.abs(mic_data).max()
            sys_level = np.abs(sys_data).max()
            mix_level = np.abs(mixed_data).max()

            is_speech = vad_engine.is_speech(mixed_data)

            if process_count % 50 == 1:
                logger.debug("chunk #%d | mic=%.4f sys=%.4f mix=%.4f | speech=%s buffer=%d total_speech=%d",
                             process_count, mic_level, sys_level, mix_level,
                             is_speech, len(buffer), speech_count)

            if is_speech:
                speech_count += 1
                buffer.append(mixed_data)
                if len(buffer) == 1:
                    logger.info("Speech started (chunk #%d)", process_count)
            else:
                current_duration = len(buffer) * self.config.chunk_size / self.config.sample_rate

                if current_duration > self.config.silence_limit:
                    logger.info("Speech ended, flushing: %d chunks, %.2fs", len(buffer), current_duration)
                    self._flush_buffer(buffer)
                    buffer = []
                elif buffer:
                    buffer.append(mixed_data)
                    if len(buffer) > 10:
                        logger.debug("Buffer overflow (>10 silence chunks), clearing")
                        buffer = []

    def _flush_buffer(self, buffer):
        if not buffer:
            return

        full_audio = np.concatenate(buffer)
        duration = len(full_audio) / self.config.sample_rate
        timestamp = time.time() - duration

        rms = np.sqrt(np.mean(full_audio ** 2))
        peak = np.abs(full_audio).max()
        logger.info("Flushing buffer: %d chunks, %.2fs, peak=%.4f, rms=%.4f",
                    len(buffer), duration, peak, rms)

        audio_chunk = AudioChunk(
            data=full_audio,
            sample_rate=self.config.sample_rate,
            timestamp=timestamp,
            duration=duration,
            is_speech=True
        )

        self.output_queue.put(audio_chunk)
        logger.debug("AudioChunk queued (output_queue size ~%d)", self.output_queue.qsize())

    def run(self):
        logger.info("Starting service: sample_rate=%d, chunk_size=%d, vad_threshold=%.2f",
                    self.config.sample_rate, self.config.chunk_size, self.config.vad_threshold)

        vad_engine = self.vad_factory()
        provider = self.device_provider

        mic_stream = provider.open_microphone(self.config.sample_rate, self.config.chunk_size)
        sys_stream = provider.open_system_loopback(self.config.sample_rate, self.config.chunk_size)
        if sys_stream is None:
            logger.warning("No loopback device found, system audio will not be captured")

        mic_thread = threading.Thread(
            target=self._capture_worker,
            args=(mic_stream, self._mic_queue, None, "mic"),
            name="mic-capture",
        )
        sys_thread = threading.Thread(
            target=self._capture_worker,
            args=(sys_stream, self._sys_queue, self.config.sample_rate, "loopback"),
            name="sys-capture",
        )

        mic_thread.start()
        sys_thread.start()
        logger.info("Capture threads started")

        control_thread = None
        if self.control_queue is not None:
            control_thread = threading.Thread(
                target=self._control_listener, daemon=True, name="control-listener"
            )
            control_thread.start()

        try:
            self._process_audio_stream(vad_engine)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        finally:
            logger.info("Shutting down...")
            self._stop_event.set()
            mic_thread.join()
            sys_thread.join()
            if control_thread is not None:
                control_thread.join(timeout=1.0)
            provider.terminate()
            logger.info("Shutdown complete")


def run_ingest_service(output_queue: ProcessQueue,
                       config: AudioConfig,
                       q_health: ProcessQueue,
                       log_file: str | None = None,
                       control_queue: ProcessQueue | None = None):
    setup_logging(log_file=log_file)
    try:
        service = AudioIngestService(output_queue, config, control_queue=control_queue)
        service.run()
    except Exception as exc:
        logger.exception("Fatal error in %s", WORKER_NAME)
        try:
            q_health.put((WORKER_NAME, repr(exc)), timeout=1.0)
        except Exception:
            pass
    finally:
        try:
            output_queue.put(None, timeout=1.0)
        except Exception:
            pass


if __name__ == "__main__":
    setup_logging()

    audio_queue = ProcessQueue()
    health_queue = ProcessQueue()

    p = Process(target=run_ingest_service, args=(audio_queue, AudioConfig(), health_queue))
    p.start()

    try:
        while True:
            audio_chunk = audio_queue.get()
            if audio_chunk is None:
                break
            logger.info("Received audio chunk: duration=%.2fs, samples=%d",
                        audio_chunk.duration, len(audio_chunk.data))
    except KeyboardInterrupt:
        pass
    finally:
        p.terminate()
        p.join()
