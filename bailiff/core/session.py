import logging
import multiprocessing
import queue
import threading
from multiprocessing.queues import Queue as ProcessQueue

from bailiff.core.config import AudioConfig
from bailiff.core.db import get_session, init_db
from bailiff.features.assistant.service import run_assistant_service
from bailiff.features.audio_ingest.service import run_ingest_service
from bailiff.features.diarization.merge import run_merge_service
from bailiff.features.diarization.service import run_diarization_service
from bailiff.features.memory.service import run_memory_service
from bailiff.features.memory.storage import MeetingStorage
from bailiff.features.transcription.service import run_transcription_service

logger = logging.getLogger("bailiff.core.session")


class SessionManager:
    # q_health carries tuples of (worker_name: str, exception_repr: str) from children to the supervisor.
    def __init__(self, log_file="bailiff.log", graceful_timeout: float = 5.0):
        self.log_file = log_file
        self.graceful_timeout = graceful_timeout

        init_db()

        self.q_audio_raw: ProcessQueue = multiprocessing.Queue(maxsize=200)
        self.q_audio_tx: ProcessQueue = multiprocessing.Queue(maxsize=200)
        self.q_audio_diar: ProcessQueue = multiprocessing.Queue(maxsize=200)

        self.q_text: ProcessQueue = multiprocessing.Queue(maxsize=500)
        self.q_diarization: ProcessQueue = multiprocessing.Queue(maxsize=500)
        self.q_merged: ProcessQueue = multiprocessing.Queue(maxsize=500)

        self.q_memory: ProcessQueue = multiprocessing.Queue(maxsize=500)
        self.q_question: ProcessQueue = multiprocessing.Queue(maxsize=50)
        self.q_answer: ProcessQueue = multiprocessing.Queue(maxsize=50)
        self.q_rag: ProcessQueue = multiprocessing.Queue(maxsize=50)

        self.q_health: ProcessQueue = multiprocessing.Queue(maxsize=200)

        self.session_id = self._create_session()

        self._running = threading.Event()
        self._fanout_thread: threading.Thread | None = None
        self._supervisor_thread: threading.Thread | None = None
        self.processes: list[multiprocessing.Process] = []

    def _create_session(self):
        db = get_session()
        try:
            storage = MeetingStorage(db)
            session = storage.create_session()
            return session.id
        finally:
            db.close()

    def _audio_fanout(self):
        while self._running.is_set():
            try:
                chunk = self.q_audio_raw.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                self._safe_put(self.q_audio_tx, None)
                self._safe_put(self.q_audio_diar, None)
                break
            self._safe_put(self.q_audio_tx, chunk)
            self._safe_put(self.q_audio_diar, chunk)

    def _supervisor(self):
        while self._running.is_set():
            try:
                event = self.q_health.get(timeout=1.0)
            except queue.Empty:
                continue
            if event is None:
                break
            worker_name, exc_repr = event
            logger.critical("Fatal error in worker %s: %s", worker_name, exc_repr)
            threading.Thread(target=self.stop, daemon=True, name="supervisor-stop").start()
            break

    def start(self):
        self._running.set()

        self._fanout_thread = threading.Thread(
            target=self._audio_fanout, daemon=True, name="audio-fanout"
        )
        self._fanout_thread.start()

        self.processes = [
            multiprocessing.Process(
                target=run_ingest_service,
                args=(self.q_audio_raw, AudioConfig(), self.log_file),
                daemon=True,
                name="audio-ingest",
            ),
            multiprocessing.Process(
                target=run_transcription_service,
                args=(self.q_audio_tx, self.q_text, self.log_file),
                daemon=True,
                name="transcription",
            ),
            multiprocessing.Process(
                target=run_diarization_service,
                args=(self.q_audio_diar, self.q_diarization, self.log_file),
                daemon=True,
                name="diarization",
            ),
            multiprocessing.Process(
                target=run_merge_service,
                args=(self.q_text, self.q_diarization, self.q_merged, self.log_file),
                daemon=True,
                name="merge",
            ),
            multiprocessing.Process(
                target=run_memory_service,
                args=(self.q_memory, self.q_rag, self.session_id, self.log_file),
                daemon=True,
                name="memory",
            ),
            multiprocessing.Process(
                target=run_assistant_service,
                args=(self.q_question, self.q_answer, self.q_memory, self.q_rag, self.session_id, self.log_file),
                daemon=True,
                name="assistant",
            ),
        ]

        for p in self.processes:
            p.start()

        self._supervisor_thread = threading.Thread(
            target=self._supervisor, daemon=True, name="health-supervisor"
        )
        self._supervisor_thread.start()

    def stop(self):
        if not self._running.is_set():
            return
        self._running.clear()

        self._safe_put(self.q_audio_raw, None)
        self._safe_put(self.q_text, None)
        self._safe_put(self.q_diarization, None)
        self._safe_put(self.q_memory, None)
        self._safe_put(self.q_question, None)

        for p in self.processes:
            p.join(timeout=self.graceful_timeout)

        for p in self.processes:
            if p.is_alive():
                logger.warning("Process %s did not exit gracefully, terminating", p.name)
                p.terminate()
                p.join(timeout=2.0)

        if self._fanout_thread and self._fanout_thread.is_alive():
            self._fanout_thread.join(timeout=2.0)

        if self._supervisor_thread and self._supervisor_thread.is_alive():
            self._safe_put(self.q_health, None)
            self._supervisor_thread.join(timeout=2.0)

        for q in [
            self.q_audio_raw, self.q_audio_tx, self.q_audio_diar,
            self.q_text, self.q_diarization, self.q_merged,
            self.q_memory, self.q_question, self.q_answer, self.q_rag,
            self.q_health,
        ]:
            try:
                q.close()
                q.cancel_join_thread()
            except Exception as e:
                logger.error("Error closing queue: %s", e)

    @staticmethod
    def _safe_put(q: ProcessQueue, item) -> None:
        try:
            q.put(item, timeout=1.0)
        except Exception:
            pass
