import logging
from multiprocessing import Queue as ProcessQueue
from typing import Callable

from bailiff.core.config import settings
from bailiff.core.logging import setup_logging
from bailiff.features.diarization.engine import DiarizationEngine

logger = logging.getLogger("bailiff.features.diarization.service")


class DiarizationService:
    """
    Service wrapper for running the DiarizationEngine.
    """
    def __init__(
        self,
        input_queue: ProcessQueue,
        output_queue: ProcessQueue,
        engine_factory: Callable[..., DiarizationEngine] | None = None,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.engine_factory = engine_factory or (
            lambda iq, oq: DiarizationEngine(
                iq,
                oq,
                model_source=settings.models.voice_embedding,
                threshold=settings.diarization.threshold,
                inertia_weight=settings.diarization.inertia_weight,
                max_speakers=settings.diarization.max_speakers,
            )
        )

    def run(self):
        logger.info("Starting diarization service")
        engine = self.engine_factory(self.input_queue, self.output_queue)
        logger.info("Diarization engine initialized, streaming...")
        engine.run()
        logger.info("Diarization service stopped")


def run_diarization_service(
    input_queue: ProcessQueue,
    output_queue: ProcessQueue,
    q_health: ProcessQueue,
    log_file: str | None = None,
):
    setup_logging(log_file=log_file, worker_name="diarization")
    try:
        service = DiarizationService(input_queue, output_queue)
        service.run()
    except Exception as exc:
        try:
            q_health.put(("diarization", repr(exc)), timeout=1.0)
        except Exception:
            pass
        raise
