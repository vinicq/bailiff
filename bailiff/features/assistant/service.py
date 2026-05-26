import logging
import queue
from multiprocessing.queues import Queue as ProcessQueue

from bailiff.core.logging import setup_logging
from bailiff.features.assistant.llm import LLMClient, LLMClientSettings
from bailiff.features.assistant.rag import RagEngine

logger = logging.getLogger("bailiff.assistant.service")


class AssistantService:
    """
    Background service that handles user questions using RAG and LLM.

    This service listens for questions on the question queue, retrieves relevant context using the
    RAG engine (communicating with MemoryService), and produces answers via the LLM.
    """
    def __init__(self,
        question_queue: ProcessQueue,
        answer_queue: ProcessQueue,
        memory_queue: ProcessQueue,
        rag_queue: ProcessQueue,
        session_id: int
    ):
        self.question_queue = question_queue
        self.answer_queue = answer_queue
        self.memory_queue = memory_queue
        self.rag_queue = rag_queue
        self.rag_engine = None
        self.llm = None
        self.vector_db = None
        self.session_id = str(session_id)

    def run(self):
        from bailiff.core.config import settings

        api_key = settings.models.llm_api_key.get_secret_value() if settings.models.llm_api_key else None
        base_url = settings.models.llm_base_url
        model = settings.models.llm_assistant

        if api_key is None and settings.models.llm_provider != "ollama":
            raise RuntimeError("llm_api_key is required when llm_provider != 'ollama'")

        if not model:
            logger.error("LLM Model not configured.")
            return

        llm_settings = LLMClientSettings(api_key=api_key, base_url=base_url, model=model)

        self.llm = LLMClient(llm_settings)
        self.rag_engine = RagEngine(llm=self.llm, memory_queue=self.memory_queue, rag_queue=self.rag_queue)

        while True:
            try:
                question = self.question_queue.get(timeout=0.1)
                if question is None:
                    break

                logger.info(f"Thinking about question: {question}")

                answer = self.rag_engine.answer_question(question, session_id=self.session_id)
                self.answer_queue.put(answer)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Error answering question: %s", e)
                continue


def run_assistant_service(
    question_queue: ProcessQueue,
    answer_queue: ProcessQueue,
    memory_queue: ProcessQueue,
    rag_queue: ProcessQueue,
    session_id: int,
    q_health: ProcessQueue,
    log_file: str,
):
    setup_logging(log_file=log_file, worker_name="assistant")
    try:
        service = AssistantService(question_queue, answer_queue, memory_queue, rag_queue, session_id)
        service.run()
    except Exception as exc:
        try:
            q_health.put(("assistant", repr(exc)), timeout=1.0)
        except Exception:
            pass
        raise
