import logging
import os
from collections import deque
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from bailiff.core.config import settings
from bailiff.core.events import TranscriptionSegment

logger = logging.getLogger("bailiff.memory.vector_db")


def _onnx_cache_present() -> bool:
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    extracted = ONNXMiniLM_L6_V2.DOWNLOAD_PATH / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME
    return extracted.exists() and any(extracted.iterdir())


def _build_embedding_function():
    if os.environ.get("BAILIFF_OFFLINE") == "1" and not _onnx_cache_present():
        raise RuntimeError(
            "Chroma default embedder requires network; pre-cache or set BAILIFF_OFFLINE=0 (offline mode)"
        )
    return embedding_functions.DefaultEmbeddingFunction()


class VectorMemory:
    """
    Manages semantic storage and retrieval using ChromaDB.

    Handles embedding and storage of transcript segments for vector-based similarity search.
    Maintains a rolling context window to cluster short segments before embedding.
    """
    MAX_SEGMENT_LENGTH = 500

    def __init__(self, persist_path: str | Path = "./chromadb"):
        path = Path(persist_path)
        if not path.is_absolute():
            path = Path(settings.app.data_dir) / path
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)

        self.persist_path = path
        self.client = chromadb.PersistentClient(path=str(path))
        self.embedding_fn = _build_embedding_function()

        self.collection = self.client.get_or_create_collection(
            name="meeting_context",
            embedding_function=self.embedding_fn,
        )

        self.context_window = deque(maxlen=10)
        self.last_session_id = None

    def add_segment(self, session_id: str, segment: TranscriptionSegment):
        if self.last_session_id != session_id:
            self.context_window.clear()
            self.last_session_id = session_id

        truncated_text = segment.text[:self.MAX_SEGMENT_LENGTH]
        self.context_window.append(truncated_text)
        context_text = "\n".join(self.context_window)

        timestamp_ms = int(segment.start_time * 1000)
        doc_id = f"{session_id}_{timestamp_ms}"

        logger.info(f"Adding segment '{segment.text}' to session '{session_id}'")

        self.collection.upsert(
            documents=[context_text],
            metadatas=[{"session_id": session_id, "speaker": segment.speaker, "start_time": segment.start_time, "end_time": segment.end_time}],
            ids=[doc_id]
        )

        logger.info(f"Added segment '{segment.text}' to session '{session_id}' with ID '{doc_id}'")

        return doc_id

    def search(self, query: str, session_id: str | None = None, k: int = 5) -> list[str]:
        logger.info(f"Searching for '{query}' in session '{session_id}'")

        where = {"session_id": session_id} if session_id else None

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where
        )

        logger.info(f"Found {len(results['documents'][0])} results")

        return results['documents'][0]
