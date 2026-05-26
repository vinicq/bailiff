import logging
from multiprocessing.queues import Queue as ProcessQueue

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

from bailiff.core.events import AudioChunk, DiarizationResult

logger = logging.getLogger("bailiff.features.diarization.engine")


class DiarizationEngine:
    """
    Speaker diarization via SpeechBrain ECAPA-TDNN Speaker Embedding.

    Receives AudioChunk objects from `audio_queue`, extracts embeddings,
    clusters them using a simple algorithm with cosine similarity,
    and pushes DiarizationResult objects to `output_queue`.
    """

    def __init__(
        self,
        audio_queue: ProcessQueue,
        output_queue: ProcessQueue,
        model_source: str,
        threshold: float,
        inertia_weight: float,
        max_speakers: int = 16,
    ):
        self.audio_queue = audio_queue
        self.output_queue = output_queue
        self.threshold = threshold
        self.inertia_weight = inertia_weight
        self.max_speakers = max_speakers

        logger.info("Initializing SpeechBrain Speaker Embedding with model: %s", model_source)

        try:
            self.classifier = EncoderClassifier.from_hparams(
                source=model_source,
                run_opts={"device": "cpu"},
            )
            logger.info("SpeechBrain Classifier initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize SpeechBrain Classifier: %s", e)
            raise

        self.speakers: dict[str, dict] = {}
        self.next_id = 0
        self.last_speaker: str | None = None

    def _compute_embedding(self, audio_chunk: AudioChunk):
        samples = audio_chunk.data
        if samples.ndim > 1:
            samples = samples[0] if samples.shape[0] < samples.shape[1] else samples[:, 0]

        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        signal = torch.from_numpy(samples).unsqueeze(0)

        with torch.no_grad():
            embeddings = self.classifier.encode_batch(signal)

        return embeddings.squeeze().cpu().numpy()

    def identify(self, audio_chunk: AudioChunk) -> str:
        emb = self._compute_embedding(audio_chunk)
        if emb is None:
            return "unknown"

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        else:
            return "unknown"

        best_speaker = None
        max_similarity = -1.0

        for spk_id, data in self.speakers.items():
            centroid = data["embedding"]
            sim = float(np.dot(emb, centroid))

            if spk_id == self.last_speaker:
                sim += self.inertia_weight

            if sim > max_similarity:
                max_similarity = sim
                best_speaker = spk_id

        at_cap = len(self.speakers) >= self.max_speakers
        if max_similarity > self.threshold or (at_cap and best_speaker is not None):
            count = self.speakers[best_speaker]["count"]
            old_emb = self.speakers[best_speaker]["embedding"]

            new_emb = (old_emb * count + emb) / (count + 1)
            new_norm = np.linalg.norm(new_emb)
            if new_norm > 0:
                new_emb = new_emb / new_norm

            self.speakers[best_speaker]["embedding"] = new_emb
            self.speakers[best_speaker]["count"] += 1

            self.last_speaker = best_speaker
            logger.debug("Matched %s with similarity %.4f", best_speaker, max_similarity)
            return best_speaker

        new_name = f"Speaker {self.next_id}"
        self.speakers[new_name] = {"count": 1, "embedding": emb}
        logger.debug(
            "New speaker %s created. Max similarity was %.4f (Threshold: %.2f)",
            new_name,
            max_similarity,
            self.threshold,
        )
        self.next_id += 1
        self.last_speaker = new_name
        return new_name

    def run(self):
        logger.info("Diarization engine running (SpeechBrain ECAPA-TDNN)")

        while True:
            chunk: AudioChunk = self.audio_queue.get()
            if chunk is None:
                logger.info("Received poison pill, stopping diarization engine.")
                break

            speaker = self.identify(chunk)

            result = DiarizationResult(
                speaker=speaker,
                start_time=chunk.timestamp,
                end_time=chunk.timestamp + chunk.duration,
            )

            logger.debug("Speaker %s [%.2f-%.2f]", speaker, result.start_time, result.end_time)
            self.output_queue.put(result)

        logger.info("Diarization engine finished")
