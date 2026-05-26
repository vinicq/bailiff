import numpy as np
import pytest

from bailiff.core.events import AudioChunk
from bailiff.features.diarization.engine import DiarizationEngine


def _make_engine(threshold=0.5, inertia_weight=0.1, max_speakers=16):
    engine = DiarizationEngine.__new__(DiarizationEngine)
    engine.audio_queue = None
    engine.output_queue = None
    engine.threshold = threshold
    engine.inertia_weight = inertia_weight
    engine.max_speakers = max_speakers
    engine.classifier = None
    engine.speakers = {}
    engine.next_id = 0
    engine.last_speaker = None
    return engine


def _chunk():
    return AudioChunk(
        data=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        timestamp=0.0,
        duration=1.0,
    )


def _patch_embedding(engine, vector):
    vec = np.asarray(vector, dtype=np.float32)
    engine._compute_embedding = lambda _chunk: vec


def test_near_identical_embeddings_get_same_id():
    engine = _make_engine(threshold=0.5)

    _patch_embedding(engine, [1.0, 0.0, 0.0])
    first = engine.identify(_chunk())

    _patch_embedding(engine, [0.99, 0.05, 0.0])
    second = engine.identify(_chunk())

    assert first == second == "Speaker 0"


def test_orthogonal_embeddings_get_different_ids():
    engine = _make_engine(threshold=0.5)

    _patch_embedding(engine, [1.0, 0.0, 0.0])
    first = engine.identify(_chunk())

    _patch_embedding(engine, [0.0, 1.0, 0.0])
    second = engine.identify(_chunk())

    assert first == "Speaker 0"
    assert second == "Speaker 1"


def test_inertia_bias_keeps_previous_speaker_in_tiebreak():
    engine = _make_engine(threshold=0.5, inertia_weight=0.2)

    _patch_embedding(engine, [1.0, 0.0, 0.0])
    engine.identify(_chunk())

    _patch_embedding(engine, [0.0, 1.0, 0.0])
    engine.identify(_chunk())

    assert engine.last_speaker == "Speaker 1"

    _patch_embedding(engine, [0.6, 0.6, 0.0])
    third = engine.identify(_chunk())

    assert third == "Speaker 1"


def test_max_speakers_cap_routes_to_closest_existing():
    engine = _make_engine(threshold=0.9, max_speakers=2)

    _patch_embedding(engine, [1.0, 0.0, 0.0])
    a = engine.identify(_chunk())
    _patch_embedding(engine, [0.0, 1.0, 0.0])
    b = engine.identify(_chunk())

    assert a == "Speaker 0"
    assert b == "Speaker 1"
    assert len(engine.speakers) == 2

    _patch_embedding(engine, [0.0, 0.0, 1.0])
    c = engine.identify(_chunk())

    assert c in {"Speaker 0", "Speaker 1"}
    assert len(engine.speakers) == 2
