import multiprocessing
import threading
import time

import pytest


@pytest.fixture
def fresh_session_manager(tmp_path, monkeypatch):
    from bailiff.core.config import settings
    import bailiff.core.db as db
    monkeypatch.setattr(settings.app, "data_dir", str(tmp_path / "data"))
    for name in ("engine", "SessionLocal"):
        if name in db.__dict__:
            monkeypatch.delitem(db.__dict__, name, raising=False)

    from bailiff.core.session import SessionManager
    sm = SessionManager(log_file=str(tmp_path / "bailiff.log"))
    yield sm
    sm.stop()


def test_audio_queues_are_bounded(fresh_session_manager):
    sm = fresh_session_manager
    for q in (sm.q_audio_raw, sm.q_audio_tx, sm.q_audio_diar):
        assert q._maxsize == 200


def test_text_queues_are_bounded(fresh_session_manager):
    sm = fresh_session_manager
    for q in (sm.q_text, sm.q_diarization, sm.q_merged, sm.q_memory):
        assert q._maxsize == 500


def test_assistant_queues_are_bounded(fresh_session_manager):
    sm = fresh_session_manager
    for q in (sm.q_question, sm.q_answer, sm.q_rag):
        assert q._maxsize == 50


def test_health_channel_exists_and_is_bounded(fresh_session_manager):
    sm = fresh_session_manager
    assert sm.q_health is not None
    assert sm.q_health._maxsize == 200


def test_stop_is_idempotent_before_start(fresh_session_manager):
    fresh_session_manager.stop()
    fresh_session_manager.stop()


def test_fanout_propagates_poison_pill_to_both_consumers(fresh_session_manager):
    sm = fresh_session_manager
    sm._running.set()

    fanout = threading.Thread(target=sm._audio_fanout, daemon=True, name="fanout-test")
    fanout.start()

    sm.q_audio_raw.put(None)
    fanout.join(timeout=3.0)

    assert sm.q_audio_tx.get(timeout=1.0) is None
    assert sm.q_audio_diar.get(timeout=1.0) is None


def test_fanout_duplicates_chunks_to_both_consumers(fresh_session_manager):
    sm = fresh_session_manager
    sm._running.set()

    fanout = threading.Thread(target=sm._audio_fanout, daemon=True, name="fanout-test")
    fanout.start()

    sm.q_audio_raw.put("chunk-a")
    sm.q_audio_raw.put("chunk-b")
    sm.q_audio_raw.put(None)
    fanout.join(timeout=3.0)

    tx_items = [sm.q_audio_tx.get(timeout=1.0) for _ in range(3)]
    diar_items = [sm.q_audio_diar.get(timeout=1.0) for _ in range(3)]
    assert tx_items == ["chunk-a", "chunk-b", None]
    assert diar_items == ["chunk-a", "chunk-b", None]


def test_session_manager_persists_a_session_row(fresh_session_manager):
    from bailiff.core.db import get_session
    from bailiff.features.memory.storage import MeetingStorage

    db = get_session()
    try:
        storage = MeetingStorage(db)
        session = storage.get_session(fresh_session_manager.session_id)
        assert session is not None
        assert session.id == fresh_session_manager.session_id
    finally:
        db.close()
