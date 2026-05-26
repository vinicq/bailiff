import multiprocessing

import pytest


def test_transcription_publishes_fatal_to_q_health(tmp_path, monkeypatch):
    import bailiff.features.transcription.service as svc
    from bailiff.features.transcription import engine as eng

    def _boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(eng.WhisperEngine, "__init__", _boom)

    q_in = multiprocessing.Queue()
    q_out = multiprocessing.Queue()
    q_health = multiprocessing.Queue()
    log_file = str(tmp_path / "bailiff.log")

    with pytest.raises(RuntimeError, match="boom"):
        svc.run_transcription_service(q_in, q_out, q_health, log_file)

    name, exc_repr = q_health.get(timeout=2.0)
    assert name == "transcription"
    assert exc_repr == "RuntimeError('boom')"


def test_diarization_publishes_fatal_to_q_health(tmp_path, monkeypatch):
    import bailiff.features.diarization.service as svc

    def _boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(svc.DiarizationService, "run", _boom)

    q_in = multiprocessing.Queue()
    q_out = multiprocessing.Queue()
    q_health = multiprocessing.Queue()
    log_file = str(tmp_path / "bailiff.log")

    with pytest.raises(RuntimeError, match="boom"):
        svc.run_diarization_service(q_in, q_out, q_health, log_file)

    name, exc_repr = q_health.get(timeout=2.0)
    assert name == "diarization"
    assert exc_repr == "RuntimeError('boom')"


def test_memory_publishes_fatal_to_q_health(tmp_path, monkeypatch):
    import bailiff.features.memory.service as svc

    def _boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(svc.MemoryService, "run", _boom)

    q_in = multiprocessing.Queue()
    q_rag = multiprocessing.Queue()
    q_health = multiprocessing.Queue()
    log_file = str(tmp_path / "bailiff.log")

    with pytest.raises(RuntimeError, match="boom"):
        svc.run_memory_service(q_in, q_rag, 1, q_health, log_file)

    name, exc_repr = q_health.get(timeout=2.0)
    assert name == "memory"
    assert exc_repr == "RuntimeError('boom')"


def test_assistant_publishes_fatal_to_q_health(tmp_path, monkeypatch):
    import bailiff.features.assistant.service as svc

    def _boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(svc.AssistantService, "run", _boom)

    q_question = multiprocessing.Queue()
    q_answer = multiprocessing.Queue()
    q_memory = multiprocessing.Queue()
    q_rag = multiprocessing.Queue()
    q_health = multiprocessing.Queue()
    log_file = str(tmp_path / "bailiff.log")

    with pytest.raises(RuntimeError, match="boom"):
        svc.run_assistant_service(q_question, q_answer, q_memory, q_rag, 1, q_health, log_file)

    name, exc_repr = q_health.get(timeout=2.0)
    assert name == "assistant"
    assert exc_repr == "RuntimeError('boom')"
