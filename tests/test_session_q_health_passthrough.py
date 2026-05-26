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


def test_session_manager_passes_q_health_to_every_process(fresh_session_manager):
    sm = fresh_session_manager
    specs = sm._build_processes()
    assert len(specs) == 6

    expected_names = {"audio-ingest", "transcription", "diarization", "merge", "memory", "assistant"}
    names = {name for name, _t, _a in specs}
    assert names == expected_names

    for name, _target, args in specs:
        assert sm.q_health in args, f"{name} does not receive q_health"
        idx = args.index(sm.q_health)
        assert args[idx + 1] == sm.log_file, f"{name} has q_health not immediately before log_file"
