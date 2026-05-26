import pytest


@pytest.fixture
def reset_db_globals(monkeypatch):
    import bailiff.core.db as db
    for name in ("engine", "SessionLocal"):
        if name in db.__dict__:
            monkeypatch.delitem(db.__dict__, name, raising=False)
    yield db


def test_init_db_creates_data_dir_if_missing(tmp_path, monkeypatch, reset_db_globals):
    from bailiff.core.config import settings
    target = tmp_path / "fresh-data-dir"
    monkeypatch.setattr(settings.app, "data_dir", str(target))
    assert not target.exists()

    reset_db_globals.init_db()

    assert target.is_dir()
    assert (target / "bailiff.db").exists()


def test_init_db_is_idempotent(tmp_path, monkeypatch, reset_db_globals):
    from bailiff.core.config import settings
    monkeypatch.setattr(settings.app, "data_dir", str(tmp_path / "data"))

    reset_db_globals.init_db()
    engine_first = reset_db_globals.engine

    reset_db_globals.init_db()
    engine_second = reset_db_globals.engine

    assert engine_first is engine_second


def test_session_local_triggers_lazy_init(tmp_path, monkeypatch, reset_db_globals):
    from bailiff.core.config import settings
    monkeypatch.setattr(settings.app, "data_dir", str(tmp_path / "data"))

    assert "SessionLocal" not in reset_db_globals.__dict__

    SessionLocal = reset_db_globals.SessionLocal

    assert SessionLocal is not None
    assert "SessionLocal" in reset_db_globals.__dict__
    assert "engine" in reset_db_globals.__dict__


def test_get_session_returns_usable_session(tmp_path, monkeypatch, reset_db_globals):
    from bailiff.core.config import settings
    monkeypatch.setattr(settings.app, "data_dir", str(tmp_path / "data"))

    session = reset_db_globals.get_session()
    try:
        from sqlalchemy import text
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1
    finally:
        session.close()


def test_getattr_raises_for_unknown_name(reset_db_globals):
    with pytest.raises(AttributeError):
        _ = reset_db_globals.not_a_real_attribute
