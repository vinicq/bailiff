import os
import sys
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def settings_with_tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BAILIFF_APP__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)

    import bailiff.core.config as cfg
    monkeypatch.setattr(cfg, "_settings_instance", None)
    settings = cfg.load_settings()
    yield settings
    monkeypatch.setattr(cfg, "_settings_instance", None)


@pytest.fixture
def ephemeral_chroma():
    import chromadb
    client = chromadb.EphemeralClient()
    yield client


@pytest.fixture
def in_memory_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from bailiff.features.memory.models import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture
def clean_config_env(monkeypatch, tmp_path):
    for key in list(os.environ.keys()):
        if key.startswith("BAILIFF_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    import bailiff.core.config as cfg
    monkeypatch.setattr(cfg, "_settings_instance", None)
    yield cfg
    monkeypatch.setattr(cfg, "_settings_instance", None)
