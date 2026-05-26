import os
from pathlib import Path

import pytest

from bailiff.core.events import TranscriptionSegment


def _segment(text="hello world", start=0.0):
    return TranscriptionSegment(
        text=text,
        start_time=start,
        end_time=start + 1.0,
        duration=1.0,
    )


def test_persistent_client_uses_path_kwarg(tmp_path):
    from bailiff.features.memory.vector_db import VectorMemory

    store_path = tmp_path / "chroma_abs"
    vm = VectorMemory(persist_path=store_path)
    vm.add_segment("s1", _segment("alpha beta gamma"))

    assert store_path.exists()
    assert store_path.is_dir()
    assert any(store_path.iterdir())
    assert vm.persist_path == store_path.resolve()


def test_relative_persist_path_resolves_under_data_dir(tmp_path, monkeypatch):
    from bailiff.core.config import settings
    from bailiff.features.memory.vector_db import VectorMemory

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings.app, "data_dir", str(data_dir))

    vm = VectorMemory(persist_path="chroma")
    vm.add_segment("s1", _segment("hello"))

    expected = (data_dir / "chroma").resolve()
    assert vm.persist_path == expected
    assert expected.exists()
    assert any(expected.iterdir())


def test_offline_mode_raises_when_embedder_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("BAILIFF_OFFLINE", "1")

    import bailiff.features.memory.vector_db as vdb

    fake_cache = tmp_path / "empty_cache"
    fake_cache.mkdir()
    monkeypatch.setattr(vdb, "_onnx_cache_present", lambda: False)

    with pytest.raises(RuntimeError, match="offline"):
        vdb.VectorMemory(persist_path=tmp_path / "chroma_offline")
