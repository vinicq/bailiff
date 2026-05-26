import logging
from logging.handlers import RotatingFileHandler

import pytest

from bailiff.core import logging as bailiff_logging


@pytest.fixture
def isolated_loggers(monkeypatch):
    root = logging.getLogger()
    bailiff = logging.getLogger("bailiff")
    saved_root = list(root.handlers)
    saved_bailiff = list(bailiff.handlers)
    monkeypatch.delenv("BAILIFF_TUI_MODE", raising=False)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_root:
        root.addHandler(h)
    for h in list(bailiff.handlers):
        bailiff.removeHandler(h)
    for h in saved_bailiff:
        bailiff.addHandler(h)


def test_worker_mode_attaches_only_rotating_file_handler(tmp_path, isolated_loggers):
    log_path = tmp_path / "worker.log"
    bailiff_logging.setup_logging(log_file=str(log_path))

    bailiff_logger = logging.getLogger("bailiff")
    handler_types = [type(h).__name__ for h in bailiff_logger.handlers]
    assert "RotatingFileHandler" in handler_types
    assert "TextualHandler" not in handler_types


def test_rotating_file_handler_uses_size_cap(tmp_path, isolated_loggers):
    log_path = tmp_path / "size.log"
    bailiff_logging.setup_logging(log_file=str(log_path))

    bailiff_logger = logging.getLogger("bailiff")
    rotating = [h for h in bailiff_logger.handlers if isinstance(h, RotatingFileHandler)]
    assert rotating, "expected at least one RotatingFileHandler"
    assert rotating[0].maxBytes == 10 * 1024 * 1024
    assert rotating[0].backupCount == 3


def test_tui_mode_via_kwarg_attaches_textual_handler(tmp_path, isolated_loggers):
    log_path = tmp_path / "tui.log"
    bailiff_logging.setup_logging(log_file=str(log_path), is_tui=True)

    bailiff_logger = logging.getLogger("bailiff")
    handler_types = [type(h).__name__ for h in bailiff_logger.handlers]
    assert "TextualHandler" in handler_types


def test_tui_mode_via_env_var_attaches_textual_handler(tmp_path, monkeypatch, isolated_loggers):
    log_path = tmp_path / "tui-env.log"
    monkeypatch.setenv("BAILIFF_TUI_MODE", "1")
    bailiff_logging.setup_logging(log_file=str(log_path))

    bailiff_logger = logging.getLogger("bailiff")
    handler_types = [type(h).__name__ for h in bailiff_logger.handlers]
    assert "TextualHandler" in handler_types


def test_setup_logging_creates_parent_dir(tmp_path, isolated_loggers):
    nested = tmp_path / "deep" / "nested" / "logs" / "bailiff.log"
    assert not nested.parent.exists()

    bailiff_logging.setup_logging(log_file=str(nested))

    assert nested.parent.is_dir()


def test_worker_name_is_embedded_in_formatter(tmp_path, isolated_loggers):
    log_path = tmp_path / "named.log"
    bailiff_logging.setup_logging(log_file=str(log_path), worker_name="transcription")

    bailiff_logger = logging.getLogger("bailiff")
    rotating = next(h for h in bailiff_logger.handlers if isinstance(h, RotatingFileHandler))
    assert "transcription" in rotating.formatter._fmt
