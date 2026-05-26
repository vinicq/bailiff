import logging
from logging.handlers import RotatingFileHandler

import pytest

from bailiff.core import logging as bailiff_logging
from bailiff.core.logging import PIIFilter, SecretRedactingFormatter


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


def test_pii_filter_demotes_transcription_engine_records_to_debug(tmp_path, isolated_loggers):
    log_path = tmp_path / "pii.log"
    bailiff_logging.setup_logging(log_file=str(log_path))

    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    capture = Capture(level=logging.DEBUG)
    capture.addFilter(PIIFilter())

    engine_logger = logging.getLogger("bailiff.features.transcription.engine")
    engine_logger.addHandler(capture)
    try:
        engine_logger.info("raw transcript leaked here")
    finally:
        engine_logger.removeHandler(capture)

    assert captured, "expected the engine record to be captured"
    rec = captured[-1]
    assert rec.levelname == "DEBUG"
    assert rec.levelno == logging.DEBUG
    assert "raw transcript leaked here" in rec.getMessage()


def test_secret_str_is_redacted_in_formatted_message(tmp_path, isolated_loggers):
    log_path = tmp_path / "secret.log"
    bailiff_logging.setup_logging(log_file=str(log_path))

    logger = logging.getLogger("bailiff.test.secret")
    logger.warning("loaded config api_key=SecretStr('topsecret') ok")

    for h in logging.getLogger("bailiff").handlers:
        if isinstance(h, RotatingFileHandler):
            h.flush()

    content = log_path.read_text(encoding="utf-8")
    assert "topsecret" not in content
    assert "SecretStr(***)" in content


def test_unrelated_logger_records_pass_through_unchanged(tmp_path, isolated_loggers):
    log_path = tmp_path / "pass.log"
    bailiff_logging.setup_logging(log_file=str(log_path))

    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    capture = Capture(level=logging.DEBUG)
    capture.addFilter(PIIFilter())

    session_logger = logging.getLogger("bailiff.core.session")
    session_logger.addHandler(capture)
    try:
        session_logger.info("session started normally")
    finally:
        session_logger.removeHandler(capture)

    assert captured
    rec = captured[-1]
    assert rec.levelname == "INFO"
    assert rec.levelno == logging.INFO


def test_secret_redacting_formatter_directly():
    formatter = SecretRedactingFormatter(fmt="%(message)s")
    record = logging.LogRecord(
        name="bailiff.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="api_key=SecretStr('abc123') and token=SecretStr('xyz')",
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert "abc123" not in rendered
    assert "xyz" not in rendered
    assert rendered.count("SecretStr(***)") == 2
