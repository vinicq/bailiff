"""Logging setup. Workers call setup_logging(); only the TUI calls setup_logging(is_tui=True).

Setting env var BAILIFF_TUI_MODE=1 has the same effect as is_tui=True for callers that
cannot pass kwargs (e.g. the UI process bootstrap before SessionManager exists).
"""
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from bailiff.core.config import settings


_SECRET_STR_PATTERN = re.compile(r"SecretStr\('[^']*'\)")
_PII_LOGGER_PREFIXES = ("transcription.engine", "diarization.engine")
_PII_MESSAGE_MARKERS = ("transcript=", "text=")


class PIIFilter(logging.Filter):
    """Demote records that may carry transcript / diarization PII to DEBUG."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno <= logging.DEBUG:
            return True

        name = record.name or ""
        message = record.getMessage()

        is_pii_logger = any(prefix in name for prefix in _PII_LOGGER_PREFIXES)
        is_pii_marker = any(marker in message for marker in _PII_MESSAGE_MARKERS)

        if is_pii_logger or is_pii_marker:
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True


class SecretRedactingFormatter(logging.Formatter):
    """Replace SecretStr('...') occurrences in the rendered message with SecretStr(***)."""

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return _SECRET_STR_PATTERN.sub("SecretStr(***)", rendered)


def setup_logging(
    level: int = logging.DEBUG,
    log_file: str | None = None,
    is_tui: bool = False,
    worker_name: str | None = None,
) -> None:
    if os.environ.get("BAILIFF_TUI_MODE") == "1":
        is_tui = True

    log_path = Path(log_file) if log_file else Path(settings.app.data_dir) / settings.app.log_file
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(processName)s] %(levelname)s %(name)s: %(message)s"
    if worker_name:
        fmt = f"%(asctime)s [%(processName)s:{worker_name}] %(levelname)s %(name)s: %(message)s"
    formatter = SecretRedactingFormatter(fmt=fmt, datefmt="%H:%M:%S")

    pii_filter = PIIFilter()

    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    file_handler.addFilter(pii_filter)

    handlers: list[logging.Handler] = [file_handler]
    if is_tui:
        from textual.logging import TextualHandler

        tui_handler = TextualHandler()
        tui_handler.setFormatter(formatter)
        tui_handler.setLevel(level)
        tui_handler.addFilter(pii_filter)
        handlers.append(tui_handler)

    bailiff_logger = logging.getLogger("bailiff")
    bailiff_logger.setLevel(level)
    bailiff_logger.propagate = False
    for h in list(bailiff_logger.handlers):
        bailiff_logger.removeHandler(h)
    for h in handlers:
        bailiff_logger.addHandler(h)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    root.setLevel(logging.WARNING)
