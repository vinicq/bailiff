import multiprocessing
import queue
import threading
import time

import pytest

from bailiff.core.events import DiarizationResult, TranscriptionSegment
from bailiff.features.diarization.merge import MergeService


def _drain_until(q, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return q.get(timeout=0.2)
        except queue.Empty:
            continue
    raise AssertionError("output queue stayed empty")


def test_matching_diar_then_segment_assigns_speaker():
    tx_q = multiprocessing.Queue()
    diar_q = multiprocessing.Queue()
    out_q = multiprocessing.Queue()

    service = MergeService(tx_q, diar_q, out_q, merge_timeout=8.0, segment_timeout=3.0)
    thread = threading.Thread(target=service.run, daemon=True)
    thread.start()

    diar_q.put(DiarizationResult(speaker="Speaker 0", start_time=0.0, end_time=2.0))
    tx_q.put(TranscriptionSegment(text="hello", start_time=0.5, end_time=1.0, duration=0.5))

    result = _drain_until(out_q)

    assert result.text == "hello"
    assert result.speaker == "Speaker 0"

    tx_q.put(None)
    thread.join(timeout=3.0)


def test_segment_without_diar_falls_back_to_unknown():
    tx_q = multiprocessing.Queue()
    diar_q = multiprocessing.Queue()
    out_q = multiprocessing.Queue()

    service = MergeService(tx_q, diar_q, out_q, merge_timeout=8.0, segment_timeout=0.3)
    thread = threading.Thread(target=service.run, daemon=True)
    thread.start()

    tx_q.put(TranscriptionSegment(text="orphan", start_time=10.0, end_time=11.0, duration=1.0))

    result = _drain_until(out_q, timeout=3.0)

    assert result.text == "orphan"
    assert result.speaker == "unknown"

    tx_q.put(None)
    thread.join(timeout=3.0)


def test_non_overlapping_diar_yields_unknown():
    tx_q = multiprocessing.Queue()
    diar_q = multiprocessing.Queue()
    out_q = multiprocessing.Queue()

    service = MergeService(tx_q, diar_q, out_q, merge_timeout=8.0, segment_timeout=0.3)
    thread = threading.Thread(target=service.run, daemon=True)
    thread.start()

    diar_q.put(DiarizationResult(speaker="Speaker 0", start_time=0.0, end_time=1.0))
    tx_q.put(TranscriptionSegment(text="late", start_time=5.0, end_time=6.0, duration=1.0))

    result = _drain_until(out_q, timeout=3.0)

    assert result.text == "late"
    assert result.speaker == "unknown"

    tx_q.put(None)
    thread.join(timeout=3.0)
