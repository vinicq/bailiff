import pytest

from bailiff.core.events import TranscriptionSegment
from bailiff.features.memory.storage import MeetingStorage


def _segment(text, start, end, speaker="Speaker 0"):
    return TranscriptionSegment(
        text=text,
        start_time=start,
        end_time=end,
        duration=end - start,
        speaker=speaker,
    )


def test_create_session_assigns_id_and_name(in_memory_db):
    storage = MeetingStorage(in_memory_db)
    session = storage.create_session(name="Standup")
    assert session.id is not None
    assert session.name == "Standup"


def test_save_and_query_transcripts(in_memory_db):
    storage = MeetingStorage(in_memory_db)
    session = storage.create_session(name="Test")

    storage.save_transcript(session.id, _segment("first thing", 0.0, 1.0))
    storage.save_transcript(session.id, _segment("second thing", 1.0, 2.0, "Speaker 1"))

    transcripts = storage.get_transcripts(session.id)
    assert len(transcripts) == 2
    assert transcripts[0].text == "first thing"
    assert transcripts[0].speaker == "Speaker 0"
    assert transcripts[1].text == "second thing"
    assert transcripts[1].speaker == "Speaker 1"


def test_get_session_by_id(in_memory_db):
    storage = MeetingStorage(in_memory_db)
    created = storage.create_session(name="Lookup")
    fetched = storage.get_session(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == "Lookup"


def test_get_session_returns_none_for_missing(in_memory_db):
    storage = MeetingStorage(in_memory_db)
    assert storage.get_session(99999) is None


def test_transcripts_isolated_per_session(in_memory_db):
    storage = MeetingStorage(in_memory_db)
    s1 = storage.create_session(name="A")
    s2 = storage.create_session(name="B")

    storage.save_transcript(s1.id, _segment("alpha", 0.0, 1.0))
    storage.save_transcript(s2.id, _segment("beta", 0.0, 1.0))

    a_only = storage.get_transcripts(s1.id)
    b_only = storage.get_transcripts(s2.id)
    assert [t.text for t in a_only] == ["alpha"]
    assert [t.text for t in b_only] == ["beta"]
