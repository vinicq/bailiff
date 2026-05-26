import pytest
import chromadb
from chromadb.utils import embedding_functions


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(name="meeting_context", embedding_function=ef)


def _upsert(collection, session_id, doc_id, text, speaker="Speaker 0", start=0.0, end=1.0):
    collection.upsert(
        documents=[text],
        metadatas=[{"session_id": session_id, "speaker": speaker, "start_time": start, "end_time": end}],
        ids=[doc_id],
    )


def test_session_id_isolation(collection):
    _upsert(collection, "s1", "s1_1", "alpha beta about cats")
    _upsert(collection, "s1", "s1_2", "dogs love bones")
    _upsert(collection, "s2", "s2_1", "alpha beta about cats")

    results = collection.query(
        query_texts=["alpha beta"],
        n_results=5,
        where={"session_id": "s1"},
    )

    ids = results["ids"][0]
    assert "s1_1" in ids
    assert "s2_1" not in ids
    for meta in results["metadatas"][0]:
        assert meta["session_id"] == "s1"


def test_n_results_caps_returned_documents(collection):
    for i in range(6):
        _upsert(collection, "sx", f"sx_{i}", f"document number {i} about beta")

    results = collection.query(query_texts=["beta"], n_results=3, where={"session_id": "sx"})

    assert len(results["documents"][0]) == 3
    assert len(results["ids"][0]) == 3


def test_query_returns_in_relevance_order(collection):
    _upsert(collection, "ord", "ord_match", "exact match for quantum mechanics topic")
    _upsert(collection, "ord", "ord_far", "weather forecast tomorrow")
    _upsert(collection, "ord", "ord_mid", "physics class about waves")

    results = collection.query(
        query_texts=["quantum mechanics"],
        n_results=3,
        where={"session_id": "ord"},
    )

    ids = results["ids"][0]
    assert ids[0] == "ord_match"
    distances = results["distances"][0]
    assert distances == sorted(distances)
