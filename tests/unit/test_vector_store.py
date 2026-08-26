"""
Tests for the vector store.
Uses real Qdrant in local persistent mode with a temp directory.
"""

import pytest
from study_llm.storage.vector_store import (
    VectorStore,
    QdrantVectorStore,
    VectorRecord,
    stable_point_id,
)


@pytest.fixture()
def store(tmp_path):
    """Create a Qdrant vector store in a temp directory."""
    vs = QdrantVectorStore(path=str(tmp_path / "vectordb"))
    yield vs
    vs.close()


def make_record(chunk_id: str, document_id: str, text: str = "") -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        vector=[1.0, 0.0, 0.0] if "docA" in chunk_id else [0.0, 1.0, 0.0],
        filename="test.txt",
        relative_path="test.txt",
        page=1,
        section="Intro",
        content_hash="abc123",
        text=text or f"content of {chunk_id}",
    )


def test_stable_point_id_is_deterministic():
    """Point IDs must be identical across calls (and processes)."""
    id1 = stable_point_id("doc-0001")
    id2 = stable_point_id("doc-0001")
    assert id1 == id2
    assert stable_point_id("doc-0002") != id1


def test_stable_point_id_fits_qdrant_range():
    """Qdrant requires unsigned 64-bit IDs."""
    for cid in ["doc-0001", "x", "", "ünïcødé-chunk"]:
        pid = stable_point_id(cid)
        assert 0 <= pid < 2**64


def test_vector_store_interface():
    """VectorStore is abstract."""
    with pytest.raises(TypeError):
        VectorStore()


def test_upsert_and_count(store):
    records = [
        make_record("docA-0001", "docA"),
        make_record("docA-0002", "docA"),
        make_record("docB-0001", "docB"),
    ]
    store.upsert(records)

    assert store.count_all() == 3
    assert store.count_chunks_for_document("docA") == 2
    assert store.count_chunks_for_document("docB") == 1


def test_upsert_is_idempotent(store):
    """Re-upserting same chunk IDs must not duplicate vectors."""
    records = [
        make_record("docA-0001", "docA"),
        make_record("docA-0002", "docA"),
    ]
    store.upsert(records)
    store.upsert(records)

    assert store.count_all() == 2


def test_search_returns_relevant_first(store):
    store.upsert([
        make_record("docA-0001", "docA"),   # vector [1, 0, 0]
        make_record("docB-0001", "docB"),   # vector [0, 1, 0]
    ])
    store.ensure_collection(3)  # no-op if exists

    results = store.search([1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "docA-0001"
    assert results[0].score >= results[1].score


def test_search_filters_by_document_ids(store):
    store.upsert([
        make_record("docA-0001", "docA"),
        make_record("docB-0001", "docB"),
    ])

    results = store.search([1.0, 0.0, 0.0], top_k=5, document_ids=["docB"])

    assert len(results) == 1
    assert results[0].document_id == "docB"


def test_delete_document_removes_only_that_document(store):
    store.upsert([
        make_record("docA-0001", "docA"),
        make_record("docA-0002", "docA"),
        make_record("docB-0001", "docB"),
    ])

    deleted = store.delete_document("docA")

    assert deleted == 2
    assert store.count_chunks_for_document("docA") == 0
    assert store.count_chunks_for_document("docB") == 1
    assert store.count_all() == 1

    # Deleting again returns 0
    assert store.delete_document("docA") == 0


def test_delete_nonexistent_document(store):
    assert store.delete_document("no-such-doc") == 0


def test_ensure_collection_dimension_change(store):
    """ensure_collection on an existing collection must not wipe data."""
    store.upsert([make_record("docA-0001", "docA")])
    store.ensure_collection(768)

    assert store.count_all() == 1