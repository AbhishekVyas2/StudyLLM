"""
Tests for the indexing pipeline, service, and startup reconciliation.
"""

import hashlib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from study_llm.indexing.pipeline import (
    IndexingPipeline, _hash_file, _generate_document_id
)
from study_llm.storage.metadata_db import MetadataDB


@pytest.fixture()
def env(tmp_path):
    """Pipeline with mock embedding + real temp Qdrant + SQLite."""
    from study_llm.storage.vector_store import QdrantVectorStore

    provider = Mock()
    provider.model_name = "mock-embed"
    provider.embed_batch.side_effect = lambda texts: [
        [float(len(t) % 7), 1.0, 0.0, 0.5] for t in texts
    ]

    store = QdrantVectorStore(path=str(tmp_path / "vdb"))
    db = MetadataDB(db_path=str(tmp_path / "meta.db"))
    pipe = IndexingPipeline(
        embedding_provider=provider,
        vector_store=store,
        metadata_db=db,
        chunk_size=128,
        data_dir=str(tmp_path),
    )
    return {
        "pipeline": pipe,
        "store": store,
        "db": db,
        "provider": provider,
        "data_dir": tmp_path,
    }


def test_hash_file_deterministic(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("hello", encoding="utf-8")
    f2.write_text("hello", encoding="utf-8")
    assert _hash_file(f1) == _hash_file(f2)
    assert _hash_file(f1) == hashlib.sha256(b"hello").hexdigest()


def test_doc_id_from_relpath():
    assert _generate_document_id("a/b.txt") == _generate_document_id("a/b.txt")
    assert _generate_document_id("a/b.txt") != _generate_document_id("c/d.txt")


def test_index_file_success(env, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Word " * 200, encoding="utf-8")

    ok = env["pipeline"].index_file(f)
    assert ok is True

    doc_id = _generate_document_id("doc.txt")
    rec = env["db"].get_document_by_id(doc_id)
    assert rec.status == "indexed"
    assert rec.chunk_count > 0
    assert env["store"].count_chunks_for_document(doc_id) == rec.chunk_count


def test_index_unchanged_skip(env, tmp_path):
    """Re-indexing identical content must not add vectors."""
    f = tmp_path / "doc.txt"
    f.write_text("Word " * 50, encoding="utf-8")
    env["pipeline"].index_file(f)
    before = env["store"].count_all()
    ok = env["pipeline"].index_file(f)
    assert ok is True
    assert env["store"].count_all() == before


def test_index_changed_content_reindexes(env, tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("First version. " * 30, encoding="utf-8")
    env["pipeline"].index_file(f)
    old_chunks = env["db"].get_document_by_id(
        _generate_document_id("doc.txt")).chunk_count

    f.write_text("Second version with entirely different text. " * 40,
                 encoding="utf-8")
    ok = env["pipeline"].index_file(f)
    assert ok is True

    rec = env["db"].get_document_by_id(_generate_document_id("doc.txt"))
    assert rec.chunk_count > 0
    # Old vectors replaced, no duplicates
    assert env["store"].count_chunks_for_document(rec.document_id)         == rec.chunk_count


def test_index_failure_marks_failed(env, tmp_path):
    from study_llm.documents.parser import ParsedDocument
    f = tmp_path / "bad.md"
    f.write_text("# Title\n\nSome content here. " * 20, encoding="utf-8")

    with patch(
        "study_llm.indexing.pipeline.parse_document",
        side_effect=ValueError("parse exploded"),
    ):
        ok = env["pipeline"].index_file(f)

    assert ok is False
    rec = env["db"].get_document_by_id(_generate_document_id("bad.md"))
    assert rec.status == "failed"


def test_service_processes_queue(env):
    from study_llm.indexing.service import IndexingService
    svc = IndexingService(env["pipeline"])
    f = env["data_dir"] / "q.txt"
    f.write_text("Queue test content. " * 100, encoding="utf-8")

    svc.start()
    svc.enqueue_file(f)
    # Wait for worker to drain
    for _ in range(100):
        if svc.pending_count() == 0 and not svc.is_busy():
            break
        time.sleep(0.05)
    svc.stop()

    rec = env["db"].get_document_by_id(_generate_document_id("q.txt"))
    assert rec is not None and rec.status == "indexed"


def test_reconciler_new_changed_deleted(env):
    from study_llm.indexing.reconciler import Reconciler

    a = env["data_dir"] / "a.txt"
    b = env["data_dir"] / "b.txt"
    c = env["data_dir"] / "c.txt"
    a.write_text("Alpha document content. " * 30, encoding="utf-8")
    b.write_text("Beta document content. " * 30, encoding="utf-8")
    env["pipeline"].index_file(a)
    env["pipeline"].index_file(b)
    c.write_text("Gamma doc. " * 10, encoding="utf-8")  # never indexed -> new

    # Simulate deletion of b from disk (remove record's file expectation)
    b.unlink()

    rec = Reconciler(env["data_dir"], env["db"], env["store"])
    result = rec.reconcile()
    relpaths = {
        str(p.relative_to(env["data_dir"])).replace("\\", "/")
        for p in result["new"]
    }

    assert "c.txt" in relpaths                        # c: new
    assert any(r.relative_path == "b.txt" for r in result["deleted"])  # deleted
    # a unchanged
    assert all(not str(p).endswith("a.txt") for p in result["new"])

    n = rec.apply_deletions(result["deleted"])
    assert n == 1
    assert env["db"].get_document_by_id(
        _generate_document_id("b.txt")) is None
    assert env["store"].count_chunks_for_document(
        _generate_document_id("b.txt")) == 0


import time  # used by queue test above
