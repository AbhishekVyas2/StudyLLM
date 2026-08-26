"""
Tests for the SQLite metadata database.
"""

import pytest
from study_llm.storage.metadata_db import MetadataDB, DocumentRecord


@pytest.fixture()
def db(tmp_path):
    """Create a metadata DB in a temp directory."""
    database = MetadataDB(db_path=str(tmp_path / "metadata.db"))
    yield database
    database.close()


def make_doc(
    document_id: str = "doc1",
    relative_path: str = "doc1.txt",
    sha256: str = "a" * 64,
    status: str = "indexed",
    chunk_count: int = 3
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        relative_path=relative_path,
        filename="doc1.txt",
        file_extension=".txt",
        sha256=sha256,
        file_size=1234,
        modified_time=1700000000.0,
        status=status,
        parser_version="docling-1.0",
        chunking_version="1",
        embedding_model="BAAI/bge-m3",
        chunk_count=chunk_count
    )


def test_schema_created(db):
    """Schema is created automatically on init."""
    tables = [
        r[0] for r in
        db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    assert "documents" in tables


def test_add_and_get_by_path(db):
    doc = make_doc()
    db.add_or_update_document(doc)

    fetched = db.get_document_by_path("doc1.txt")
    assert fetched is not None
    assert fetched.document_id == "doc1"
    assert fetched.sha256 == "a" * 64
    assert fetched.status == "indexed"
    assert fetched.chunk_count == 3


def test_add_and_get_by_id(db):
    same_name_different_subdir = make_doc(
        document_id="doc2", relative_path="subdir/doc1.txt"
    )
    db.add_or_update_document(make_doc())
    db.add_or_update_document(same_name_different_subdir)

    fetched = db.get_document_by_id("doc2")
    assert fetched is not None
    assert fetched.relative_path == "subdir/doc1.txt"


def test_update_status_indexed_sets_fields(db):
    db.add_or_update_document(make_doc(status="pending"))
    db.update_status(
        "doc1",
        status="indexed",
        chunk_count=7,
        embedding_model="bge-m3"
    )
    fetched = db.get_document_by_id("doc1")
    assert fetched.status == "indexed"
    assert fetched.indexed_at is not None
    assert fetched.chunk_count == 7
    assert fetched.embedding_model == "bge-m3"


def test_update_status_simple_form(db):
    db.add_or_update_document(make_doc(status="pending"))
    db.update_status("doc1", status="failed")
    assert db.get_document_by_id("doc1").status == "failed"


def test_delete_document_removes_row(db):
    db.add_or_update_document(make_doc())
    assert db.delete_document("doc1") is True
    assert db.get_document_by_id("doc1") is None
    assert db.delete_document("no-such-doc") is False


def test_count_queries(db):
    db.add_or_update_document(make_doc(document_id="d1", relative_path="d1.txt"))
    db.add_or_update_document(
        make_doc(document_id="d2", relative_path="d2.txt", status="pending")
    )
    assert db.count_documents() == 2
    assert db.count_indexed_documents() == 1