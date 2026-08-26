"""
SQLite metadata database for StudyLLM.
Tracks document/index state for reliable incremental indexing.
"""

import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Document metadata stored in SQLite."""
    document_id: str
    relative_path: str
    filename: str
    file_extension: str
    sha256: str
    file_size: int
    modified_time: float
    indexed_at: Optional[str] = None
    status: str = "pending"  # pending | indexing | indexed | failed
    parser_version: str = ""
    chunking_version: str = ""
    embedding_model: str = ""
    chunk_count: int = 0


class MetadataDB:
    """SQLite-backed document metadata store."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "storage/metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self):
        """Create tables if they don't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    relative_path TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    file_extension TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    modified_time REAL NOT NULL,
                    indexed_at TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    parser_version TEXT DEFAULT '',
                    chunking_version TEXT DEFAULT '',
                    embedding_model TEXT DEFAULT '',
                    chunk_count INTEGER DEFAULT 0
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256)
            """)

    def add_or_update_document(self, doc: DocumentRecord) -> None:
        """Insert or update a document record."""
        with self.conn:
            self.conn.execute("""
                INSERT INTO documents (
                    document_id, relative_path, filename, file_extension,
                    sha256, file_size, modified_time, indexed_at, status,
                    parser_version, chunking_version, embedding_model, chunk_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    relative_path=excluded.relative_path,
                    filename=excluded.filename,
                    file_extension=excluded.file_extension,
                    sha256=excluded.sha256,
                    file_size=excluded.file_size,
                    modified_time=excluded.modified_time,
                    indexed_at=excluded.indexed_at,
                    status=excluded.status,
                    parser_version=excluded.parser_version,
                    chunking_version=excluded.chunking_version,
                    embedding_model=excluded.embedding_model,
                    chunk_count=excluded.chunk_count
            """, (
                doc.document_id, doc.relative_path, doc.filename,
                doc.file_extension, doc.sha256, doc.file_size,
                doc.modified_time, doc.indexed_at, doc.status,
                doc.parser_version, doc.chunking_version,
                doc.embedding_model, doc.chunk_count
            ))

    def get_document_by_path(self, relative_path: str) -> Optional[DocumentRecord]:
        """Get a document record by its relative path."""
        row = self.conn.execute(
            "SELECT * FROM documents WHERE relative_path = ?", (relative_path,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_document_by_id(self, document_id: str) -> Optional[DocumentRecord]:
        """Get a document record by its ID."""
        row = self.conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_all_documents(self) -> List[DocumentRecord]:
        """Get all document records."""
        rows = self.conn.execute("SELECT * FROM documents").fetchall()
        return [self._row_to_record(r) for r in rows]

    def delete_document(self, document_id: str) -> bool:
        """Delete a document record. Returns True if deleted."""
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
        return cursor.rowcount > 0

    def update_status(
        self,
        document_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        embedding_model: Optional[str] = None
    ) -> None:
        """Update a document's indexing status."""
        with self.conn:
            if chunk_count is not None and embedding_model is not None:
                self.conn.execute("""
                    UPDATE documents
                    SET status = ?, indexed_at = ?, chunk_count = ?, embedding_model = ?
                    WHERE document_id = ?
                """, (
                    status, datetime.now().isoformat(), chunk_count,
                    embedding_model, document_id
                ))
            else:
                self.conn.execute(
                    "UPDATE documents SET status = ? WHERE document_id = ?",
                    (status, document_id)
                )

    def count_documents(self) -> int:
        """Count documents in the database."""
        row = self.conn.execute("SELECT COUNT(*) as c FROM documents").fetchone()
        return row['c']

    def count_indexed_documents(self) -> int:
        """Count successfully indexed documents."""
        row = self.conn.execute(
            "SELECT COUNT(*) as c FROM documents WHERE status = 'indexed'"
        ).fetchone()
        return row['c']

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        """Convert a database row to a DocumentRecord."""
        return DocumentRecord(
            document_id=row['document_id'],
            relative_path=row['relative_path'],
            filename=row['filename'],
            file_extension=row['file_extension'],
            sha256=row['sha256'],
            file_size=row['file_size'],
            modified_time=row['modified_time'],
            indexed_at=row['indexed_at'],
            status=row['status'],
            parser_version=row['parser_version'] or "",
            chunking_version=row['chunking_version'] or "",
            embedding_model=row['embedding_model'] or "",
            chunk_count=row['chunk_count'] or 0
        )

    def close(self):
        """Close the database connection."""
        self.conn.close()


# Global instance
_metadata_db = None


def get_metadata_db(db_path: str = "storage/metadata.db") -> MetadataDB:
    """Get global metadata DB instance."""
    global _metadata_db
    if _metadata_db is None:
        _metadata_db = MetadataDB(db_path=db_path)
    return _metadata_db