"""
Indexing pipeline for StudyLLM.
parse -> chunk -> embed -> store (Qdrant) + metadata (SQLite).
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import List

from study_llm.documents.parser import parse_document
from study_llm.chunking.text_chunker import TextChunker
from study_llm.embeddings.embedding_provider import EmbeddingProvider
from study_llm.storage.metadata_db import DocumentRecord

logger = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    """SHA-256 of file content, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class IndexingPipeline:
    """Indexes a single document end-to-end."""

    def __init__(self,
                 embedding_provider: EmbeddingProvider,
                 vector_store,
                 metadata_db,
                 chunk_size: int = 512,
                 data_dir: str = "data",
                 version: str = "1"):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.chunker = TextChunker(chunk_size=chunk_size)
        self.data_dir = Path(data_dir).resolve()
        self.version = version

    def index_file(self, file_path: Path) -> bool:
        """
        Parse, chunk, embed and store one file.

        Args:
            file_path: Absolute path to the file to index.

        Returns:
            True on success, False on failure (document marked failed).
        """
        t0 = time.time()
        data_root = self.data_dir
        rel_path = str(file_path.relative_to(data_root)).replace("\\", "/")
        doc_id = _generate_document_id(rel_path)

        # Existing record for cleanup on re-index
        existing = self.metadata_db.get_document_by_id(doc_id)

        try:
            sha256 = _hash_file(file_path)
            stat = file_path.stat()
        except OSError as e:
            logger.error(f"Cannot read {file_path}: {e}")
            return False

        if existing and existing.status == "indexed" and existing.sha256 == sha256:
            logger.debug(f"Unchanged, skipping: {rel_path}")
            return True

        # Mark as indexing
        self.metadata_db.add_or_update_document(DocumentRecord(
            document_id=doc_id,
            relative_path=rel_path,
            filename=file_path.name,
            file_extension=file_path.suffix.lower(),
            sha256=sha256,
            file_size=stat.st_size,
            modified_time=stat.st_mtime,
            status="indexing",
        ))
        self._cleanup_old_vectors(doc_id)

        try:
            parsed = parse_document(file_path)
            if parsed.error or not (parsed.content or "").strip():
                raise ValueError(parsed.error or "no extractable text")

            chunks = self.chunker.chunk_document(parsed)

            if not chunks:
                raise ValueError("chunker produced no chunks")

            texts = [c.content for c in chunks]
            vectors = self.embedding_provider.embed_batch(texts)

            from study_llm.storage.vector_store import VectorRecord
            records = [
                VectorRecord(
                    chunk_id=c.chunk_id,
                    document_id=doc_id,
                    vector=v,
                    filename=file_path.name,
                    relative_path=rel_path,
                    page=c.page,
                    section=c.section,
                    content_hash=sha256,
                    text=c.content,
                )
                for c, v in zip(chunks, vectors)
            ]
            self.vector_store.ensure_collection(len(vectors[0]))
            self.vector_store.upsert(records)

            self.metadata_db.update_status(
                doc_id,
                status="indexed",
                chunk_count=len(chunks),
                embedding_model=self.embedding_provider.model_name
                if hasattr(self.embedding_provider, "model_name")
                else "unknown",
            )
            logger.info(
                f"Indexed {rel_path}: {len(chunks)} chunks "
                f"in {time.time() - t0:.1f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to index {rel_path}: {e}")
            self.metadata_db.update_status(doc_id, status="failed")
            return False

    def _cleanup_old_vectors(self, document_id: str) -> None:
        """Remove stale vectors before re-indexing a document."""
        count = self.vector_store.delete_document(document_id)
        if count:
            logger.debug(f"Removed {count} old vectors for {document_id}")


def _generate_document_id(relative_path: str) -> str:
    """Stable document ID derived from its path within data/."""
    return hashlib.md5(relative_path.encode("utf-8")).hexdigest()[:16]
