"""
Startup reconciliation for StudyLLM.
Compares the filesystem (source of truth) with metadata DB and vector store.
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {
    ".txt", ".text", ".md", ".markdown", ".pdf", ".docx",
    ".pptx", ".xlsx", ".html", ".csv"
}


class Reconciler:
    """Reconcile data/ with SQLite + Qdrant at startup."""

    def __init__(self, data_root: Path, metadata_db, vector_store):
        self.data_root = Path(data_root).resolve()
        self.metadata_db = metadata_db
        self.vector_store = vector_store

    def scan_filesystem(self) -> List[Path]:
        """All supported files currently in data/."""
        files = []
        for p in sorted(self.data_root.rglob("*")):
            if (p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
                    and not p.name.startswith("~$")):
                files.append(p)
        return files

    def reconcile(self) -> dict:
        """
        Compare filesystem with metadata DB.

        Returns:
            {"new": [...], "changed": [...], "deleted": [...], "unchanged": N}
        """
        fs_files = self.scan_filesystem()
        fs_by_relpath = {
            str(p.relative_to(self.data_root)).replace("\\", "/"): p
            for p in fs_files
        }

        new, changed, deleted, unchanged = [], [], [], 0

        from study_llm.indexing.pipeline import _hash_file, _generate_document_id
        for rel_path, path in fs_by_relpath.items():
            doc_id = _generate_document_id(rel_path)
            record = self.metadata_db.get_document_by_id(doc_id)

            if record is None:
                new.append(path)
                continue

            if record.status in ("pending", "failed") or record.sha256 != _hash_file(path):
                changed.append(path)
                continue

            vec_count = self.vector_store.count_chunks_for_document(doc_id)
            if vec_count == 0:
                changed.append(path)
                continue

            unchanged += 1

        for record in self.metadata_db.get_all_documents():
            if record.relative_path not in fs_by_relpath:
                deleted.append(record)

        return {
            "new": new,
            "changed": changed,
            "deleted": deleted,
            "unchanged": unchanged,
        }

    def apply_deletions(self, deleted_docs: list) -> int:
        """Delete vectors + metadata for documents no longer on disk."""
        total = 0
        for record in deleted_docs:
            try:
                n_vec = self.vector_store.delete_document(record.document_id)
            except Exception as e:
                logger.error(
                    f"Failed to delete vectors for {record.relative_path}: {e}"
                )
                continue
            self.metadata_db.delete_document(record.document_id)
            logger.info(f"Reconciler removed {record.relative_path} ({n_vec} vectors)")
            total += 1
        return total
