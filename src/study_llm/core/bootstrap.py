"""
Bootstrap for StudyLLM: builds the full runtime stack.
Wires config -> embeddings -> vector store -> metadata -> indexing -> watcher.
"""

import logging
from pathlib import Path

from study_llm.core.config import get_config

logger = logging.getLogger(__name__)


class Runtime:
    """Assembled StudyLLM runtime components."""

    def __init__(self):
        self.config = get_config()
        self.embedding_provider = None
        self.vector_store = None
        self.metadata_db = None
        self.pipeline = None
        self.service = None
        self.watcher = None
        self.reconciler = None

    def build(self):
        """Construct all components."""
        from study_llm.embeddings.embedding_provider import (
            CachedEmbeddingProvider, BGEM3EmbeddingProvider,
        )
        from study_llm.storage.vector_store import QdrantVectorStore
        from study_llm.storage.metadata_db import MetadataDB
        from study_llm.indexing.pipeline import IndexingPipeline
        from study_llm.indexing.service import IndexingService
        from study_llm.indexing.reconciler import Reconciler
        from study_llm.indexing.watcher import Watcher

        data_dir = "data"

        # Embeddings (cached on disk)
        base = BGEM3EmbeddingProvider(
            model_name=self.config.embedding_model
        )
        self.embedding_provider = CachedEmbeddingProvider(base)

        # Storage
        self.vector_store = QdrantVectorStore(path=self.config.vector_db_path)
        self.metadata_db = MetadataDB(db_path="storage/metadata.db")

        # Indexing
        self.pipeline = IndexingPipeline(
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            metadata_db=self.metadata_db,
            chunk_size=self.config.chunk_size,
            data_dir=data_dir,
        )
        self.service = IndexingService(self.pipeline)

        # Lifecycle
        self.reconciler = Reconciler(data_dir, self.metadata_db,
                                     self.vector_store)
        self.watcher = Watcher(
            data_dir,
            on_index=self.service.enqueue_file,
            on_delete=self._handle_delete,
        )
        return self

    def _handle_delete(self, relative_path: str):
        """A file was deleted from data/ - remove its knowledge."""
        try:
            doc_id = self.pipeline._generate_document_id(relative_path) \
                if hasattr(self.pipeline, "_generate_document_id") else None
            if doc_id is None:
                from study_llm.indexing.pipeline import _generate_document_id
                doc_id = _generate_document_id(relative_path)
            n = self.vector_store.delete_document(doc_id)
            self.metadata_db.delete_document(doc_id)
            logger.info(f"Deleted {relative_path}: removed {n} vectors")
        except Exception as e:
            logger.error(f"Delete handling failed for {relative_path}: {e}")

    def reconcile_and_enqueue(self) -> dict:
        """Run startup reconciliation, enqueue new/changed, apply deletions."""
        result = self.reconciler.reconcile()
        self.reconciler.apply_deletions(result["deleted"])
        for p in result["new"] + result["changed"]:
            self.service.enqueue_file(p)
        return result

    def start_background(self):
        """Start the indexing worker and filesystem watcher."""
        self.service.start()
        self.watcher.start()

    def shutdown(self):
        """Stop background work and release resources."""
        try:
            self.watcher.stop()
        except Exception:
            pass
        try:
            self.service.stop()
        except Exception:
            pass
        try:
            self.vector_store.close()
        except Exception:
            pass
        try:
            self.metadata_db.close()
        except Exception:
            pass


_runtime = None


def get_runtime() -> Runtime:
    """Build (once) and return the global runtime."""
    global _runtime
    if _runtime is None:
        _runtime = Runtime().build()
    return _runtime
