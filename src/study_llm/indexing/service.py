"""
Background indexing service for StudyLLM.
Single-worker queue that indexes documents one at a time.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

from study_llm.indexing.pipeline import IndexingPipeline

logger = logging.getLogger(__name__)


class IndexingService:
    """Queues files for background indexing by a single worker thread."""

    def __init__(self, pipeline: IndexingPipeline):
        self.pipeline = pipeline
        self._q: "queue.Queue[Optional[Path]]" = queue.Queue()
        self._worker = None
        self._stop_event = threading.Event()
        self.cancel_requested = threading.Event()

    def start(self):
        """Start the background worker."""
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._run, name="studyllm-indexer", daemon=True
        )
        self._worker.start()

    def stop(self):
        """Stop the worker after draining the queue."""
        self._q.put(None)
        if self._worker:
            self._worker.join(timeout=10)
            self._worker = None

    def enqueue_file(self, path: Path):
        """Queue a file for indexing."""
        self._q.put(path)

    def enqueue_all(self, paths):
        for p in paths:
            self.enqueue_file(p)

    def pending_count(self) -> int:
        """Number of files waiting in the queue."""
        return self._q.qsize()

    def is_busy(self) -> bool:
        """True while a file is being indexed."""
        return getattr(self, "_busy", False)

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                self._busy = True
                self.pipeline.index_file(item)
            except Exception as e:
                logger.error(f"Indexing error for {item}: {e}")
            finally:
                self._busy = False
                self._q.task_done()


# Global instance
_service = None


def get_indexing_service(pipeline=None) -> IndexingService:
    global _service
    if pipeline is None:
        from study_llm.core.config import get_config
        default_pipeline = build_default_pipeline()
        _service = IndexingService(default_pipeline)
    else:
        IndexingService(pipeline)  # bug - fix below
        _service = IndexingService(pipeline)
    return _service
