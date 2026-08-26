"""
Reranker for StudyLLM RAG pipeline.
Optional second-stage reranking of retrieved chunks.
"""

import abc
import logging
from typing import List

from study_llm.storage.vector_store import SearchResult

logger = logging.getLogger(__name__)


class Reranker(abc.ABC):
    """Abstract interface for a reranker."""

    @abc.abstractmethod
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results by relevance to the query. Best first."""
        pass


class NoOpReranker(Reranker):
    """Pass-through reranker that preserves retrieval order."""

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        return results


class BGEReranker(Reranker):
    """BGE cross-encoder reranker via sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "auto"):
        self.model_name = model_name
        self._device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker: {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Reranker loaded")

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Score and reorder results with a cross-encoder."""
        if not results:
            return []

        self._load_model()
        texts = [r.text or "" for r in results]
        scores = self._model.predict([(query, t) for t in texts])
        ranked = sorted(
            zip(results, scores), key=lambda pair: pair[1], reverse=True
        )
        return [r for r, _ in ranked]


# Global instance
_reranker = None


def get_reranker(enabled: bool = False) -> Reranker:
    """Get global reranker (NoOp unless enabled)."""
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker() if enabled else NoOpReranker()
    return _reranker