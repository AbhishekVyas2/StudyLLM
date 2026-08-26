"""
Retriever for StudyLLM RAG pipeline.
Embeds the query and searches the vector store.
"""

import logging
from typing import List, Optional

from study_llm.embeddings.embedding_provider import (
    EmbeddingProvider,
    CachedEmbeddingProvider,
)
from study_llm.storage.vector_store import SearchResult, get_vector_store

logger = logging.getLogger(__name__)


class Retriever:
    """Query embedding + vector search over indexed documents."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store=None,
        top_k: int = 10
    ):
        """
        Initialize retriever.

        Args:
            embedding_provider: Provider for query embeddings
            vector_store: Vector store instance (default global Qdrant store)
            top_k: Number of chunks to retrieve before reranking
        """
        from study_llm.storage.vector_store import get_vector_store  # avoid circular import at module load
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store or get_vector_store()
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Retrieve the most similar chunks for a query.

        Args:
            query: User's question
            top_k: Override the configured number of results
            document_ids: Restrict search to these documents

        Returns:
            List of SearchResults, best first
        """
        if not query.strip():
            return []

        k = top_k or self.top_k
        query_vector = self.embedding_provider.embed(query)

        try:
            results = self.vector_store.search(
                query_vector=query_vector,
                top_k=k,
                document_ids=document_ids
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        logger.debug(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
        return results