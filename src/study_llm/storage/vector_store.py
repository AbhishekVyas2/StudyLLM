"""
Vector store abstraction and Qdrant implementation for StudyLLM.
Handles storage, retrieval, and deterministic deletion of document vectors.
"""

import abc
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def stable_point_id(chunk_id: str) -> int:
    """
    Derive a deterministic unsigned 64-bit Qdrant point ID from a chunk ID.

    Python's built-in hash() is randomized per process, so it must never be
    used for persisted IDs. This uses SHA-256 truncated to 64 bits instead.
    """
    digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass
class VectorRecord:
    """A vector with its payload metadata."""
    chunk_id: str
    document_id: str
    vector: List[float]
    filename: str
    relative_path: str = ""
    page: Optional[int] = None
    section: Optional[str] = None
    content_hash: str = ""
    text: str = ""


@dataclass
class SearchResult:
    """A search result from the vector store."""
    chunk_id: str
    document_id: str
    score: float
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    text: str = ""
    relative_path: str = ""


class VectorStore(abc.ABC):
    """Abstract interface for a vector store."""

    @abc.abstractmethod
    def ensure_collection(self, dimension: int):
        """Ensure the collection exists with the given embedding dimension."""
        pass

    @abc.abstractmethod
    def upsert(self, records: List[VectorRecord]):
        """Insert or update vectors."""
        pass

    @abc.abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete all vectors belonging to a document. Returns number deleted."""
        pass

    @abc.abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass

    @abc.abstractmethod
    def count_chunks_for_document(self, document_id: str) -> int:
        """Count vectors belonging to a document."""
        pass

    @abc.abstractmethod
    def count_all(self) -> int:
        """Count all vectors in the collection."""
        pass

    @abc.abstractmethod
    def close(self):
        """Release resources."""
        pass


def _get_qdrant():
    """Import and return qdrant_client components lazily."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            VectorParams,
            PointStruct,
            Filter,
            FieldCondition,
            MatchValue,
        )
        return QdrantClient, Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    except ImportError as e:
        raise ImportError(
            "qdrant-client is not installed. Install with: pip install qdrant-client"
        ) from e


class QdrantVectorStore(VectorStore):
    """Qdrant-based local vector store."""

    COLLECTION_NAME = "studyllm"

    def __init__(self, path: str = "storage/vectordb"):
        """
        Initialize local Qdrant vector store.

        Args:
            path: Directory path for persistent Qdrant storage
        """
        QdrantClient, Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue = _get_qdrant()

        self._Distance = Distance
        self._VectorParams = VectorParams
        self._PointStruct = PointStruct
        self._Filter = Filter
        self._FieldCondition = FieldCondition
        self._MatchValue = MatchValue

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))
        self._dimension: Optional[int] = None

    def ensure_collection(self, dimension: int):
        """Ensure the collection exists with the given embedding dimension."""
        self._dimension = dimension
        if not self.client.collection_exists(self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=self._VectorParams(
                    size=dimension,
                    distance=self._Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection '{self.COLLECTION_NAME}' (dim={dimension})")

    def upsert(self, records: List[VectorRecord]):
        """Insert or update vectors."""
        if not records:
            return

        if self._dimension is None:
            self.ensure_collection(len(records[0].vector))

        points = [
            self._PointStruct(
                id=stable_point_id(r.chunk_id),
                vector=r.vector,
                payload={
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "filename": r.filename,
                    "relative_path": r.relative_path,
                    "page": r.page,
                    "section": r.section,
                    "content_hash": r.content_hash,
                    "text": r.text,
                }
            )
            for r in records
        ]

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points[i:i + batch_size]
            )

        logger.debug(f"Upserted {len(records)} vectors")

    def delete_document(self, document_id: str) -> int:
        """Delete all vectors belonging to a document. Returns count deleted."""
        count = self.count_chunks_for_document(document_id)
        if count == 0:
            return 0

        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=self._Filter(
                must=[
                    self._FieldCondition(
                        key="document_id",
                        match=self._MatchValue(value=document_id)
                    )
                ]
            )
        )

        logger.info(f"Deleted {count} vectors for document {document_id}")
        return count

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        query_filter = None
        if document_ids:
            query_filter = self._Filter(
                must=[
                    self._FieldCondition(
                        key="document_id",
                        match=self._MatchValue(value=did)
                    ) for did in document_ids
                ]
            )

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )

        return [
            SearchResult(
                chunk_id=p.payload.get("chunk_id", ""),
                document_id=p.payload.get("document_id", ""),
                score=p.score,
                filename=p.payload.get("filename", ""),
                page=p.payload.get("page"),
                section=p.payload.get("section"),
                text=p.payload.get("text", ""),
                relative_path=p.payload.get("relative_path", "")
            )
            for p in results.points
        ]

    def _collection_ready(self) -> bool:
        """True if the collection exists."""
        return self.client.collection_exists(self.COLLECTION_NAME)

    def count_chunks_for_document(self, document_id: str) -> int:
        """Count vectors belonging to a document."""
        if not self._collection_ready():
            return 0
        result = self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=self._Filter(
                must=[
                    self._FieldCondition(
                        key="document_id",
                        match=self._MatchValue(value=document_id)
                    )
                ]
            ),
            exact=True
        )
        return result.count

    def count_all(self) -> int:
        """Count all vectors in the collection."""
        if not self._collection_ready():
            return 0
        result = self.client.count(collection_name=self.COLLECTION_NAME, exact=True)
        return result.count

    def close(self):
        """Release resources."""
        try:
            self.client.close()
        except Exception:
            pass


# Global instance
_vector_store = None


def get_vector_store(path: str = "storage/vectordb") -> QdrantVectorStore:
    """Get global Qdrant vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore(path=path)
    return _vector_store