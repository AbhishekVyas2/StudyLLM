"""
Embedding provider abstraction for StudyLLM.
Provides an interface for generating embeddings from text.
"""

import abc
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingProvider(abc.ABC):
    """Abstract base class for embedding providers."""

    @abc.abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of embeddings produced by this provider."""
        pass

    @abc.abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass

    @abc.abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass


class BGEM3EmbeddingProvider(EmbeddingProvider):
    """BGE-M3 embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto", cache_dir: Optional[Path] = None):
        """
        Initialize BGE-M3 embedding provider.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cpu', 'cuda', or 'auto')
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path("storage/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._dimension = None
        self._device = self._resolve_device(device)

    def _resolve_device(self, device: str) -> str:
        """Resolve device setting."""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                return "cpu"
            except ImportError:
                return "cpu"
        return device

    def _load_model(self):
        """Lazily load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name} on {self._device}")
                self._model = SentenceTransformer(self.model_name, device=self._device)
                # Get dimension from model
                test_embedding = self._model.encode(["test"], convert_to_numpy=True)
                self._dimension = test_embedding.shape[1]
                logger.info(f"Embedding model loaded. Dimension: {self._dimension}")
            except ImportError as e:
                logger.error("sentence-transformers not installed. "
                             "Install with: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        self._load_model()

        # BGE-M3 uses normalized embeddings by default
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32
        )

        return embeddings.tolist()

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_batch([text])[0]


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wrapper that adds caching to any embedding provider."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_file: Optional[Path] = Path("storage/embeddings/cache.json")
    ):
        """
        Initialize cached embedding provider.

        Args:
            provider: The underlying embedding provider
            cache_file: Path to cache file (None for memory-only)
        """
        self.provider = provider
        self.cache_file = cache_file
        self._cache: dict = {}
        if self.cache_file is not None:
            self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load embedding cache from disk."""
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save embedding cache to disk."""
        if self.cache_file is None:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.provider.get_dimension()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with caching."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        to_embed = []
        to_embed_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_embed.append(text)
                to_embed_indices.append(i)

        # Embed uncached texts
        if to_embed:
            new_embeddings = self.provider.embed_batch(to_embed)

            for idx, embedding in zip(to_embed_indices, new_embeddings):
                key = self._cache_key(texts[idx])
                self._cache[key] = embedding
                results[idx] = embedding

            # Save cache
            self._save_cache()

        return results

    def embed(self, text: str) -> List[float]:
        """Generate embedding with caching."""
        return self.embed_batch([text])[0]


# Global instances
_bge_provider = None
_cached_provider = None


def get_bge_embedding_provider() -> BGEM3EmbeddingProvider:
    """Get global BGE-M3 embedding provider."""
    global _bge_provider
    if _bge_provider is None:
        _bge_provider = BGEM3EmbeddingProvider()
    return _bge_provider


def get_cached_embedding_provider() -> CachedEmbeddingProvider:
    """Get global cached embedding provider."""
    global _cached_provider
    if _cached_provider is None:
        _cached_provider = CachedEmbeddingProvider(get_bge_embedding_provider())
    return _cached_provider