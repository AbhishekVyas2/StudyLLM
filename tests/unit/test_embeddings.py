"""
Tests for embeddings.
"""

from unittest.mock import Mock, patch
from study_llm.embeddings.embedding_provider import (
    EmbeddingProvider,
    BGEM3EmbeddingProvider,
    CachedEmbeddingProvider
)


class _FakeArray:
    """Minimal stand-in for a numpy array: only .shape is accessed."""

    def __init__(self, shape):
        self.shape = shape


def test_embedding_provider_interface():
    """Test that EmbeddingProvider is abstract."""
    import pytest
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_cached_embedding_provider_basic():
    """Test basic caching functionality."""
    # Create mock provider
    mock_provider = Mock()
    mock_provider.get_dimension.return_value = 384
    mock_provider.embed_batch.return_value = [[0.1, 0.2, 0.3]]

    cached = CachedEmbeddingProvider(
        mock_provider,
        cache_file=None  # Use memory only
    )

    # First call - should hit provider
    result1 = cached.embed("test text")
    assert result1 == [0.1, 0.2, 0.3]
    mock_provider.embed_batch.assert_called_once()

    # Reset mock
    mock_provider.reset_mock()

    # Second call with same text - should use cache
    result2 = cached.embed("test text")
    assert result2 == [0.1, 0.2, 0.3]
    mock_provider.embed_batch.assert_not_called()


def test_cached_embedding_provider_different_texts():
    """Test that different texts get different embeddings."""
    mock_provider = Mock()
    mock_provider.get_dimension.return_value = 384
    mock_provider.embed_batch.side_effect = [
        [[0.1, 0.2, 0.3]],
        [[0.4, 0.5, 0.6]]
    ]

    cached = CachedEmbeddingProvider(mock_provider, cache_file=None)

    result1 = cached.embed("text one")
    result2 = cached.embed("text two")

    assert result1 == [0.1, 0.2, 0.3]
    assert result2 == [0.4, 0.5, 0.6]
    assert result1 != result2


def test_bge_provider_import_error():
    """Test that BGE provider handles missing dependencies gracefully."""
    import sys
    saved = sys.modules.pop('sentence_transformers', None)
    try:
        # Simulate sentence_transformers not being installed
        sys.modules['sentence_transformers'] = None
        provider = BGEM3EmbeddingProvider()
        try:
            provider.get_dimension()
            assert False, "Expected an error when model unavailable"
        except Exception:
            pass  # Expected (ImportError or similar)
    finally:
        # Restore prior state exactly, avoiding module-entry leakage
        if saved is None:
            sys.modules.pop('sentence_transformers', None)
        else:
            sys.modules['sentence_transformers'] = saved


def test_bge_provider_init():
    """Test BGE provider initialization without loading real numpy/torch."""
    import sys
    import types

    mock_instance = Mock()
    mock_instance.encode.return_value = _FakeArray((1, 1024))

    # Create a fake sentence_transformers module whose SentenceTransformer
    # class returns our mock instance.
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = Mock(return_value=mock_instance)

    saved = sys.modules.get('sentence_transformers')
    sys.modules['sentence_transformers'] = fake_module
    try:
        provider = BGEM3EmbeddingProvider()
        # get_dimension triggers model load
        dim = provider.get_dimension()

        assert dim == 1024
        fake_module.SentenceTransformer.assert_called_once()
    finally:
        if saved is None:
            sys.modules.pop('sentence_transformers', None)
        else:
            sys.modules['sentence_transformers'] = saved
