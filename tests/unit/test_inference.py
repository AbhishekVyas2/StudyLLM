"""
Tests for LLM inference components.
"""

import pytest
from unittest.mock import Mock, patch
from study_llm.inference.llama_cpp import LlamaCppInference


def test_llama_cpp_import_error():
    """Test that ImportError is raised when llama-cpp-python is not available."""
    with patch('study_llm.inference.llama_cpp.LLAMA_CPP_AVAILABLE', False):
        with pytest.raises(ImportError, match="llama-cpp-python is not installed"):
            LlamaCppInference("dummy/path")


def test_llama_cpp_inference_creation():
    """Test that we can create a LlamaCppInference instance (mocked)."""
    with patch('study_llm.inference.llama_cpp.LLAMA_CPP_AVAILABLE', True):
        with patch('study_llm.inference.llama_cpp.Llama') as mock_llama:
            mock_instance = Mock()
            mock_llama.return_value = mock_instance

            inference = LlamaCppInference("dummy/path")
            assert inference is not None
            assert inference.model_path.name == "path"
            assert inference.model_path.parent.name == "dummy"
            assert inference.n_ctx == 2048
            assert inference.n_gpu_layers == 0


def test_is_loaded():
    """Test the is_loaded method."""
    with patch('study_llm.inference.llama_cpp.LLAMA_CPP_AVAILABLE', True):
        with patch('study_llm.inference.llama_cpp.Llama') as mock_llama:
            mock_instance = Mock()
            mock_llama.return_value = mock_instance

            inference = LlamaCppInference("dummy/path")
            assert inference.is_loaded() is True

            # Test when model is None
            inference.model = None
            assert inference.is_loaded() is False


@patch('study_llm.inference.llama_cpp.LLAMA_CPP_AVAILABLE', True)
@patch('study_llm.inference.llama_cpp.Llama')
def test_generate_method(mock_llama):
    """Test the generate method."""
    mock_instance = Mock()
    mock_instance.return_value = {
        'choices': [{'text': 'Generated text'}]
    }
    mock_llama.return_value = mock_instance

    inference = LlamaCppInference("dummy/path")
    result = inference.generate("Test prompt")

    assert result == "Generated text"
    mock_instance.assert_called_once_with(
        "Test prompt",
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=None,
        echo=False
    )


@patch('study_llm.inference.llama_cpp.LLAMA_CPP_AVAILABLE', True)
@patch('study_llm.inference.llama_cpp.Llama')
def test_generate_with_stream(mock_llama):
    """Test the generate method with streaming."""
    mock_instance = Mock()
    mock_generator = Mock()
    mock_instance.generate.return_value = mock_generator
    mock_llama.return_value = mock_instance

    inference = LlamaCppInference("dummy/path")
    result = inference.generate("Test prompt", stream=True)

    assert result == mock_generator
    mock_instance.generate.assert_called_once_with(
        "Test prompt",
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=None,
        stream=True
    )


def test_create_embedding_placeholder():
    """Test that create_embedding returns a placeholder."""
    with patch('study_llm.inference.llama_cpp.LLAMA_CPP_AVAILABLE', True):
        with patch('study_llm.inference.llama_cpp.Llama') as mock_llama:
            mock_instance = Mock()
            mock_llama.return_value = mock_instance

            inference = LlamaCppInference("dummy/path")
            embedding = inference.create_embedding("Test text")

            # Should return a list of zeros with placeholder size
            assert isinstance(embedding, list)
            assert len(embedding) == 384  # Placeholder size
            assert all(x == 0.0 for x in embedding)