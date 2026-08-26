"""
LLM inference using llama.cpp for GGUF models.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Generator
import threading

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

logger = logging.getLogger(__name__)


class LlamaCppInference:
    """Handles LLM inference using llama.cpp for GGUF models."""

    def __init__(self, model_path: Path, n_ctx: int = 2048, n_gpu_layers: int = 0):
        """
        Initialize the LlamaCpp inference engine.

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context size for the model
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            )

        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.model: Optional[Llama] = None
        self._lock = threading.Lock()

        self._load_model()

    def _load_model(self):
        """Load the GGUF model."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            logger.info(f"Context size: {self.n_ctx}, GPU layers: {self.n_gpu_layers}")

            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False  # Set to True for debugging
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            raise

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Generate text from the model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            repeat_penalty: Penalty for repeating tokens
            stop: List of stop sequences
            stream: Whether to stream the response

        Returns:
            Generated text (string) or generator of tokens if stream=True
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")

        with self._lock:
            try:
                if stream:
                    return self.model.generate(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        stop=stop,
                        stream=True
                    )
                else:
                    result = self.model(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        stop=stop,
                        echo=False
                    )
                    return result['choices'][0]['text']
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                raise

    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for the given text.
        Note: This is a placeholder - llama.cpp doesn't provide embeddings directly.
        For actual embeddings, we'd use a separate embedding model like BGE-M3.

        Args:
            text: Input text

        Returns:
            Embedding vector (placeholder - returns zeros)
        """
        # This is a placeholder. Actual embeddings would come from a separate model.
        logger.warning("create_embedding is not implemented for llama.cpp - "
                      "use a separate embedding model like BGE-M3")
        # Return a dummy embedding of appropriate size (would match embedding model)
        return [0.0] * 384  # Placeholder size

    def unload(self):
        """Unload the model to free memory."""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                logger.info("Model unloaded")


# Global instance management
_llama_instances: dict[str, LlamaCppInference] = {}


def get_llama_cpp_inference(
    model_path: str | Path,
    n_ctx: int = 2048,
    n_gpu_layers: int = 0
) -> LlamaCppInference:
    """
    Get or create a LlamaCppInference instance for the given model path.

    Args:
        model_path: Path to the GGUF model file
        n_ctx: Context size
        n_gpu_layers: Number of GPU layers

    Returns:
        LlamaCppInference instance
    """
    model_path = str(Path(model_path).absolute())
    if model_path not in _llama_instances:
        _llama_instances[model_path] = LlamaCppInference(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers
        )
    return _llama_instances[model_path]


def unload_all_models():
    """Unload all loaded models."""
    global _llama_instances
    for instance in _llama_instances.values():
        instance.unload()
    _llama_instances.clear()