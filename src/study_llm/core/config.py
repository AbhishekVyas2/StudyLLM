"""
Configuration management for StudyLLM.
Handles loading, saving, and accessing application settings.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class StudyLLMConfig:
    """Configuration settings for StudyLLM."""
    # Hardware settings
    hardware_tier_override: Optional[int] = None

    # Model settings
    auto_load_model: bool = True
    preferred_quantization: str = "Q4_K_M"
    ctx_size: int = 2048  # Context size
    gpu_layers: int = 0   # Number of layers to offload to GPU (0 = CPU only)

    # Embedding settings
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 32

    # Vector database settings
    vector_db_path: str = "storage/vectordb"
    vector_db_collection: str = "studyllm"

    # Document processing settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size_mb: int = 100

    # RAG settings
    retrieval_top_k: int = 10
    reranker_top_k: int = 3
    use_reranker: bool = True
    context_budget_tokens: int = 1500

    # Performance settings
    indexing_workers: int = 2
    background_indexing: bool = True
    show_indexing_progress: bool = True

    # UI settings
    use_colors: bool = True
    show_sources: bool = True

    def __post_init__(self):
        """Ensure storage directories exist."""
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("storage").mkdir(exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StudyLLMConfig':
        """Create from dictionary."""
        # Filter out any keys that don't exist in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def save(self, config_path: Path = Path("config/settings.json")):
        """Save configuration to file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    @classmethod
    def load(cls, config_path: Path = Path("config/settings.json")) -> 'StudyLLMConfig':
        """Load configuration from file."""
        if not config_path.exists():
            logger.info(f"No configuration file found at {config_path}, using defaults")
            return cls()

        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()


# Global config instance
_config = None


def get_config() -> StudyLLMConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = StudyLLMConfig.load()
    return _config


def reload_config():
    """Reload configuration from file."""
    global _config
    _config = StudyLLMConfig.load()
    return _config