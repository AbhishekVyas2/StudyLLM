"""
Model management for StudyLLM.
Handles discovery, validation, and loading of GGUF models.
"""

import os
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import struct

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a GGUF model."""
    path: Path
    name: str
    architecture: Optional[str]
    parameter_count: Optional[float]  # in billions
    quantization: Optional[str]
    size_gb: float
    is_valid: bool
    error_message: Optional[str] = None


class ModelManager:
    """Manages GGUF models for StudyLLM."""

    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)

    def discover_models(self) -> List[ModelInfo]:
        """Discover all GGUF models in the models directory."""
        models = []
        if not self.models_dir.exists():
            return models

        for model_path in self.models_dir.glob("*.gguf"):
            try:
                model_info = self._analyze_model(model_path)
                models.append(model_info)
            except Exception as e:
                logger.warning(f"Failed to analyze model {model_path}: {e}")
                models.append(ModelInfo(
                    path=model_path,
                    name=model_path.name,
                    architecture=None,
                    parameter_count=None,
                    quantization=None,
                    size_gb=model_path.stat().st_size / (1024**3),
                    is_valid=False,
                    error_message=str(e)
                ))

        # Sort by name for consistent ordering
        models.sort(key=lambda x: x.name.lower())
        return models

    def _analyze_model(self, model_path: Path) -> ModelInfo:
        """Analyze a GGUF model file to extract metadata."""
        stats = model_path.stat()
        size_gb = stats.st_size / (1024**3)

        # Try to parse GGUF header for metadata
        architecture, parameter_count, quantization = self._parse_gguf_header(model_path)

        # Extract name from filename (remove extension and common suffixes)
        name = model_path.stem
        # Remove common quantization suffixes for cleaner display
        for suffix in ["-Q4_K_M", "-Q4_K_S", "-Q4_0", "-Q4_1", "-Q5_K_M", "-Q5_K_S",
                      "-Q5_0", "-Q5_1", "-Q6_K", "-Q8_0", "-F16", "-F32"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        return ModelInfo(
            path=model_path,
            name=name,
            architecture=architecture,
            parameter_count=parameter_count,
            quantization=quantization,
            size_gb=size_gb,
            is_valid=True
        )

    def _parse_gguf_header(self, model_path: Path) -> tuple[Optional[str], Optional[float], Optional[str]]:
        """Parse GGUF header to extract model metadata.

        Implements the real GGUF v2/v3 specification:
          - magic "GGUF", version u32
          - tensor_count u64, metadata_kv_count u64
          - each KV pair: string key, u32 value type, typed value
          - strings are u64 length + bytes; arrays carry their own
            element type so sizes are computed exactly.
        Only metadata is read — tensors are skipped entirely.
        """
        try:
            with open(model_path, 'rb') as f:
                if f.read(4) != b'GGUF':
                    return None, None, None
                version = struct.unpack('<I', f.read(4))[0]
                if version < 2:
                    logger.debug(f"{model_path.name}: legacy GGUF v{version}, "
                                 "best-effort metadata parse")
                # Counts are u64 in v2/v3 (v1 used u32; those files are rare)
                fmt_q = '<Q' if version >= 2 else '<I'
                tensor_count = struct.unpack(fmt_q, f.read(8 if version >= 2 else 4))[0]
                kv_count = struct.unpack(fmt_q, f.read(8 if version >= 2 else 4))[0]

                def read_string() -> str:
                    n = struct.unpack('<Q', f.read(8))[0] if version >= 2 \
                        else struct.unpack('<I', f.read(4))[0]
                    return f.read(n).decode('utf-8', errors='replace')

                scalar_fmts = {
                    0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I',
                    5: '<i', 6: '<f', 10: '<Q', 11: '<q', 12: '<d',
                }

                def skip_value(vtype: int):
                    if vtype in scalar_fmts:
                        f.seek(struct.calcsize(scalar_fmts[vtype]), 1)
                    elif vtype == 7:  # BOOL
                        f.seek(1, 1)
                    elif vtype == 8:  # STRING
                        read_string()
                    elif vtype == 9:  # ARRAY
                        elem_type = struct.unpack('<I', f.read(4))[0]
                        n = struct.unpack('<Q', f.read(8))[0] if version >= 2 \
                            else struct.unpack('<I', f.read(4))[0]
                        for _ in range(min(n, 10_000_000)):
                            skip_value(elem_type)
                    else:
                        raise ValueError(f"unknown GGUF value type {vtype}")

                architecture = None
                parameter_count = None
                file_type = None
                size_label = None

                for _ in range(kv_count):
                    key = read_string()
                    vtype = struct.unpack('<I', f.read(4))[0]

                    value = None
                    if vtype in scalar_fmts:
                        value = struct.unpack(scalar_fmts[vtype], f.read(
                            struct.calcsize(scalar_fmts[vtype])))[0]
                    elif vtype == 7:
                        value = bool(f.read(1)[0])
                    elif vtype == 8:
                        value = read_string()
                    else:
                        skip_value(vtype)

                    if key == "general.architecture":
                        architecture = str(value)
                    elif key == "general.parameter_count" and value:
                        parameter_count = float(value) / 1e9
                    elif key == "general.size_label" and value:
                        size_label = str(value)
                    elif key == "general.file_type":
                        file_type = value

                # Newer GGUFs omit general.parameter_count; general.size_label
                # (e.g. "8B", "0.6B", "30B-A3B") carries the same information.
                if parameter_count is None and size_label:
                    m = re.match(r"([0-9]+(?:\.[0-9]+)?)B", size_label)
                    if m:
                        parameter_count = float(m.group(1))

                quantization = self._file_type_to_quant(file_type)
                return architecture, parameter_count, quantization

        except Exception as e:
            logger.debug(f"Failed to parse GGUF header for {model_path}: {e}")
            return None, None, None

    @staticmethod
    def _file_type_to_quant(file_type: Optional[int]) -> Optional[str]:
        """Map GGUF general.file_type codes to quantization labels."""
        mapping = {
            0: "F32", 1: "F16",
            2: "Q4_0", 3: "Q4_1",
            7: "Q8_0",
            8: "Q5_0", 9: "Q5_1",
            10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
            14: "Q4_K_S", 15: "Q4_K_M",
            16: "Q5_K_S", 17: "Q5_K_M",
            18: "Q6_K",
            19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
            22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
            26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
            30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
        }
        if file_type in mapping:
            return mapping[file_type]
        if file_type is not None and file_type >= 32:
            return "F16" if (file_type % 2 == 0) else "Q4_K_M"  # most-reliable fallback
        return None

    def get_recommended_model(self, performance_tier) -> Optional[ModelInfo]:
        """Get a recommended model based on performance tier."""
        models = self.discover_models()
        valid_models = [m for m in models if m.is_valid]

        if not valid_models:
            return None

        # Filter models by recommended size based on tier
        recommended_size = performance_tier.recommended_model_size_billions
        size_tolerance = 0.5  # Allow ±0.5B tolerance

        suitable_models = []
        for model in valid_models:
            if model.parameter_count is None:
                # If we can't determine parameter count, include it but prioritize lower
                suitable_models.append((model, float('inf')))
            else:
                size_diff = abs(model.parameter_count - recommended_size)
                suitable_models.append((model, size_diff))

        # Sort by size difference (closest to recommended first)
        suitable_models.sort(key=lambda x: x[1])

        if suitable_models:
            return suitable_models[0][0]

        # Fallback to smallest valid model
        valid_models.sort(key=lambda x: x.parameter_count or float('inf'))
        return valid_models[0] if valid_models else None

    def validate_model(self, model_path: Path) -> bool:
        """Validate that a model file is a valid GGUF."""
        try:
            info = self._analyze_model(model_path)
            return info.is_valid
        except Exception:
            return False


# Global instance for easy access
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager