"""
Tests for hardware detection.
"""

import pytest
from study_llm.hardware.detector import HardwareDetector, get_performance_tier


def test_hardware_detector_creation():
    """Test that we can create a hardware detector."""
    detector = HardwareDetector()
    assert detector is not None
    assert hasattr(detector, 'specs')


def test_get_performance_tier():
    """Test that we can get a performance tier."""
    tier = get_performance_tier()
    assert tier is not None
    assert 1 <= tier.tier <= 5
    assert isinstance(tier.description, str)
    assert isinstance(tier.recommended_model_size_billions, float)
    assert isinstance(tier.notes, str)


def test_hardware_specs_attributes():
    """Test that hardware specs have expected attributes."""
    detector = HardwareDetector()
    specs = detector.specs

    # Check that all expected attributes exist
    assert hasattr(specs, 'cpu_model')
    assert hasattr(specs, 'cpu_cores')
    assert hasattr(specs, 'cpu_threads')
    assert hasattr(specs, 'ram_total_gb')
    assert hasattr(specs, 'ram_available_gb')
    assert hasattr(specs, 'gpu_model')
    assert hasattr(specs, 'vram_total_gb')
    assert hasattr(specs, 'vram_available_gb')
    assert hasattr(specs, 'os_name')
    assert hasattr(specs, 'os_version')
    assert hasattr(specs, 'disk_free_gb')

    # Check types
    assert isinstance(specs.cpu_model, str)
    assert isinstance(specs.cpu_cores, int)
    assert isinstance(specs.cpu_threads, int)
    assert isinstance(specs.ram_total_gb, float)
    assert isinstance(specs.ram_available_gb, float)
    assert specs.gpu_model is None or isinstance(specs.gpu_model, str)
    assert specs.vram_total_gb is None or isinstance(specs.vram_total_gb, float)
    assert specs.vram_available_gb is None or isinstance(specs.vram_available_gb, float)
    assert isinstance(specs.os_name, str)
    assert isinstance(specs.os_version, str)
    assert isinstance(specs.disk_free_gb, float)