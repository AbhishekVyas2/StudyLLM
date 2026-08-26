"""
Hardware detection for StudyLLM.
Detects CPU, RAM, GPU, VRAM and calculates performance tier (1-5).
"""

import platform
import subprocess
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareSpecs:
    """Hardware specifications detected by StudyLLM."""
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_model: Optional[str]
    vram_total_gb: Optional[float]
    vram_available_gb: Optional[float]
    os_name: str
    os_version: str
    disk_free_gb: float


@dataclass
class PerformanceTier:
    """Performance tier recommendation."""
    tier: int  # 1-5
    description: str
    recommended_model_size_billions: float
    notes: str


class HardwareDetector:
    """Detects system hardware and recommends performance tier."""

    def __init__(self):
        self.specs = self._detect_hardware()

    def _detect_hardware(self) -> HardwareSpecs:
        """Detect all hardware specifications."""
        return HardwareSpecs(
            cpu_model=self._get_cpu_model(),
            cpu_cores=self._get_cpu_cores(),
            cpu_threads=self._get_cpu_threads(),
            ram_total_gb=self._get_ram_total(),
            ram_available_gb=self._get_ram_available(),
            gpu_model=self._get_gpu_model(),
            vram_total_gb=self._get_vram_total(),
            vram_available_gb=self._get_vram_available(),
            os_name=platform.system(),
            os_version=platform.version(),
            disk_free_gb=self._get_disk_free()
        )

    @staticmethod
    def _wmic_value(args) -> str:
        """Run a wmic query and return the first non-header data value.

        wmic pads output with blank lines (and \r line endings), so
        naive `split('\n')[1]` often grabs an empty string.
        """
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=10
        )
        values = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        return values[1] if len(values) > 1 else ""

    def _get_cpu_model(self) -> str:
        """Get CPU model name."""
        try:
            if platform.system() == "Windows":
                model = self._wmic_value(["wmic", "cpu", "get", "name"])
                if model:
                    return model
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=10
                )
                return result.stdout.strip()
        except Exception:
            pass
        return "Unknown CPU"

    def _get_cpu_cores(self) -> int:
        """Get number of CPU cores."""
        try:
            import os
            return os.cpu_count() or 1
        except Exception:
            return 1

    def _get_cpu_threads(self) -> int:
        """Get number of CPU threads."""
        try:
            import os
            return os.cpu_count() or 1
        except Exception:
            return 1

    def _get_ram_total(self) -> float:
        """Get total RAM in GB."""
        try:
            if platform.system() == "Windows":
                raw = self._wmic_value(
                    ["wmic", "computersystem", "get", "totalphysicalmemory"]
                )
                if raw.isdigit():
                    return int(raw) / (1024**3)
            elif platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal" in line:
                            mem_kb = int(line.split()[1])
                            return mem_kb / (1024**2)  # Convert to GB
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=10
                )
                total_bytes = int(result.stdout.strip())
                return total_bytes / (1024**3)
        except Exception:
            pass
        return 0.0

    def _get_ram_available(self) -> float:
        """Get available RAM in GB."""
        try:
            if platform.system() == "Windows":
                raw = self._wmic_value(["wmic", "OS", "get", "FreePhysicalMemory"])
                if raw.isdigit():
                    return int(raw) / (1024**2)  # Convert to GB
            elif platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    mem_free = 0
                    buffers = 0
                    cached = 0
                    for line in f:
                        if "MemFree" in line:
                            mem_free = int(line.split()[1])
                        elif "Buffers" in line:
                            buffers = int(line.split()[1])
                        elif "Cached" in line:
                            cached = int(line.split()[1])
                    return (mem_free + buffers + cached) / (1024**2)  # Convert to GB
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True, text=True, timeout=10
                )
                lines = result.stdout.strip().split('\n')
                page_size = 4096  # Typical page size on macOS
                free_mem = 0
                for line in lines:
                    if "Pages free" in line:
                        free_mem = int(line.split()[2].rstrip('.'))
                        break
                return (free_mem * page_size) / (1024**3)  # Convert to GB
        except Exception:
            pass
        return 0.0

    def _get_gpu_model(self) -> Optional[str]:
        """Get GPU model name."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    capture_output=True, text=True, timeout=10
                )
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip() and line.strip() != "Name":
                        return line.strip()
            elif platform.system() == "Linux":
                result = subprocess.run(
                    ["lspci", "-vnn", "|", "grep", "-i", "vga"],
                    capture_output=True, text=True, timeout=10, shell=True
                )
                if result.stdout.strip():
                    # Extract GPU name from lspci output
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        return lines[0].split(':')[-1].strip()
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=10
                )
                # Parse macOS GPU info (simplified)
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "Chipset Model" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
        return None

    def _get_vram_total(self) -> Optional[float]:
        """Get total VRAM in GB."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "AdapterRAM"],
                    capture_output=True, text=True, timeout=10
                )
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip() and line.strip() != "AdapterRAM":
                        try:
                            vram_bytes = int(line.strip())
                            return vram_bytes / (1024**3)
                        except ValueError:
                            continue
            elif platform.system() == "Linux":
                # Try to get VRAM info from various sources
                try:
                    result = subprocess.run(
                        ["lspci", "-vv"],
                        capture_output=True, text=True, timeout=10
                    )
                    # Look for prefetchable memory size
                    vram_match = re.search(
                        r"Memory at .* \[size=([0-9]+)M\]",
                        result.stdout
                    )
                    if vram_match:
                        vram_mb = int(vram_match.group(1))
                        return vram_mb / 1024  # Convert to GB
                except Exception:
                    pass
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=10
                )
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "VRAM" in line:
                        try:
                            vram_str = line.split(":")[1].strip()
                            if "MB" in vram_str:
                                vram_mb = float(vram_str.replace("MB", "").strip())
                                return vram_mb / 1024
                            elif "GB" in vram_str:
                                return float(vram_str.replace("GB", "").strip())
                        except (ValueError, IndexError):
                            continue
        except Exception:
            pass
        return None

    def _get_vram_available(self) -> Optional[float]:
        """Get available VRAM in GB."""
        # For simplicity, we'll assume most VRAM is available
        # In a more sophisticated implementation, we'd check current usage
        vram_total = self._get_vram_total()
        if vram_total is not None:
            # Assume 80% available for simplicity
            return vram_total * 0.8
        return None

    def _get_disk_free(self) -> float:
        """Get free disk space in GB."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            return free / (1024**3)
        except Exception:
            return 0.0

    def get_performance_tier(self) -> PerformanceTier:
        """
        Calculate performance tier based on hardware specs.

        Tiers are deliberately conservative — a model that barely fits in RAM
        still runs unusably slow if it can't be GPU-accelerated. Sizes assume
        Q4_K_M quantization.

        Tiers:
        1: Very low-end (< 8 GB RAM, CPU-only, 1-2B models)
        2: Low-end (8-16 GB RAM or weak GPU, 2-4B models)
        3: Mainstream (16+ GB RAM with 6+ GB VRAM, 7B models)
        4: High-end (32 GB RAM, 10+ GB VRAM, 13-14B models)
        5: Enthusiast (32+ GB RAM, 20+ GB VRAM, 30B+ models)
        """
        ram = self.specs.ram_total_gb
        vram = self.specs.vram_total_gb or 0  # Default to 0 if no GPU

        # Determine tier based on RAM and VRAM.
        # VRAM requirements are set high enough that the recommended model
        # actually fits on the GPU — otherwise CPU-only speed is what matters,
        # and then only small models are usable.
        if ram >= 32 and vram >= 20:
            tier = 5
            description = "Enthusiast"
            recommended_size = 30.0
            notes = "High-end CPU and GPU, can run largest models"
        elif ram >= 32 and vram >= 10:
            tier = 4
            description = "High-end"
            recommended_size = 13.0
            notes = "High-end system with good GPU"
        elif ram >= 16 and vram >= 6:
            tier = 3
            description = "Mainstream"
            recommended_size = 7.0
            notes = "GPU can hold most of the recommended model"
        elif ram >= 8:
            tier = 2
            description = "Low-end"
            recommended_size = 3.0
            notes = "Limited acceleration — stick to small models for speed"
        else:
            tier = 1
            description = "Very low-end"
            recommended_size = 1.5
            notes = "Very limited resources, use smallest practical models"

        # Weak GPU relative to the tier means the model runs on CPU —
        # step down so the recommendation stays responsive.
        if tier >= 3 and vram < 6:
            tier -= 1
            description = f"{description.replace(' (Weak GPU for size)', '')} (Weak GPU for size)"
            notes += " Detected weak GPU, adjusting tier down."

        # Adjust tier upward if we have exceptional specs
        if tier <= 3 and vram >= 12 and ram >= 24:
            tier = min(5, tier + 1)
            description = f"{description} (Strong GPU)"
            notes += " Detected strong GPU, adjusting tier up."

        return PerformanceTier(
            tier=tier,
            description=description,
            recommended_model_size_billions=recommended_size,
            notes=notes
        )

    def get_specs_summary(self) -> str:
        """Get a formatted summary of hardware specs."""
        lines = [
            f"CPU       : {self.specs.cpu_model}",
            f"CPU Cores : {self.specs.cpu_cores}",
            f"RAM       : {self.specs.ram_total_gb:.1f} GB",
            f"GPU       : {self.specs.gpu_model or 'Not detected'}",
            f"VRAM      : {self.specs.vram_total_gb:.1f} GB" if self.specs.vram_total_gb else "VRAM      : Not detected",
            f"OS        : {self.specs.os_name} {self.specs.os_version}",
            f"Disk Free : {self.specs.disk_free_gb:.1f} GB"
        ]
        return "\n".join(lines)


# Global instance for easy access
_hardware_detector = None


def get_hardware_detector() -> HardwareDetector:
    """Get the global hardware detector instance."""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector


def get_performance_tier() -> PerformanceTier:
    """Get the current performance tier recommendation."""
    return get_hardware_detector().get_performance_tier()


def get_hardware_specs() -> HardwareSpecs:
    """Get the current hardware specifications."""
    return get_hardware_detector().specs