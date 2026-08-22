"""
Environment and Hardware Telemetry utility for experimental reproducibility.
"""

import os
import sys
import platform
import psutil
import torch
from typing import Dict, Any


def get_environment_info() -> Dict[str, Any]:
    """Inspect and return verified environment and hardware metadata."""
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU)"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda_avail else 0.0
    bf16_supported = torch.cuda.is_bf16_supported() if cuda_avail else False
    cuda_version = torch.version.cuda if cuda_avail else None

    mem = psutil.virtual_memory()

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "bf16_supported": bf16_supported,
        "system_ram_total_gb": round(mem.total / (1024**3), 2),
        "system_ram_available_gb": round(mem.available / (1024**3), 2),
        "cpu_count_logical": os.cpu_count() or 1,
    }


def get_peak_gpu_memory_mb() -> float:
    """Return peak GPU memory allocated in MB if CUDA is active, else 0.0."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024**2), 2)
    return 0.0


def get_process_memory_mb() -> float:
    """Return current process Resident Set Size (RSS) memory in MB."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024**2), 2)
