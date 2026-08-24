#!/usr/bin/env python3
"""
Environment and Hardware Diagnostics Tool for Gemma 4 31B IT Local Deployment.
Verifies CUDA availability, GPU architecture, VRAM capacity, and bitsandbytes quantization support.
"""

import sys
import platform


def check_environment():
    print("=" * 60)
    print("  Gemma 4 31B IT Local Deployment Environment Check")
    print("=" * 60)
    print(f"Python Version: {platform.python_version()} ({platform.architecture()[0]})")
    print(f"OS Platform   : {platform.system()} {platform.release()}")

    # 1. PyTorch & CUDA Check
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            
            print(f"GPU Count      : {gpu_count}")
            print(f"GPU Name       : {gpu_name}")
            print(f"Compute Cap    : {capability[0]}.{capability[1]}")
            print(f"Total VRAM     : {total_vram_gb:.2f} GB")
            print(f"bfloat16 Supp. : {torch.cuda.is_bf16_supported()}")

            if total_vram_gb >= 30.0:
                print("\n[OK] VRAM capacity is sufficient for Gemma 4 31B IT (4-bit NF4 footprint ~18-20GB VRAM, 8-bit ~30GB VRAM).")
            else:
                print(f"\n[INFO] Detected {total_vram_gb:.2f} GB VRAM. For 31B model, 4-bit NF4 quantization or CPU offloading is recommended.")
        else:
            print("\n[WARN] CUDA is not currently active on this environment. Use --mock flag for local development.")
    except ImportError:
        print("\n[ERROR] PyTorch is not installed. Install torch with CUDA support.")

    # 2. Transformers & BitsAndBytes Check
    print("\n--- Dependencies Check ---")
    packages = ["transformers", "accelerate", "bitsandbytes", "pydantic", "PIL", "yaml", "rich"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  [✓] {pkg:<15}: {version}")
        except ImportError:
            print(f"  [✗] {pkg:<15}: NOT INSTALLED")

    print("=" * 60)


if __name__ == "__main__":
    check_environment()
