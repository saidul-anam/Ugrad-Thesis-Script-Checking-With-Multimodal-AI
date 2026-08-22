"""
Image hashing utility for duplicate detection, dataset integrity, and leakage prevention.
"""

import hashlib
from typing import Union
from pathlib import Path
from PIL import Image
import io


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_image_hash(image: Image.Image) -> str:
    """Compute SHA-256 hash of PIL Image pixel bytes to detect duplicates."""
    hasher = hashlib.sha256()
    # Normalize image to RGB bytes for consistent hashing regardless of format
    rgb_img = image.convert("RGB")
    hasher.update(rgb_img.tobytes())
    return hasher.hexdigest()
