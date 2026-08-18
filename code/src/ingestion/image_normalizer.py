"""
Image normalizer and preprocessor module.
Handles aspect ratio preservation, maximum dimension downscaling, and standard color format.
"""

from typing import List, Tuple
from PIL import Image


def normalize_image(
    image: Image.Image,
    max_side: int = 2048,
    resample: Image.Resampling = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Normalize image to RGB, downscaling if dimensions exceed max_side while preserving aspect ratio.
    """
    img = image.convert("RGB")
    w, h = img.size

    if max(w, h) > max_side:
        if w >= h:
            new_w = max_side
            new_h = int(h * (max_side / w))
        else:
            new_h = max_side
            new_w = int(w * (max_side / h))
        img = img.resize((new_w, new_h), resample=resample)

    return img


def normalize_images(
    images: List[Image.Image],
    max_side: int = 2048
) -> List[Image.Image]:
    """Normalize a batch of page images."""
    return [normalize_image(img, max_side=max_side) for img in images]
