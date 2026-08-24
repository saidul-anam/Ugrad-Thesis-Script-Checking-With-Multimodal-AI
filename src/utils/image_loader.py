from PIL import Image
import os
from typing import Union, Tuple


def load_and_preprocess_image(
    image_input: Union[str, Image.Image],
    max_dimension: int = 2048
) -> Image.Image:
    """
    Load an image from filepath or PIL Image, convert to RGB, and resize if larger than max_dimension.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Script image not found at path: {image_input}")
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return img
