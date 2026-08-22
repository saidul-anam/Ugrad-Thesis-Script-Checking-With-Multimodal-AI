"""
Safe Image Preprocessing for Handwritten Examination Scripts.
Ensures image format compatibility while strictly preserving handwriting strokes.
"""

from typing import Optional, Tuple
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import unicodedata
import re

from src.utils.hashing import compute_image_hash, compute_file_hash


class ImagePreprocessor:
    """
    Modular preprocessor for handwritten answer script images.
    Preserves original resolution and stroke morphology unless explicitly configured.
    """

    def __init__(
        self,
        enabled: bool = True,
        resize: bool = True,
        max_image_side: int = 2048,
        min_image_side: int = 600,
        deskew: bool = False,
        denoise: bool = False,
        contrast_normalization: bool = False,
        convert_to_rgb: bool = True
    ):
        self.enabled = enabled
        self.resize = resize
        self.max_image_side = max_image_side
        self.min_image_side = min_image_side
        self.deskew = deskew
        self.denoise = denoise
        self.contrast_normalization = contrast_normalization
        self.convert_to_rgb = convert_to_rgb

    def load_and_preprocess(self, image_path: str) -> Tuple[Image.Image, Image.Image, str]:
        """
        Load an image file, maintain the untouched original image,
        and optionally return a preprocessed copy alongside its stable SHA-256 hash.

        Returns:
            (original_image, processed_image, image_hash)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # Check if PDF
        if path.suffix.lower() == ".pdf":
            try:
                import fitz
                doc = fitz.open(str(path))
                page = doc[0]  # First page
                pix = page.get_pixmap(dpi=300)
                original_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
            except ImportError:
                raise ImportError("PyMuPDF (fitz) is required to render PDF inputs.")
        else:
            original_img = Image.open(str(path))

        image_hash = compute_image_hash(original_img)

        if not self.enabled:
            processed_img = original_img.convert("RGB") if self.convert_to_rgb else original_img
            return original_img, processed_img, image_hash

        processed_img = original_img.copy()

        # 1. RGB conversion
        if self.convert_to_rgb and processed_img.mode != "RGB":
            processed_img = processed_img.convert("RGB")

        # 2. Orientation fix based on EXIF tag
        try:
            processed_img = ImageOps.exif_transpose(processed_img)
        except Exception:
            pass

        # 3. Geometry-preserving resize
        if self.resize:
            w, h = processed_img.size
            max_side = max(w, h)
            min_side = min(w, h)

            if max_side > self.max_image_side:
                scale = self.max_image_side / float(max_side)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                processed_img = processed_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            elif min_side < self.min_image_side and max_side < self.max_image_side:
                scale = self.min_image_side / float(min_side)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                processed_img = processed_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # 4. Optional contrast normalization
        if self.contrast_normalization:
            processed_img = ImageOps.autocontrast(processed_img, cutoff=1)

        return original_img, processed_img, image_hash


def safe_normalize_text(text: str) -> str:
    """
    Perform ONLY safe, non-destructive normalization:
    - Unicode NFC normalization
    - Standardize carriage returns to standard newlines
    - Strip leading/trailing whitespace
    - Condense multiple spaces into single spaces on the same line
    - Strictly does NOT alter spelling, grammar, punctuation, or words.
    """
    if not text:
        return ""

    # Unicode NFC normalization
    norm = unicodedata.normalize("NFC", text)

    # Standardize line endings
    norm = norm.replace("\r\n", "\n").replace("\r", "\n")

    # Clean horizontal whitespace per line while preserving line breaks
    lines = norm.split("\n")
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

    # Join lines and strip external whitespace
    result = "\n".join(cleaned_lines).strip()
    return result
