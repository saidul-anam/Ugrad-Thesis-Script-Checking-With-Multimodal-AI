"""
Abstract Base Interface for Plug-and-Play OCR Backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from PIL import Image
from src.ocr.schemas import OCRResult


class OCRBackend(ABC):
    """
    Abstract interface for single-pass handwritten text extraction.
    All OCR engines (Infinite-OCR, TrOCR, EasyOCR, etc.) must implement this contract.
    """

    @abstractmethod
    def extract(
        self,
        image: Image.Image,
        image_path: str,
        script_id: str,
        **kwargs: Any
    ) -> OCRResult:
        """
        Execute single-pass OCR on the provided handwritten script image.

        Args:
            image: PIL Image object of the handwritten examination answer.
            image_path: Source path to the image file.
            script_id: Anonymized unique identifier for the student script.

        Returns:
            Strictly validated OCRResult Pydantic object.
        """
        raise NotImplementedError
