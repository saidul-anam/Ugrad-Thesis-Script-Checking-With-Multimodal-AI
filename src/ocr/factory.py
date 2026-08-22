"""
OCR Backend Factory.
Instantiates the configured OCR backend (Infinite-OCR, TrOCR, EasyOCR, or Mock).
"""

from typing import Dict, Any, Optional
from src.ocr.base import OCRBackend
from src.ocr.infinite_ocr import InfiniteOCRBackend
from src.ocr.trocr_backend import TrOCRBackend
from src.ocr.easyocr_backend import EasyOCRBackend
from src.ocr.mock_backend import MockOCRBackend


def get_ocr_backend(config: Optional[Dict[str, Any]] = None) -> OCRBackend:
    """
    Instantiate the appropriate OCRBackend subclass based on configuration.
    """
    config = config or {}
    backend_type = config.get("backend", "infinite_ocr").lower().strip()
    model_name = config.get("model_name", "nanonets/Nanonets-OCR2-3B")
    device = config.get("device", "auto")

    if backend_type in ["infinite_ocr", "nanonets", "qwen", "qwen_vl"]:
        return InfiniteOCRBackend(
            model_name=model_name,
            device=device,
            max_new_tokens=config.get("max_new_tokens", 4096)
        )
    elif backend_type in ["trocr", "trocr_handwritten"]:
        return TrOCRBackend(
            model_name=model_name if "trocr" in model_name.lower() else "microsoft/trocr-base-handwritten",
            device=device,
            max_new_tokens=config.get("max_new_tokens", 256)
        )
    elif backend_type in ["easyocr", "craft"]:
        return EasyOCRBackend(
            device=device,
            language=config.get("language", "en")
        )
    elif backend_type in ["mock", "test"]:
        return MockOCRBackend(
            mock_text=config.get("mock_text", "The student answers that honesty is the best policy in human life."),
            mock_confidence=config.get("mock_confidence", 0.94),
            confidence_available=config.get("confidence_available", True)
        )
    else:
        # Default fallback to InfiniteOCRBackend
        return InfiniteOCRBackend(
            model_name=model_name,
            device=device
        )
