"""
OCR Subsystem package.
"""

from src.ocr.base import OCRBackend
from src.ocr.schemas import OCRResult, OCRContent, OCRSegment, InputMetadata, ExecutionMetadata, GroundTruthSample
from src.ocr.factory import get_ocr_backend

__all__ = [
    "OCRBackend",
    "OCRResult",
    "OCRContent",
    "OCRSegment",
    "InputMetadata",
    "ExecutionMetadata",
    "GroundTruthSample",
    "get_ocr_backend"
]
