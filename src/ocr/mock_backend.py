"""
Deterministic Mock OCR Backend for Unit Testing and Fast CI Verification.
"""

import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from PIL import Image

from src.ocr.base import OCRBackend
from src.ocr.schemas import OCRResult, OCRContent, InputMetadata, ExecutionMetadata, OCRSegment
from src.preprocessing.image import safe_normalize_text
from src.utils.hashing import compute_image_hash


class MockOCRBackend(OCRBackend):
    """
    Mock OCR backend returning predefined transcriptions and confidence for test suites.
    """

    def __init__(
        self,
        mock_text: str = "The student answers that honesty is the best policy in human life.",
        mock_confidence: float = 0.94,
        confidence_available: bool = True
    ):
        self.mock_text = mock_text
        self.mock_confidence = mock_confidence
        self.confidence_available = confidence_available

    def extract(
        self,
        image: Image.Image,
        image_path: str,
        script_id: str,
        **kwargs: Any
    ) -> OCRResult:
        start_time = time.perf_counter()
        image_hash = compute_image_hash(image)

        raw_text = self.mock_text
        normalized_text = safe_normalize_text(raw_text)

        conf = self.mock_confidence if self.confidence_available else None
        conf_type = "raw" if self.confidence_available else "none"

        segments = [
            OCRSegment(text=line, confidence=conf)
            for line in raw_text.split("\n") if line.strip()
        ]

        elapsed_time = round(time.perf_counter() - start_time, 4)

        return OCRResult(
            script_id=script_id,
            input=InputMetadata(
                image_path=image_path,
                image_hash=image_hash,
                image_size=[image.width, image.height]
            ),
            ocr=OCRContent(
                backend="mock",
                model="mock-deterministic-v1",
                raw_text=raw_text,
                normalized_text=normalized_text,
                confidence=conf,
                confidence_type=conf_type,
                confidence_available=self.confidence_available,
                segments=segments
            ),
            metadata=ExecutionMetadata(
                processing_time_seconds=elapsed_time,
                device="cpu",
                gpu_memory_peak_mb=0.0,
                cpu_memory_mb=10.0,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        )
