"""
Unit tests for OCR Pydantic Schemas.
"""

import unittest
from pydantic import ValidationError
from src.ocr.schemas import OCRResult, OCRContent, InputMetadata, ExecutionMetadata, OCRSegment


class TestOCRSchemas(unittest.TestCase):

    def test_valid_ocr_result(self):
        res = OCRResult(
            script_id="EXAM_001",
            input=InputMetadata(
                image_path="data/samples/exam_001.jpg",
                image_hash="abc123hash",
                image_size=[1000, 1500]
            ),
            ocr=OCRContent(
                backend="infinite_ocr",
                model="nanonets/Nanonets-OCR2-3B",
                raw_text="The quick brown fox.",
                normalized_text="The quick brown fox.",
                confidence=0.92,
                confidence_type="raw",
                confidence_available=True,
                segments=[OCRSegment(text="The quick brown fox.", confidence=0.92)]
            ),
            metadata=ExecutionMetadata(
                processing_time_seconds=1.23,
                device="cpu",
                timestamp="2026-08-22T00:00:00Z"
            )
        )
        self.assertEqual(res.script_id, "EXAM_001")
        self.assertEqual(res.ocr.confidence, 0.92)
        json_str = res.to_json()
        self.assertIn("EXAM_001", json_str)

    def test_invalid_confidence_above_one(self):
        with self.assertRaises(ValidationError):
            OCRContent(
                backend="test",
                model="test",
                raw_text="text",
                normalized_text="text",
                confidence=1.5  # Invalid > 1.0
            )

    def test_invalid_confidence_below_zero(self):
        with self.assertRaises(ValidationError):
            OCRContent(
                backend="test",
                model="test",
                raw_text="text",
                normalized_text="text",
                confidence=-0.1  # Invalid < 0.0
            )

    def test_missing_confidence_is_handled(self):
        content = OCRContent(
            backend="test",
            model="test",
            raw_text="text",
            normalized_text="text",
            confidence=None,
            confidence_available=False,
            confidence_type="none"
        )
        self.assertIsNone(content.confidence)
        self.assertFalse(content.confidence_available)


if __name__ == "__main__":
    unittest.main()
