"""
Unit tests for OCR Extraction and Preprocessing.
"""

import unittest
from PIL import Image
from src.ocr.mock_backend import MockOCRBackend
from src.preprocessing.image import safe_normalize_text


class TestOCRBackend(unittest.TestCase):

    def test_mock_backend_extraction(self):
        backend = MockOCRBackend(
            mock_text="Answer to Question Number 1:\nHonesty is the best policy.",
            mock_confidence=0.95
        )
        img = Image.new("RGB", (300, 100), color=(255, 255, 255))
        result = backend.extract(
            image=img,
            image_path="test_sample.jpg",
            script_id="TEST_SCRIPT_01"
        )
        self.assertEqual(result.script_id, "TEST_SCRIPT_01")
        self.assertEqual(result.ocr.confidence, 0.95)
        self.assertTrue(result.ocr.confidence_available)
        self.assertIn("Honesty is the best policy.", result.ocr.normalized_text)

    def test_safe_normalize_text_preserves_words(self):
        raw = "  Hello   World  \r\n This is   a line.  "
        normalized = safe_normalize_text(raw)
        self.assertEqual(normalized, "Hello World\nThis is a line.")


if __name__ == "__main__":
    unittest.main()
