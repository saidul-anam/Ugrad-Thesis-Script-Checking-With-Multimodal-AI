"""
Unit tests for document and image ingestion.
"""

import unittest
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.image_normalizer import normalize_image, normalize_images
from ingestion.page_router import PageRouter
from ingestion.pdf_loader import load_images_from_file


class TestIngestion(unittest.TestCase):

    def test_image_normalization(self):
        # Create a large image
        large_img = Image.new("RGB", (3000, 1500), color=(200, 200, 200))
        norm_img = normalize_image(large_img, max_side=2048)
        self.assertEqual(norm_img.size[0], 2048)
        self.assertEqual(norm_img.size[1], 1024)

    def test_page_router_manifest(self):
        p1 = Image.new("RGB", (100, 100))
        p2 = Image.new("RGB", (100, 100))
        p3 = Image.new("RGB", (100, 100))
        pages = [p1, p2, p3]

        manifest = {"Q1": [0, 1], "Q2": [2]}
        routed = PageRouter.route_pages(pages, manifest=manifest)
        self.assertEqual(len(routed["Q1"]), 2)
        self.assertEqual(len(routed["Q2"]), 1)

    def test_pdf_loading_real_sample(self):
        sample_pdf = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "SE_11_Q1_0001.pdf")
        if os.path.exists(sample_pdf):
            pages = load_images_from_file(sample_pdf, dpi=150)
            self.assertGreaterEqual(len(pages), 1)
            self.assertIsInstance(pages[0], Image.Image)


if __name__ == "__main__":
    unittest.main()
