import numpy as np
import cv2
from PIL import Image
from src.pipeline.stage0_strikethrough_detector import StrikethroughDetector


def test_strikethrough_detector_blank_image():
    detector = StrikethroughDetector()
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    res = detector.detect(img)
    assert not res.has_strikethrough
    assert res.region_count == 0


def test_strikethrough_detector_detects_stroke():
    detector = StrikethroughDetector(min_line_width=20, max_line_height=8)
    arr = np.full((300, 300), 255, dtype=np.uint8)

    # Simulate text letters (dots/strokes above and below the strike line)
    cv2.putText(arr, "sample", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    # Draw horizontal strikethrough across the text
    cv2.line(arr, (45, 142), (180, 142), 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    assert res.has_strikethrough
    assert res.region_count >= 1
    assert any(r.w >= 20 for r in res.regions)


def test_strikethrough_detector_rejects_bottom_underline():
    detector = StrikethroughDetector(min_line_width=30, max_line_height=6)
    arr = np.full((300, 300), 255, dtype=np.uint8)

    # Text well above line
    cv2.putText(arr, "header", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    # Underline with no text ink below it
    cv2.line(arr, (40, 130), (200, 130), 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    # Underline without ink below should not be treated as a strikethrough traversing letters
    assert res.region_count == 0
