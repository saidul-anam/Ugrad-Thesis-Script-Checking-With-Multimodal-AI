import cv2
import numpy as np
from PIL import Image
from src.pipeline.stage0_red_ink_detector import RedInkDetector, RedInkDetectionResult


def test_dual_zone_blank_image():
    detector = RedInkDetector(min_pixel_threshold=500, margin_pixel_threshold=300)
    blank = Image.new("RGB", (1000, 1500), color=(255, 255, 255))
    res = detector.detect(blank)
    assert not res.has_red_ink
    assert not res.margin_has_red_ink
    assert not res.body_has_red_ink
    assert res.red_pixel_count == 0


def test_dual_zone_margin_only():
    """Red score number written only in the left margin (X < 18%)."""
    detector = RedInkDetector(min_pixel_threshold=400, margin_pixel_threshold=400, margin_width_ratio=0.18)
    arr = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    # Draw red mark in left margin (x in 50..120, y in 200..350)
    cv2.putText(arr, "10", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 8)
    img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    res = detector.detect(img)
    assert res.has_red_ink
    assert res.margin_has_red_ink
    assert not res.body_has_red_ink
    assert res.margin_red_pixel_count > 400


def test_dual_zone_body_ticks_and_inpainting():
    """Red checkmark written in student answer body (X > 18%) is detected and inpainted."""
    detector = RedInkDetector(
        min_pixel_threshold=400,
        margin_pixel_threshold=400,
        margin_width_ratio=0.18,
        enable_inpainting=True
    )
    arr = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    # Draw red checkmark tick in body (x in 400..600, y in 400..600)
    cv2.line(arr, (400, 500), (450, 560), (0, 0, 255), 10)
    cv2.line(arr, (450, 560), (600, 420), (0, 0, 255), 10)
    img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    res = detector.detect(img)
    assert res.has_red_ink
    assert not res.margin_has_red_ink  # Stage 0b bypass trigger!
    assert res.body_has_red_ink
    assert res.body_red_pixel_count > 400

    # Verify clean_image has inpainted the red checkmark
    assert res.clean_image is not None
    clean_arr = np.array(res.clean_image.convert("RGB"))
    clean_bgr = cv2.cvtColor(clean_arr, cv2.COLOR_BGR2RGB)
    hsv_clean = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_clean, np.array([0, 60, 60]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv_clean, np.array([170, 60, 60]), np.array([180, 255, 255]))
    remaining_red = cv2.countNonZero(cv2.bitwise_or(mask1, mask2))
    assert remaining_red < 30
