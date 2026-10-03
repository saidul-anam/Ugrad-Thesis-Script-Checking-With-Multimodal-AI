import numpy as np
import pytest
from src.pipeline.stage0_strikethrough_detector import StrikethroughDetector


def test_compact_single_word_stroke_detection():
    """Verify that compact cancellations (16px to 22px wide) are detected with lower floor."""
    detector = StrikethroughDetector(min_line_width=16)

    # 400x400 blank canvas
    img = np.full((400, 400), 255, dtype=np.uint8)

    # Draw simulated word glyphs: 'the'
    # Text baseline around y=200
    # Letter ascender/descender ink
    img[190:210, 100:104] = 0   # 't'
    img[185:210, 106:110] = 0   # 'h'
    img[195:210, 112:116] = 0   # 'e'

    # Strike line cutting through the 18px word at y=200
    img[200, 99:118] = 0  # 19px horizontal line

    res = detector.detect(img)
    assert res.has_strikethrough is True
    assert res.region_count >= 1
    # Check that the region encompasses the strike
    reg = res.regions[0]
    assert 95 <= reg.x <= 105
    assert reg.w >= 16


def test_flowchart_box_remains_rejected():
    """Verify that closed diagram/flowchart boxes are still rejected and never flagged as strikethroughs."""
    detector = StrikethroughDetector(min_line_width=16)

    img = np.full((400, 400), 255, dtype=np.uint8)

    # Draw a 100x60 closed rectangular flowchart box
    # Top and bottom horizontal borders
    img[150, 100:200] = 0
    img[210, 100:200] = 0
    # Left and right vertical borders
    img[150:211, 100] = 0
    img[150:211, 200] = 0

    res = detector.detect(img)
    # The box perimeter borders should be rejected by corner connectivity and perimeter overlap
    assert res.region_count == 0
    assert res.has_strikethrough is False
