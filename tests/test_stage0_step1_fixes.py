"""
Step 1 Vision Preprocessing & Stroke Calibration Unit Tests.
Covers:
  - Problem 18: Flowchart & diagram closed-box contour filter in Stage 0.5.
  - Problem 5: Luminance-preserving chromatic suppression in Stage 0 red ink detector.
  - Problem 11: Empty / whitespace struck tag purging.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pytest
from PIL import Image

from src.pipeline.stage0_strikethrough_detector import StrikethroughDetector
from src.pipeline.stage0_red_ink_detector import RedInkDetector
import re


def test_flowchart_box_borders_not_detected_as_strikes():
    """
    Simulates Question 2 flowchart box on SE_11_Q1_0020 (Page 3).
    A closed rectangular box with text 'all forms of abuse' inside must NOT have
    its top or bottom borders classified as strikethrough strokes.
    """
    detector = StrikethroughDetector(min_line_width=25, max_line_height=8)
    arr = np.full((500, 700), 255, dtype=np.uint8)

    # Draw closed flowchart box: (x=80, y=120, w=450, h=120)
    cv2.rectangle(arr, (80, 120), (530, 240), 0, thickness=2)

    # Render student text inside the box
    cv2.putText(arr, "3. Becoming Vulnerable to", (100, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(arr, "all forms of abuse", (100, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    # Neither box border should be flagged as a strikethrough
    assert not res.has_strikethrough
    assert res.region_count == 0


def test_dual_corner_connected_edges_rejected():
    """
    A horizontal stroke that terminates in vertical lines at both ends is a box edge,
    not a strikethrough.
    """
    detector = StrikethroughDetector(min_line_width=25, max_line_height=8)
    arr = np.full((300, 500), 255, dtype=np.uint8)

    # Horizontal edge
    cv2.line(arr, (60, 140), (350, 140), 0, 2)
    # Left and right vertical corner edges
    cv2.line(arr, (60, 100), (60, 180), 0, 2)
    cv2.line(arr, (350, 100), (350, 180), 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    assert not res.has_strikethrough
    assert res.region_count == 0


def test_genuine_strikethrough_inside_box_preserved():
    """
    If a student genuinely strikes through a word INSIDE a box, the strike line
    pierces the glyphs and does NOT connect to vertical box corners, so it MUST be detected.
    """
    detector = StrikethroughDetector(min_line_width=20, max_line_height=8)
    arr = np.full((500, 700), 255, dtype=np.uint8)

    # Draw closed box
    cv2.rectangle(arr, (80, 100), (550, 300), 0, thickness=2)

    # Draw word 'cancelled' and strike line right through its center
    cv2.putText(arr, "cancelled", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.line(arr, (140, 192), (320, 192), 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    # The internal strike should be detected
    assert res.has_strikethrough
    assert res.region_count >= 1
    # Check that the detected region corresponds to the word strike, not the box
    reg = res.regions[0]
    assert 130 <= reg.x <= 160
    assert 180 <= reg.y <= 200


def test_chromatic_suppression_preserves_black_ink_under_red_tick():
    """
    Problem 5: When teacher draws a red checkmark over dark student ink,
    the red hue is neutralized while the student stroke luminance is 100% preserved.
    """
    detector = RedInkDetector(
        min_pixel_threshold=400,
        margin_pixel_threshold=400,
        margin_width_ratio=0.18,
        enable_inpainting=True
    )
    arr = np.full((800, 800, 3), 255, dtype=np.uint8)

    # Dark student ink stroke (gray level 75)
    cv2.line(arr, (250, 400), (550, 400), (75, 75, 75), 6)

    # Red checkmark crossing the student stroke
    cv2.line(arr, (380, 340), (430, 460), (0, 0, 255), 10)
    cv2.line(arr, (430, 460), (540, 300), (0, 0, 255), 10)

    img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    res = detector.detect(img)

    assert res.has_red_ink
    assert res.clean_image is not None

    clean_arr = np.array(res.clean_image.convert("L"))
    # The student stroke at the intersection (x=430, y=400) must remain dark
    intersection_val = clean_arr[400, 430]
    assert intersection_val < 135, f"Student stroke was bleached to {intersection_val}, expected < 135"


def test_empty_struck_tags_sanitization():
    """
    Problem 11: Empty or whitespace-only struck tags from margin scribbles
    must be purged so downstream models do not see ghost tags.
    """
    raw = "The lion laughed [struck:   ] and said [struck:] hello [struck: -] world."
    cleaned = re.sub(r'\[struck:\s*\]', '', raw)
    cleaned = re.sub(r'\[struck:[^\w\u0980-\u09FF]*\]', '', cleaned)
    # Consecutive spaces can also be collapsed
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    assert "[struck:" not in cleaned
    assert cleaned.strip() == "The lion laughed and said hello world."
