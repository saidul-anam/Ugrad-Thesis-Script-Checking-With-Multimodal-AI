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


def test_strikethrough_detector_multi_word():
    detector = StrikethroughDetector(min_line_width=20, max_line_height=8)
    arr = np.full((400, 600), 255, dtype=np.uint8)

    # Long text line with strike across multiple words
    cv2.putText(arr, "In 1980 the percentage was", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.line(arr, (30, 142), (450, 142), 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    assert res.has_strikethrough
    assert res.multi_word_count >= 1
    assert any(r.is_multi_word for r in res.regions)


def test_strikethrough_detector_rejects_faint_bleedthrough():
    detector = StrikethroughDetector(min_line_width=20, max_line_height=8)
    arr = np.full((300, 300), 255, dtype=np.uint8)

    # Simulate faint reverse-side bleed-through ink (gray level 190, barely darker than 255)
    cv2.putText(arr, "faint text", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 190, 1)
    cv2.line(arr, (45, 142), (180, 142), 180, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    # Faint ghost ink should be rejected by the optical density floor
    assert not res.has_strikethrough
    assert res.region_count == 0


def test_strikethrough_detector_rejects_table_gridlines():
    detector = StrikethroughDetector(min_line_width=20, max_line_height=8)
    arr = np.full((300, 300), 255, dtype=np.uint8)

    # Draw horizontal table row line
    cv2.line(arr, (30, 100), (250, 100), 0, 2)
    # Draw 4 vertical column dividing lines crossing through the horizontal line
    for vx in (40, 90, 140, 190, 240):
        cv2.line(arr, (vx, 80), (vx, 140), 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    # Table gridline intersecting vertical dividers must not be classified as a strikethrough
    assert not res.has_strikethrough
    assert res.region_count == 0


def test_strikethrough_detector_detects_diagonal_slash():
    detector = StrikethroughDetector(min_line_width=20, max_line_height=15)
    arr = np.full((300, 400), 255, dtype=np.uint8)

    # Text letters across line
    cv2.putText(arr, "crossed out clause", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    # Draw diagonal slash at ~15 degrees across text
    cv2.line(arr, (35, 160), (320, 135), 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    assert res.has_strikethrough
    assert res.region_count >= 1
    reg = res.regions[0]
    assert reg.y_pct > 0.0 and reg.y2_pct > reg.y_pct
    assert reg.x_pct > 0.0 and reg.x2_pct > reg.x_pct
    assert reg.w >= 20


def test_strikethrough_detector_normalized_coordinates():
    detector = StrikethroughDetector(min_line_width=20, max_line_height=8)
    arr = np.full((500, 1000), 255, dtype=np.uint8)

    # Text at y=250 (50% down page)
    cv2.putText(arr, "middle text line", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.line(arr, (190, 242), (450, 242), 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    assert res.has_strikethrough
    reg = res.regions[0]
    # Check that percentages are in range [0, 100]
    assert 0.0 <= reg.y_pct <= 100.0
    assert 0.0 <= reg.y2_pct <= 100.0
    assert 0.0 <= reg.x_pct <= 100.0
    assert 0.0 <= reg.x2_pct <= 100.0
    # Y should be around 45%-55%
    assert 40.0 <= reg.y_pct <= 55.0


def test_strikethrough_detector_rejects_flowchart_box():
    """Simulate a flowchart box (like Question 2 in SE_11_Q1_0020) with text inside.
    The top and bottom box borders must NOT be detected as strikethrough strikes."""
    detector = StrikethroughDetector(min_line_width=25, max_line_height=8)
    arr = np.full((400, 600), 255, dtype=np.uint8)

    # Draw closed flowchart box: (x=100, y=100, w=350, h=100)
    cv2.rectangle(arr, (100, 100), (450, 200), 0, thickness=2)

    # Write text inside the flowchart box
    cv2.putText(arr, "all forms of abuse", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    # Box borders should NOT be detected as strikethrough strikes
    assert not res.has_strikethrough
    assert res.region_count == 0


def test_strikethrough_detector_rejects_dual_corner_horizontal_edge():
    """A horizontal line terminating in vertical lines at both ends is a box edge, not a strike."""
    detector = StrikethroughDetector(min_line_width=25, max_line_height=8)
    arr = np.full((300, 400), 255, dtype=np.uint8)

    # Draw horizontal line
    cv2.line(arr, (50, 150), (250, 150), 0, 2)
    # Draw vertical lines at left and right ends (corners)
    cv2.line(arr, (50, 130), (50, 170), 0, 2)
    cv2.line(arr, (250, 130), (250, 170), 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    assert not res.has_strikethrough
    assert res.region_count == 0


def test_strikethrough_detector_header_underline_classified_correctly():
    """Header underline with descending letters (e.g. 'Question', 'g') must be classified as underline, NOT strikethrough."""
    detector = StrikethroughDetector(min_line_width=25, max_line_height=8)
    arr = np.full((300, 500), 255, dtype=np.uint8)

    # Text at y=100
    cv2.putText(arr, "Ans to Question No-01", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    # Underline drawn right along the bottom baseline (y=108), touching descenders
    cv2.line(arr, (45, 108), (420, 108), 0, 2)
    # A few stray descender pixels below y=108
    cv2.circle(arr, (200, 114), 2, 0, -1)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    # Must NOT be marked as strikethrough
    assert not res.has_strikethrough
    assert res.region_count == 0
    # Must be classified as underline
    assert len(res.underlines) >= 1
    assert res.underlines[0].is_underline


def test_strikethrough_detector_steep_diagonal_slash():
    """Steep diagonal slash at 45-60 degrees must be detected and clustered into a block."""
    detector = StrikethroughDetector(min_line_width=20, max_line_height=15)
    arr = np.full((400, 400), 255, dtype=np.uint8)

    # Multi-line text
    cv2.putText(arr, "Line one text here", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(arr, "Line two text here", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(arr, "Line three text here", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    # Steep diagonal slash cutting through all 3 lines at ~50 degrees (from x=50,y=200 to x=200,y=80)
    cv2.line(arr, (50, 200), (200, 80), 0, 3)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    assert res.has_strikethrough
    assert len(res.blocks) >= 1
    assert any(b.stroke_type == "steep_diagonal" for b in res.blocks)


def test_strikethrough_detector_x_cross_block():
    """Intersecting opposite diagonal slashes forming an X cross must produce an x_cross block."""
    detector = StrikethroughDetector(min_line_width=20, max_line_height=15)
    arr = np.full((400, 400), 255, dtype=np.uint8)

    cv2.putText(arr, "First draft line", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(arr, "Second draft line", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    # Draw X cross: forward slash and backslash crossing in center
    cv2.line(arr, (50, 80), (220, 160), 0, 3)
    cv2.line(arr, (50, 160), (220, 80), 0, 3)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    assert res.has_strikethrough
    assert len(res.blocks) >= 1
    assert any(b.stroke_type == "x_cross" for b in res.blocks)


def test_strikethrough_detector_multiline_block_clustering():
    """3 consecutive parallel horizontal lines must be clustered into a single parallel_horizontal block."""
    detector = StrikethroughDetector(min_line_width=20, max_line_height=8)
    arr = np.full((500, 600), 255, dtype=np.uint8)

    # 3 lines of text with a horizontal stroke through each
    for idx, y in enumerate([120, 160, 200]):
        cv2.putText(arr, f"Cancelled draft line number {idx+1}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        cv2.line(arr, (45, y - 8), (480, y - 8), 0, 2)

    img = Image.fromarray(arr)
    res = detector.detect(img)

    assert res.has_strikethrough
    assert res.region_count >= 3
    assert len(res.blocks) >= 1
    block = res.blocks[0]
    assert block.line_count >= 3
    assert block.stroke_type == "parallel_horizontal"


