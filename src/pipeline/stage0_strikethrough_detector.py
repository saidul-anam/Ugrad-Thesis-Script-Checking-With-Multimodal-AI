"""
Stage 0.5: Fast OpenCV Morphological Strikethrough Pre-Detector.
Runs on CPU in <15ms per page. Detects horizontal line strokes and diagonal cross-outs
traversing handwritten text, providing visual bounding-box priors for Stage 1.
"""

from typing import List, Tuple, Union, Optional
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field


class StrikethroughRegion(BaseModel):
    """Detected bounding box of a struck-through line or word region."""
    x: int
    y: int
    w: int
    h: int
    y_pct: float = 0.0
    y2_pct: float = 0.0
    x_pct: float = 0.0
    x2_pct: float = 0.0
    angle: float = 0.0
    confidence: float = 1.0
    is_multi_word: bool = False
    is_underline: bool = False


class StrikethroughBlock(BaseModel):
    """Detected composite multi-line crossed-out block (e.g. 2-6 lines or X-cross)."""
    y_pct: float = 0.0
    y2_pct: float = 0.0
    x_pct: float = 0.0
    x2_pct: float = 0.0
    line_count: int = 1
    stroke_type: str = "parallel_horizontal"  # 'parallel_horizontal', 'steep_diagonal', 'x_cross'
    confidence: float = 1.0


class StrikethroughDetectionResult(BaseModel):
    """Output of Stage 0.5 strikethrough detector."""
    has_strikethrough: bool = False
    region_count: int = 0
    multi_word_count: int = 0
    regions: List[StrikethroughRegion] = Field(default_factory=list)
    underlines: List[StrikethroughRegion] = Field(default_factory=list)
    blocks: List[StrikethroughBlock] = Field(default_factory=list)
    details: str = ""


class StrikethroughDetector:
    """
    Detects horizontal and diagonal strikethrough strokes across text.
    Uses CLAHE contrast enhancement and morphological line kernels to isolate thin, extended stroke segments.
    """

    def __init__(
        self,
        min_line_width: int = 16,
        max_line_height: int = 6,
        binarization_thresh: int = 140,
        use_clahe: bool = True
    ):
        self.min_line_width = min_line_width
        self.max_line_height = max_line_height
        self.binarization_thresh = binarization_thresh
        self.use_clahe = use_clahe

    def detect(
        self,
        image_input: Union[Image.Image, np.ndarray, str],
        teacher_mask: Optional[np.ndarray] = None
    ) -> StrikethroughDetectionResult:
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                return StrikethroughDetectionResult(details=f"Could not load {image_input}")
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        elif isinstance(image_input, Image.Image):
            gray = np.array(image_input.convert("L"))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_input
        else:
            return StrikethroughDetectionResult(details=f"Unsupported image type {type(image_input)}")

        h, w = gray.shape[:2]

        # 1. Otsu binarization (invert so text ink is white on black background)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Horizontal morphological structuring element to isolate straight strike lines
        horiz_len = max(5, min(max(1, w - 2), max(self.min_line_width, int(w * 0.012))))
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
        horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

        # 3. Horizontal bridge closing on isolated lines to connect broken/dotted pen segments
        bridge_len = max(3, min(15, max(3, int(w * 0.1))))
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_len, 1))
        bridged_lines = cv2.morphologyEx(horiz_lines, cv2.MORPH_CLOSE, bridge_kernel)

        # 3a. Page-Wide Notebook Ruling Line Detection & Suppression
        # Genuine notebook ruling lines span across the entire page (length > 80% of page width).
        # Short stroke segments lying along these ruling lines are suppressed.
        ruling_len = max(100, int(w * 0.82))
        ruling_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ruling_len, 1))
        page_ruling_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ruling_kernel)
        ruling_mask = cv2.dilate(page_ruling_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

        # 3b. Multi-Angle & Diagonal Line Detection via HoughLinesP
        # Catches diagonal cross-outs and slashes (-35 deg to +35 deg) across multi-word clauses
        diagonal_mask = np.zeros_like(binary)
        hough_min_len = max(14, int(horiz_len * 0.8))
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=max(20, int(hough_min_len * 0.6)),
            minLineLength=hough_min_len,
            maxLineGap=max(4, min(12, int(w * 0.01)))
        )

        hough_angles = {}
        detected_x_cross_blocks: List[StrikethroughBlock] = []
        valid_hough_lines = []

        if lines is not None and len(lines) > 0:
            lines_reshaped = lines.reshape(-1, 4)
            for x1, y1, x2, y2 in lines_reshaped:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                angle_deg = float(np.rad2deg(np.arctan2(dy, dx)))
                if angle_deg < -90:
                    angle_deg += 180
                elif angle_deg > 90:
                    angle_deg -= 180

                # Filter for genuine diagonal strikes (between 4 and 75 degrees)
                abs_ang = abs(angle_deg)
                if 4.0 <= abs_ang <= 75.0:
                    seg_len = float(np.hypot(dx, dy))
                    if abs_ang < 7.0:
                        min_req_len = max(60, int(w * 0.12))
                    elif abs_ang <= 35.0:
                        min_req_len = max(30, int(w * 0.05))
                    else:
                        # Steep diagonal / multi-line cross-outs (35 to 75 deg)
                        min_req_len = max(35, int(min(w, h) * 0.05))

                    if seg_len >= max(hough_min_len, min_req_len):
                        # Stroke continuity check: a genuine pen stroke has solid ink along the line.
                        num_samples = max(10, int(seg_len))
                        xs = np.linspace(x1, x2, num_samples).astype(int)
                        ys = np.linspace(y1, y2, num_samples).astype(int)
                        valid_pts = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                        if np.sum(valid_pts) > 0:
                            ink_ratio = float(np.sum(binary[ys[valid_pts], xs[valid_pts]] > 0)) / float(np.sum(valid_pts))
                            if ink_ratio >= 0.55:
                                cv2.line(diagonal_mask, (x1, y1), (x2, y2), 255, thickness=2)
                                key = (int(x1 // 20), int(y1 // 20))
                                hough_angles[key] = angle_deg
                                valid_hough_lines.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), angle_deg))

            # Detect pairs of intersecting opposite-sloped lines forming an X-cross
            for i in range(len(valid_hough_lines)):
                for j in range(i + 1, len(valid_hough_lines)):
                    ax1, ay1, ax2, ay2, ang_a = valid_hough_lines[i]
                    bx1, by1, bx2, by2, ang_b = valid_hough_lines[j]
                    if (ang_a * ang_b < -100) and (abs(ang_a) >= 20.0 and abs(ang_b) >= 20.0):
                        # Opposing diagonal slopes: check bounding box intersection
                        ix1 = max(ax1, bx1)
                        iy1 = max(ay1, by1)
                        ix2 = min(ax2, bx2)
                        iy2 = min(ay2, by2)
                        if ix1 < ix2 and iy1 < iy2:
                            x_span = max(ax2, bx2) - min(ax1, bx1)
                            y_span = max(ay2, by2) - min(ay1, by1)
                            if x_span >= max(40, int(w * 0.05)) and y_span >= max(30, int(h * 0.02)):
                                detected_x_cross_blocks.append(StrikethroughBlock(
                                    y_pct=round((float(min(ay1, by1)) / float(h)) * 100.0, 1),
                                    y2_pct=round((float(max(ay2, by2)) / float(h)) * 100.0, 1),
                                    x_pct=round((float(min(ax1, bx1)) / float(w)) * 100.0, 1),
                                    x2_pct=round((float(max(ax2, bx2)) / float(w)) * 100.0, 1),
                                    line_count=max(2, int(round(y_span / 30.0))),
                                    stroke_type="x_cross",
                                    confidence=0.98
                                ))

        # Combine horizontal bridged lines and diagonal Hough segments
        merged_lines = cv2.bitwise_or(bridged_lines, diagonal_mask)

        # 4. Vertical structuring element to detect vertical table / grid column dividers
        vert_len = max(3, min(max(3, int(h * 0.5)), max(18, int(h * 0.015)))) if h >= 6 else 1
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
        vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)

        # 4b. Flowchart & Closed Diagram Box Detection
        # Students frequently draw boxes around flowchart nodes or question parts (e.g. Q2 flowchart in SE_11_Q1_0020).
        # We detect closed box boundaries to prevent box edges from triggering false-positive strikethroughs.
        box_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, box_close_kernel)
        box_contours, _ = cv2.findContours(closed_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        box_border_mask = np.zeros((h, w), dtype=np.uint8)
        min_box_w = max(50, int(w * 0.05))
        min_box_h = max(35, int(h * 0.03))

        for cnt in box_contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw >= min_box_w and bh >= min_box_h and bw < int(w * 0.95):
                cnt_area = cv2.contourArea(cnt)
                rect_area = float(bw * bh)
                area_ratio = float(cnt_area / max(1.0, rect_area))
                arc_len = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * arc_len, True)
                interior_density = float(np.mean(closed_binary[by:by + bh, bx:bx + bw] > 0))
                # Structural container boxes are rectangular, convex, and hollow
                if 4 <= len(approx) <= 8 and cv2.isContourConvex(approx) and area_ratio >= 0.80 and interior_density <= 0.35:
                    cv2.drawContours(box_border_mask, [cnt], -1, 255, thickness=max(2, min(5, int(min(bw, bh) * 0.04))))

        # 5. Find connected components on isolated lines
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(merged_lines, connectivity=8)

        detected_regions: List[StrikethroughRegion] = []
        detected_underlines: List[StrikethroughRegion] = []
        multi_word_count = 0
        multi_word_threshold = max(100, int(w * 0.12))

        for i in range(1, num_labels):
            rx = int(stats[i, cv2.CC_STAT_LEFT])
            ry = int(stats[i, cv2.CC_STAT_TOP])
            rw = int(stats[i, cv2.CC_STAT_WIDTH])
            rh = int(stats[i, cv2.CC_STAT_HEIGHT])

            has_diag_ink = np.any(diagonal_mask[ry:ry + rh, rx:rx + rw] > 0)
            stroke_angle = 0.0
            if has_diag_ink and hough_angles:
                for (kx, ky), ang in hough_angles.items():
                    if rx - 20 <= kx * 20 <= rx + rw + 20 and ry - 20 <= ky * 20 <= ry + rh + 20:
                        stroke_angle = ang
                        break
            if stroke_angle == 0.0 and has_diag_ink:
                stroke_angle = float(np.rad2deg(np.arctan2(rh, rw)))

            # Exclude full-width page rules / margins / underlines
            max_allowed_h = max(self.max_line_height, int(rw * 2.5)) if (abs(stroke_angle) >= 4.0 or has_diag_ink) else (self.max_line_height + 4)
            if rw >= hough_min_len and rh <= max_allowed_h and rw < int(w * 0.85):
                # 5a. Optical Density Check: Genuine pen ink has dark stroke core.
                # Faint reverse-side bleed-through ink has high grayscale values (>150-160).
                stroke_roi_gray = gray[ry:ry + rh, rx:rx + rw]
                if stroke_roi_gray.size > 0 and np.min(stroke_roi_gray) > 145:
                    # Stroke is too faint / ghostly (bleed-through from reverse page)
                    continue

                # 5a-2. Notebook Ruling Line Suppression:
                # Reject short strokes that lie directly on page-wide notebook ruling lines
                if rw < int(w * 0.75):
                    stroke_ruling_overlap = ruling_mask[ry:ry + rh, rx:rx + rw]
                    if stroke_ruling_overlap.size > 0:
                        ruling_overlap_ratio = float(np.sum(stroke_ruling_overlap > 0)) / float(stroke_ruling_overlap.size)
                        if ruling_overlap_ratio >= 0.45 and not has_diag_ink:
                            # Stroke is collinear with printed notebook ruling line
                            continue

                # 5b. Table Gridline Rejection:
                # Count vertical grid lines that extend BOTH well above AND well below the stroke.
                # Real table column dividers cross across table cells. Letter ascenders/descenders only extend to one side.
                v_check_dist = max(12, int(h * 0.012))
                min_ext = max(9, int(v_check_dist * 0.75))
                above_slice = vert_lines[max(0, ry - v_check_dist):ry, rx:rx + rw]
                below_slice = vert_lines[ry + rh:min(h, ry + rh + v_check_dist), rx:rx + rw]
                v_through = (np.sum(above_slice > 0, axis=0) >= min_ext) & (np.sum(below_slice > 0, axis=0) >= min_ext)
                cols = np.where(v_through)[0]
                if len(cols) > 0:
                    distinct_cols = []
                    for c in cols:
                        if not distinct_cols or (c - distinct_cols[-1]) >= 15:
                            distinct_cols.append(c)
                    num_v_crossings = len(distinct_cols)
                else:
                    num_v_crossings = 0

                if num_v_crossings >= 3:
                    # Intersects 3+ structural table grid dividers: this is a table row line, not a strikethrough
                    continue

                # 5b-2. Flowchart / Diagram Box Perimeter Overlap Rejection
                stroke_box_overlap = box_border_mask[ry:ry + rh, rx:rx + rw]
                if stroke_box_overlap.size > 0:
                    overlap_ratio = float(np.sum(stroke_box_overlap > 0)) / float(stroke_box_overlap.size)
                    if overlap_ratio >= 0.35:
                        # Stroke lies on a detected flowchart box or diagram perimeter
                        continue

                # 5b-2b. Teacher Grading Ink & Underline Rejection
                # Reject candidate strokes that overlap with detected red teacher ink (e.g. teacher checkmarks, underlines)
                if teacher_mask is not None:
                    if teacher_mask.shape[:2] == (h, w):
                        stroke_teacher_overlap = teacher_mask[ry:ry + rh, rx:rx + rw]
                        if stroke_teacher_overlap.size > 0:
                            overlap_teacher_ratio = float(np.sum(stroke_teacher_overlap > 0)) / float(stroke_teacher_overlap.size)
                            if overlap_teacher_ratio >= 0.25:
                                # Overlaps significantly with teacher grading marks: reject
                                continue

                # 5b-3. Dual-Endpoint Corner Connectivity Rejection:
                # A horizontal box/cell border terminates into perpendicular vertical strokes at BOTH ends.
                # Check strictly at the stroke endpoints (not reaching deep into word letters).
                v_check_h = max(14, min(40, int(h * 0.02)))
                left_x1 = max(0, rx - 3)
                left_x2 = min(w, rx + 3)
                left_v_up = np.sum(vert_lines[max(0, ry - v_check_h):ry, left_x1:left_x2] > 0)
                left_v_down = np.sum(vert_lines[ry + rh:min(h, ry + rh + v_check_h), left_x1:left_x2] > 0)
                has_left_corner = (left_v_up > 8) or (left_v_down > 8)

                right_x1 = max(0, rx + rw - 3)
                right_x2 = min(w, rx + rw + 3)
                right_v_up = np.sum(vert_lines[max(0, ry - v_check_h):ry, right_x1:right_x2] > 0)
                right_v_down = np.sum(vert_lines[ry + rh:min(h, ry + rh + v_check_h), right_x1:right_x2] > 0)
                has_right_corner = (right_v_up > 8) or (right_v_down > 8)

                if has_left_corner and has_right_corner:
                    # Both endpoints connect to structural vertical borders: this is a box/cell boundary
                    continue

                # 5c. Immediate Proximity Glyph Piercing Check
                # A strikethrough cuts through handwritten letters.
                if has_diag_ink:
                    box_ink = np.sum(binary[ry:ry + rh, rx:rx + rw] > 0)
                    if box_ink < max(20, int(rw * 0.15)):
                        continue
                else:
                    imm_top = max(0, ry - 8)
                    imm_bot = min(h, ry + rh + 12)
                    imm_above = np.sum(binary[imm_top:ry, rx:rx + rw] > 0)
                    imm_below = np.sum(binary[ry + rh:imm_bot, rx:rx + rw] > 0)
                    min_imm_px = max(2, int(rw * 0.02))

                    # Broader context ink verification
                    y_top = max(0, ry - 14)
                    y_bot = min(h, ry + rh + 14)
                    above_pixels = np.sum(binary[y_top:ry, rx:rx + rw] > 0)
                    below_pixels = np.sum(binary[ry + rh:y_bot, rx:rx + rw] > 0)
                    min_text_px = max(10, int(rw * 0.1))

                    # Estimate stroke angle
                    comp_key = (int(rx // 20), int(ry // 20))
                    stroke_angle = hough_angles.get(comp_key, 0.0)
                    if stroke_angle == 0.0 and rh > 3:
                        stroke_angle = float(np.rad2deg(np.arctan2(rh, rw)))

                    if imm_above < min_imm_px or above_pixels < min_text_px:
                        # No text above the stroke: stray noise or isolated margin rule
                        continue

                    # Descender-aware analysis:
                    # Check the horizontal column occupancy of ink below the stroke.
                    # Underlines beneath words with descenders (g, j, p, q, y) only have ink in sparse narrow columns.
                    below_slice = binary[ry + rh:imm_bot, rx:rx + rw]
                    cols_with_ink = np.sum(below_slice > 0, axis=0) > 0
                    col_occupancy = float(np.mean(cols_with_ink)) if cols_with_ink.size > 0 else 0.0
                    tot_context_ink = float(above_pixels + below_pixels)
                    ratio_above = (float(above_pixels) / tot_context_ink) if tot_context_ink > 0 else 1.0

                    # 5c-2. Letter 't' Crossbar Discrimination:
                    # A letter 't' crossbar sits near the top of the letter height (above the x-height/midline),
                    # so >=78% of surrounding glyph ink is strictly BELOW the crossbar (ratio_above < 0.22).
                    # Genuine word strikethroughs cut through the midline of the word, where ink is balanced above and below.
                    if rw <= 50 and ratio_above < 0.26 and not has_diag_ink and abs(stroke_angle) <= 10.0:
                        # Isolated crossbar on 't' ascender, not a strikethrough
                        continue

                    is_descender_underline = (col_occupancy <= 0.25 and ratio_above >= 0.65 and rw >= 25)
                    is_clear_underline = (imm_below < min_imm_px)

                    if (is_clear_underline or is_descender_underline) and not has_diag_ink and abs(stroke_angle) <= 10.0:
                        detected_underlines.append(StrikethroughRegion(
                            x=rx, y=ry, w=rw, h=rh,
                            y_pct=round((float(ry) / float(h)) * 100.0, 1),
                            y2_pct=round((float(ry + rh) / float(h)) * 100.0, 1),
                            x_pct=round((float(rx) / float(w)) * 100.0, 1),
                            x2_pct=round((float(rx + rw) / float(w)) * 100.0, 1),
                            angle=round(stroke_angle, 1),
                            confidence=min(1.0, float(rw / 100.0) + 0.3),
                            is_multi_word=rw >= multi_word_threshold,
                            is_underline=True
                        ))
                        continue

                    # 5d. Underline vs. Strikethrough Baseline Discriminator (for lines with descenders below)
                    # An underline runs beneath a text line (or along the baseline), where:
                    # 1. >75% of surrounding text ink is strictly above the stroke.
                    # 2. The stroke y-position is at or below the 75th percentile of the text bounding band (rho >= 0.75).
                    if not has_diag_ink and abs(stroke_angle) <= 10.0:
                        v_band = max(20, min(60, int(h * 0.035)))
                        y_top_band = max(0, ry - v_band)
                        y_bot_band = min(h, ry + rh + v_band)

                        band_above = float(np.sum(binary[y_top_band:ry, rx:rx + rw] > 0))
                        band_below = float(np.sum(binary[ry + rh:y_bot_band, rx:rx + rw] > 0))
                        tot_band_ink = band_above + band_below

                        if tot_band_ink > 0:
                            ratio_above = band_above / tot_band_ink
                            local_text = binary[y_top_band:y_bot_band, rx:rx + rw]
                            text_rows = np.where(local_text > 0)[0]
                            if len(text_rows) > 0:
                                y_min_text = float(np.percentile(text_rows, 5))
                                y_max_text = float(np.percentile(text_rows, 95))
                                text_span = max(1.0, y_max_text - y_min_text)
                                stroke_mid_y = float(ry - y_top_band) + rh / 2.0
                                rho = (stroke_mid_y - y_min_text) / text_span

                                if (rho >= 0.75 and ratio_above >= 0.75) or ratio_above >= 0.85:
                                    # Conclusively an underline underneath a heading, answer title, or word
                                    detected_underlines.append(StrikethroughRegion(
                                        x=rx, y=ry, w=rw, h=rh,
                                        y_pct=round((float(ry) / float(h)) * 100.0, 1),
                                        y2_pct=round((float(ry + rh) / float(h)) * 100.0, 1),
                                        x_pct=round((float(rx) / float(w)) * 100.0, 1),
                                        x2_pct=round((float(rx + rw) / float(w)) * 100.0, 1),
                                        angle=round(stroke_angle, 1),
                                        confidence=min(1.0, float(rw / 100.0) + 0.3),
                                        is_multi_word=rw >= multi_word_threshold,
                                        is_underline=True
                                    ))
                                    continue

                is_multi = rw >= multi_word_threshold
                if is_multi:
                    multi_word_count += 1

                y_pct = round((float(ry) / float(h)) * 100.0, 1)
                y2_pct = round((float(ry + rh) / float(h)) * 100.0, 1)
                x_pct = round((float(rx) / float(w)) * 100.0, 1)
                x2_pct = round((float(rx + rw) / float(w)) * 100.0, 1)

                detected_regions.append(StrikethroughRegion(
                    x=rx,
                    y=ry,
                    w=rw,
                    h=rh,
                    y_pct=y_pct,
                    y2_pct=y2_pct,
                    x_pct=x_pct,
                    x2_pct=x2_pct,
                    angle=round(stroke_angle, 1),
                    confidence=min(1.0, float(rw / 100.0) + 0.3),
                    is_multi_word=is_multi,
                    is_underline=False
                ))

        # 6. Composite Multi-Line Block Clustering (Parallel Horizontal, Steep Diagonal, and X-Cross)
        blocks: List[StrikethroughBlock] = []
        clustered_region_indices = set()

        # Add pre-detected X-cross blocks
        for xb in detected_x_cross_blocks:
            blocks.append(xb)

        # 6a. Steep Diagonals & Multi-line Slashes (|angle| >= 30 deg or (rh >= 35px and |angle| >= 20 deg))
        for idx, r in enumerate(detected_regions):
            if abs(r.angle) >= 30.0 or (r.h >= max(35, int(h * 0.035)) and abs(r.angle) >= 20.0):
                blocks.append(StrikethroughBlock(
                    y_pct=r.y_pct,
                    y2_pct=r.y2_pct,
                    x_pct=r.x_pct,
                    x2_pct=r.x2_pct,
                    line_count=max(2, int(round(r.h / 30.0))),
                    stroke_type="steep_diagonal",
                    confidence=r.confidence
                ))
                clustered_region_indices.add(idx)

        # Ruled Notebook Paper Detection & Filtering:
        # If a page contains many thin horizontal segments distributed across the entire page,
        # filter out spurious paper rulings BEFORE clustering parallel horizontal blocks!
        if len(detected_regions) > 6:
            horizontal_strikes = [
                r for i, r in enumerate(detected_regions)
                if abs(r.angle) <= 2.5 and r.h <= 5 and not r.is_multi_word
            ]
            if len(horizontal_strikes) >= 6:
                y_positions = sorted(r.y for r in horizontal_strikes)
                y_span = y_positions[-1] - y_positions[0]
                if y_span > int(h * 0.3):
                    detected_regions = [
                        r for r in detected_regions
                        if abs(r.angle) >= 3.5
                        or r.h >= 6
                        or (r.is_multi_word and abs(r.angle) >= 1.5)
                    ]
                    multi_word_count = sum(1 for r in detected_regions if r.is_multi_word)
                    clustered_region_indices = set()

        # 6b. Parallel Horizontal Line Clustering
        # First group regions that belong to the same line of text (within ~10-12px vertically).
        # This prevents multiple struck words or fragmented segments on the same line from falsely breaking multi-line clusters.
        horiz_indices = [
            i for i, r in enumerate(detected_regions)
            if i not in clustered_region_indices and abs(r.angle) <= 15.0
        ]
        horiz_indices.sort(key=lambda i: detected_regions[i].y)

        line_h_est = max(24, min(80, int(h * 0.04)))
        max_y_gap = max(55, int(line_h_est * 2.2))
        same_line_v_thresh = max(8, int(line_h_est * 0.35))

        # Group horizontal regions into distinct text line bands
        line_groups: List[List[int]] = []
        for idx in horiz_indices:
            r = detected_regions[idx]
            placed = False
            for group in line_groups:
                grp_y = float(np.mean([detected_regions[gi].y for gi in group]))
                if abs(r.y - grp_y) <= same_line_v_thresh:
                    group.append(idx)
                    placed = True
                    break
            if not placed:
                line_groups.append([idx])

        # Sort line groups from top to bottom
        line_groups.sort(key=lambda grp: min(detected_regions[gi].y for gi in grp))

        # Cluster consecutive text lines into multi-line blocks
        clusters: List[List[List[int]]] = []
        curr_cluster_lines: List[List[int]] = []

        for grp in line_groups:
            if not curr_cluster_lines:
                curr_cluster_lines.append(grp)
            else:
                prev_grp = curr_cluster_lines[-1]
                prev_min_y = min(detected_regions[gi].y for gi in prev_grp)
                prev_min_x = min(detected_regions[gi].x for gi in prev_grp)
                prev_max_x = max(detected_regions[gi].x + detected_regions[gi].w for gi in prev_grp)
                prev_w = max(1, prev_max_x - prev_min_x)

                curr_min_y = min(detected_regions[gi].y for gi in grp)
                curr_min_x = min(detected_regions[gi].x for gi in grp)
                curr_max_x = max(detected_regions[gi].x + detected_regions[gi].w for gi in grp)
                curr_w = max(1, curr_max_x - curr_min_x)

                y_dist = curr_min_y - prev_min_y
                x_overlap = min(curr_max_x, prev_max_x) - max(curr_min_x, prev_min_x)
                min_w = min(curr_w, prev_w)
                overlap_ratio = float(x_overlap / max(1.0, float(min_w)))

                if 8 <= y_dist <= max_y_gap and overlap_ratio >= 0.25:
                    curr_cluster_lines.append(grp)
                else:
                    if len(curr_cluster_lines) >= 2:
                        clusters.append(list(curr_cluster_lines))
                    curr_cluster_lines = [grp]

        if len(curr_cluster_lines) >= 2:
            clusters.append(list(curr_cluster_lines))

        for c_lines in clusters:
            all_indices = [idx for line in c_lines for idx in line]
            for idx in all_indices:
                clustered_region_indices.add(idx)
            c_regs = [detected_regions[idx] for idx in all_indices]
            min_y = min(r.y for r in c_regs)
            max_y2 = max(r.y + r.h for r in c_regs)
            min_x = min(r.x for r in c_regs)
            max_x2 = max(r.x + r.w for r in c_regs)
            blocks.append(StrikethroughBlock(
                y_pct=round((float(min_y) / float(h)) * 100.0, 1),
                y2_pct=round((float(max_y2) / float(h)) * 100.0, 1),
                x_pct=round((float(min_x) / float(w)) * 100.0, 1),
                x2_pct=round((float(max_x2) / float(w)) * 100.0, 1),
                line_count=len(c_lines),
                stroke_type="parallel_horizontal",
                confidence=0.95
            ))

        has_strike = len(detected_regions) > 0 or len(blocks) > 0
        details = (
            f"Detected {len(detected_regions)} strikethrough stroke(s) "
            f"({multi_word_count} multi-word clause strike(s), {len(blocks)} multi-line block(s)), "
            f"{len(detected_underlines)} underline(s)."
        )

        return StrikethroughDetectionResult(
            has_strikethrough=has_strike,
            region_count=len(detected_regions),
            multi_word_count=multi_word_count,
            regions=detected_regions,
            underlines=detected_underlines,
            blocks=blocks,
            details=details
        )
