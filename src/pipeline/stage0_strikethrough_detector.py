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


class StrikethroughDetectionResult(BaseModel):
    """Output of Stage 0.5 strikethrough detector."""
    has_strikethrough: bool = False
    region_count: int = 0
    multi_word_count: int = 0
    regions: List[StrikethroughRegion] = Field(default_factory=list)
    underlines: List[StrikethroughRegion] = Field(default_factory=list)
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

                # Filter for genuine diagonal strikes (between 4 and 35 degrees)
                # Purely horizontal strikes (|angle| < 4 deg) are handled via continuous morphological opening (horiz_lines),
                # preventing HoughLinesP from bridging letter crossbars across plain text lines into false strikes.
                abs_ang = abs(angle_deg)
                if 4.0 <= abs_ang <= 35.0:
                    seg_len = float(np.hypot(dx, dy))
                    min_req_len = max(60, int(w * 0.12)) if abs_ang < 7.0 else max(35, int(w * 0.08))
                    if seg_len >= max(hough_min_len, min_req_len):
                        # Stroke continuity check: a genuine pen stroke has solid ink along the line.
                        # Spurious Hough lines bridging gaps between letters across words have low ink ratio (<40%).
                        num_samples = max(10, int(seg_len))
                        xs = np.linspace(x1, x2, num_samples).astype(int)
                        ys = np.linspace(y1, y2, num_samples).astype(int)
                        valid_pts = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                        if np.sum(valid_pts) > 0:
                            ink_ratio = float(np.sum(binary[ys[valid_pts], xs[valid_pts]] > 0)) / float(np.sum(valid_pts))
                            if ink_ratio >= 0.60:
                                cv2.line(diagonal_mask, (x1, y1), (x2, y2), 255, thickness=2)
                                key = (int(x1 // 20), int(y1 // 20))
                                hough_angles[key] = angle_deg

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
            comp_key = (int(rx // 20), int(ry // 20))
            stroke_angle = hough_angles.get(comp_key, 0.0)
            if stroke_angle == 0.0 and has_diag_ink:
                stroke_angle = float(np.rad2deg(np.arctan2(rh, rw)))

            # Exclude full-width page rules / margins / underlines
            max_allowed_h = max(self.max_line_height, int(rw * 0.75)) if (abs(stroke_angle) >= 4.0 or has_diag_ink) else (self.max_line_height + 4)
            if rw >= hough_min_len and rh <= max_allowed_h and rw < int(w * 0.85):
                # 5a. Optical Density Check: Genuine pen ink has dark stroke core.
                # Faint reverse-side bleed-through ink has high grayscale values (>150-160).
                stroke_roi_gray = gray[ry:ry + rh, rx:rx + rw]
                if stroke_roi_gray.size > 0 and np.min(stroke_roi_gray) > 145:
                    # Stroke is too faint / ghostly (bleed-through from reverse page)
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
                    imm_bot = min(h, ry + rh + 8)
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

                    if imm_below < min_imm_px or below_pixels < min_text_px:
                        # Text above but no piercing ink below: clear underline
                        if not has_diag_ink and abs(stroke_angle) <= 10.0:
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

        # Ruled Notebook Paper Detection & Filtering:
        # If a page contains many thin horizontal segments (angle ~ 0 deg) distributed across multiple vertical heights,
        # it is ruled notebook paper where lines are paper rulings, not student cross-outs.
        if len(detected_regions) > 6:
            horizontal_strikes = [r for r in detected_regions if abs(r.angle) <= 2.5 and r.h <= 5]
            if len(horizontal_strikes) >= 6:
                y_positions = sorted(r.y for r in horizontal_strikes)
                y_span = y_positions[-1] - y_positions[0]
                if y_span > int(h * 0.3):
                    # Ruled notebook paper detected: filter out baseline horizontal lines.
                    # Keep genuine diagonal slashes (|angle| >= 3.5 deg) or heavy thick cross-outs (rh >= 6).
                    detected_regions = [
                        r for r in detected_regions
                        if abs(r.angle) >= 3.5 or r.h >= 6 or (r.is_multi_word and abs(r.angle) >= 2.0)
                    ]
                    multi_word_count = sum(1 for r in detected_regions if r.is_multi_word)

        has_strike = len(detected_regions) > 0
        details = f"Detected {len(detected_regions)} strikethrough stroke(s) ({multi_word_count} multi-word clause strike(s)), {len(detected_underlines)} underline(s)."

        return StrikethroughDetectionResult(
            has_strikethrough=has_strike,
            region_count=len(detected_regions),
            multi_word_count=multi_word_count,
            regions=detected_regions,
            underlines=detected_underlines,
            details=details
        )
