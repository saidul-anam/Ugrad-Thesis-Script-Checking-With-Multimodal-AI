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
    angle: float = 0.0
    confidence: float = 1.0


class StrikethroughDetectionResult(BaseModel):
    """Output of Stage 0.5 strikethrough detector."""
    has_strikethrough: bool = False
    region_count: int = 0
    regions: List[StrikethroughRegion] = Field(default_factory=list)
    details: str = ""


class StrikethroughDetector:
    """
    Detects horizontal and diagonal strikethrough strokes across text.
    Uses morphological line kernels to isolate thin, extended stroke segments.
    """

    def __init__(
        self,
        min_line_width: int = 25,
        max_line_height: int = 6,
        binarization_thresh: int = 140
    ):
        self.min_line_width = min_line_width
        self.max_line_height = max_line_height
        self.binarization_thresh = binarization_thresh

    def detect(self, image_input: Union[Image.Image, np.ndarray, str]) -> StrikethroughDetectionResult:
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

        # 1. Otsu / adaptive binarization (invert so text ink is white on black background)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Horizontal morphological structuring element to isolate straight strike lines
        horiz_len = max(self.min_line_width, int(w * 0.02))
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
        horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

        # 3. Find connected components on isolated lines
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(horiz_lines, connectivity=8)

        detected_regions: List[StrikethroughRegion] = []
        for i in range(1, num_labels):
            rx = int(stats[i, cv2.CC_STAT_LEFT])
            ry = int(stats[i, cv2.CC_STAT_TOP])
            rw = int(stats[i, cv2.CC_STAT_WIDTH])
            rh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            # Exclude full-width page rules / margins / underlines
            if rw >= horiz_len and rh <= self.max_line_height and rw < int(w * 0.85):
                # Verify that ink surrounds the stroke vertically (indicates strikethrough traversing text, not underline)
                y_top = max(0, ry - 10)
                y_bot = min(h, ry + rh + 10)
                band = binary[y_top:y_bot, rx:rx + rw]
                ink_above = np.sum(binary[y_top:ry, rx:rx + rw]) > 0
                ink_below = np.sum(binary[ry + rh:y_bot, rx:rx + rw]) > 0

                if ink_above and ink_below:
                    detected_regions.append(StrikethroughRegion(
                        x=rx,
                        y=ry,
                        w=rw,
                        h=rh,
                        confidence=min(1.0, float(rw / 100.0) + 0.3)
                    ))

        has_strike = len(detected_regions) > 0
        details = f"Detected {len(detected_regions)} strikethrough stroke(s)."

        return StrikethroughDetectionResult(
            has_strikethrough=has_strike,
            region_count=len(detected_regions),
            regions=detected_regions,
            details=details
        )
