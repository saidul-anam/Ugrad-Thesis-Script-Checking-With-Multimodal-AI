"""Red-ink detection and erasure (pure computer vision).

The teacher's marks (score, ticks, crosses, circles) are in red ballpoint. The
student's answer and the printed question are black/blue-black. We isolate red
pixels in HSV, then inpaint them away so the grader never sees teacher ink.

INVARIANT: the grader must only ever see the erased image produced here.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def red_mask(bgr: np.ndarray) -> np.ndarray:
    """Return a uint8 {0,255} mask of red-ink pixels.

    Red hue wraps around the 0/180 boundary in OpenCV's HSV, so we OR two
    ranges. We also require the red channel to clearly dominate green/blue in
    BGR, which rejects the dark near-neutral strokes of black/blue-black pen
    that can drift to a reddish hue at low saturation.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 60, 40], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([165, 60, 40], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    b, g, r = cv2.split(bgr.astype(np.int16))
    dominant = ((r - g > 40) & (r - b > 40)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, dominant)

    # Drop specks, then reconnect strokes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def erase_red(bgr: np.ndarray) -> np.ndarray:
    """Remove red ink by inpainting the (dilated) red mask from its surroundings."""
    mask = red_mask(bgr)
    dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(bgr, dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def red_fraction(bgr: np.ndarray) -> float:
    """Share of pixels flagged as red — a blank/unmarked script trends toward 0."""
    return float(np.count_nonzero(red_mask(bgr))) / bgr[:, :, 0].size


def erase_file(src: Path, dst: Path) -> None:
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"could not read image: {src}")
    cv2.imwrite(str(dst), erase_red(img))


def overlay_file(src: Path, dst: Path) -> None:
    """Write a QA image: red mask painted bright green over the original."""
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"could not read image: {src}")
    mask = red_mask(img)
    vis = img.copy()
    vis[mask > 0] = (0, 255, 0)
    cv2.imwrite(str(dst), vis)
