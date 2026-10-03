"""
Module: band_tiler.py
Native-DPI Overlapping Horizontal Band Tiler and Multi-Band Transcript Stitcher.

Eliminates downsampling blur on high-resolution handwritten exam scans by slicing
pages into overlapping horizontal bands at 100% native resolution, while preserving
stroke connectivity and micro-details (e.g. cursive loops, fine strikethroughs).
"""

import re
import math
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from difflib import SequenceMatcher
from pydantic import BaseModel, Field, ConfigDict


class PageBand(BaseModel):
    """A horizontal strip/band of a page at native resolution."""
    band_index: int
    total_bands: int
    y_start_px: int
    y_end_px: int
    y_start_pct: float
    y_end_pct: float
    width_px: int
    height_px: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BandTiler:
    """
    Slices a high-resolution page into 2-4 overlapping horizontal bands
    at native DPI to ensure characters remain at 60-80px height without downscaling.
    """

    def __init__(
        self,
        target_band_height: int = 950,
        min_overlap_ratio: float = 0.25,
        max_page_height_for_single_band: int = 1350
    ):
        self.target_band_height = target_band_height
        self.min_overlap_ratio = min_overlap_ratio
        self.max_page_height_for_single_band = max_page_height_for_single_band

    def tile_page(self, image: Image.Image) -> Tuple[List[Image.Image], List[PageBand]]:
        """
        Partition page into overlapping horizontal bands.
        Returns:
            band_images: List of cropped PIL Images (native DPI)
            band_metadata: List of PageBand metadata objects
        """
        w, h = image.size

        # If page is small enough to fit within native VLM budget without downscaling, single band
        if h <= self.max_page_height_for_single_band:
            meta = PageBand(
                band_index=0,
                total_bands=1,
                y_start_px=0,
                y_end_px=h,
                y_start_pct=0.0,
                y_end_pct=100.0,
                width_px=w,
                height_px=h
            )
            return [image], [meta]

        # Calculate number of bands (typically 3 for a 2800-3400px page)
        num_bands = max(2, min(4, math.ceil(h / float(self.target_band_height))))

        # Calculate band height with overlap
        # Total covered height = N * band_h - (N - 1) * overlap = h
        # Let overlap = min_overlap_ratio * band_h
        # band_h * (N - (N - 1) * min_overlap_ratio) = h
        denom = float(num_bands) - float(num_bands - 1) * self.min_overlap_ratio
        band_h = int(math.ceil(float(h) / max(1.0, denom)))
        band_h = max(band_h, int(h * 0.35))
        band_h = min(band_h, h)

        step = int(float(h - band_h) / float(num_bands - 1)) if num_bands > 1 else h

        band_images: List[Image.Image] = []
        band_metadata: List[PageBand] = []

        for idx in range(num_bands):
            y_start = idx * step
            y_end = min(h, y_start + band_h)
            if idx == num_bands - 1:
                y_end = h
                y_start = max(0, h - band_h)

            crop_box = (0, y_start, w, y_end)
            band_img = image.crop(crop_box)
            band_images.append(band_img)

            band_metadata.append(PageBand(
                band_index=idx,
                total_bands=num_bands,
                y_start_px=y_start,
                y_end_px=y_end,
                y_start_pct=round((y_start / float(h)) * 100.0, 1),
                y_end_pct=round((y_end / float(h)) * 100.0, 1),
                width_px=w,
                height_px=y_end - y_start
            ))

        return band_images, band_metadata


def stitch_band_transcripts(band_texts: List[str]) -> str:
    """
    Seamlessly merge overlapping band transcriptions into a continuous full-page transcript.
    Uses line-level SequenceMatcher to detect and eliminate duplicate lines in the overlap zones
    while preserving natural line breaks and paragraph spacing.
    """
    if not band_texts:
        return ""
    if len(band_texts) == 1:
        return band_texts[0].strip()

    merged_lines: List[str] = [line.rstrip() for line in band_texts[0].strip().split("\n")]

    for b_idx in range(1, len(band_texts)):
        next_lines = [line.rstrip() for line in band_texts[b_idx].strip().split("\n")]
        if not next_lines:
            continue
        if not merged_lines:
            merged_lines = next_lines
            continue

        # Look for overlap: compare non-empty lines from the tail of merged_lines with the head of next_lines
        tail_non_empty = [l for l in merged_lines if l.strip()]
        head_non_empty = [l for l in next_lines if l.strip()]

        best_overlap_head = 0
        best_match_score = 0.0

        max_lookback = min(len(tail_non_empty), 8)
        max_lookahead = min(len(head_non_empty), 8)

        for overlap_len in range(1, min(max_lookback, max_lookahead) + 1):
            tail_slice = tail_non_empty[-overlap_len:]
            head_slice = head_non_empty[:overlap_len]

            tail_str = " ".join(tail_slice).lower()
            head_str = " ".join(head_slice).lower()

            sm = SequenceMatcher(None, tail_str, head_str)
            score = sm.ratio()

            if score > 0.75 and score > best_match_score:
                best_match_score = score
                best_overlap_head = overlap_len

        if best_match_score > 0.75:
            matched_head_lines = head_non_empty[:best_overlap_head]
            last_matched_line = matched_head_lines[-1]
            cut_idx = 0
            for idx, nl in enumerate(next_lines):
                if SequenceMatcher(None, nl.strip().lower(), last_matched_line.strip().lower()).ratio() >= 0.85:
                    cut_idx = idx + 1
                    break
            merged_lines.extend(next_lines[cut_idx:])
        else:
            # Fallback: check first 3 non-empty lines of next_lines
            cut_idx = 0
            for nl in head_non_empty[:3]:
                nl_clean = nl.strip().lower()
                matched = False
                for ml in tail_non_empty[-4:]:
                    if SequenceMatcher(None, ml.strip().lower(), nl_clean).ratio() >= 0.82:
                        matched = True
                        break
                if matched:
                    for idx, raw_nl in enumerate(next_lines):
                        if raw_nl.strip() == nl:
                            cut_idx = idx + 1
                            break
                else:
                    break
            merged_lines.extend(next_lines[cut_idx:])

    result = "\n".join(merged_lines)
    result = re.sub(r'\n{3,}', '\n\n', result).strip()
    return result
