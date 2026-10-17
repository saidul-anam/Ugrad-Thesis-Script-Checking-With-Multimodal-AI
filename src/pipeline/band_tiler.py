"""
Module: band_tiler.py
Native-DPI Overlapping Horizontal Band Tiler and Multi-Band Transcript Stitcher.

Eliminates downsampling blur on high-resolution handwritten exam scans by slicing
pages into overlapping horizontal bands at 100% native resolution, while preserving
stroke connectivity and micro-details (e.g. cursive loops, fine strikethroughs).
"""

import re
import math
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
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


_TRUNCATED_RE = re.compile(r"\[truncated\]", re.IGNORECASE)


def _band_tokens(text: str) -> List[Tuple[Optional[str], int, int, int]]:
    """
    Word tokens of a transcript as (key, start, end, line_no). The key is the lower-cased word without
    punctuation; punctuation-only tokens are skipped. A token carrying [truncated] gets key None, so a
    word cut by the band edge never anchors an overlap.
    """
    toks: List[Tuple[Optional[str], int, int, int]] = []
    line_no = 0
    for m in re.finditer(r"\S+|\n", text):
        if m.group() == "\n":
            line_no += 1
            continue
        if _TRUNCATED_RE.search(m.group()):
            key = None
        else:
            key = re.sub(r"[\W_]", "", m.group().lower())
            if not key:
                continue
        toks.append((key, m.start(), m.end(), line_no))
    return toks


def _line_spans(toks) -> Dict[int, Tuple[int, int]]:
    """line_no -> (first token index, last token index)."""
    spans: Dict[int, Tuple[int, int]] = {}
    for idx, t in enumerate(toks):
        lo, _ = spans.get(t[3], (idx, idx))
        spans[t[3]] = (lo, idx)
    return spans


def _overlap_seam(a_toks, b_toks, a_window: int, b_window: int) -> Optional[Tuple[int, int]]:
    """
    Find where band B continues band A: the longest run of identical words between the tail of A and the
    head of B. Returns (index in A, index in B) of the run's last word, or None when the bands share no
    such run.

    The tiler overlaps neighbouring bands by a quarter to a third of a band's height, so the shared text
    always lies in the last half of A and the first half of B; the search is limited to those halves
    (widened to whole lines) so a phrase the student repeated elsewhere cannot be taken for the overlap.
    A run is accepted only if it contains a complete line of both bands and more than one word: a line
    that lies wholly inside the overlap is read in full by both bands, while a single word (a lone
    "Ans:") can match by chance.
    """
    if not a_toks or not b_toks:
        return None
    a_spans, b_spans = _line_spans(a_toks), _line_spans(b_toks)
    a_lo = a_spans[a_toks[max(0, len(a_toks) - a_window)][3]][0]
    b_hi = b_spans[b_toks[min(len(b_toks), b_window) - 1][3]][1] + 1
    A = [t[0] for t in a_toks[a_lo:]]
    B = [t[0] for t in b_toks[:b_hi]]

    best_n, best_x, best_y = 0, -1, -1
    prev = [0] * (len(B) + 1)
    for x, ka in enumerate(A):
        cur = [0] * (len(B) + 1)
        if ka is not None:
            for y, kb in enumerate(B):
                if ka == kb:
                    n = cur[y + 1] = prev[y] + 1
                    if n > best_n or (n == best_n and x > best_x):   # ties: the run nearest A's end
                        best_n, best_x, best_y = n, x, y
        prev = cur
    if best_n < 2:
        return None

    a_end, b_end = a_lo + best_x, best_y
    a_start, b_start = a_end - best_n + 1, b_end - best_n + 1

    def has_full_line(toks, spans, lo, hi):
        return any(spans[toks[i][3]][0] >= lo and spans[toks[i][3]][1] <= hi for i in range(lo, hi + 1))

    if not (has_full_line(a_toks, a_spans, a_start, a_end) and has_full_line(b_toks, b_spans, b_start, b_end)):
        return None
    return a_end, b_end


def _drop_marker_only_edge_line(lines: List[str], from_end: bool) -> List[str]:
    """Remove a first/last line that holds nothing but a [truncated] marker (no student text)."""
    order = range(len(lines) - 1, -1, -1) if from_end else range(len(lines))
    for idx in order:
        if lines[idx].strip():
            if not _TRUNCATED_RE.sub("", lines[idx]).strip():
                return lines[:idx] + lines[idx + 1:]
            break
    return lines


def _similar_lines_merge(upper_lines: List[str], lower_lines: List[str]) -> str:
    """
    Join two bands that share no exact run of words, which happens when the two bands misread the overlap
    differently ("normal trees" / "norormal troes"). This is the pipeline's original line matcher, with its
    original ratios: the k (at most 8) last non-empty lines of the upper band that best resemble the k first
    lines of the lower band (SequenceMatcher ratio above 0.75) are taken as the overlap and dropped from the
    lower band. Failing that, leading lower-band lines that closely match one of the upper band's last four
    lines (ratio 0.82 or more) are dropped. With no similar lines both bands are kept whole.
    """
    merged_lines = [line.rstrip() for line in upper_lines]
    next_lines = [line.rstrip() for line in lower_lines]
    tail_non_empty = [l for l in merged_lines if l.strip()]
    head_non_empty = [l for l in next_lines if l.strip()]
    if not tail_non_empty or not head_non_empty:
        return "\n".join(merged_lines + next_lines)

    best_overlap_head, best_match_score = 0, 0.0
    for overlap_len in range(1, min(len(tail_non_empty), len(head_non_empty), 8) + 1):
        tail_str = " ".join(tail_non_empty[-overlap_len:]).lower()
        head_str = " ".join(head_non_empty[:overlap_len]).lower()
        score = SequenceMatcher(None, tail_str, head_str).ratio()
        if score > 0.75 and score > best_match_score:
            best_match_score, best_overlap_head = score, overlap_len

    cut_idx = 0
    if best_match_score > 0.75:
        last_matched_line = head_non_empty[best_overlap_head - 1].strip().lower()
        for idx, nl in enumerate(next_lines):
            if SequenceMatcher(None, nl.strip().lower(), last_matched_line).ratio() >= 0.85:
                cut_idx = idx + 1
                break
    else:
        for nl in head_non_empty[:3]:
            if not any(SequenceMatcher(None, ml.strip().lower(), nl.strip().lower()).ratio() >= 0.82
                       for ml in tail_non_empty[-4:]):
                break
            for idx, raw_nl in enumerate(next_lines):
                if raw_nl.strip() == nl:
                    cut_idx = idx + 1
                    break
    return "\n".join(merged_lines + next_lines[cut_idx:])


def stitch_band_transcripts(band_texts: List[str]) -> str:
    """
    Merge overlapping band transcriptions into one page transcript.

    Neighbouring bands both read the text in their overlap, and the line at each band edge is usually cut
    (read partly, or marked [truncated]). Each band is joined to the text so far at the longest run of
    identical words in the overlap (see _overlap_seam): the text before the run comes from the upper band,
    the text after it from the lower band. Partial edge lines on both sides fall away because each lies on
    the far side of the seam for the band that cut it. When no shared run is found (the bands read the
    overlap differently, or share no text), edge lines holding nothing but a [truncated] marker are dropped
    and the bands are joined by line similarity (see _similar_lines_merge).
    """
    texts = [t.strip() for t in band_texts if t and t.strip()]
    if not texts:
        return ""

    merged = texts[0]
    prev_count = len(_band_tokens(merged))
    for nxt in texts[1:]:
        a_toks, b_toks = _band_tokens(merged), _band_tokens(nxt)
        seam = _overlap_seam(a_toks, b_toks, a_window=max(1, prev_count // 2), b_window=max(1, len(b_toks) // 2))
        if seam is not None:
            merged = merged[:a_toks[seam[0]][2]] + nxt[b_toks[seam[1]][2]:]
        else:
            a_lines = _drop_marker_only_edge_line(merged.rstrip().split("\n"), from_end=True)
            b_lines = _drop_marker_only_edge_line(nxt.lstrip().split("\n"), from_end=False)
            merged = _similar_lines_merge(a_lines, b_lines)
        prev_count = len(b_toks)

    merged = "\n".join(line.rstrip() for line in merged.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", merged).strip()
