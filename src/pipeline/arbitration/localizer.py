"""
Line localization: find the physical line of handwriting that contains a candidate token and
return a crop of it.

Two strategies, in order:
  (a) ask the VLM for the bounding box of the line reading the context sentence
      ([y1, x1, y2, x2] on a 0-1000 scale), then VALIDATE it by re-transcribing the crop and
      fuzzy-matching against the context;
  (b) OpenCV horizontal projection-profile line segmentation on the Stage 0 clean image, then
      try the lines around the expected position (from the transcript line index) and keep the
      crop whose re-read best matches the context.

Crop re-reads are cached per (page, bbox) and reused as the identity sample of the consensus.
"""

import difflib
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image

from src.core.schemas import ArbitrationCandidate
from src.pipeline.arbitration.candidate_selector import tokenize, levenshtein
from src.prompts.stage3b_arbitration import (
    LINE_BBOX_SYSTEM_PROMPT,
    LINE_CROP_SYSTEM_PROMPT,
    LINE_CROP_VERBATIM_PROMPT,
    build_line_bbox_prompt,
)


# ---------------------------------------------------------------------------
# Page attribution (moved out of the orchestrator so it can run per candidate)
# ---------------------------------------------------------------------------
def attribute_error_to_page(
    erroneous_text: str,
    context: str,
    page_transcripts: Dict[int, str],
) -> Optional[int]:
    """Page whose transcript contains the erroneous span, else the context head/tail, else None."""
    needle = (erroneous_text or "").strip().lower()
    ctx = " ".join((context or "").split()).lower()
    for page_no in sorted(page_transcripts):
        p_text = " ".join((page_transcripts[page_no] or "").split()).lower()
        if needle and needle in p_text:
            if not ctx or ctx[:30] in p_text or ctx[-20:] in p_text:
                return page_no
    for page_no in sorted(page_transcripts):
        p_text = " ".join((page_transcripts[page_no] or "").split()).lower()
        if needle and needle in p_text:
            return page_no
    for page_no in sorted(page_transcripts):
        p_text = " ".join((page_transcripts[page_no] or "").split()).lower()
        if ctx and (ctx[:30] in p_text or (len(ctx) > 15 and ctx[-20:] in p_text)):
            return page_no
    return None


def match_ratio(crop_text: str, context: str, candidate_token: str = "", intended_token: str = "") -> float:
    """
    Best difflib ratio between the crop re-read and any token window of the context of the same
    length (a context sentence usually spans more than one physical line). A small bonus is given
    when the disputed token (or the intended one) is present within 2 edits.
    """
    c_toks = tokenize(crop_text)
    x_toks = tokenize(context)
    if not c_toks or not x_toks:
        return 0.0
    win = max(1, min(len(c_toks), len(x_toks)))
    best = 0.0
    c_str = " ".join(c_toks)
    for i in range(0, max(1, len(x_toks) - win + 1)):
        seg = " ".join(x_toks[i:i + win])
        r = difflib.SequenceMatcher(None, c_str, seg, autojunk=False).ratio()
        if r > best:
            best = r
    bonus = 0.0
    for tok in (candidate_token, intended_token):
        if tok and any(levenshtein(t, tok.lower()) <= 2 for t in c_toks):
            bonus = 0.1
            break
    return min(1.0, best + bonus)


@dataclass
class LineCrop:
    page_no: int
    bbox_px: Tuple[int, int, int, int]   # x1, y1, x2, y2
    image: Image.Image
    transcript: str
    method: str                          # bbox | projection
    match_ratio: float
    path: Optional[str] = None


class LineLocalizer:
    def __init__(
        self,
        engine,
        page_images: List[Tuple[int, Image.Image, str]],
        page_transcripts: Dict[int, str],
        clean_image_fn: Optional[Callable[[int], Image.Image]] = None,
        use_bbox: bool = True,
        min_ratio: float = 0.6,
        search_window: int = 3,
        crop_pad_px: int = 14,
        crop_min_height_px: int = 96,
        crop_dir: Optional[str] = None,
    ):
        self.engine = engine
        self.pages: Dict[int, Image.Image] = {p_no: img for p_no, img, _ in page_images}
        self.page_transcripts = page_transcripts
        self.clean_image_fn = clean_image_fn
        self.use_bbox = use_bbox
        self.min_ratio = min_ratio
        self.search_window = search_window
        self.crop_pad_px = crop_pad_px
        self.crop_min_height_px = crop_min_height_px
        self.crop_dir = crop_dir
        self._lines_cache: Dict[int, List[Tuple[int, int, int, int]]] = {}
        self._clean_cache: Dict[int, Image.Image] = {}
        self._read_cache: Dict[Tuple[int, Tuple[int, int, int, int]], str] = {}
        self.model_calls = 0
        self.token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # ---------------------------------------------------------------- utils
    def _accumulate_usage(self) -> None:
        u = self.engine.get_last_usage() or {}
        for k in self.token_usage:
            self.token_usage[k] += int(u.get(k, 0) or 0)

    def _clean_image(self, page_no: int) -> Image.Image:
        if page_no not in self._clean_cache:
            img = None
            if self.clean_image_fn is not None:
                try:
                    img = self.clean_image_fn(page_no)
                except Exception:
                    img = None
            self._clean_cache[page_no] = (img or self.pages[page_no]).convert("RGB")
        return self._clean_cache[page_no]

    def crop(self, page_no: int, bbox: Tuple[int, int, int, int], pad: Optional[int] = None) -> Image.Image:
        img = self._clean_image(page_no)
        w, h = img.size
        pad = self.crop_pad_px if pad is None else pad
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1 - pad // 2), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad // 2), min(h, y2 + pad)
        c = img.crop((x1, y1, x2, y2))
        if c.size[1] < self.crop_min_height_px and c.size[1] > 0:
            c = c.resize((c.size[0] * 2, c.size[1] * 2), Image.LANCZOS)
        return c

    def transcribe_crop(self, page_no: int, bbox: Tuple[int, int, int, int]) -> str:
        key = (page_no, tuple(bbox))
        if key in self._read_cache:
            return self._read_cache[key]
        img = self.crop(page_no, bbox)
        try:
            out = self.engine.generate_multimodal(
                image=img,
                prompt=LINE_CROP_VERBATIM_PROMPT,
                system_prompt=LINE_CROP_SYSTEM_PROMPT,
                temperature=0.0,
                top_p=0.1,
                max_new_tokens=96,
                thinking_mode=False,
            )
            self.model_calls += 1
            self._accumulate_usage()
        except Exception:
            out = ""
        text = _first_line(out)
        self._read_cache[key] = text
        return text

    # ------------------------------------------------------- (a) VLM bbox
    def _bbox_via_vlm(self, page_no: int, context: str) -> Optional[Tuple[int, int, int, int]]:
        img = self._clean_image(page_no)
        w, h = img.size
        try:
            raw = self.engine.generate_multimodal(
                image=img,
                prompt=build_line_bbox_prompt(context),
                system_prompt=LINE_BBOX_SYSTEM_PROMPT,
                temperature=0.0,
                top_p=0.1,
                max_new_tokens=64,
                thinking_mode=False,
            )
            self.model_calls += 1
            self._accumulate_usage()
        except Exception:
            return None
        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", raw or "")
        if not m:
            return None
        y1, x1, y2, x2 = (int(m.group(i)) for i in range(1, 5))
        if max(y1, x1, y2, x2) > 1000:
            return None
        X1, Y1 = int(x1 / 1000 * w), int(y1 / 1000 * h)
        X2, Y2 = int(x2 / 1000 * w), int(y2 / 1000 * h)
        if X2 <= X1 or Y2 <= Y1:
            return None
        bh, bw = Y2 - Y1, X2 - X1
        if bh < 0.006 * h or bh > 0.18 * h or bw < 0.06 * w:
            return None
        return (X1, Y1, X2, Y2)

    # ------------------------------------------------ (b) projection lines
    def projection_lines(self, page_no: int) -> List[Tuple[int, int, int, int]]:
        if page_no in self._lines_cache:
            return self._lines_cache[page_no]
        img = self._clean_image(page_no)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        otsu_thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # reverse-side bleed-through is lighter than the student's ink: keep only clearly dark pixels
        binary = ((gray < otsu_thr * 0.85).astype(np.uint8)) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # bleed-through speckle
        # ignore a thin frame at the borders (scanner edges, margin rule)
        bx = int(0.02 * w)
        by = int(0.02 * h)
        binary[:by, :] = 0
        binary[h - by:, :] = 0
        binary[:, :bx] = 0
        binary[:, w - bx:] = 0
        rows = binary.sum(axis=1) / 255.0
        k = max(3, int(0.004 * h) | 1)
        rows_s = np.convolve(rows, np.ones(k) / k, mode="same")
        # threshold relative to the typical ink row (median), not the maximum: a single heavy
        # underline or header rule must not suppress lightly written lines
        nz = rows_s[rows_s > 0]
        thr = max(2.0, 0.25 * float(np.percentile(nz, 50))) if nz.size else 2.0
        active = rows_s > thr
        lines: List[Tuple[int, int]] = []
        start = None
        min_h = max(10, int(0.008 * h))
        for y, a in enumerate(active):
            if a and start is None:
                start = y
            elif not a and start is not None:
                if y - start >= min_h:
                    lines.append((start, y))
                start = None
        if start is not None and h - start >= min_h:
            lines.append((start, h))
        # merge very small gaps
        merged: List[Tuple[int, int]] = []
        for y1, y2 in lines:
            if merged and y1 - merged[-1][1] < max(4, int(0.003 * h)):
                merged[-1] = (merged[-1][0], y2)
            else:
                merged.append((y1, y2))
        # cursive ascenders/descenders often bridge neighbouring lines so several lines fuse into one
        # band: split tall bands at the valleys of the profile, using the page's own line pitch
        pitch = _estimate_line_pitch(rows_s, h)
        split: List[Tuple[int, int]] = []
        for y1, y2 in merged:
            split.extend(_split_band_at_valleys(rows_s, y1, y2, pitch))
        merged = [(y1, y2) for y1, y2 in split if y2 - y1 >= min_h]
        out: List[Tuple[int, int, int, int]] = []
        for y1, y2 in merged:
            band = binary[y1:y2, :]
            cols = np.where(band.sum(axis=0) > 0)[0]
            if cols.size == 0:
                continue
            x1, x2 = int(cols.min()), int(cols.max()) + 1
            if x2 - x1 < 0.05 * w:
                continue
            out.append((x1, y1, x2, y2))
        self._lines_cache[page_no] = out
        return out

    def _expected_line_fraction(self, page_no: int, context: str, candidate_token: str) -> float:
        lines = [l for l in (self.page_transcripts.get(page_no) or "").splitlines() if l.strip()]
        if not lines:
            return 0.5
        ctx = " ".join(context.split()).lower()
        cand = candidate_token.lower()
        best_i, best_r = 0, -1.0
        for i, l in enumerate(lines):
            ll = l.lower()
            r = difflib.SequenceMatcher(None, ll, ctx[:len(ll) + 10], autojunk=False).ratio()
            if cand and re.search(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", ll):
                r += 0.5
            if r > best_r:
                best_i, best_r = i, r
        return (best_i + 0.5) / len(lines)

    # ------------------------------------------------------------- locate
    def locate(self, cand: ArbitrationCandidate) -> Optional[LineCrop]:
        page_no = cand.page_no
        if page_no is None or page_no not in self.pages:
            page_no = attribute_error_to_page(cand.erroneous_text, cand.context_sentence, self.page_transcripts)
        if page_no is None or page_no not in self.pages:
            return None
        cand.page_no = page_no
        context = cand.context_sentence or cand.erroneous_text

        best: Optional[LineCrop] = None
        if self.use_bbox:
            bbox = self._bbox_via_vlm(page_no, context)
            if bbox is not None:
                text = self.transcribe_crop(page_no, bbox)
                r = match_ratio(text, context, cand.candidate_token, cand.intended_token)
                if r >= self.min_ratio:
                    best = LineCrop(page_no, bbox, self.crop(page_no, bbox), text, "bbox", r)

        if best is None:
            lines = self.projection_lines(page_no)
            if lines:
                frac = self._expected_line_fraction(page_no, context, cand.candidate_token)
                centre = int(round(frac * len(lines) - 0.5))
                order = sorted(range(len(lines)), key=lambda i: abs(i - centre))
                n_transcript_lines = len([l for l in (self.page_transcripts.get(page_no) or "").splitlines() if l.strip()])
                n_try = 2 * self.search_window + 1
                if n_transcript_lines and len(lines) > 1.5 * n_transcript_lines:
                    n_try = min(len(lines), max(n_try, 10))   # segmentation found spurious lines: widen the search
                for i in order[:n_try]:
                    bbox = lines[i]
                    text = self.transcribe_crop(page_no, bbox)
                    r = match_ratio(text, context, cand.candidate_token, cand.intended_token)
                    if best is None or r > best.match_ratio:
                        best = LineCrop(page_no, bbox, self.crop(page_no, bbox), text, "projection", r)
                    if r >= 0.9:
                        break
                if best is not None and best.match_ratio < self.min_ratio:
                    best = None

        if best is not None and self.crop_dir:
            try:
                os.makedirs(self.crop_dir, exist_ok=True)
                safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", cand.candidate_id)
                path = os.path.join(self.crop_dir, f"{safe_id}_p{page_no}_{cand.candidate_token}.png")
                best.image.save(path)
                best.path = path
            except Exception:
                best.path = None
        return best


def _estimate_line_pitch(rows_s: np.ndarray, h: int) -> int:
    """Dominant line spacing (px) from the autocorrelation of the row profile; falls back to 4.5% of h."""
    fallback = max(20, int(0.045 * h))
    x = rows_s - rows_s.mean()
    if x.size < 50 or float(np.abs(x).sum()) == 0.0:
        return fallback
    ac = np.correlate(x, x, mode="full")[x.size - 1:]
    lo, hi = max(8, int(0.02 * h)), max(9, int(0.1 * h))
    if hi <= lo or hi >= ac.size:
        return fallback
    seg = ac[lo:hi]
    lag = lo + int(np.argmax(seg))
    if seg.max() <= 0:
        return fallback
    return int(lag)


def _split_band_at_valleys(rows_s: np.ndarray, y1: int, y2: int, pitch: int) -> List[Tuple[int, int]]:
    """Split a band taller than ~1.5 line pitches at the profile valleys between line cores."""
    height = y2 - y1
    if pitch <= 0 or height < 1.5 * pitch:
        return [(y1, y2)]
    seg = rows_s[y1:y2]
    k = max(3, int(pitch * 0.3) | 1)
    seg_s = np.convolve(seg, np.ones(k) / k, mode="same")
    half = max(2, int(pitch * 0.35))
    cuts: List[int] = []
    i = half
    last_cut = -10 ** 9
    while i < seg_s.size - half:
        window = seg_s[i - half:i + half + 1]
        if seg_s[i] <= window.min() and (i - last_cut) >= int(pitch * 0.6):
            left_peak = seg_s[max(0, i - pitch):i].max() if i > 0 else 0.0
            right_peak = seg_s[i:min(seg_s.size, i + pitch)].max()
            if seg_s[i] < 0.75 * min(left_peak, right_peak):
                cuts.append(i)
                last_cut = i
                i += int(pitch * 0.5)
                continue
        i += 1
    if not cuts:
        return [(y1, y2)]
    bounds = [0] + cuts + [seg_s.size]
    return [(y1 + a, y1 + b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 0]


def _first_line(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
    for line in t.splitlines():
        if line.strip():
            return line.strip().strip('"').strip()
    return t
