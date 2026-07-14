"""Full-marks (denominator) extraction.

Priority per CLAUDE.md: rubric text → marks-column total → marks stated in the
question. Metrics are normalized by this denominator, so a 1-point miss on a
10-marker is not weighted like one on a 2-marker.
"""
from __future__ import annotations

import re
from typing import Optional

from .normalize_bn import bn_digits_to_ascii

# A number (Bengali or ASCII, optional decimal) sitting immediately before the
# word "নম্বর" (mark). OCR sometimes splits it as "নম্ব র"; allow inner spaces.
_ALLOC_RE = re.compile(r"([০-৯0-9]+(?:[.।][০-৯0-9]+)?)\s*ন\s*ম্\s*ব\s*র")


def _to_float(tok: str) -> Optional[float]:
    tok = bn_digits_to_ascii(tok).replace("।", ".")
    try:
        return float(tok)
    except ValueError:
        return None


def full_marks_from_rubric(rubric: str) -> Optional[float]:
    """Sum the per-clause allocations (the number right before each 'নম্বর')."""
    if not rubric or not rubric.strip():
        return None
    vals = [_to_float(m.group(1)) for m in _ALLOC_RE.finditer(rubric)]
    vals = [v for v in vals if v is not None]
    total = round(sum(vals), 4)
    return total if total > 0 else None


def full_marks_from_marks_col(marks: str) -> Optional[float]:
    """Full marks implied by an 'a/b' token (fallback when the rubric is silent).

    Semantics are denominator-dependent (see ground_truth.parse_marks_token):
      b in {5,10} → full = b   (7/10 → out of 10)
      b <= 2      → full = a/b (5/2 → 2.5 total, half-mark units)
    """
    from .ground_truth import parse_marks_token
    tok = parse_marks_token(marks)
    return tok["full"] if tok else None


def full_marks_from_question(text: str) -> Optional[float]:
    """Marks stated inline in the question, e.g. '(5)' or 'Marks: 5'."""
    if not text:
        return None
    m = re.search(r"marks?\s*[:\-]?\s*(\d+(?:\.\d+)?)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"[\(\[]\s*(\d+(?:\.\d+)?)\s*[\)\]]\s*$", text.strip())
    if m:
        return float(m.group(1))
    return None


def resolve_full_marks(row: dict, question_text: str = "") -> dict:
    """Apply the fallback chain. Returns {full_marks, source}."""
    fm = full_marks_from_rubric(row.get("rubric", ""))
    if fm:
        return {"full_marks": fm, "source": "rubric"}
    fm = full_marks_from_marks_col(row.get("marks", ""))
    if fm:
        return {"full_marks": fm, "source": "marks_col"}
    fm = full_marks_from_question(question_text or row.get("questionEN", ""))
    if fm:
        return {"full_marks": fm, "source": "question"}
    return {"full_marks": None, "source": "none"}
