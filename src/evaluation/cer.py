"""
Character Error Rate (CER) Metric Calculation.
"""

from typing import Tuple
import unicodedata
import re


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Standard dynamic programming Levenshtein distance."""
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_for_cer(text: str) -> str:
    """Normalize text for invariant CER calculation (Unicode NFC, lowercase, condense whitespace)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_cer(hypothesis: str, reference: str, normalize: bool = False) -> float:
    """
    Calculate Character Error Rate (CER) = EditDistance(hyp, ref) / max(1, len(ref)).
    """
    hyp = normalize_for_cer(hypothesis) if normalize else hypothesis
    ref = normalize_for_cer(reference) if normalize else reference

    if not ref:
        return 0.0 if not hyp else 1.0

    edit_dist = compute_levenshtein_distance(hyp, ref)
    return float(edit_dist / len(ref))
