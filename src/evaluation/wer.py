"""
Word Error Rate (WER) Metric Calculation.
"""

from typing import List
import unicodedata
import re
import string


def compute_word_levenshtein(w1: List[str], w2: List[str]) -> int:
    """Token-level Levenshtein distance."""
    if len(w1) < len(w2):
        return compute_word_levenshtein(w2, w1)

    if len(w2) == 0:
        return len(w1)

    previous_row = range(len(w2) + 1)
    for i, t1 in enumerate(w1):
        current_row = [i + 1]
        for j, t2 in enumerate(w2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (t1 != t2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_for_wer(text: str) -> List[str]:
    """Tokenize and normalize text for English WER evaluation (lowercase, strip standard punctuation)."""
    if not text:
        return []
    text = unicodedata.normalize("NFC", text).lower()
    # Strip non-word punctuation
    text = re.sub(r"[^\w\s]", "", text)
    tokens = [t for t in text.split() if t.strip()]
    return tokens


def compute_wer(hypothesis: str, reference: str, normalize: bool = False) -> float:
    """
    Calculate Word Error Rate (WER) = TokenEditDistance(hyp, ref) / max(1, len(ref_tokens)).
    """
    if normalize:
        hyp_tokens = normalize_for_wer(hypothesis)
        ref_tokens = normalize_for_wer(reference)
    else:
        hyp_tokens = hypothesis.split() if hypothesis else []
        ref_tokens = reference.split() if reference else []

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    edit_dist = compute_word_levenshtein(hyp_tokens, ref_tokens)
    return float(edit_dist / len(ref_tokens))
