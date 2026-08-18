"""
OCR Evaluation Metrics: Character Error Rate (CER) and Word Error Rate (WER)
with Bangla Unicode Normalization support.
"""

import unicodedata
from typing import List, Tuple


def normalize_bangla_text(text: str) -> str:
    """
    Apply standard Unicode normalization (NFC) and clean whitespace
    for fair OCR evaluation in Bangla and English.
    """
    if not text:
        return ""
    # NFC Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Normalize varied whitespace and line breaks
    text = " ".join(text.split())
    return text


def levenshtein_distance(seq1: List[str] | str, seq2: List[str] | str) -> int:
    """Compute standard Levenshtein edit distance between two sequences (chars or tokens)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1]   # Substitution
                )

    return dp[m][n]


def compute_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Compute Character Error Rate (CER) = EditDistance(ref, hyp) / len(ref).
    """
    if normalize:
        reference = normalize_bangla_text(reference)
        hypothesis = normalize_bangla_text(hypothesis)

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    dist = levenshtein_distance(reference, hypothesis)
    return dist / len(reference)


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Compute Word Error Rate (WER) = EditDistance(ref_words, hyp_words) / len(ref_words).
    """
    if normalize:
        reference = normalize_bangla_text(reference)
        hypothesis = normalize_bangla_text(hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def evaluate_ocr_batch(references: List[str], hypotheses: List[str]) -> Tuple[float, float]:
    """
    Compute corpus-level aggregate CER and WER over lists of reference and hypothesis transcriptions.
    """
    total_char_dist = 0
    total_char_len = 0
    total_word_dist = 0
    total_word_len = 0

    for ref, hyp in zip(references, hypotheses):
        norm_ref = normalize_bangla_text(ref)
        norm_hyp = normalize_bangla_text(hyp)

        # CER
        total_char_dist += levenshtein_distance(norm_ref, norm_hyp)
        total_char_len += len(norm_ref)

        # WER
        ref_words = norm_ref.split()
        hyp_words = norm_hyp.split()
        total_word_dist += levenshtein_distance(ref_words, hyp_words)
        total_word_len += len(ref_words)

    mean_cer = total_char_dist / max(1, total_char_len)
    mean_wer = total_word_dist / max(1, total_word_len)

    return mean_cer, mean_wer
