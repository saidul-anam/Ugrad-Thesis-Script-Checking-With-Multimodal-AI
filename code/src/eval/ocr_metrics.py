"""
OCR Evaluation Metrics: Character Error Rate (CER) and Word Error Rate (WER)
with Bangla/English Unicode Normalization, Punctuation Stripping, and Case Normalization.
"""

import re
import unicodedata
from typing import List, Tuple, Dict, Any


def clean_text_for_ocr_eval(
    text: str,
    remove_punct: bool = True,
    to_lower: bool = True,
    strip_tags: bool = True
) -> str:
    """
    Apply comprehensive text cleaning for fair, standardized OCR evaluation:
    1. Unicode NFC normalization
    2. Strips OCR metadata tags like [struck: ...], [illegible], [cut: ...]
    3. Normalizes smart quotes, dashes, hyphens
    4. Strips punctuation marks (if remove_punct=True)
    5. Converts to lower case (if to_lower=True)
    6. Collapses all whitespace and line breaks to single spaces
    """
    if not text:
        return ""

    # 1. Unicode NFC Normalization
    text = unicodedata.normalize("NFC", str(text))

    # 2. Strip OCR metadata tags
    if strip_tags:
        text = re.sub(r"\[struck:\s*[^\]]*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[(illegible|cut:[^\]]*|unclear)\]", "", text, flags=re.IGNORECASE)

    # 3. Standardize typographical variants
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", " ").replace("–", " ").replace("-", " ")

    # 4. Remove punctuation marks if requested
    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text)

    # 5. Lowercase if requested
    if to_lower:
        text = text.lower()

    # 6. Normalize varied whitespace and line breaks
    text = " ".join(text.split())
    return text


def normalize_bangla_text(text: str) -> str:
    """
    Backward-compatible text normalization function.
    Cleans punctuation and whitespace for fair OCR evaluation.
    """
    return clean_text_for_ocr_eval(text, remove_punct=True, to_lower=True)


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


def compute_cer(reference: str, hypothesis: str, clean: bool = True) -> float:
    """
    Compute Character Error Rate (CER) = EditDistance(ref, hyp) / len(ref).
    When clean=True, applies unicode, lowercase, and punctuation normalization.
    """
    if clean:
        reference = clean_text_for_ocr_eval(reference, remove_punct=True, to_lower=True)
        hypothesis = clean_text_for_ocr_eval(hypothesis, remove_punct=True, to_lower=True)

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    dist = levenshtein_distance(reference, hypothesis)
    return dist / len(reference)


def compute_wer(reference: str, hypothesis: str, clean: bool = True) -> float:
    """
    Compute Word Error Rate (WER) = EditDistance(ref_words, hyp_words) / len(ref_words).
    When clean=True, strips attached punctuation and converts to lowercase before splitting.
    """
    if clean:
        reference = clean_text_for_ocr_eval(reference, remove_punct=True, to_lower=True)
        hypothesis = clean_text_for_ocr_eval(hypothesis, remove_punct=True, to_lower=True)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def compute_raw_wer_cer(reference: str, hypothesis: str) -> Tuple[float, float]:
    """Compute strict verbatim (raw) CER and WER without removing punctuation."""
    cer = compute_cer(reference, hypothesis, clean=False)
    wer = compute_wer(reference, hypothesis, clean=False)
    return cer, wer


def compute_normalized_wer_cer(reference: str, hypothesis: str) -> Tuple[float, float]:
    """Compute fair normalized CER and WER with punctuation & case cleaning."""
    cer = compute_cer(reference, hypothesis, clean=True)
    wer = compute_wer(reference, hypothesis, clean=True)
    return cer, wer


def evaluate_ocr_batch(references: List[str], hypotheses: List[str], clean: bool = True) -> Tuple[float, float]:
    """
    Compute corpus-level aggregate CER and WER over lists of reference and hypothesis transcriptions.
    """
    total_char_dist = 0
    total_char_len = 0
    total_word_dist = 0
    total_word_len = 0

    for ref, hyp in zip(references, hypotheses):
        norm_ref = clean_text_for_ocr_eval(ref, remove_punct=True, to_lower=True) if clean else ref
        norm_hyp = clean_text_for_ocr_eval(hyp, remove_punct=True, to_lower=True) if clean else hyp

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
