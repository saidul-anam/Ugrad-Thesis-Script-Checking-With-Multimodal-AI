"""Bengali text/number normalization and the per-row question-text router.

INVARIANT (CLAUDE.md): questionBN has two corruption modes — route per row,
never blind-convert. We choose *which text to trust*, and only Unicode-normalize
genuine Bengali. We never reverse-decode blindly, because a bad decode silently
mangles real Bengali; garbled Bengali-only rows are flagged so the grader leans
on the image instead.
"""
from __future__ import annotations

import re
import unicodedata

_BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_BN_TO_ASCII = {ord(d): str(i) for i, d in enumerate(_BN_DIGITS)}

# Bengali Unicode block.
_BENGALI_RE = re.compile(r"[ঀ-৿]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def bn_digits_to_ascii(text: str) -> str:
    return (text or "").translate(_BN_TO_ASCII)


def nfc(text: str) -> str:
    """Canonical Unicode composition — merges decomposed Bengali conjunct forms."""
    return unicodedata.normalize("NFC", text or "")


def bengali_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha() or _BENGALI_RE.match(c)]
    if not letters:
        return 0.0
    bn = sum(1 for c in letters if _BENGALI_RE.match(c))
    return bn / len(letters)


# Signature of legacy-Bijoy garble rendered as Bengali Unicode: an abnormally
# high rate of hasant (্), visarga (ঃ), and the rare glyphs ৎ ঢ় ং clustered
# together — real Bengali prose uses these far more sparingly.
_GARBLE_CHARS = "্ঃৎঢ়ং"


def looks_bijoy_garbled(text: str) -> bool:
    t = nfc(text)
    bn = [c for c in t if _BENGALI_RE.match(c)]
    if len(bn) < 6:
        return False
    hits = sum(1 for c in t if c in _GARBLE_CHARS)
    return hits / len(bn) > 0.28


def route_question(row: dict) -> dict:
    """Pick the trustworthy question text for a row.

    Returns {text, source, note}. source ∈
      EN            — English present; garbled BN is English-typed-in-Bijoy.
      BN_UNICODE    — genuine Bengali, Unicode-normalized only.
      BN_GARBLED    — Bengali-only and Bijoy-garbled; text is unreliable, so the
                      grader must read the printed question off the image.
      NONE          — no usable question text.
    """
    en = (row.get("questionEN") or "").strip()
    bn = (row.get("questionBN") or "").strip()

    if _LATIN_WORD_RE.search(en):
        return {"text": nfc(en), "source": "EN", "note": ""}

    if bn:
        norm = nfc(bn)
        if looks_bijoy_garbled(norm):
            return {"text": norm, "source": "BN_GARBLED",
                    "note": "Bijoy-garbled Bengali; defer to image OCR"}
        return {"text": norm, "source": "BN_UNICODE", "note": ""}

    if en:
        return {"text": nfc(en), "source": "EN", "note": ""}
    return {"text": "", "source": "NONE", "note": "no question text"}
