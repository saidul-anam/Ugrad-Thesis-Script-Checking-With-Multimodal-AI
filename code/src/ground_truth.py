"""Ground-truth assembly and the two-source cross-check.

Authoritative score = the red mark read off the image (harvester). The `marks`
column is a noisy cross-check only; its 'a/b' half-mark tokens are corrupt and
mean different things per row, so we test the red mark against *every* plausible
reading of the token and agree if any matches.

  Agree    → high-confidence GOLD label.
  Disagree → FLAG for a human look.

The share of FLAG rows is the gold-label error bar that caps any accuracy claim.
"""
from __future__ import annotations

import re
from typing import Optional

TOL = 0.01


def parse_marks_token(marks: str) -> Optional[dict]:
    """Decode an 'a/b' marks token into {full, score, kind}.

    Empirically (verified against rubric-derived full marks over the whole set)
    the token means different things by denominator:

      b in {5, 10}  → SCORE/FULL: numerator a is the teacher's score, b the total
                      (e.g. 7/10 = scored 7 out of 10). Cross-checkable.
      b <= 2        → HALF-MARK total: the value a/b IS the full marks
                      (e.g. 5/2 = 2.5 total), written in half-mark units. The
                      score is NOT in the token — it lives only in the red ink.

    Returns None for a non 'a/b' token.
    """
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", marks or "")
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if b == 0:
        return None
    if b >= 3:  # only 5 and 10 occur; treat as score / full
        return {"full": float(b), "score": float(a), "kind": "score_over_full"}
    return {"full": round(a / b, 4), "score": None, "kind": "half_mark_total"}


def cross_check(red_mark: Optional[float], marks: str) -> dict:
    """Cross-check the harvested red score against the marks column.

    Only 'score/full' tokens carry an independent score to check against; the
    half-mark tokens carry only the total, so the red mark is the sole score and
    is used unverified.

    status ∈ GOLD | FLAG | NO_REDMARK | NO_MARKSCOL
    """
    tok = parse_marks_token(marks)
    if red_mark is None:
        return {"status": "NO_REDMARK", "gold_score": None,
                "token_score": tok.get("score") if tok else None, "matched": None}
    if not tok or tok["score"] is None:
        # No independent score to verify against; trust the red mark, unverified.
        return {"status": "NO_MARKSCOL", "gold_score": red_mark,
                "token_score": None, "matched": None}
    if abs(tok["score"] - red_mark) <= TOL:
        return {"status": "GOLD", "gold_score": red_mark,
                "token_score": tok["score"], "matched": tok["score"]}
    return {"status": "FLAG", "gold_score": red_mark,
            "token_score": tok["score"], "matched": None}
