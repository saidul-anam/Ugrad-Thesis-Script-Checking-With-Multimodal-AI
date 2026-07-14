"""Analysis — per-subject metrics and a ranked disagreement table.

Metrics are normalized by full marks. We always report the gold-label error rate
(share of cross-check FLAG rows); it caps how precise any accuracy claim can be.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import pandas as pd


def _norm_err(gold: float, pred: float, full: Optional[float]) -> Optional[float]:
    if full and full > 0:
        return abs(gold - pred) / full
    return None


def _txt(v) -> str:
    """Coerce a cell to a string, treating None and pandas NaN as empty."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v)


def _usable(r) -> bool:
    """A row counts toward metrics only if it has a trustworthy gold score and
    denominator. Prefer the explicit flag; fall back to status for older runs."""
    if "gold_usable" in r and r.get("gold_usable") is not None:
        return bool(r.get("gold_usable"))
    return r.get("status") in ("GOLD", "NO_MARKSCOL")


def metrics_by_subject(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """df needs: subject, gold_score, full_marks, and score_<variant>."""
    col = f"score_{variant}"
    rows = []
    groups = defaultdict(list)
    for _, r in df.iterrows():
        if not _usable(r):
            continue
        gold, pred, full = r.get("gold_score"), r.get(col), r.get("full_marks")
        if gold is None or pred is None or (isinstance(pred, float) and math.isnan(pred)):
            continue
        groups[r["subject"]].append((float(gold), float(pred), full))
        groups["ALL"].append((float(gold), float(pred), full))

    for subject, vals in groups.items():
        n = len(vals)
        if not n:
            continue
        exact = sum(1 for g, p, _ in vals if abs(g - p) < 1e-6) / n
        within = sum(1 for g, p, _ in vals if abs(g - p) <= 0.5) / n
        nerrs = [e for g, p, f in vals if (e := _norm_err(g, p, f)) is not None]
        nmae = sum(nerrs) / len(nerrs) if nerrs else None
        rows.append({
            "subject": subject, "variant": variant, "n": n,
            "exact_agreement": round(exact, 4),
            "within_0.5": round(within, 4),
            "normalized_mae": round(nmae, 4) if nmae is not None else None,
        })
    cols = ["subject", "variant", "n", "exact_agreement", "within_0.5", "normalized_mae"]
    if not rows:  # no gradeable rows (e.g. an all-blank / unmarked batch)
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values(["variant", "subject"])


def gold_label_error_rate(df: pd.DataFrame) -> dict:
    counts = df["status"].value_counts().to_dict()
    checked = counts.get("GOLD", 0) + counts.get("FLAG", 0)
    flag = counts.get("FLAG", 0)
    return {
        "cross_checked_rows": checked,
        "flagged_rows": flag,
        "gold_label_error_rate": round(flag / checked, 4) if checked else None,
        "status_counts": counts,
    }


def disagreement_table(df: pd.DataFrame, variant: str, top: int = 40) -> pd.DataFrame:
    col = f"score_{variant}"
    recs = []
    for _, r in df.iterrows():
        if not _usable(r):
            continue
        gold, pred, full = r.get("gold_score"), r.get(col), r.get("full_marks")
        if gold is None or pred is None or (isinstance(pred, float) and math.isnan(pred)):
            continue
        ne = _norm_err(float(gold), float(pred), full)
        recs.append({
            "id": r["id"], "subject": r["subject"],
            "gold_score": gold, f"gemini_{variant}": pred, "full_marks": full,
            "abs_err": round(abs(float(gold) - float(pred)), 3),
            "normalized_err": round(ne, 3) if ne is not None else None,
            "transcription": _txt(r.get(f"transcription_{variant}"))[:200],
            "reasoning": _txt(r.get(f"reasoning_{variant}"))[:300],
        })
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    return out.sort_values("normalized_err", ascending=False, na_position="last").head(top)
