"""Emit output/evaluation_avg.csv — one row per (task_id x model), averaging the
k runs. Keeps the per-run spread (min/max/range/std) and evidence rate, since the
whole reason for k runs is to see within-model noise around the mean.

Derived + regenerable from the per-run records; never edit by hand.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from src.config import EvalConfig
from src.eval.models import FOUR_CRITERIA, band_for

AVG_COLUMNS = (
    ["task_id", "script_id", "question_no", "question_type", "max_mark",
     "model", "provider", "n_runs"]
    + ["mean_total_score", "total_score_min", "total_score_max",
       "total_score_range", "total_score_std"]
    + [f"mean_{c}" for c in FOUR_CRITERIA] + ["mean_raw_total"]
    + ["performance_band_of_mean", "cap_applied_rate", "cap_reason_modal"]
    + ["is_attempted", "synthesised", "evidence_not_found_rate"]
    + ["model_version", "thinking_level", "rubric_hash", "prompt_hash"]
)


def _r(x: float, n: int = 3) -> float:
    return round(x, n)


def _aggregate(group: list[dict]) -> dict:
    m0 = group[0]["_metadata"]
    totals = [float(r["score_breakdown"]["total_score"]) for r in group]
    max_mark = int(m0["max_mark"])
    mean_total = statistics.mean(totals)
    n = len(group)

    cap_rate = sum(1 for r in group if r["structural_audit"]["cap_applied"]) / n
    reasons = Counter(r["structural_audit"]["applied_cap_reason"] for r in group)
    evid_rate = sum(1 for r in group if r["_metadata"].get("evidence_not_found")) / n

    row = {
        "task_id": m0["task_id"], "script_id": m0["script_id"],
        "question_no": m0["question_no"], "question_type": m0["question_type"],
        "max_mark": max_mark, "model": m0["model"],
        "provider": m0.get("provider") or "", "n_runs": n,
        "mean_total_score": _r(mean_total),
        "total_score_min": min(totals), "total_score_max": max(totals),
        "total_score_range": _r(max(totals) - min(totals)),
        "total_score_std": _r(statistics.stdev(totals)) if n >= 2 else 0.0,
        "mean_raw_total": _r(statistics.mean(float(r["raw_subscores"]["raw_total"]) for r in group)),
        "performance_band_of_mean": band_for(round(mean_total), max_mark),
        "cap_applied_rate": _r(cap_rate),
        "cap_reason_modal": reasons.most_common(1)[0][0],
        "is_attempted": group[0]["attempt_status"]["is_attempted"],
        "synthesised": m0.get("synthesised", False),
        "evidence_not_found_rate": _r(evid_rate),
        "model_version": m0.get("model_id") or "", "thinking_level": m0["thinking_level"],
        "rubric_hash": m0["rubric_hash"], "prompt_hash": m0["prompt_hash"],
    }
    for c in FOUR_CRITERIA:
        row[f"mean_{c}"] = _r(statistics.mean(float(r["score_breakdown"][c]) for r in group))
    return row


def emit_averaged(cfg: EvalConfig) -> tuple[Path, int]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for p in sorted(cfg.paths.evaluations.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        m = rec["_metadata"]
        groups.setdefault((m["task_id"], m["model"]), []).append(rec)

    rows = [_aggregate(g) for g in groups.values()]
    rows.sort(key=lambda r: (r["task_id"], r["model"]))

    out_path = cfg.paths.evaluation_csv.parent / "evaluation_avg.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AVG_COLUMNS, quoting=csv.QUOTE_ALL,
                           lineterminator="\r\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_path, len(rows)
