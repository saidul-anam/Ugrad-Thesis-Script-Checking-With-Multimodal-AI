"""Emit output/evaluation.csv (long format) + evaluation_runs.csv.

evaluation.csv: one row per (task_id x model x run). Long format so a third
grader later becomes more rows, not a schema change.
evaluation_runs.csv: one row per API call (synthesised records make none).

Conventions (schema v3): utf-8-sig, QUOTE_ALL, CRLF. Rationale/evidence fields
contain commas and newlines — a real CSV parser is mandatory downstream.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.config import EvalConfig
from src.eval.models import FOUR_CRITERIA

EVAL_COLUMNS = (
    ["task_id", "script_id", "question_no", "question_type", "max_mark",
     "model", "provider", "run_index"]
    + [f"raw_{c}" for c in FOUR_CRITERIA] + ["raw_total"]
    + list(FOUR_CRITERIA) + ["total_score", "capped_from"]
    + ["cap_applied", "applied_cap_reason", "performance_band",
       "is_attempted", "synthesised"]
    + [f"{c}_rationale" for c in FOUR_CRITERIA]
    + [f"{c}_evidence" for c in FOUR_CRITERIA]
    + ["evidence_not_found", "frequent_errors", "positive_aspects",
       "feedback_summary", "model_version", "thinking_level",
       "rubric_hash", "prompt_hash", "timestamp"]
)

RUNS_COLUMNS = [
    "run_id", "task_id", "model", "provider", "model_id", "thinking_level",
    "run_index", "call_index", "input_tokens", "output_tokens", "thoughts_tokens",
    "cost_usd", "latency_ms", "attempt_number", "http_status", "timestamp",
]


def _num(v: object) -> str:
    """Render a score: '' for None, drop trailing .0 for whole numbers."""
    if v is None or v == "":
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _join(items: object) -> str:
    return ";".join(str(x) for x in (items or []))


def build_eval_row(rec: dict) -> dict:
    m = rec["_metadata"]
    raw = rec["raw_subscores"]
    sb = rec["score_breakdown"]
    audit = rec["structural_audit"]
    reasoning = rec["criterion_reasoning"]
    errors = rec["error_analysis"]

    row = {
        "task_id": m["task_id"], "script_id": m["script_id"],
        "question_no": m["question_no"], "question_type": m["question_type"],
        "max_mark": m["max_mark"], "model": m["model"],
        "provider": m.get("provider") or "", "run_index": m["run_index"],
        "raw_total": _num(raw["raw_total"]),
        "total_score": _num(sb["total_score"]), "capped_from": _num(sb.get("capped_from")),
        "cap_applied": audit["cap_applied"], "applied_cap_reason": audit["applied_cap_reason"],
        "performance_band": rec["performance_band"],
        "is_attempted": rec["attempt_status"]["is_attempted"],
        "synthesised": m.get("synthesised", False),
        "evidence_not_found": _join(m.get("evidence_not_found")),
        "frequent_errors": _join(errors.get("frequent_errors")),
        "positive_aspects": _join(errors.get("positive_aspects")),
        "feedback_summary": rec["feedback_summary"],
        "model_version": m.get("model_id") or "", "thinking_level": m["thinking_level"],
        "rubric_hash": m["rubric_hash"], "prompt_hash": m["prompt_hash"],
        "timestamp": m["timestamp"],
    }
    for c in FOUR_CRITERIA:
        row[f"raw_{c}"] = _num(raw[c])
        row[c] = _num(sb[c])
        row[f"{c}_rationale"] = reasoning[c]["rationale"]
        row[f"{c}_evidence"] = reasoning[c]["evidence"]
    return row


def build_runs_rows(rec: dict) -> list[dict]:
    m = rec["_metadata"]
    rows = []
    for i, call in enumerate(m.get("calls", [])):
        rows.append({
            "run_id": f"{m['task_id']}__{m['model']}__run{m['run_index']}__call{i}",
            "task_id": m["task_id"], "model": m["model"],
            "provider": m.get("provider") or "", "model_id": m.get("model_id") or "",
            "thinking_level": m["thinking_level"], "run_index": m["run_index"],
            "call_index": i, "input_tokens": call["input_tokens"],
            "output_tokens": call["output_tokens"],
            "thoughts_tokens": call.get("thoughts_tokens", 0),
            "cost_usd": call["cost_usd"], "latency_ms": call["latency_ms"],
            "attempt_number": call["attempt_number"], "http_status": call["http_status"],
            "timestamp": m["timestamp"],
        })
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL,
                           lineterminator="\r\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _sort_key(rec: dict):
    m = rec["_metadata"]
    return (m["task_id"], m["model"], m["run_index"])


def emit_all(cfg: EvalConfig) -> tuple[Path, Path, int, int]:
    records = []
    for p in sorted(cfg.paths.evaluations.glob("*.json")):
        records.append(json.loads(p.read_text(encoding="utf-8")))
    records.sort(key=_sort_key)

    eval_rows = [build_eval_row(r) for r in records]
    runs_rows = [row for r in records for row in build_runs_rows(r)]

    _write_csv(cfg.paths.evaluation_csv, EVAL_COLUMNS, eval_rows)
    _write_csv(cfg.paths.evaluation_runs_csv, RUNS_COLUMNS, runs_rows)
    return cfg.paths.evaluation_csv, cfg.paths.evaluation_runs_csv, len(eval_rows), len(runs_rows)
