"""Evaluation driver: iterate (row x model x run), cache, parse, validate, write.

Cache key = SHA256(prompt bytes + model id + thinking_level + run index).
Resumable: a completed record whose stored cache_key still matches is skipped.
not_attempted rows are synthesised directly (no API call).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from pydantic import ValidationError

from src.config import EvalConfig
from src.eval.grader_client import GraderClient
from src.eval.models import Evaluation, check_evidence, synthesise_blank
from src.eval.prompt_builder import (
    BuiltPrompt,
    build_prompt,
    load_rubric,
    load_user_template,
)
from src.extract import strip_code_fences
from src.llm_client import LLMError, RawResponse

logger = logging.getLogger("eval")


class EvalError(RuntimeError):
    """A (task, model, run) failed to grade. Carries the label (fail loudly)."""

    def __init__(self, label: str, message: str) -> None:
        self.label = label
        super().__init__(f"[{label}] {message}")


@dataclass
class EvalResult:
    task_id: str
    model: str
    run_index: int
    status: str                    # graded | cached | synthesised
    total_score: int | None = None
    cost_usd: float = 0.0
    wall_time_s: float = 0.0
    evidence_missing: list[str] = field(default_factory=list)


def load_rows(cfg: EvalConfig, script_id: str | None, limit: int | None) -> list[dict]:
    with cfg.paths.extraction_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if script_id:
        rows = [r for r in rows if r["script_id"] == script_id]
    if limit is not None:
        keep = list(dict.fromkeys(r["script_id"] for r in rows))[:limit]
        rows = [r for r in rows if r["script_id"] in set(keep)]
    return rows


def record_path(cfg: EvalConfig, task_id: str, model: str, run: int) -> Path:
    return cfg.paths.evaluations / f"{task_id}__{model}__run{run}.json"


def cache_key(prompt_bytes: bytes, model_id: str, thinking_level: str, run: int) -> str:
    h = hashlib.sha256()
    h.update(prompt_bytes)
    h.update(model_id.encode("utf-8"))
    h.update(thinking_level.encode("utf-8"))
    h.update(str(run).encode("utf-8"))
    return h.hexdigest()


def _grade_and_parse(
    grader: GraderClient, bp: BuiltPrompt, label: str
) -> tuple[Evaluation, list[RawResponse]]:
    """One grade call, retried once on EITHER invalid JSON or a schema-validation
    failure (both models occasionally omit a required field). Then fail loudly.
    """
    calls: list[RawResponse] = []

    def attempt(user_content: str) -> Evaluation:
        resp = grader.grade(user_content, label=label)
        calls.append(resp)
        parsed = json.loads(strip_code_fences(resp.text))  # JSONDecodeError
        return Evaluation.model_validate(parsed)           # ValidationError

    try:
        return attempt(bp.user_content), calls
    except (json.JSONDecodeError, ValidationError) as e1:
        logger.warning("[%s] invalid/incomplete output, retrying once: %s", label, e1)
        retry_user = (
            f"{bp.user_content}\n\nYour previous reply was not valid JSON or was "
            f"missing required fields ({e1}). Return ONLY the complete JSON object "
            f"with EVERY field present, no fence."
        )
        try:
            return attempt(retry_user), calls
        except (json.JSONDecodeError, ValidationError) as e2:
            raise EvalError(label, f"invalid/incomplete after retry: {e2}") from e2


def _base_meta(cfg: EvalConfig, row: dict, bp: BuiltPrompt, model: str, run: int) -> dict:
    return {
        "task_id": row["task_id"],
        "script_id": row["script_id"],
        "question_no": row["question_no"],
        "question_type": row["question_type"],
        "max_mark": int(row["max_mark"]),
        "model": model,
        "thinking_level": cfg.thinking_level,
        "run_index": run,
        "rubric_hash": bp.rubric_hash,
        "prompt_hash": bp.prompt_hash,
        "cap_exempt_applied": bp.cap_exempt_applied,
        "cap_threshold": bp.cap_threshold,
        "cap_note": bp.cap_note,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _write_record(path: Path, evaluation: Evaluation, meta: dict) -> None:
    out = evaluation.model_dump()
    out["_metadata"] = meta
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_one(
    row: dict, bp: BuiltPrompt, model: str, run: int,
    cfg: EvalConfig, grader: GraderClient | None, *, force: bool,
) -> EvalResult:
    task_id = row["task_id"]
    label = f"{task_id}/{model}/run{run}"
    out = record_path(cfg, task_id, model, run)
    started = monotonic()

    # not_attempted -> synthesise directly, no API.
    if row.get("extraction_status") == "not_attempted":
        if not force and out.exists():
            return EvalResult(task_id, model, run, "synthesised", 0, 0.0, 0.0)
        ev = synthesise_blank(row["question_type"], int(row["max_mark"]))
        meta = _base_meta(cfg, row, bp, model, run)
        meta.update({
            "provider": None, "model_id": None, "cache_key": "synthesised",
            "input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0,
            "cost_usd": 0.0, "latency_ms": 0, "attempt_number": 0,
            "http_status": None, "evidence_not_found": [], "synthesised": True,
            "calls": [],
        })
        _write_record(out, ev, meta)
        return EvalResult(task_id, model, run, "synthesised", 0, 0.0, 0.0)

    assert grader is not None
    ck = cache_key(bp.prompt_bytes, grader.model_id, cfg.thinking_level, run)

    # Resume: completed record whose cache_key still matches.
    if not force and out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
            if prev.get("_metadata", {}).get("cache_key") == ck:
                total = prev.get("score_breakdown", {}).get("total_score")
                return EvalResult(task_id, model, run, "cached", total,
                                  prev["_metadata"].get("cost_usd", 0.0), 0.0)
        except (OSError, json.JSONDecodeError):
            pass

    try:
        ev, calls = _grade_and_parse(grader, bp, label)
    except LLMError as exc:
        raise EvalError(label, str(exc)) from exc

    evidence_missing = check_evidence(ev, row.get("extracted_text", ""))
    total_in = sum(c.input_tokens for c in calls)
    total_out = sum(c.output_tokens for c in calls)
    thoughts = sum(int(c.settings.get("thoughts_tokens", 0)) for c in calls)
    total_cost = sum(c.cost_usd for c in calls)
    last = calls[-1]
    # one entry per API call (>1 only when a JSON-parse retry fired)
    call_records = [{
        "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
        "thoughts_tokens": int(c.settings.get("thoughts_tokens", 0)),
        "cost_usd": round(c.cost_usd, 6), "latency_ms": c.latency_ms,
        "attempt_number": c.attempt_number, "http_status": c.http_status,
    } for c in calls]

    meta = _base_meta(cfg, row, bp, model, run)
    meta.update({
        "provider": grader.provider, "model_id": grader.model_id, "cache_key": ck,
        "input_tokens": total_in, "output_tokens": total_out,
        "thoughts_tokens": thoughts, "cost_usd": round(total_cost, 6),
        "latency_ms": last.latency_ms, "attempt_number": last.attempt_number,
        "http_status": last.http_status, "evidence_not_found": evidence_missing,
        "synthesised": False, "calls": call_records,
    })
    _write_record(out, ev, meta)
    return EvalResult(
        task_id, model, run, "graded", ev.score_breakdown.total_score,
        total_cost, monotonic() - started, evidence_missing,
    )


def build_prompts_for_rows(cfg: EvalConfig, rows: list[dict]) -> dict[str, BuiltPrompt]:
    rubric, rhash = load_rubric(cfg)
    tmpl = load_user_template(cfg)
    return {r["task_id"]: build_prompt(r, rubric, rhash, tmpl, cfg) for r in rows}
