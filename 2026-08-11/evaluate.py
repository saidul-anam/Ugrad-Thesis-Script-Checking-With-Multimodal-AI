"""Evaluation stage: score each task unit in dataset/master_{backend}.csv
against the frozen rubric (prompts/rubric_v1.txt) and record structured JSON.

Usage:
    python evaluate.py                       # gemini grader, run 1
    python evaluate.py --run 2               # repeat run
    python evaluate.py --model <model-id>    # override grader model

One unit, one API call. Raw responses are saved verbatim before parsing.
Scores are never repaired, clamped, or recomputed; internal inconsistencies
are logged to logs/schema_violations.csv as findings.
Resumable: if evaluations/raw/{task_id}_run{N}.json exists, it is re-parsed
from disk instead of re-calling the API.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time

from dotenv import load_dotenv

from common import ROOT, append_csv_row, append_run_log, utc_now

RUBRIC_FILE = ROOT / "prompts" / "rubric_v1.txt"
DATASET_CSV = ROOT / "dataset" / "master_gemini.csv"
EVAL_RAW_DIR = ROOT / "evaluations" / "raw"
RESULTS_CSV = ROOT / "results" / "evaluation_log.csv"
SCHEMA_VIOLATIONS_CSV = ROOT / "logs" / "schema_violations.csv"

PROMPT_VERSION = "rubric_v1"
MAX_429_RETRIES = 5
BACKOFF_BASE_S = 2.0

RESULT_COLUMNS = [
    "script_id", "task_id", "task_type", "model", "model_version", "run_number",
    "total_score", "performance_band", "context_content_data",
    "structure_format_brevity", "language_mechanics",
    "originality_comparisons_paraphrase", "cap_applied", "cap_reason",
    "is_attempted", "frequent_errors", "positive_aspects", "feedback_summary",
    "raw_json_path", "run_timestamp", "prompt_version", "temperature",
]

BAND_RANGE_RE = re.compile(r"\((\d+)\s*[-–]\s*(\d+)\)|\((\d+)\)")


def build_user_message(row: dict) -> str:
    parts = [
        f"TASK TYPE: {row['task_type']}",
        "",
        "TASK PROMPT GIVEN TO STUDENT:",
        row["prompt_given"],
        "",
    ]
    if row["stimulus_data"].strip():
        parts += ["SOURCE DATA PROVIDED TO STUDENT:", row["stimulus_data"], ""]
    parts += ["STUDENT RESPONSE:", row["text_sent_to_grader"]]
    return "\n".join(parts)


def log_violation(task_id: str, run: int, vtype: str, detail: str) -> None:
    append_csv_row(
        SCHEMA_VIOLATIONS_CSV,
        ["timestamp", "task_id", "run_number", "violation_type", "detail"],
        [utc_now(), task_id, run, vtype, detail],
    )
    print(f"    VIOLATION [{vtype}]: {detail}")


def band_range(band: str) -> tuple[int, int] | None:
    """'Band 4 (8-10)' -> (8, 10); 'Band 0 (0)' -> (0, 0)."""
    m = BAND_RANGE_RE.search(band or "")
    if not m:
        return None
    if m.group(3) is not None:
        v = int(m.group(3))
        return (v, v)
    return (int(m.group(1)), int(m.group(2)))


def validate(task_id: str, run: int, parsed: dict) -> None:
    """Log internal inconsistencies. Never corrects anything."""
    try:
        sb = parsed["score_breakdown"]
        subs = [
            sb["context_content_data"], sb["structure_format_brevity"],
            sb["language_mechanics"], sb["originality_comparisons_paraphrase"],
        ]
        total = sb["total_score"]
    except (KeyError, TypeError) as e:
        log_violation(task_id, run, "missing_field", f"score_breakdown incomplete: {e!r}")
        return
    if sum(subs) != total:
        log_violation(task_id, run, "subscore_sum_mismatch",
                      f"sub-scores {subs} sum to {sum(subs)}, total_score is {total}")
    band = parsed.get("performance_band", "")
    rng = band_range(band)
    if rng is None:
        log_violation(task_id, run, "unrecognised_band", f"performance_band={band!r}")
    elif not (rng[0] <= total <= rng[1]):
        log_violation(task_id, run, "score_outside_band",
                      f"total_score {total} outside {band!r} range {rng}")


class Grader:
    def __init__(self, model: str):
        from google import genai
        from google.genai import types

        self.types = types
        self.client = genai.Client(vertexai=True, api_key=os.environ["GOOGLE_API_KEY"])
        self.model = model
        self.model_version = model  # refined from responses
        self.rubric = RUBRIC_FILE.read_text()
        self.temperature = 0.0
        self.max_output_tokens = 16384
        self.top_p = None  # not set; API default

    def settings(self) -> dict:
        return {
            "model": self.model,
            "model_version_reported": self.model_version,
            "api_mode": "vertex_ai_express",
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_p": self.top_p,
            "response_mime_type": "application/json",
            "system_prompt_file": RUBRIC_FILE.name,
            "prompt_version": PROMPT_VERSION,
            "backoff_429": {"max_retries": MAX_429_RETRIES, "base_s": BACKOFF_BASE_S},
        }

    def grade(self, user_message: str) -> tuple[dict, dict, float]:
        """Returns (raw response dict, usage dict, duration)."""
        t0 = time.time()
        retries_429 = 0
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_message,
                    config=self.types.GenerateContentConfig(
                        system_instruction=self.rubric,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        response_mime_type="application/json",
                    ),
                )
                break
            except Exception as e:
                is_429 = getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
                if not is_429 or retries_429 >= MAX_429_RETRIES:
                    raise
                delay = BACKOFF_BASE_S * (2 ** retries_429) + random.uniform(0, 1)
                retries_429 += 1
                print(f"    429 rate limit; sleeping {delay:.1f}s "
                      f"(retry {retries_429}/{MAX_429_RETRIES})")
                time.sleep(delay)
        duration = time.time() - t0
        if response.model_version:
            self.model_version = response.model_version
        um = response.usage_metadata
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", None),
            "output_tokens": getattr(um, "candidates_token_count", None),
            "thoughts_tokens": getattr(um, "thoughts_token_count", None),
            "total_tokens": getattr(um, "total_token_count", None),
            "retries_429": retries_429,
        }
        return response.model_dump(mode="json", exclude_none=True), usage, duration


def response_text(raw: dict) -> str | None:
    """Extract the text payload from a stored raw response dict."""
    try:
        parts = raw["candidates"][0]["content"]["parts"]
        texts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
        return "".join(texts) if texts else None
    except (KeyError, IndexError, TypeError):
        return None


def upsert_results(new_rows: list[dict]) -> None:
    """Append rows, replacing any prior row with the same (model, run, task)."""
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    existing: list[dict] = []
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open() as f:
            existing = list(csv.DictReader(f))
    new_keys = {(r["model"], str(r["run_number"]), r["task_id"]) for r in new_rows}
    kept = [r for r in existing
            if (r["model"], str(r["run_number"]), r["task_id"]) not in new_keys]
    with RESULTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for r in kept + new_rows:
            w.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-3.6-flash", help="grader model id")
    parser.add_argument("--run", type=int, default=1, help="run number N")
    args = parser.parse_args()

    load_dotenv()
    grader = Grader(args.model)
    EVAL_RAW_DIR.mkdir(parents=True, exist_ok=True)

    with DATASET_CSV.open() as f:
        rows = list(csv.DictReader(f))
    print(f"grading {len(rows)} task units from {DATASET_CSV.name} "
          f"[model={args.model}, run={args.run}, temperature={grader.temperature}]")

    results: list[dict] = []
    excluded: list[str] = []
    usage_totals: dict[str, int] = {}

    for row in rows:
        task_id = row["task_id"]
        raw_path = EVAL_RAW_DIR / f"{task_id}_run{args.run}.json"
        print(f"  {task_id}:", end=" ", flush=True)

        parsed = None
        for attempt in (1, 2):
            if raw_path.exists() and attempt == 1:
                raw = json.loads(raw_path.read_text())
                print("(raw exists, re-parsing)", end=" ")
            else:
                user_message = build_user_message(row)
                try:
                    raw, usage, duration = grader.grade(user_message)
                except Exception as e:
                    log_violation(task_id, args.run, "api_error", repr(e)[:300])
                    raw = None
                    if attempt == 2:
                        break
                    continue
                raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=1))
                append_run_log(f"eval_{grader.model}", {
                    "timestamp": utc_now(), "task_id": task_id, "run": args.run,
                    "attempt": attempt, "settings": grader.settings(),
                    "usage": usage, "duration_s": round(duration, 2),
                })
                for k, v in usage.items():
                    if isinstance(v, int):
                        usage_totals[k] = usage_totals.get(k, 0) + v
                print(f"({duration:.0f}s)", end=" ")

            text = response_text(raw) if raw else None
            if text is not None:
                try:
                    parsed = json.loads(text)
                    break
                except json.JSONDecodeError as e:
                    log_violation(task_id, args.run, "invalid_json",
                                  f"attempt {attempt}: {e}; raw saved at {raw_path.name}")
            else:
                log_violation(task_id, args.run, "empty_response",
                              f"attempt {attempt}: no text part in response")
            if attempt == 1 and raw_path.exists():
                # force a fresh API call on the retry
                raw_path.rename(raw_path.with_suffix(".invalid.json"))

        if parsed is None:
            excluded.append(task_id)
            print("EXCLUDED (no valid JSON after retry)")
            continue

        validate(task_id, args.run, parsed)

        sb = parsed.get("score_breakdown") or {}
        audit = parsed.get("structural_audit") or {}
        attempt_status = parsed.get("attempt_status") or {}
        errs = (parsed.get("error_analysis") or {}).get("frequent_errors") or []
        pos = (parsed.get("error_analysis") or {}).get("positive_aspects") or []
        results.append({
            "script_id": row["script_id"],
            "task_id": task_id,
            "task_type": row["task_type"],
            "model": "gemini",
            "model_version": grader.model_version,
            "run_number": args.run,
            "total_score": sb.get("total_score", ""),
            "performance_band": parsed.get("performance_band", ""),
            "context_content_data": sb.get("context_content_data", ""),
            "structure_format_brevity": sb.get("structure_format_brevity", ""),
            "language_mechanics": sb.get("language_mechanics", ""),
            "originality_comparisons_paraphrase": sb.get("originality_comparisons_paraphrase", ""),
            "cap_applied": audit.get("cap_applied", ""),
            "cap_reason": audit.get("applied_cap_reason", ""),
            "is_attempted": attempt_status.get("is_attempted", ""),
            "frequent_errors": " | ".join(str(x) for x in errs),
            "positive_aspects": " | ".join(str(x) for x in pos),
            "feedback_summary": parsed.get("feedback_summary", ""),
            "raw_json_path": str(raw_path.relative_to(ROOT)),
            "run_timestamp": utc_now(),
            "prompt_version": PROMPT_VERSION,
            "temperature": grader.temperature,
        })
        print(f"score {sb.get('total_score', '?')}/10, {parsed.get('performance_band', '?')}")

    upsert_results(results)
    print(f"\nwrote {len(results)} rows to {RESULTS_CSV.relative_to(ROOT)}"
          + (f"; EXCLUDED: {excluded}" if excluded else ""))
    print(f"usage totals: {usage_totals}")

    # ---- console summary ----
    print(f"\n{'task_id':<12} {'task_type':<12} {'score':>5}  {'band':<14} cap")
    for r in results:
        band_short = (r["performance_band"] or "?").split("(")[0].strip()
        print(f"{r['task_id']:<12} {r['task_type']:<12} {str(r['total_score']):>5}  "
              f"{band_short:<14} {r['cap_applied']}")

    para = EVAL_RAW_DIR / f"PARA_001_run{args.run}.json"
    if para.exists():
        text = response_text(json.loads(para.read_text()))
        if text:
            print("\n=== PARA_001 full JSON ===")
            try:
                print(json.dumps(json.loads(text), indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
