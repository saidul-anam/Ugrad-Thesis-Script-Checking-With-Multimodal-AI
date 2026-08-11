"""Shared paths, data types, and logging for the transcription pipeline."""
from __future__ import annotations

import csv
import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCANS_DIR = ROOT / "scans"
PAGE_IMAGES_DIR = ROOT / "page_images"
TRANSCRIPTS_RAW_DIR = ROOT / "transcripts_raw"
TRANSCRIPTS_DIR = ROOT / "transcripts"
TRANSCRIPTS_EXCLUDED_DIR = ROOT / "transcripts_excluded"
LOGS_DIR = ROOT / "logs"
RAW_RESPONSES_DIR = LOGS_DIR / "raw_responses"
REPORTS_DIR = ROOT / "reports"
DATASET_DIR = ROOT / "dataset"
PROMPT_FILE = ROOT / "prompts" / "ocr_prompt_v1.txt"
TASK_STEMS_FILE = ROOT / "prompts" / "task_stems.csv"

API_FAILURES_CSV = LOGS_DIR / "api_failures.csv"
SEGMENTATION_FAILURES_CSV = LOGS_DIR / "segmentation_failures.csv"

RENDER_DPI = 300

# question_no -> (task id prefix, task type). Everything else is excluded.
TASK_MAP = {
    3: ("SUMMARY", "Summary"),
    7: ("PARA", "Paragraph"),
    8: ("CHART", "Graph/Chart"),
    9: ("STORY", "Story"),
    10: ("LETTER", "Letter"),
}


@dataclass
class PageResult:
    """Result of transcribing one page image. Backend-agnostic."""
    text: str | None
    usage: dict
    raw_response: object  # JSON-serialisable raw API response, or None
    error: str | None = None
    duration_s: float = 0.0


def strip_struck(text: str) -> str:
    """Remove [struck: ...] spans (marker and contents) and collapse the
    double spaces that stripping leaves behind. Used to build the grader
    input (text_sent_to_grader); the stored transcript is never modified.
    [illegible] and [cut: ...] are left in place — they represent genuinely
    missing text, not presentation noise."""
    out = re.sub(r"\[struck:[^\]]*\]", "", text)
    out = re.sub(r" {2,}", " ", out)
    out = re.sub(r" +$", "", out, flags=re.MULTILINE)
    return re.sub(r"^ +", "", out, flags=re.MULTILINE)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def script_num(script_id: str) -> int:
    """SE_11_Q1_0001 -> 1"""
    m = re.search(r"(\d+)$", script_id)
    if not m:
        raise ValueError(f"cannot extract script number from {script_id!r}")
    return int(m.group(1))


def page_txt_path(backend_name: str, script_id: str, page_no: int) -> Path:
    return TRANSCRIPTS_RAW_DIR / backend_name / f"{script_id}_p{page_no:02d}.txt"


def page_image_path(script_id: str, page_no: int) -> Path:
    return PAGE_IMAGES_DIR / f"{script_id}_p{page_no:02d}.png"


def raw_response_path(backend_name: str, script_id: str, page_no: int) -> Path:
    return RAW_RESPONSES_DIR / backend_name / f"{script_id}_p{page_no:02d}.json"


def append_csv_row(path: Path, header: list[str], row: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


def log_api_failure(script_id: str, page_no: int, backend: str, attempt: int, error: str) -> None:
    append_csv_row(
        API_FAILURES_CSV,
        ["timestamp", "script_id", "page", "backend", "attempt", "error"],
        [utc_now(), script_id, page_no, backend, attempt, error],
    )


def append_run_log(backend_name: str, record: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"run_log_{backend_name}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_run_log(backend_name: str) -> list[dict]:
    path = LOGS_DIR / f"run_log_{backend_name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
