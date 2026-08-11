"""Stage 4: assemble dataset/master_{backend}.csv from segmented transcripts.

Usage:
    python assemble.py --backend gemini
    python assemble.py --backend mistral_ocr

Joins prompts/task_stems.csv on question_no. Transcripts are copied into the
CSV exactly as stored — no cleanup of any kind.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys

from common import (
    DATASET_DIR,
    LOGS_DIR,
    TASK_MAP,
    TASK_STEMS_FILE,
    TRANSCRIPTS_DIR,
    read_run_log,
    strip_struck,
    utc_now,
)

COLUMNS = [
    "script_id", "task_id", "task_type", "question_no", "marks", "prompt_given",
    "stimulus_data", "transcribed_text", "text_sent_to_grader",
    "scan_file_reference", "page_range",
    "transcription_backend", "backend_version", "backend_settings",
    "illegible_count", "cut_count", "struck_count", "word_count",
    "run_timestamp", "notes",
]


def load_stems() -> dict[int, dict]:
    with TASK_STEMS_FILE.open() as f:
        return {int(r["question_no"]): r for r in csv.DictReader(f)}


def load_meta(backend: str, unit: str) -> dict:
    path = LOGS_DIR / "segment_meta" / backend / f"{unit}.meta"
    meta = {}
    if path.exists():
        for line in path.read_text().splitlines():
            k, _, v = line.partition("=")
            meta[k] = v
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["gemini", "mistral_ocr"])
    args = parser.parse_args()
    backend = args.backend

    stems = load_stems()
    prefix_to_q = {v[0]: q for q, v in TASK_MAP.items()}

    # Per-script backend version/settings and latest call timestamp, from the
    # transcription run log (successful calls only).
    runs: dict[str, dict] = {}
    for r in read_run_log(backend):
        if r.get("error"):
            continue
        s = runs.setdefault(r["script_id"], {"ts": "", "version": "", "settings": {}})
        s["ts"] = max(s["ts"], r["timestamp"])
        s["version"] = r["backend_version"]
        s["settings"] = r["settings"]

    unit_dir = TRANSCRIPTS_DIR / backend
    if not unit_dir.is_dir():
        print(f"no transcripts for backend {backend!r}", file=sys.stderr)
        return 1

    DATASET_DIR.mkdir(exist_ok=True)
    out_path = DATASET_DIR / f"master_{backend}.csv"
    rows = 0
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for txt_path in sorted(unit_dir.glob("*.txt")):
            unit = txt_path.stem                      # e.g. PARA_001
            m = re.match(r"([A-Z]+)_(\d{3})$", unit)
            if not m:
                print(f"skipping unrecognised file {txt_path.name}", file=sys.stderr)
                continue
            prefix = m.group(1)
            q = prefix_to_q[prefix]
            stem = stems.get(q, {})
            meta = load_meta(backend, unit)
            script_id = meta.get("script_id", "")
            run = runs.get(script_id, {})
            text = txt_path.read_text()
            n_headers = int(meta.get("n_header_occurrences", "1"))
            notes = "" if n_headers == 1 else (
                f"answer header appeared {n_headers} times; chunks concatenated in page order"
            )
            writer.writerow({
                "script_id": script_id,
                "task_id": unit,
                "task_type": TASK_MAP[q][1],
                "question_no": q,
                "marks": stem.get("marks", ""),
                "prompt_given": stem.get("prompt_given", ""),
                "stimulus_data": stem.get("stimulus_data", ""),
                "transcribed_text": text,
                "text_sent_to_grader": strip_struck(text),
                "scan_file_reference": f"{script_id}.pdf",
                "page_range": meta.get("page_range", ""),
                "transcription_backend": backend,
                "backend_version": run.get("version", ""),
                "backend_settings": json.dumps(run.get("settings", {}), ensure_ascii=False),
                "illegible_count": text.count("[illegible"),
                "cut_count": text.count("[cut:"),
                "struck_count": text.count("[struck:"),
                "word_count": len(text.split()),
                "run_timestamp": run.get("ts", utc_now()),
                "notes": notes,
            })
            rows += 1
    print(f"wrote {out_path} ({rows} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
