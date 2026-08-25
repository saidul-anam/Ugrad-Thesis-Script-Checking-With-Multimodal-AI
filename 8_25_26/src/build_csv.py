"""Derive extraction.csv (+ extraction_runs.csv) from the canonical JSON.

The per-script JSON in data/transcripts/ is the source of truth. This module
produces the *derived, regenerable* answer-grain CSV described in
docs/extraction_csv_schema_v3.md — one row per (script x in-scope question).

Verbatim rule: `extracted_text_raw` is the transcript byte-for-byte.
`extracted_text` is the defined grading view — struck text removed, insertions
applied, uncertainty markers unwrapped — but spelling/grammar errors are never
touched. Question context comes from data/Question/task_stems.csv.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.config import Config
from src.validate import (
    CONFIDENCE_MIN,
    LETTER_QUESTIONS,
    PARAGRAPH_QUESTIONS,
    _isolated_words,
    normalize_qno,
)

# task_type (task_stems.csv) -> question_type enum + task_id prefix
TYPE_MAP = {
    "Summary": ("Summary", "SUMMARY"),
    "Paragraph": ("Paragraph", "PARA"),
    "Graph/Chart": ("Graph_Chart", "CHART"),
    "Story": ("Story", "STORY"),
    "Letter": ("Letter_Email", "LETTER"),
    "Theme": ("Theme", "THEME"),
}

LETTER_COMPONENT_PATTERNS = {
    "date": re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b|\b\d{1,2}(st|nd|rd|th)?\s+"
                       r"(January|February|March|April|May|June|July|August|September|"
                       r"October|November|December)", re.IGNORECASE),
    "salutation": re.compile(r"\bDear\b", re.IGNORECASE),
    "close": re.compile(r"\b(Yours?|Sincerely|Faithfully|lovingly|your friend|"
                        r"your loving)\b", re.IGNORECASE),
}

CSV_COLUMNS = [
    # identity
    "task_id", "script_id", "question_no", "question_type", "max_mark",
    # ground truth (reserved-empty)
    "teacher_mark", "teacher_mark_confidence", "teacher_mark_source",
    # question context
    "question", "source_text",
    # transcript
    "extracted_text_raw", "extracted_text", "student_title", "page_range",
    # attempt status
    "extraction_status", "error_message",
    # transcription quality
    "confidence", "illegible_count", "unclear_count", "cut_count",
    "struck_count", "inserted_count", "ambiguous_authorship_count",
    "red_ink_suspected", "needs_review", "review_flags",
    # computed metrics
    "word_count", "char_count", "source_word_count", "length_ratio",
    "exceeds_one_third", "verbatim_overlap_pct", "paragraph_break_count",
    "letter_components_found", "letter_components_missing_count",
    # provenance
    "transcription_model_version", "transcription_provider",
    "transcription_prompt_version", "prompt_hash", "transcription_settings",
    "extraction_timestamp",
]

RUNS_COLUMNS = [
    "run_id", "script_id", "model_id", "provider", "thinking_level",
    "pages_sent", "input_tokens", "output_tokens", "cost_usd", "latency_ms",
    "attempt_number", "http_status", "timestamp",
]


@dataclass
class QuestionMeta:
    question_type: str
    prefix: str
    max_mark: int
    question: str
    source_text: str


# ---------------------------------------------------------------------------
# Marker handling — verbatim raw vs resolved grading view
# ---------------------------------------------------------------------------
def resolve_markers(raw: str) -> str:
    """Grading view: remove struck text, apply insertions, unwrap unclear/cut.

    Never alters spelling. [illegible] is kept (it marks a real gap).
    """
    text = re.sub(r"~~.*?~~", "", raw)                       # struck -> gone
    text = re.sub(r"\{inserted:\s*(.*?)\}", r"\1", text)     # insertion applied
    text = re.sub(r"\[unclear:\s*(.*?)\]", r"\1", text)      # best reading
    text = re.sub(r"\[cut:\s*(.*?)\]", r"\1", text)          # partial kept
    return text


def _marker_counts(raw: str) -> dict[str, int]:
    return {
        "illegible_count": len(re.findall(r"\[illegible\]", raw)),
        "unclear_count": len(re.findall(r"\[unclear:", raw)),
        "cut_count": len(re.findall(r"\[cut:", raw)),
        "struck_count": len(re.findall(r"~~.*?~~", raw)),
        "inserted_count": len(re.findall(r"\{inserted:", raw)),
    }


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", re.sub(r"\[illegible\]", " ", text))


def _norm_tokens(text: str) -> list[str]:
    return [t.lower().strip("'") for t in re.findall(r"[A-Za-z0-9']+", text)]


def verbatim_overlap_pct(student: str, source: str) -> float:
    """Fraction of student tokens inside a run of >=7 consecutive tokens shared
    with the source (case-folded, punctuation stripped)."""
    s = _norm_tokens(student)
    t = _norm_tokens(source)
    if not s or not t:
        return 0.0
    covered = [False] * len(s)
    n, m = len(s), len(t)
    for i in range(n):
        best = 0
        for j in range(m):
            k = 0
            while i + k < n and j + k < m and s[i + k] == t[j + k]:
                k += 1
            best = max(best, k)
        if best >= 7:
            for k in range(best):
                covered[i + k] = True
    return round(sum(covered) / n, 4)


def _letter_components(text: str) -> list[str]:
    """Heuristic letter-part detection (spot-verify by hand — drives a cap)."""
    found = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # address: comma-bearing short lines near the top, or a From/To block
    if any(("," in ln and len(ln.split()) <= 6) for ln in lines[:4]) or \
            re.search(r"\b(From|To)\b", text):
        found.append("address")
    for name, pat in LETTER_COMPONENT_PATTERNS.items():
        if pat.search(text):
            found.append(name)
    # body: at least one long line of prose
    if any(len(ln.split()) >= 12 for ln in lines):
        found.append("body")
    # signature: a short final line that isn't the close phrase
    if lines and len(lines[-1].split()) <= 4 and not LETTER_COMPONENT_PATTERNS[
        "close"
    ].search(lines[-1]):
        found.append("signature")
    # preserve canonical order
    order = ["address", "date", "salutation", "body", "close", "signature"]
    return [c for c in order if c in found]


# ---------------------------------------------------------------------------
# Loading inputs
# ---------------------------------------------------------------------------
def load_task_stems(path: Path) -> dict[str, QuestionMeta]:
    out: dict[str, QuestionMeta] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qno = normalize_qno(row["question_no"])
            task_type = row["task_type"].strip()
            enum, prefix = TYPE_MAP[task_type]
            prompt_given = row["prompt_given"]
            stimulus = row.get("stimulus_data", "") or ""
            question = prompt_given + (f"\n\n{stimulus}" if stimulus else "")
            # source_text: passage/poem alone (Q3, Q11); empty otherwise.
            if qno == "3":
                source_text = re.sub(
                    r"^\s*Summarize the following text\.\s*", "", prompt_given
                )
            elif qno == "11":
                source_text = stimulus
            else:
                source_text = ""
            out[qno] = QuestionMeta(
                question_type=enum,
                prefix=prefix,
                max_mark=int(row["marks"]),
                question=question,
                source_text=source_text,
            )
    return out


def _page_range(pages: list) -> str:
    nums = [int(p) for p in pages if isinstance(p, (int, float))]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    return str(lo) if lo == hi else f"{lo}-{hi}"


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------
def build_rows(
    cfg: Config, stems: dict[str, QuestionMeta]
) -> tuple[list[dict], list[dict]]:
    targets = [normalize_qno(q) for q in cfg.target_questions]
    transcripts = sorted(cfg.paths.transcripts.glob("*.json"))
    type_counter: dict[str, int] = {}
    rows: list[dict] = []
    runs: list[dict] = []

    for path in transcripts:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("_metadata", {})
        script_id = meta.get("script_id", path.stem)
        answers = {normalize_qno(a["question_number"]): a for a in data.get("answers", [])}

        # provenance is identical for every row of this script
        prov = {
            "transcription_model_version": meta.get("model_id", ""),
            "transcription_provider": meta.get("provider", ""),
            "transcription_prompt_version": meta.get("prompt_version", ""),
            "prompt_hash": meta.get("prompt_hash", ""),
            "transcription_settings": json.dumps(meta.get("settings", {})),
            "extraction_timestamp": meta.get("timestamp", ""),
        }

        for qno in targets:
            qmeta = stems[qno]
            type_counter[qmeta.prefix] = type_counter.get(qmeta.prefix, 0) + 1
            task_id = f"{qmeta.prefix}_{type_counter[qmeta.prefix]:03d}"
            rows.append(
                _build_row(script_id, qno, qmeta, task_id, answers.get(qno), prov)
            )

        for call in meta.get("calls", []):
            runs.append({
                "run_id": f"{script_id}-{call.get('chunk_index', 0)}",
                "script_id": script_id,
                "model_id": meta.get("model_id", ""),
                "provider": meta.get("provider", ""),
                "thinking_level": meta.get("thinking_level", ""),
                "pages_sent": call.get("pages_sent", ""),
                "input_tokens": call.get("input_tokens", ""),
                "output_tokens": call.get("output_tokens", ""),
                "cost_usd": call.get("cost_usd", ""),
                "latency_ms": call.get("latency_ms", ""),
                "attempt_number": call.get("attempt_number", ""),
                "http_status": call.get("http_status", ""),
                "timestamp": meta.get("timestamp", ""),
            })
    return rows, runs


def _build_row(
    script_id: str, qno: str, qmeta: QuestionMeta, task_id: str,
    answer: dict | None, prov: dict,
) -> dict:
    base = {
        "task_id": task_id,
        "script_id": script_id,
        "question_no": qno,
        "question_type": qmeta.question_type,
        "max_mark": qmeta.max_mark,
        "teacher_mark": "",
        "teacher_mark_confidence": "",
        "teacher_mark_source": "absent",
        "question": qmeta.question,
        "source_text": qmeta.source_text,
        **prov,
    }

    if answer is None:
        # A skipped question is a legitimate Band 0, not a pipeline error.
        base.update({
            "extracted_text_raw": "", "extracted_text": "", "student_title": "",
            "page_range": "", "extraction_status": "not_attempted",
            "error_message": "question not found in transcript (verified absent)",
            "confidence": "", "illegible_count": "", "unclear_count": "",
            "cut_count": "", "struck_count": "", "inserted_count": "",
            "ambiguous_authorship_count": "", "red_ink_suspected": "",
            "needs_review": "", "review_flags": "",
            "word_count": 0, "char_count": 0, "source_word_count": "",
            "length_ratio": "", "exceeds_one_third": "", "verbatim_overlap_pct": "",
            "paragraph_break_count": "", "letter_components_found": "",
            "letter_components_missing_count": "",
        })
        return base

    raw = answer.get("transcript", "")
    resolved = resolve_markers(raw)
    counts = _marker_counts(raw)
    conf = float(answer.get("confidence", 0.0))
    word_count = len(_word_tokens(resolved))

    # quality flags (row-level)
    flags: list[str] = []
    red_ink = qno in (set(m for m in ["3", "7", "8", "9", "11"])) and bool(
        _isolated_words(raw)
    ) and qno not in LETTER_QUESTIONS
    if conf < CONFIDENCE_MIN:
        flags.append("low_confidence")
    if red_ink:
        flags.append("suspected_red_ink")

    paragraph_breaks = resolved.count("\n\n")
    if qno in PARAGRAPH_QUESTIONS:
        segs = [s for s in resolved.split("\n\n") if s.strip()]
        mean_words = (sum(len(_word_tokens(s)) for s in segs) / len(segs)) if segs else 0
        if paragraph_breaks > 3 or (segs and mean_words < 15):
            flags.append("wrap_as_paragraph")

    # source-dependent metrics (Q3, Q11)
    src_wc: object = ""
    length_ratio: object = ""
    exceeds_third: object = ""
    overlap: object = ""
    if qno in ("3", "11") and qmeta.source_text:
        src_wc = len(_word_tokens(qmeta.source_text))
        length_ratio = round(word_count / src_wc, 4) if src_wc else ""
        overlap = verbatim_overlap_pct(resolved, qmeta.source_text)
    if qno == "3" and isinstance(src_wc, int) and src_wc:
        exceeds_third = word_count > (src_wc / 3)

    # letter metrics (Q10)
    letter_found: object = ""
    letter_missing: object = ""
    if qno == "10":
        comps = _letter_components(resolved)
        letter_found = ";".join(comps)
        letter_missing = 6 - len(comps)

    base.update({
        "extracted_text_raw": raw,
        "extracted_text": resolved,
        "student_title": answer.get("student_title") or "",
        "page_range": _page_range(answer.get("pages", [])),
        "extraction_status": "ok",
        "error_message": "",
        "confidence": conf,
        **counts,
        "ambiguous_authorship_count": len(answer.get("ambiguous_authorship", []) or []),
        "red_ink_suspected": red_ink,
        "needs_review": bool(flags),
        "review_flags": ";".join(flags),
        "word_count": word_count,
        "char_count": len(resolved),
        "source_word_count": src_wc,
        "length_ratio": length_ratio,
        "exceeds_one_third": exceeds_third,
        "verbatim_overlap_pct": overlap,
        "paragraph_break_count": paragraph_breaks,
        "letter_components_found": letter_found,
        "letter_components_missing_count": letter_missing,
    })
    return base


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    # utf-8-sig, QUOTE_ALL, CRLF per schema FILE CONVENTIONS.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, quoting=csv.QUOTE_ALL, lineterminator="\r\n"
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_assertions(rows: list[dict], n_scripts: int) -> list[str]:
    """Schema VALIDATION ASSERTIONS. Returns human-readable PASS/WARN/FAIL lines."""
    out: list[str] = []

    def check(cond: bool, msg: str, warn: bool = False) -> None:
        tag = "PASS" if cond else ("WARN" if warn else "FAIL")
        out.append(f"[{tag}] {msg}")

    ids = [r["task_id"] for r in rows]
    check(len(ids) == len(set(ids)), "task_id unique")
    check(len(rows) == n_scripts * 6, f"row count == scripts*6 ({len(rows)} == {n_scripts*6})")

    pairs = [(r["script_id"], r["question_no"]) for r in rows]
    check(len(pairs) == len(set(pairs)), "each (script, question) present exactly once")

    mark_map = {"3": 10, "7": 10, "8": 10, "9": 7, "10": 5, "11": 8}
    check(all(int(r["max_mark"]) == mark_map[r["question_no"]] for r in rows),
          "question_type<->max_mark mapping holds")

    errors = sum(1 for r in rows if r["extraction_status"] == "error")
    check(errors == 0, f"extraction_status=='error' count is 0 (got {errors})")

    check(all(r["teacher_mark"] == "" and r["teacher_mark_source"] == "absent"
              for r in rows), "teacher_mark empty & source 'absent' on every row")

    ok = [r for r in rows if r["extraction_status"] == "ok"]
    check(all(r["extracted_text"] != "" for r in ok), "extracted_text non-empty for ok rows")
    check(all("~~" not in r["extracted_text"] and "{inserted" not in r["extracted_text"]
              and "[struck:" not in r["extracted_text"] for r in rows),
          "extracted_text has no struck/inserted markers")

    hashes = {r["prompt_hash"] for r in rows}
    check(len(hashes) == 1, f"prompt_hash identical across all rows ({len(hashes)} distinct)")

    uncertainty = sum(int(r["illegible_count"] or 0) + int(r["unclear_count"] or 0)
                      + int(r["cut_count"] or 0) for r in ok)
    check(uncertainty > 0,
          f"total illegible+unclear+cut > 0 (got {uncertainty}) — 0 may mean guessing",
          warn=True)
    return out


def build_all(cfg: Config) -> tuple[Path, Path, list[str], int]:
    stems = load_task_stems(cfg.paths.raw_pdfs.parent / "Question" / "task_stems.csv")
    rows, runs = build_rows(cfg, stems)
    n_scripts = len(sorted(cfg.paths.transcripts.glob("*.json")))

    csv_path = cfg.paths.raw_pdfs.parent / "extraction.csv"
    runs_path = cfg.paths.raw_pdfs.parent / "extraction_runs.csv"
    _write_csv(csv_path, CSV_COLUMNS, rows)
    _write_csv(runs_path, RUNS_COLUMNS, runs)

    assertions = run_assertions(rows, n_scripts)
    return csv_path, runs_path, assertions, len(rows)
