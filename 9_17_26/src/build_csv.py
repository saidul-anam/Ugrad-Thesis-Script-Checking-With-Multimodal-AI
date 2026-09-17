"""Derive extraction.csv (+ extraction_runs.csv) from the canonical JSON.

The per-script JSON in data/transcripts/ is the source of truth. This module
produces the *derived, regenerable* answer-grain CSV described in
docs/extraction_csv_schema_v3.md — one row per (script x in-scope question).

Everything paper-specific comes from the exam set: which questions are in scope,
their marks and types, and the question text itself (from that set's
task_stems.csv). Row metrics are keyed off `question_type`, never off a hard-coded
question number, so a Class VII Summary at Q7 and a Class XI Summary at Q3 are
measured the same way.

Verbatim rule: `extracted_text_raw` is the transcript byte-for-byte.
`extracted_text` is the defined grading view — struck text removed, insertions
applied, uncertainty markers unwrapped — but spelling/grammar errors are never
touched.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.config import Config, ExamSetConfig, exam_set_of
from src.validate import CONFIDENCE_MIN, _isolated_words, normalize_qno

# task_type (task_stems.csv) -> question_type enum + task_id prefix
TYPE_MAP = {
    # Class XI extended-response types
    "Summary": ("Summary", "SUMMARY"),
    "Paragraph": ("Paragraph", "PARA"),
    "Graph/Chart": ("Graph_Chart", "CHART"),
    "Story": ("Story", "STORY"),
    "Letter": ("Letter_Email", "LETTER"),
    "Theme": ("Theme", "THEME"),
    # Class VI/VII reading types (item-scored against an answer key)
    "MCQ": ("MCQ", "MCQ"),
    "Short_Answer": ("Short_Answer", "SHORT"),
    "Gap_Fill": ("Gap_Fill", "GAP"),
    "Vocabulary": ("Vocabulary", "VOCAB"),
    "Table_Completion": ("Table_Completion", "TABLE"),
    "True_False": ("True_False", "TF"),
    "Sentence_Transformation": ("Sentence_Transformation", "TRANS"),
    "Preposition_Use": ("Preposition_Use", "PREP"),
    "Rearrange": ("Rearrange", "REARR"),
    "Poem_Short_Answer": ("Poem_Short_Answer", "POEM"),
    # Class VI/VII writing type
    "Dialogue": ("Dialogue", "DIALOG"),
}

# Question types whose answer is continuous prose. Only these get the
# absorbed-correction (isolated single word on its own line) check: an item-scored
# answer, a letter's address block and a dialogue's turns all legitimately put a
# lone word on a line.
PROSE_TYPES = {"Summary", "Paragraph", "Graph_Chart", "Story", "Theme"}

# Types whose length/verbatim ratio against the source passage drives a cap.
SUMMARY_TYPES = {"Summary"}

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
    "task_id", "exam_set", "script_id", "question_no", "question_type", "max_mark",
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
    "run_id", "exam_set", "script_id", "model_id", "provider", "thinking_level",
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
# An opener the transcriber never closed. Rare, but it happens — one Class VII
# script came back with `d) ~~joy` and no closing pair — and left alone the raw
# markup lands in the text a grader reads. Each pattern closes at the end of its
# line, which is where the answer it belongs to ends.
_DANGLING = (
    (re.compile(r"~~[^\n]*"), ""),                    # struck, unclosed -> gone
    (re.compile(r"\{inserted:\s*([^\n]*)"), r"\1"),   # keep the inserted words
    (re.compile(r"\[unclear:\s*([^\n]*)"), r"\1"),    # keep the best reading
    (re.compile(r"\[cut:\s*([^\n]*)"), r"\1"),        # keep the partial word
)


def resolve_balanced_markers(raw: str) -> str:
    """Marker resolution proper: every opener matched by its closer."""
    text = re.sub(r"~~.*?~~", "", raw)                       # struck -> gone
    text = re.sub(r"\{inserted:\s*(.*?)\}", r"\1", text)     # insertion applied
    text = re.sub(r"\[unclear:\s*(.*?)\]", r"\1", text)      # best reading
    text = re.sub(r"\[cut:\s*(.*?)\]", r"\1", text)          # partial kept
    return text


def resolve_markers(raw: str) -> str:
    """Grading view: remove struck text, apply insertions, unwrap unclear/cut.

    Never alters spelling. [illegible] is kept (it marks a real gap). An
    unterminated marker is resolved to the end of its line, and the row is
    flagged `unbalanced_marker`: the grading view must never carry transcription
    markup, but a transcriber that dropped a closer may have dropped text too,
    so a human should look at the page.
    """
    text = resolve_balanced_markers(raw)
    for pattern, replacement in _DANGLING:
        text = pattern.sub(replacement, text)
    return text


def has_unbalanced_markers(raw: str) -> bool:
    """True if any marker was opened and never closed. Flagged, not hidden."""
    return resolve_balanced_markers(raw) != resolve_markers(raw)


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
def _legacy_source_text(qno: str, prompt_given: str, stimulus: str) -> str:
    """Class XI stems carry no source_text column; derive it as before.

    Q3 (Summary) is the passage embedded in the prompt; Q11 (Theme) is the poem
    in stimulus_data. Everything else has none.
    """
    if qno == "3":
        return re.sub(r"^\s*Summarize the following text\.\s*", "", prompt_given)
    if qno == "11":
        return stimulus
    return ""


def load_task_stems(path: Path) -> dict[str, QuestionMeta]:
    """Read one exam set's task_stems.csv into {question_no: QuestionMeta}.

    `source_text` is an optional column: the passage a length/verbatim metric is
    measured against. When the column is absent the Class XI rule is applied.
    """
    out: dict[str, QuestionMeta] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        has_source_col = "source_text" in (reader.fieldnames or [])
        for row in reader:
            qno = normalize_qno(row["question_no"])
            task_type = row["task_type"].strip()
            if task_type not in TYPE_MAP:
                raise KeyError(
                    f"{path.name} Q{qno}: unknown task_type '{task_type}' "
                    f"(have: {sorted(TYPE_MAP)})"
                )
            enum, prefix = TYPE_MAP[task_type]
            prompt_given = row["prompt_given"]
            stimulus = row.get("stimulus_data", "") or ""
            question = prompt_given + (f"\n\n{stimulus}" if stimulus else "")
            source_text = (
                (row.get("source_text") or "") if has_source_col
                else _legacy_source_text(qno, prompt_given, stimulus)
            )
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


def _transcripts_for_set(cfg: Config, set_id: str) -> list[Path]:
    return [
        p for p in sorted(cfg.paths.transcripts.glob("*.json"))
        if exam_set_of(p.stem) == set_id
    ]


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------
def build_rows(
    cfg: Config, set_id: str, es: ExamSetConfig, stems: dict[str, QuestionMeta]
) -> tuple[list[dict], list[dict]]:
    targets = [normalize_qno(q) for q in es.target_questions]
    missing_stems = [q for q in targets if q not in stems]
    if missing_stems:
        raise KeyError(
            f"{es.task_stems.name} has no row for target question(s) "
            f"{', '.join(missing_stems)}"
        )
    type_counter: dict[str, int] = {}
    rows: list[dict] = []
    runs: list[dict] = []

    for path in _transcripts_for_set(cfg, set_id):
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
                _build_row(set_id, script_id, qno, qmeta, task_id,
                           answers.get(qno), prov)
            )

        for call in meta.get("calls", []):
            runs.append({
                "run_id": f"{script_id}-{call.get('chunk_index', 0)}",
                "exam_set": set_id,
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
    set_id: str, script_id: str, qno: str, qmeta: QuestionMeta, task_id: str,
    answer: dict | None, prov: dict,
) -> dict:
    base = {
        "task_id": task_id,
        "exam_set": set_id,
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
    qtype = qmeta.question_type

    # quality flags (row-level)
    flags: list[str] = []
    red_ink = qtype in PROSE_TYPES and bool(_isolated_words(raw))
    if conf < CONFIDENCE_MIN:
        flags.append("low_confidence")
    if red_ink:
        flags.append("suspected_red_ink")
    if has_unbalanced_markers(raw):
        flags.append("unbalanced_marker")

    paragraph_breaks = resolved.count("\n\n")
    if qtype == "Paragraph":
        segs = [s for s in resolved.split("\n\n") if s.strip()]
        mean_words = (sum(len(_word_tokens(s)) for s in segs) / len(segs)) if segs else 0
        if paragraph_breaks > 3 or (segs and mean_words < 15):
            flags.append("wrap_as_paragraph")

    # source-dependent metrics: any question whose stem supplies a source passage
    src_wc: object = ""
    length_ratio: object = ""
    exceeds_third: object = ""
    overlap: object = ""
    if qmeta.source_text:
        src_wc = len(_word_tokens(qmeta.source_text))
        length_ratio = round(word_count / src_wc, 4) if src_wc else ""
        overlap = verbatim_overlap_pct(resolved, qmeta.source_text)
        if qtype in SUMMARY_TYPES and src_wc:
            exceeds_third = word_count > (src_wc / 3)

    # letter metrics
    letter_found: object = ""
    letter_missing: object = ""
    if qtype == "Letter_Email":
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, quoting=csv.QUOTE_ALL, lineterminator="\r\n"
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_assertions(
    rows: list[dict], n_scripts: int, stems: dict[str, QuestionMeta],
    targets: list[str],
) -> list[str]:
    """Schema VALIDATION ASSERTIONS. Returns human-readable PASS/WARN/FAIL lines."""
    out: list[str] = []

    def check(cond: bool, msg: str, warn: bool = False) -> None:
        tag = "PASS" if cond else ("WARN" if warn else "FAIL")
        out.append(f"[{tag}] {msg}")

    n_q = len(targets)
    ids = [r["task_id"] for r in rows]
    check(len(ids) == len(set(ids)), "task_id unique")
    check(len(rows) == n_scripts * n_q,
          f"row count == scripts*questions ({len(rows)} == {n_scripts}*{n_q})")

    pairs = [(r["script_id"], r["question_no"]) for r in rows]
    check(len(pairs) == len(set(pairs)), "each (script, question) present exactly once")

    check(all(int(r["max_mark"]) == stems[r["question_no"]].max_mark for r in rows),
          "question_type<->max_mark mapping holds")
    check(all(r["question_type"] == stems[r["question_no"]].question_type
              for r in rows), "question_type matches the task stems")

    errors = sum(1 for r in rows if r["extraction_status"] == "error")
    check(errors == 0, f"extraction_status=='error' count is 0 (got {errors})")

    check(all(r["teacher_mark"] == "" and r["teacher_mark_source"] == "absent"
              for r in rows), "teacher_mark empty & source 'absent' on every row")

    check(len({r["exam_set"] for r in rows}) <= 1,
          "every row belongs to one exam set")

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

    not_attempted = sum(1 for r in rows if r["extraction_status"] == "not_attempted")
    check(not_attempted < len(rows) * 0.5,
          f"not_attempted rows are a minority ({not_attempted}/{len(rows)}) — a high "
          f"count means the transcriber is losing answers, not that students skipped",
          warn=True)
    return out


def build_all(cfg: Config, set_id: str) -> tuple[Path, Path, list[str], int]:
    es = cfg.exam_set(set_id)
    stems = load_task_stems(es.task_stems)
    rows, runs = build_rows(cfg, set_id, es, stems)
    n_scripts = len(_transcripts_for_set(cfg, set_id))

    _write_csv(es.extraction_csv, CSV_COLUMNS, rows)
    _write_csv(es.extraction_runs_csv, RUNS_COLUMNS, runs)

    targets = [normalize_qno(q) for q in es.target_questions]
    assertions = run_assertions(rows, n_scripts, stems, targets)
    return es.extraction_csv, es.extraction_runs_csv, assertions, len(rows)
