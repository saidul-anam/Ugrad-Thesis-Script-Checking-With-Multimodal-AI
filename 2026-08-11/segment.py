"""Stage 2: split page transcripts into task units by student answer headers.

Usage:
    python segment.py --backend gemini
    python segment.py --backend mistral_ocr

Runs identically on every backend's output — no backend-specific rules.
Headers the matcher cannot find are logged to logs/segmentation_failures.csv,
never patched around.
"""
from __future__ import annotations

import argparse
import re
import sys

from common import (
    LOGS_DIR,
    SEGMENTATION_FAILURES_CSV,
    TASK_MAP,
    TRANSCRIPTS_DIR,
    TRANSCRIPTS_EXCLUDED_DIR,
    TRANSCRIPTS_RAW_DIR,
    append_csv_row,
    script_num,
    utc_now,
)

# Header matching is deliberately loose. Observed variants across students and
# backends include:
#   "Answer of the Question Number - 7"    "Ans of Question Number-11"
#   "Ans: to the Q. No. 3"                 "Ans: do the Q No. 5"
#   "# An: to the Q No. 8"                 (OCR noise in "Ans")
#   "# ~~Ansl~~ Answer of the Question Number-10"  (struck false start)
# A header line must: be short, start with an Ans-ish word once markdown
# decoration is stripped, contain a question-ish or number-ish word, and end
# with a 1-2 digit number. The same rules apply to every backend.
DECORATION_RE = re.compile(r"[#>*_~`]")
ANS_FIRST_WORD_RE = re.compile(r"ans?\w*[:.\-]*", re.IGNORECASE)
QWORD_RE = re.compile(r"\bq\w*\b|\bn[ou]m?\w*\b|\bnumber\b", re.IGNORECASE)
TRAIL_NUM_RE = re.compile(r"\b0?(\d{1,2})\s*[.)\-]?\s*$")
MAX_HEADER_LEN = 70


def match_header(line: str) -> int | None:
    """Return the question number if the line is an answer header, else None."""
    s = DECORATION_RE.sub(" ", line).strip()
    if not s or len(s) > MAX_HEADER_LEN:
        return None
    if not ANS_FIRST_WORD_RE.fullmatch(s.split()[0]):
        return None
    if not QWORD_RE.search(s):
        return None
    m = TRAIL_NUM_RE.search(s)
    return int(m.group(1)) if m else None


def find_headers(lines: list[str]) -> list[tuple[int, int]]:
    """Return [(line_index, question_no)] for every header line."""
    out = []
    for i, line in enumerate(lines):
        q = match_header(line)
        if q is not None:
            out.append((i, q))
    return out


def segment_backend(backend_name: str) -> int:
    raw_dir = TRANSCRIPTS_RAW_DIR / backend_name
    if not raw_dir.is_dir():
        print(f"no raw transcripts for backend {backend_name!r}", file=sys.stderr)
        return 1

    out_dir = TRANSCRIPTS_DIR / backend_name
    excl_dir = TRANSCRIPTS_EXCLUDED_DIR / backend_name
    out_dir.mkdir(parents=True, exist_ok=True)
    excl_dir.mkdir(parents=True, exist_ok=True)

    # Output dirs are wholly derived from transcripts_raw; clear stale files
    # so a re-run never leaves orphans from an earlier matcher version.
    for d in (out_dir, excl_dir):
        for f in d.glob("*.txt"):
            f.unlink()

    # Re-running replaces this backend's failure rows rather than appending
    # duplicates; the CSV always reflects the current segmentation state.
    if SEGMENTATION_FAILURES_CSV.exists():
        rows = SEGMENTATION_FAILURES_CSV.read_text().splitlines()
        kept = [rows[0]] + [r for r in rows[1:] if f",{backend_name}," not in r]
        SEGMENTATION_FAILURES_CSV.write_text("\n".join(kept) + "\n")

    # Group page files by script id: SE_11_Q1_0001_p01.txt
    scripts: dict[str, list] = {}
    for p in sorted(raw_dir.glob("*_p[0-9][0-9].txt")):
        m = re.match(r"(.+)_p(\d{2})$", p.stem)
        scripts.setdefault(m.group(1), []).append((int(m.group(2)), p))

    n_units = 0
    for script_id, pages in sorted(scripts.items()):
        nnn = f"{script_num(script_id):03d}"
        # Concatenate pages, tracking which page each line came from.
        lines: list[str] = []
        line_pages: list[int] = []
        for page_no, path in sorted(pages):
            for ln in path.read_text().splitlines():
                lines.append(ln)
                line_pages.append(page_no)

        headers = find_headers(lines)
        found_qs = [q for _, q in headers]

        # Segment: text between one header and the next belongs to the former.
        segments: dict[int, dict] = {}
        for idx, (line_i, q) in enumerate(headers):
            end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
            body = lines[line_i + 1 : end]  # header line stripped
            seg_pages = line_pages[line_i:end]
            seg = segments.setdefault(q, {"chunks": [], "pages": [], "headers": []})
            seg["chunks"].append("\n".join(body).strip("\n"))
            seg["pages"].extend(seg_pages)
            seg["headers"].append(lines[line_i].strip())

        # Preamble before the first header stays auditable.
        pre_end = headers[0][0] if headers else len(lines)
        preamble = "\n".join(lines[:pre_end]).strip()
        if preamble:
            (excl_dir / f"{script_id}_preamble.txt").write_text(preamble + "\n")

        for q, seg in sorted(segments.items()):
            text = "\n\n".join(c for c in seg["chunks"] if c)
            pages_sorted = sorted(set(seg["pages"]))
            page_range = f"{pages_sorted[0]}-{pages_sorted[-1]}" if pages_sorted else ""
            if q in TASK_MAP:
                prefix, _ = TASK_MAP[q]
                path = out_dir / f"{prefix}_{nnn}.txt"
            else:
                path = excl_dir / f"Q{q}_{nnn}.txt"
            path.write_text(text + "\n")
            meta_dir = LOGS_DIR / "segment_meta" / backend_name
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / f"{path.stem}.meta").write_text(
                f"script_id={script_id}\nquestion_no={q}\npage_range={page_range}\n"
                f"headers_matched={' | '.join(seg['headers'])}\n"
                f"n_header_occurrences={len(seg['headers'])}\n"
            )
            n_units += 1

        for q in sorted(TASK_MAP):
            if q not in found_qs:
                prefix, _ = TASK_MAP[q]
                append_csv_row(
                    SEGMENTATION_FAILURES_CSV,
                    ["timestamp", "script_id", "backend", "missing_task", "question_no"],
                    [utc_now(), script_id, backend_name, prefix, q],
                )
                print(f"  MISSING: {script_id} {prefix} (Q{q}) — logged")

        print(f"{script_id}: headers found for Q{sorted(set(found_qs))}")

    print(f"[{backend_name}] wrote {n_units} task units")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["gemini", "mistral_ocr"])
    args = parser.parse_args()
    return segment_backend(args.backend)


if __name__ == "__main__":
    sys.exit(main())
