#!/usr/bin/env python3
"""
Segmentation health audit (CPU only, no model calls).

Runs the answer segmenter with engine=None over stored per-page transcripts and reports, per paper,
the symptoms of segmentation failures that can be checked without a hand-made segmentation key:

  - missing questions (no segment, or an empty one)
  - missing questions to which the teacher gave marks   <- almost always a segmentation miss
  - header-only answers (<= 8 words) to which the teacher gave marks
  - answers that contain another question's explicit header on a later line
  - answers with an empty page list, or whose pages jump by 3 or more

It also stores every script's assignment (question -> words, pages, first characters) in a JSON file,
so a later run can be diffed against it with --baseline: every changed assignment is listed.

Sources (both are scanned by default):
  digitized : data_without_pipeline/english/se_*_q1/<script>/extraction_result.json
  ocr       : outputs/extracted/english/se_*_q1/<script>/extraction_result.json

Usage:
  python scripts/audit_segmentation.py --tag before
  python scripts/audit_segmentation.py --tag after --baseline outputs/benchmarks/segmentation_audit_before_<stamp>.json
  python scripts/audit_segmentation.py --paper SE_08_Q1 --show-failures
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.schemas import ExtractionResult
from src.pipeline.answer_segmenter import (
    segment_script_into_questions,
    normalize_header_text,
    _extract_explicit_english_header,
)
from src.utils.ground_truth import load_all_teacher_marks, canonicalize_question_key
from src.utils.question_utils import load_question_for_script

SOURCES = {
    "digitized": "data_without_pipeline/english",
    "ocr": "outputs/extracted/english",
}
HEADER_ONLY_WORDS = 8


def valid_questions(question_obj) -> List[str]:
    out = []
    for sq in (question_obj.sub_questions or []) if question_obj else []:
        q_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
        name = str(sq.get("name") or sq.get("title") or "")
        if str(sq.get("task_type") or "").lower() == "class_test" or re.search(r"class[\s_-]*test\s*$", name, re.I):
            continue
        if q_no:
            out.append(q_no)
    return out


def foreign_headers(q: str, text: str, order: List[str]) -> List[str]:
    hits = []
    for line in text.split("\n")[1:]:
        if not line.strip():
            continue
        cand = _extract_explicit_english_header(normalize_header_text(line), valid_q_order=order)
        if cand and str(cand) != str(q) and re.sub(r"\(.*?\)", "", str(cand)) != re.sub(r"\(.*?\)", "", str(q)):
            hits.append(f"{cand}: {line.strip()[:60]}")
    return hits


def audit_script(path: Path, question_obj, teacher: Dict[str, float]) -> Dict[str, Any]:
    ext = ExtractionResult.model_validate(json.load(open(path, "r", encoding="utf-8")))
    with contextlib.redirect_stdout(io.StringIO()) as log:
        answers = segment_script_into_questions(ext, question_obj, engine=None)
    order = valid_questions(question_obj)
    by_q = {a.q_no: a for a in answers}
    rec: Dict[str, Any] = {"assign": {}, "missing": [], "missing_teacher": [], "header_only_teacher": [],
                           "foreign": [], "empty_pages": [], "page_jump": [], "extra": [],
                           "log": [l for l in log.getvalue().splitlines() if l.strip()]}
    for q in order:
        a = by_q.get(q)
        t = teacher.get(canonicalize_question_key(q))
        words = len(a.answer_text.split()) if a else 0
        if a:
            rec["assign"][q] = {"words": words, "pages": list(a.page_numbers), "head": a.answer_text.strip()[:80]}
        if not a or not a.answer_text.strip():
            rec["missing"].append(q)
            if t and t > 0:
                rec["missing_teacher"].append((q, t))
            continue
        if words <= HEADER_ONLY_WORDS and t and t > 0:
            rec["header_only_teacher"].append((q, t, words))
        fh = foreign_headers(q, a.answer_text, order)
        if fh:
            rec["foreign"].append((q, fh[:3]))
        if not a.page_numbers:
            rec["empty_pages"].append(q)
        else:
            pg = sorted(a.page_numbers)
            if any(b - x >= 3 for x, b in zip(pg, pg[1:])):
                rec["page_jump"].append((q, pg))
    rec["extra"] = [a.q_no for a in answers if a.q_no not in order]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", action="append", help="e.g. SE_08_Q1 (repeatable); default: all found")
    ap.add_argument("--source", choices=list(SOURCES) + ["all"], default="all")
    ap.add_argument("--baseline", default=None, help="JSON written by an earlier run, to diff assignments")
    ap.add_argument("--show-failures", action="store_true", help="print every failing question")
    ap.add_argument("--out-dir", default="outputs/benchmarks")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    teacher_all = {sid: marks for sid, (marks, _src) in load_all_teacher_marks().items()}
    wanted = {p.upper() for p in args.paper} if args.paper else None
    sources = list(SOURCES) if args.source == "all" else [args.source]

    results: Dict[str, Dict[str, Any]] = {}
    q_cache: Dict[str, Any] = {}
    for src in sources:
        root = Path(SOURCES[src])
        for res_path in sorted(root.glob("se_*_q1/*/extraction_result.json")):
            sid = res_path.parent.name.upper()
            paper = re.sub(r"_\d+$", "", sid)
            if wanted and paper not in wanted:
                continue
            if paper not in q_cache:
                q_cache[paper] = load_question_for_script(paper, lang="english")
            teacher = {canonicalize_question_key(k): float(v) for k, v in (teacher_all.get(sid) or {}).items()
                       if str(k).lower() not in ("total_marks", "total", "tot")}
            try:
                rec = audit_script(res_path, q_cache[paper], teacher)
            except Exception as ex:  # keep auditing the rest
                rec = {"error": f"{type(ex).__name__}: {ex}"}
            rec["paper"], rec["source"] = paper, src
            results[f"{src}:{sid}"] = rec

    keys = ("missing", "missing_teacher", "header_only_teacher", "foreign", "empty_pages", "page_jump")
    table = defaultdict(lambda: defaultdict(int))
    for key, rec in results.items():
        grp = f"{rec['paper']} ({rec['source']})"
        table[grp]["scripts"] += 1
        if "error" in rec:
            table[grp]["errors"] += 1
            continue
        for k in keys:
            table[grp][k] += len(rec[k])

    md = [f"# Segmentation audit{(' / ' + args.tag) if args.tag else ''}",
          f"Generated {datetime.now().isoformat(timespec='seconds')} | engine=None (deterministic layers only)", "",
          "| Paper (source) | Scripts | Missing | Missing, teacher > 0 | Header-only, teacher > 0 | Foreign header inside | Empty pages | Page jump ≥3 | Errors |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    totals = defaultdict(int)
    for grp in sorted(table):
        r = table[grp]
        md.append(f"| {grp} | {r['scripts']} | " + " | ".join(str(r[k]) for k in keys) + f" | {r['errors']} |")
        for k in list(keys) + ["scripts", "errors"]:
            totals[k] += r[k]
    md.append("| **All** | " + str(totals["scripts"]) + " | " + " | ".join(str(totals[k]) for k in keys) + f" | {totals['errors']} |")

    fails = []
    for key, rec in sorted(results.items()):
        if "error" in rec:
            fails.append(f"- {key}: ERROR {rec['error']}")
            continue
        for q, t in rec["missing_teacher"]:
            fails.append(f"- {key} Q{q}: missing, teacher {t:g}")
        for q, t, w in rec["header_only_teacher"]:
            fails.append(f"- {key} Q{q}: {w} words, teacher {t:g}")
        if args.show_failures:
            for q, fh in rec["foreign"]:
                fails.append(f"- {key} Q{q}: contains header {fh}")
    if fails:
        md += ["", "## Missing or header-only answers the teacher marked" + (" (+ foreign headers)" if args.show_failures else ""), *fails]

    if args.baseline:
        base = json.load(open(args.baseline, "r", encoding="utf-8"))["results"]
        changed = []
        for key in sorted(set(base) | set(results)):
            a, b = base.get(key, {}).get("assign", {}), results.get(key, {}).get("assign", {})
            for q in sorted(set(a) | set(b), key=lambda x: (len(x), x)):
                x, y = a.get(q), b.get(q)
                if x is None or y is None or abs(x["words"] - y["words"]) > max(3, 0.1 * max(x["words"], y["words"])) or x["pages"] != y["pages"]:
                    fx = f"{x['words']}w p{x['pages']}" if x else "—"
                    fy = f"{y['words']}w p{y['pages']}" if y else "—"
                    changed.append(f"- {key} Q{q}: {fx} -> {fy}")
        md += ["", f"## Changed assignments vs {args.baseline} ({len(changed)})", *changed]

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(args.out_dir) / f"segmentation_audit{('_' + args.tag) if args.tag else ''}_{stamp}"
    with open(str(base_path) + ".json", "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=1, default=str)
    with open(str(base_path) + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWritten: {base_path}.md / .json")


if __name__ == "__main__":
    main()
