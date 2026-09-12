#!/usr/bin/env python3
"""
Re-run the answer segmenter on existing extraction artifacts WITHOUT re-running the VLM stages.

Use after a segmenter fix: it reloads outputs/extracted/<lang>/<script>/extraction_result.json,
re-segments the (already normalized) per-page transcripts into question answers, refreshes
metadata.aligned_answers and rewrites extraction_result.json + extraction_summary.md.
Stage 4 then picks up the new segmentation.

Usage:
  python scripts/resegment_extraction.py --lang english                 # all scripts
  python scripts/resegment_extraction.py --lang english --script SE_11_Q1_0010
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.schemas import ExtractionResult
from src.pipeline.answer_segmenter import segment_script_into_questions
from src.utils.question_utils import load_question_for_script
from src.utils.export_utils import export_extraction_artifacts


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-segment extracted scripts into question answers")
    ap.add_argument("--lang", default="english")
    ap.add_argument("--extracted-dir", default=None)
    ap.add_argument("--script", default=None)
    args = ap.parse_args()
    root = Path(args.extracted_dir or f"outputs/extracted/{args.lang}")
    targets = [root / args.script] if args.script else sorted(p for p in root.iterdir() if p.is_dir())
    for d in targets:
        f = d / "extraction_result.json"
        if not f.exists():
            continue
        ext = ExtractionResult.model_validate(json.load(open(f, "r", encoding="utf-8")))
        q = load_question_for_script(ext.script_id, lang=args.lang)
        before = [(a.get("q_no"), a.get("word_count")) for a in ext.metadata.get("aligned_answers", [])]
        answers = segment_script_into_questions(ext, q)
        ext.metadata["aligned_answers"] = [a.model_dump() for a in answers]
        ext.metadata["resegmented"] = True
        export_extraction_artifacts(ext, str(d))
        after = [(a.q_no, a.word_count) for a in answers]
        print(f"{ext.script_id}: {before}\n  -> {after}")


if __name__ == "__main__":
    main()
