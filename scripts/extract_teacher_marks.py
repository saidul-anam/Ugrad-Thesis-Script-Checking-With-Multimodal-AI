#!/usr/bin/env python3
"""
Standalone Teacher Mark Extraction Tool.

Extracts human examiner red marks, question allocations, and margin totals directly
from raw PDFs without running student transcription. Saves results independently to:
  outputs/extracted/<lang>/<script_id>/stage0b_teacher_marks.json
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Early memory config
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image
from pdf2image import convert_from_path

from src.core.config import load_config
from src.engine.engine_factory import create_engine
from src.utils.script_files import list_script_files, all_script_files, describe_paper_folders
from src.pipeline.stage0_preprocessor import detect_red_ink
from src.pipeline.stage0b_teacher_marks import (
    Stage0bTeacherMarkExtractor,
    reconcile_document_teacher_marks,
)
from src.core.schemas import TeacherMarkItem


def load_question_schema(paper: str, questions_dir: str = "data/questions") -> Optional[Dict[str, Any]]:
    """Load matching question schema JSON for syllabus and rubric calibration."""
    p_lower = paper.lower().replace("-", "_")
    candidates = glob.glob(os.path.join(questions_dir, "*.json"))
    for c in candidates:
        stem = Path(c).stem.lower().replace("-", "_")
        if stem in p_lower or p_lower in stem:
            try:
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def extract_teacher_marks_for_script(
    pdf_path: str,
    extractor: Stage0bTeacherMarkExtractor,
    output_dir: str,
    question_obj: Optional[Dict[str, Any]] = None,
    force: bool = False,
    dpi: int = 150,
) -> Dict[str, Any]:
    """Extract and reconcile teacher marks for a single exam script PDF."""
    script_id = Path(pdf_path).stem
    out_file = os.path.join(output_dir, script_id, "stage0b_teacher_marks.json")

    if os.path.exists(out_file) and not force:
        print(f"[{script_id}] stage0b_teacher_marks.json already exists. Use --force to overwrite.")
        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(os.path.join(output_dir, script_id), exist_ok=True)
    print(f"\n=======================================================")
    print(f"[{script_id}] Processing Raw PDF for Teacher Marks: {pdf_path}")
    print(f"=======================================================")

    # Render pages
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as ex:
        print(f"[{script_id}] ERROR rendering PDF: {ex}")
        return {}

    # Extract max marks from question object
    question_max_marks: Dict[str, float] = {}
    valid_paper_questions: List[str] = []
    if question_obj and "questions" in question_obj:
        for q in question_obj["questions"]:
            qno = str(q.get("question_no") or q.get("q_no") or "").strip()
            max_m = float(q.get("max_marks", 0) or q.get("marks", 0) or 0)
            if qno:
                valid_paper_questions.append(qno)
                if max_m > 0:
                    question_max_marks[qno] = max_m

    all_page_marks: List[TeacherMarkItem] = []
    per_page_records: Dict[int, List[Dict[str, Any]]] = {}

    for page_no, page_img in enumerate(pages, 1):
        stage0_res = detect_red_ink(page_img)
        has_red = stage0_res.has_red_ink
        has_margin = getattr(stage0_res, "has_margin_scores", False)
        red_px = getattr(stage0_res, "red_pixel_count", 0)

        print(f"[{script_id}] Page {page_no}/{len(pages)}: red_pixels={red_px}, has_red_ink={has_red}, has_margin_scores={has_margin}")

        if not has_red and not has_margin:
            per_page_records[page_no] = []
            continue

        stage0b_res = extractor.run(
            image=page_img,
            candidate_questions=valid_paper_questions,
            question_max_marks=question_max_marks or None,
            valid_paper_questions=valid_paper_questions or None,
            temperature=0.0,
            top_p=0.1,
            max_new_tokens=1024,
            thinking_mode=True,
        )

        p_marks = stage0b_res.teacher_marks
        for m in p_marks:
            if not getattr(m, "page_no", None):
                m.page_no = page_no
        all_page_marks.extend(p_marks)
        per_page_records[page_no] = [m.model_dump() for m in p_marks]
        print(f"[{script_id}] Page {page_no}: Found {len(p_marks)} mark(s).")

    # Document-level reconciliation
    if all_page_marks:
        reconciled = reconcile_document_teacher_marks(
            all_page_marks,
            question_max_marks=question_max_marks or None,
        )
        print(f"[{script_id}] Reconciled {len(all_page_marks)} raw marks -> {len(reconciled)} final marks.")
    else:
        reconciled = []
        print(f"[{script_id}] No red teacher marks detected across any pages.")

    total_score = sum(m.marks for m in reconciled if m.marks is not None)
    result_data = {
        "script_id": script_id,
        "pdf_path": pdf_path,
        "total_marks": total_score,
        "marks_count": len(reconciled),
        "reconciled_marks": [m.model_dump() for m in reconciled],
        "per_page_raw_marks": per_page_records,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"[{script_id}] Saved standalone teacher marks to {out_file} (Total Score: {total_score})")
    return result_data


def main():
    parser = argparse.ArgumentParser(description="Standalone Teacher Mark Extraction Tool")
    parser.add_argument("--lang", default="english", choices=["english", "bangla"], help="Language/subject")
    parser.add_argument("--pdf-dir", default=None, help="Directory containing raw PDFs")
    parser.add_argument("--out-dir", default=None, help="Output directory for extracted marks")
    parser.add_argument("--script", default=None, help="Filter to single script ID (e.g. SE_11_Q1_0002)")
    parser.add_argument("--top", type=int, default=None, help="Limit to top N scripts")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--dpi", type=int, default=150, help="Rendering DPI for PDF pages")
    args = parser.parse_args()

    pdf_dir = args.pdf_dir or f"data/raw_pdfs/{args.lang}"
    out_dir = args.out_dir or f"outputs/extracted/{args.lang}"

    if not os.path.exists(pdf_dir):
        print(f"Error: PDF directory not found: {pdf_dir}")
        sys.exit(1)

    if args.script:
        # Search pdf_dir and its question-paper sub-folders (e.g. se_11_q1/, se_10_q1/)
        s_clean = Path(args.script).name.replace(".pdf", "").strip()
        pdf_files = [p for p in all_script_files(pdf_dir, (".pdf",)) if s_clean in Path(p).stem]
        if not pdf_files:
            print(f"Error: Script '{args.script}' not found in {pdf_dir}")
            sys.exit(1)
        if len({Path(p).parent for p in pdf_files}) > 1:
            print(f"Error: '{args.script}' matches scripts from several question papers: "
                  f"{', '.join(Path(p).stem for p in pdf_files)}. Use the full script ID.")
            sys.exit(1)
    else:
        pdf_files = list_script_files(pdf_dir, (".pdf",))
        hint = None if pdf_files else describe_paper_folders(pdf_dir, (".pdf",))
        if hint:
            print(f"Error: {hint}")
            sys.exit(1)

    if args.top:
        pdf_files = pdf_files[: args.top]

    print(f"Found {len(pdf_files)} PDF(s) to process in {pdf_dir}.")

    # Load configuration and initialize VLM engine
    config = load_config()
    print("[Engine] Initializing Gemma 4 / VLM Engine...")
    engine = create_engine(config.models, engine_type=config.models.default_engine)
    extractor = Stage0bTeacherMarkExtractor(engine)

    question_schema = load_question_schema(args.lang)

    for idx, pdf_p in enumerate(pdf_files, 1):
        print(f"\n--- [{idx}/{len(pdf_files)}] ---")
        extract_teacher_marks_for_script(
            pdf_path=pdf_p,
            extractor=extractor,
            output_dir=out_dir,
            question_obj=question_schema,
            force=args.force,
            dpi=args.dpi,
        )

    print("\n[Done] All teacher mark extractions complete.")


if __name__ == "__main__":
    main()
