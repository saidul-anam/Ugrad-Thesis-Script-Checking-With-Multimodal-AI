#!/usr/bin/env python3
"""
Setup Data Without Pipeline Controller.

Converts digitized/transcribed student script text files (e.g. from SE_08_Q1_extraction/SE_08_Q1_*.txt)
into valid extraction artifacts inside data_without_pipeline/ so they can be evaluated
immediately by Stage 4 without running the vision/OCR pipeline.
"""

import os
import sys
import re
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.core.schemas import (
    ExtractionResult,
    PageExtractionResult,
    Stage1TranscriptionResult,
    Stage2VerificationResult,
    Stage3ErrorResult,
    TeacherMarkItem,
)
from src.utils.question_utils import extract_question_id
from src.utils.export_utils import export_extraction_artifacts
from src.utils.ground_truth import (
    get_ground_truth_info_for_script,
    ground_truth_to_teacher_marks,
)

console = Console()


def parse_script_pages(raw_text: str) -> List[Tuple[int, str]]:
    """
    Parses a raw script text file into a list of (page_no, page_text) tuples.
    Detects page markers:
      - '--- PAGE 1 ---'
      - '--- Page 1 ---'
      - '--- Page Break ---'
    """
    cleaned = raw_text.strip()
    if not cleaned:
        return []

    # 1. Match explicit numbered page headers like '--- PAGE 1 ---'
    pattern = re.compile(r'(?i)(?:^|\n)\s*---\s*PAGE\s*(\d+)\s*---\s*\n?')
    matches = list(pattern.finditer(cleaned))

    if matches:
        pages: List[Tuple[int, str]] = []
        for i, m in enumerate(matches):
            p_no = int(m.group(1))
            start_idx = m.end()
            end_idx = matches[i + 1].start() if (i + 1 < len(matches)) else len(cleaned)
            content = cleaned[start_idx:end_idx].strip()
            pages.append((p_no, content))
        return pages

    # 2. Match unnumbered page breaks '--- Page Break ---'
    pb_splits = re.split(r'(?i)\n*---\s*Page Break\s*---\n*', cleaned)
    if len(pb_splits) > 1:
        return [(i + 1, s.strip()) for i, s in enumerate(pb_splits) if s.strip()]

    # 3. Default to single page
    return [(1, cleaned)]


def create_extraction_result_from_txt(
    script_id: str,
    raw_text: str,
    paper_id: str,
    lang: str = "english",
    source_file: Optional[str] = None,
    sync_gt: bool = True
) -> Tuple[ExtractionResult, Optional[Dict[str, float]], Optional[str]]:
    """
    Builds an ExtractionResult dataclass from raw transcribed script text.
    """
    pages_data = parse_script_pages(raw_text)
    if not pages_data:
        pages_data = [(1, raw_text.strip())]

    page_results: List[PageExtractionResult] = []
    page_texts: List[str] = []

    for p_no, p_txt in pages_data:
        w_count = len(p_txt.split())
        c_count = len(p_txt)
        struck_count = len(re.findall(r'\[struck:[^\]]*\]', p_txt))
        illegible_count = len(re.findall(r'\[illegible\]', p_txt, re.I))
        unclear_count = len(re.findall(r'\[unclear:[^\]]*\]', p_txt, re.I))

        s1 = Stage1TranscriptionResult(
            raw_transcript=p_txt,
            illegible_count=illegible_count,
            unclear_count=unclear_count,
            struck_count=struck_count,
            character_count=c_count,
            word_count=w_count,
            detected_script=lang.capitalize()
        )
        s2 = Stage2VerificationResult(
            verified_transcript=p_txt,
            silent_corrections_fixed=[],
            proposed_patches=[],
            total_corrections_count=0,
            verification_notes="Imported directly from verified digitized script text."
        )
        s3 = Stage3ErrorResult(
            errors=[],
            spelling_error_count=0,
            grammar_error_count=0,
            syntax_error_count=0,
            total_error_count=0,
            linguistic_summary="Direct text evaluation mode (Stage 4 parses content directly)."
        )

        page_results.append(PageExtractionResult(
            page_no=p_no,
            image_path="",
            has_red_ink=False,
            red_pixel_count=0,
            stage1_transcription=s1,
            stage2_verification=s2,
            stage3_errors=s3,
            teacher_marks=[]
        ))
        page_texts.append(p_txt)

    # Aggregated full script transcript
    full_transcript = "\n\n--- Page Break ---\n\n".join(page_texts)
    total_words = sum(p.stage1_transcription.word_count for p in page_results)
    total_chars = sum(p.stage1_transcription.character_count for p in page_results)
    total_struck = sum(p.stage1_transcription.struck_count for p in page_results)

    script_s1 = Stage1TranscriptionResult(
        raw_transcript=full_transcript,
        illegible_count=sum(p.stage1_transcription.illegible_count for p in page_results),
        unclear_count=sum(p.stage1_transcription.unclear_count for p in page_results),
        struck_count=total_struck,
        character_count=total_chars,
        word_count=total_words,
        detected_script=lang.capitalize()
    )
    script_s2 = Stage2VerificationResult(
        verified_transcript=full_transcript,
        silent_corrections_fixed=[],
        proposed_patches=[],
        total_corrections_count=0,
        verification_notes="Digitized exam transcript without OCR artifacts."
    )
    script_s3 = Stage3ErrorResult(
        errors=[],
        spelling_error_count=0,
        grammar_error_count=0,
        syntax_error_count=0,
        total_error_count=0,
        linguistic_summary="Direct transcript setup for Stage 4 modular evaluation."
    )

    metadata: Dict[str, Any] = {
        "source": "data_without_pipeline",
        "paper": paper_id,
        "language": lang,
        "source_file": source_file or "",
        "imported_at": datetime.now().isoformat()
    }

    gt_marks: Optional[Dict[str, float]] = None
    gt_source: Optional[str] = None
    if sync_gt:
        gt_marks, gt_source = get_ground_truth_info_for_script(script_id, lang=lang)
        if gt_marks:
            metadata["ground_truth_marks"] = gt_marks
            metadata["ground_truth_source"] = gt_source
            metadata["verified_by_human"] = True

    ext_result = ExtractionResult(
        script_id=script_id,
        image_path="",
        model_id="digitized_script_import",
        timestamp=datetime.now().isoformat(),
        has_red_ink=False,
        stage1_transcription=script_s1,
        stage2_verification=script_s2,
        stage3_errors=script_s3,
        teacher_marks=[],
        pages=page_results,
        metadata=metadata
    )

    return ext_result, gt_marks, gt_source


def is_student_script_file(file_path: Path, paper_id: Optional[str] = None) -> bool:
    """
    Distinguishes student script text files from question papers or CSVs.
    e.g. 'SE_08_Q1_0001.txt' -> True, 'SE_08_Q1.txt' -> False, 'SE_08_Q1.csv' -> False.
    """
    if file_path.suffix.lower() != ".txt":
        return False
    stem = file_path.stem
    if paper_id and stem.upper() == paper_id.upper():
        return False
    # Check for question-only patterns like SE_08_Q1 or SB_11_Q1
    if re.fullmatch(r'(?:SE|SB)_[0-9A-Za-z]+_Q[0-9A-Za-z]+', stem, re.IGNORECASE):
        return False
    # Student scripts typically end with _<number/code>
    if re.search(r'_[0-9]{3,}', stem):
        return True
    return True


def discover_input_script_files(
    input_dir: Path,
    requested_scripts: Optional[List[str]] = None,
    paper_id: Optional[str] = None
) -> List[Path]:
    """Finds matching .txt script files based on CLI inputs or directory scanning."""
    all_txt = [p for p in input_dir.glob("*.txt") if p.is_file()]

    if not requested_scripts or requested_scripts == ["all"]:
        # Filter only student script files
        filtered = [p for p in all_txt if is_student_script_file(p, paper_id=paper_id)]
        return sorted(filtered, key=lambda p: p.name)

    # User provided specific script arguments
    resolved: List[Path] = []
    seen = set()

    for item in requested_scripts:
        p_item = Path(item)
        if p_item.is_file():
            if p_item.resolve() not in seen:
                resolved.append(p_item)
                seen.add(p_item.resolve())
            continue

        # Try relative to input_dir
        cand1 = input_dir / item
        if cand1.is_file():
            if cand1.resolve() not in seen:
                resolved.append(cand1)
                seen.add(cand1.resolve())
            continue

        # Try appending .txt
        cand2 = input_dir / f"{item}.txt"
        if cand2.is_file():
            if cand2.resolve() not in seen:
                resolved.append(cand2)
                seen.add(cand2.resolve())
            continue

        # Glob match
        matched_globs = list(input_dir.glob(f"*{item}*.txt"))
        for mg in sorted(matched_globs):
            if mg.resolve() not in seen and is_student_script_file(mg, paper_id=paper_id):
                resolved.append(mg)
                seen.add(mg.resolve())

    return resolved


def setup_single_script(
    file_path: Path,
    output_root: Path,
    paper_id: str,
    lang: str = "english",
    sync_gt: bool = True
) -> Dict[str, Any]:
    """Converts one student script .txt file and exports its artifacts."""
    script_id = file_path.stem
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    ext_result, gt_marks, gt_source = create_extraction_result_from_txt(
        script_id=script_id,
        raw_text=raw_text,
        paper_id=paper_id,
        lang=lang,
        source_file=str(file_path),
        sync_gt=sync_gt
    )

    # Destination directory: data_without_pipeline/<lang>/<paper_id_lower>/<script_id>
    target_dir = output_root / lang / paper_id.lower() / script_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # Export canonical extraction artifacts
    export_extraction_artifacts(ext_result, str(target_dir))

    # Also save original script copy
    shutil.copy2(file_path, target_dir / "source_transcript.txt")

    # If ground truth was found, write ground_truth_marks.json
    if gt_marks:
        tm_items = ground_truth_to_teacher_marks(
            gt_marks,
            source_desc=f"Verified Examiner Ground Truth ({Path(gt_source or '').name}) for {script_id}"
        )
        gt_json_path = target_dir / "ground_truth_marks.json"
        with open(gt_json_path, "w", encoding="utf-8") as gf:
            json.dump([m.model_dump() for m in tm_items], gf, indent=2, ensure_ascii=False)


    return {
        "script_id": script_id,
        "pages": len(ext_result.pages),
        "words": ext_result.stage1_transcription.word_count,
        "target_dir": str(target_dir),
        "gt_count": len(gt_marks) if gt_marks else 0,
        "gt_source": Path(gt_source).name if gt_source else "None"
    }


def main():
    parser = argparse.ArgumentParser(
        description="Setup raw transcribed script text files into data_without_pipeline/ for Stage 4 evaluation."
    )
    parser.add_argument(
        "scripts",
        nargs="*",
        default=[],
        help="Script IDs, filenames, or paths to process (e.g. SE_08_Q1_0001, SE_08_Q1_0001.txt). Defaults to all in input-dir."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default="SE_08_Q1_extraction",
        help="Input directory containing transcribed script .txt files (default: SE_08_Q1_extraction)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data_without_pipeline",
        help="Output root directory for extraction artifacts (default: data_without_pipeline)"
    )
    parser.add_argument(
        "--paper",
        "-p",
        type=str,
        default=None,
        help="Target exam question paper ID (e.g. SE_08_Q1). Inferred automatically if omitted."
    )
    parser.add_argument(
        "--lang",
        "-l",
        type=str,
        choices=["english", "bangla"],
        default="english",
        help="Language / Subject of the scripts (default: english)"
    )
    parser.add_argument(
        "--sync-gt",
        dest="sync_gt",
        action="store_true",
        default=True,
        help="Synchronize official examiner marks from data/ground_truth/transcripts/teacher_marks/ (default: True)"
    )
    parser.add_argument(
        "--no-gt",
        dest="sync_gt",
        action="store_false",
        help="Do not attach ground truth teacher marks"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Maximum number of scripts to set up"
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        console.print(f"[bold red]Error: Input directory '{input_path}' does not exist.[/bold red]")
        sys.exit(1)

    # Infer paper ID if not explicitly specified
    paper_id = args.paper
    if not paper_id:
        if args.scripts and args.scripts != ["all"]:
            paper_id = extract_question_id(args.scripts[0], lang=args.lang)
        if not paper_id:
            paper_id = extract_question_id(input_path.name, lang=args.lang)
        if not paper_id:
            paper_id = "SE_08_Q1"

    console.print(Panel.fit(
        f"[bold cyan]Setup Scripts Without Pipeline Controller[/bold cyan]\n"
        f"[green]Exam Paper:[/green] {paper_id} ({args.lang.capitalize()})\n"
        f"[green]Input Directory:[/green] {input_path}\n"
        f"[green]Output Directory:[/green] {output_path}\n"
        f"[green]Sync Teacher Marks (GT):[/green] {'Yes' if args.sync_gt else 'No'}",
        border_style="cyan"
    ))

    # Discover candidate files
    script_files = discover_input_script_files(
        input_dir=input_path,
        requested_scripts=args.scripts,
        paper_id=paper_id
    )

    if args.top and len(script_files) > args.top:
        script_files = script_files[:args.top]

    if not script_files:
        console.print(f"[bold yellow]No student script .txt files found in '{input_path}'.[/bold yellow]")
        sys.exit(0)

    console.print(f"\n[bold green]Found {len(script_files)} student script(s) to set up:[/bold green]")

    records = []
    for idx, f in enumerate(script_files, 1):
        rec = setup_single_script(
            file_path=f,
            output_root=output_path,
            paper_id=paper_id,
            lang=args.lang,
            sync_gt=args.sync_gt
        )
        records.append(rec)
        gt_info = f"[cyan]{rec['gt_count']} marks ({rec['gt_source']})[/cyan]" if rec['gt_count'] > 0 else "[dim]None[/dim]"
        console.print(f"  [[bold cyan]{idx:02d}/{len(script_files):02d}[/bold cyan]] [bold white]{rec['script_id']}[/bold white] - {rec['pages']} page(s), {rec['words']} words, GT: {gt_info}")

    # Summary table
    table = Table(title=f"Setup Completed ({len(records)} Scripts Ready in '{output_path}')", border_style="cyan")
    table.add_column("No", justify="right", style="cyan")
    table.add_column("Script ID", style="bold white")
    table.add_column("Pages", justify="right", style="green")
    table.add_column("Words", justify="right", style="yellow")
    table.add_column("Teacher Marks (GT)", style="magenta")
    table.add_column("Artifact Directory", style="dim")

    for i, r in enumerate(records, 1):
        gt_disp = f"✓ {r['gt_count']} marks ({r['gt_source']})" if r['gt_count'] > 0 else "None"
        table.add_row(
            str(i),
            r["script_id"],
            str(r["pages"]),
            str(r["words"]),
            gt_disp,
            r["target_dir"]
        )

    console.print("\n", table)

    # Next steps instructions
    paper_dir_arg = f"{output_path}/{args.lang}/{paper_id.lower()}"
    console.print(Panel.fit(
        f"[bold green]✓ All {len(records)} script(s) successfully initialized in '{output_path}'![/bold green]\n\n"
        f"[bold white]To evaluate all scripts using Stage 4 Modular Evaluator, run:[/bold white]\n"
        f"  [cyan]python scripts/evaluate_scripts.py --extraction-dir {paper_dir_arg} --paper {paper_id}[/cyan]\n\n"
        f"[bold white]To evaluate a single script (e.g. {records[0]['script_id']}):[/bold white]\n"
        f"  [cyan]python scripts/evaluate_scripts.py --extraction-dir {output_path} --script-name {records[0]['script_id']}[/cyan]",
        title="Ready for Evaluation"
    ))


if __name__ == "__main__":
    main()
