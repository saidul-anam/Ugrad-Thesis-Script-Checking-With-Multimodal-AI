#!/usr/bin/env python3
"""
Question Paper Extraction Controller.

Extracts exam question prompts, instructions, and marks from question PDFs/images,
saving structured question artifacts to outputs/questions/<lang>/<question_id>.json.
"""

import os
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.panel import Panel

from src.core.config import load_config
from src.core.schemas import ExtractedQuestion
from src.utils.pdf_processor import is_pdf, extract_images_from_pdf, extract_text_from_pdf
from src.utils.question_utils import extract_question_id, save_extracted_question
from src.engine.engine_factory import create_engine

console = Console()


def parse_marks_from_text(text: str) -> Optional[float]:
    """Extract total marks from question prompt text if present (e.g. '[10 marks]', 'Total: 10')."""
    patterns = [
        r'\[\s*(?:marks?|পূর্ণমান)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:marks?|নম্বর)?\s*\]',
        r'(?:total|full)\s*marks?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)',
        r'পূর্ণমান\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)',
        r'\(([0-9]+(?:\.[0-9]+)?)\s*marks?\)'
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def extract_single_question(
    file_path: str,
    lang: str = "english",
    question_id_override: Optional[str] = None,
    engine=None,
    force_vlm: bool = False,
    output_dir: str = "outputs/questions"
) -> ExtractedQuestion:
    """Extracts a question paper into an ExtractedQuestion artifact."""
    p = Path(file_path)
    q_id = question_id_override or extract_question_id(p.name, lang=lang) or p.stem
    console.print(f"\n[bold cyan]Extracting Question: '{q_id}' from '{p.name}'...[/bold cyan]")

    extracted_text = ""

    # 1. Try digital text extraction if PDF and not forced to VLM
    if is_pdf(file_path) and not force_vlm:
        digital_text = extract_text_from_pdf(file_path)
        if len(digital_text.split()) >= 10:
            console.print(f"  [green]✓ Extracted digital text from PDF ({len(digital_text.split())} words)[/green]")
            extracted_text = digital_text

    # 2. Fall back to Vision / VLM extraction if scanned image or forced
    if not extracted_text:
        console.print("  [yellow]Scanned document / image detected. Running multimodal extraction...[/yellow]")
        if engine is None:
            raise RuntimeError("Vision / LLM engine required for scanned question extraction.")

        if is_pdf(file_path):
            pages = extract_images_from_pdf(file_path)
            images = [img for _, img, _ in pages]
        else:
            from PIL import Image
            images = [Image.open(file_path).convert("RGB")]

        vlm_texts = []
        prompt = (
            "You are an expert exam transcriber. Transcribe the complete question paper or exam question "
            "verbatim. Include the question number, title, stimulus/passage if any, instructions, "
            "and all sub-questions with allocated marks. Output only the transcribed question text."
        )

        for idx, img in enumerate(images, 1):
            t = engine.generate_multimodal(
                prompt=prompt,
                image=img,
                temperature=0.0,
                max_new_tokens=1536
            )
            vlm_texts.append(t.strip())

        extracted_text = "\n\n".join(vlm_texts)
        console.print(f"  [green]✓ VLM transcribed {len(images)} page(s) ({len(extracted_text.split())} words)[/green]")

    total_marks = parse_marks_from_text(extracted_text)

    # Derive title from first line
    lines = [line.strip() for line in extracted_text.split("\n") if line.strip()]
    title = lines[0][:100] if lines else f"Question {q_id}"

    question_obj = ExtractedQuestion(
        question_id=q_id,
        language=lang,
        title=title,
        question_text=extracted_text,
        total_marks=total_marks,
        source_file=str(file_path),
        extracted_at=datetime.now().isoformat()
    )

    saved_json = save_extracted_question(question_obj, output_dir=output_dir)
    console.print(f"  [bold green]✓ Saved Question Artifact:[/bold green] {saved_json}")
    console.print(f"  [dim]Total Marks:[/dim] {total_marks or 'N/A'}")

    return question_obj


def main():
    parser = argparse.ArgumentParser(description="Extract Exam Questions from PDF/Image files")
    parser.add_argument(
        "--lang",
        type=str,
        choices=["bangla", "english"],
        default="english",
        help="Language / Subject of questions: 'english' or 'bangla'"
    )
    parser.add_argument(
        "--pdf",
        "--image",
        "--input-file",
        dest="input_file",
        type=str,
        default=None,
        help="Path to a single question PDF or image file"
    )
    parser.add_argument(
        "--question-id",
        "--name",
        dest="question_id",
        type=str,
        default=None,
        help="Explicit question ID (e.g. 'SE_11_Q1' or 'SB_11_Q1'). Defaults to filename stem / pattern."
    )
    parser.add_argument(
        "--questions-dir",
        type=str,
        default=None,
        help="Directory containing question files (defaults to data/questions/<lang>)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/questions",
        help="Root output directory for saving extracted question JSON (default: outputs/questions)"
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Force using Vision LLM even if digital PDF text is present"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Use local OpenAI-compatible API endpoint (e.g. LM Studio on port 1234)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:1234/v1",
        help="URL of OpenAI-compatible API endpoint"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock engine for zero-VRAM testing"
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization for direct CUDA model"
    )

    args = parser.parse_args()
    questions_dir = args.questions_dir or f"data/questions/{args.lang}"

    # Collect target question files
    target_files = []
    if args.input_file:
        if os.path.exists(args.input_file):
            target_files = [args.input_file]
        else:
            console.print(f"[red]Specified file does not exist: {args.input_file}[/red]")
            return
    else:
        if os.path.exists(questions_dir):
            extensions = ("*.pdf", "*.PDF", "*.png", "*.jpg", "*.jpeg")
            found = []
            for ext in extensions:
                found.extend(Path(questions_dir).glob(ext))
            target_files = sorted([str(p) for p in set(found)])

    if not target_files:
        console.print(f"[yellow]No question files found in '{questions_dir}'.[/yellow]")
        console.print(f"Place question PDFs (e.g. 'SE_11_Q1.pdf', 'SB_11_Q1.pdf') in '{questions_dir}/' or specify --input-file.")
        console.print(f"Or download from Google Drive: python3 scripts/download_questions.py --lang {args.lang}")
        return

    console.print(Panel.fit(
        f"[bold cyan]Gemma 4 Exam Question Extractor[/bold cyan]\n"
        f"[green]Language / Subject:[/green] {args.lang.capitalize()}\n"
        f"[green]Target Question Files:[/green] {len(target_files)}\n"
        f"[green]Output Directory:[/green] {args.output_dir}/{args.lang}",
        title="Question Extractor Initialized"
    ))

    # Initialize engine only if needed
    engine = None

    for fpath in target_files:
        # Check if engine needed
        needs_engine = args.vlm or not (is_pdf(fpath) and len(extract_text_from_pdf(fpath).split()) >= 10)
        if needs_engine and engine is None:
            console.print("[bold]Initializing Engine for question vision extraction...[/bold]")
            cfg = load_config("configs/pipeline_config.yaml")
            engine = create_engine(
                cfg,
                force_mock=args.mock,
                force_api=args.api,
                api_url=args.api_url if args.api else None
            )

        extract_single_question(
            file_path=fpath,
            lang=args.lang,
            question_id_override=args.question_id if len(target_files) == 1 else None,
            engine=engine,
            force_vlm=args.vlm,
            output_dir=args.output_dir
        )

    console.print(f"\n[bold green]✓ Successfully extracted {len(target_files)} question(s)![/bold green]\n")


if __name__ == "__main__":
    main()
