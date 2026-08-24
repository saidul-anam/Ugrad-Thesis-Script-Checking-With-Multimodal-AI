#!/usr/bin/env python3
"""
Top-Level Controller Script for Gemma 4 31B IT Multimodal Script Evaluation.

Features:
- Downloads exam script PDFs from Google Drive with smart caching (skips redownloading).
- Controls how many PDFs are processed via `--top N`.
- Feeds PDFs directly into Gemma 4 31B IT multimodal pipeline.
- Automatically saves outputs at EVERY stage into dedicated folders:
    outputs/runs/<script_id>/
      ├── stage1_transcription.json & stage1_raw_transcript.txt
      ├── stage2_verification.json & stage2_verified_transcript.txt
      ├── stage3_errors.json & stage3_errors.csv
      ├── stage4_evaluation.json
      ├── complete_report.json
      └── evaluation_report.md
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.core.config import load_config
from src.engine.engine_factory import create_engine
from src.pipeline.orchestrator import ScriptCheckingPipeline
from scripts.download_drive_pdfs import download_drive_pdfs, DEFAULT_GDRIVE_FOLDER


console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Top Controller: Download & Evaluate Exam Script PDFs with Gemma 4 31B IT"
    )
    # Data source & top limits
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Number of PDF scripts to process (e.g. --top 5). Omit to process all."
    )
    parser.add_argument(
        "--gdrive-url",
        type=str,
        default=DEFAULT_GDRIVE_FOLDER,
        help="Google Drive folder URL containing script PDFs"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="data/raw_pdfs",
        help="Directory to store and read raw PDF scripts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/runs",
        help="Root directory for stage-by-stage evaluation reports"
    )
    parser.add_argument(
        "--rubric",
        type=str,
        default="configs/rubrics/bangla_creative_question.yaml",
        help="Path to rubric YAML"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Pipeline config YAML"
    )

    # Model & Execution controls
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID override (default: google/gemma-4-31b-it)"
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["4bit", "8bit", "none"],
        default=None,
        help="Quantization override (default: 4bit)"
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable reasoning / thinking mode ablation"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in Mock development mode without GPU or model weights"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download PDFs without running evaluation"
    )
    parser.add_argument(
        "--skip-evaluated",
        action="store_true",
        default=True,
        help="Skip evaluating PDFs that already have completed evaluation reports"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download PDFs from Google Drive even if already present locally"
    )

    args = parser.parse_args()

    console.print(Panel.fit(
        f"[bold cyan]Gemma 4 31B IT Multimodal Script Evaluation Controller[/bold cyan]\n"
        f"[green]Google Drive Folder:[/green] {args.gdrive_url}\n"
        f"[green]Target PDF Directory:[/green] {args.pdf_dir}\n"
        f"[green]Top Limit (--top):[/green] {args.top or 'All available PDFs'}\n"
        f"[yellow]Execution Mode:[/yellow] {'Mock (Dev PC)' if args.mock else 'CUDA RTX 5090 (Gemma 4 31B IT)'}\n"
        f"[yellow]Stage-by-Stage Output Root:[/yellow] {args.output_dir}",
        title="Top Controller Initialized"
    ))

    # 1. Download / Discover PDFs from Google Drive
    console.print("\n[bold]Step 1: Checking & Downloading Exam Script PDFs...[/bold]")
    pdf_paths = download_drive_pdfs(
        gdrive_url=args.gdrive_url,
        target_dir=args.pdf_dir,
        top_limit=args.top,
        skip_existing=not args.force_download
    )

    if not pdf_paths:
        console.print(f"[red]No PDF scripts found in '{args.pdf_dir}'.[/red]")
        console.print(f"Place your PDF files into '{args.pdf_dir}/' or check Google Drive connection.")
        return

    console.print(f"[green]Total PDFs ready for processing: {len(pdf_paths)}[/green]")
    for idx, p in enumerate(pdf_paths, 1):
        console.print(f"  [{idx}] {Path(p).name}")

    if args.download_only:
        console.print("\n[green]--download-only flag active. Download complete![/green]")
        return

    # 2. Setup Pipeline Config
    cfg = load_config(args.config)
    if args.model:
        cfg.model.model_id = args.model
    if args.quant:
        cfg.model.quantization = args.quant
    if args.thinking:
        cfg.decoding.thinking_mode = True
    if args.output_dir:
        cfg.pipeline.output_dir = args.output_dir

    # 3. Instantiate Engine & Pipeline
    console.print("\n[bold]Step 2: Initializing Inference Engine & Pipeline...[/bold]")
    engine = create_engine(cfg, force_mock=args.mock)
    pipeline = ScriptCheckingPipeline(
        engine=engine,
        config=cfg,
        rubric_path=args.rubric
    )

    # 4. Process Each PDF Script
    console.print(f"\n[bold]Step 3: Processing {len(pdf_paths)} PDF script(s) with Gemma 4...[/bold]")

    summary_records = []

    for idx, pdf_path in enumerate(pdf_paths, 1):
        script_id = Path(pdf_path).stem
        script_out_dir = os.path.join(cfg.pipeline.output_dir, script_id)
        completed_report_path = os.path.join(script_out_dir, "complete_report.json")

        if args.skip_evaluated and os.path.exists(completed_report_path):
            console.print(f"\n[{idx}/{len(pdf_paths)}] Skipping already evaluated: [cyan]{script_id}[/cyan]")
            continue

        console.print(f"\n{'='*60}")
        console.print(f"[{idx}/{len(pdf_paths)}] Processing PDF: [bold cyan]{Path(pdf_path).name}[/bold cyan]")
        console.print(f"{'='*60}")

        try:
            report = pipeline.evaluate_script(
                input_source=pdf_path,
                script_id=script_id,
                thinking_mode=cfg.decoding.thinking_mode
            )

            summary_records.append({
                "script_id": script_id,
                "words": report.stage1_transcription.word_count,
                "reverted": report.stage2_verification.total_corrections_count,
                "errors": report.stage3_errors.total_error_count,
                "score": f"{report.stage4_evaluation.final_score:.2f} / {report.stage4_evaluation.total_max_marks:.2f}",
                "pct": f"{report.stage4_evaluation.percentage:.1f}%",
                "status": "Success"
            })

        except Exception as e:
            console.print(f"[bold red]Failed evaluating {pdf_path}: {e}[/bold red]")
            summary_records.append({
                "script_id": script_id,
                "words": 0,
                "reverted": 0,
                "errors": 0,
                "score": "N/A",
                "pct": "0%",
                "status": f"Error: {e}"
            })

    # 5. Print Overall Summary Table
    if summary_records:
        table = Table(title="Batch Processing Summary: All Stages Saved")
        table.add_column("Script ID", style="cyan")
        table.add_column("Words", style="white")
        table.add_column("Silent Reverted", style="yellow")
        table.add_column("Errors", style="red")
        table.add_column("Final Score", style="green")
        table.add_column("Percentage", style="bold green")
        table.add_column("Status", style="magenta")

        for r in summary_records:
            table.add_row(
                r["script_id"],
                str(r["words"]),
                str(r["reverted"]),
                str(r["errors"]),
                r["score"],
                r["pct"],
                r["status"]
            )

        console.print("\n")
        console.print(table)
        console.print(f"\n[green]All outputs saved stage-by-stage inside: [bold]{cfg.pipeline.output_dir}/<script_id>/[/bold][/green]\n")


if __name__ == "__main__":
    main()
