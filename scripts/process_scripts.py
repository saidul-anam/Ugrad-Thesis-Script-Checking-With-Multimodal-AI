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

# Ensure repository root is on sys.path when running as a standalone script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from src.core.config import load_config
from src.engine.engine_factory import create_engine
from src.pipeline.orchestrator import ScriptCheckingPipeline
from scripts.download_drive_pdfs import (
    download_drive_pdfs,
    GDRIVE_FOLDERS,
    DEFAULT_GDRIVE_FOLDER_BANGLA,
    DEFAULT_GDRIVE_FOLDER_ENGLISH
)


console = Console()


def interactive_wizard(args):
    """
    Interactive terminal wizard that asks the user to configure all choosable aspects
    before running the batch script processor.
    """
    console.print(Panel.fit(
        "[bold cyan]🎯 Gemma 4 Multimodal Script Evaluation - Interactive Setup[/bold cyan]\n"
        "[dim]Configure options below (press Enter to accept default values in brackets)[/dim]",
        border_style="cyan"
    ))

    # 1. Language / Subject selection (Bangla vs English)
    console.print("\n[bold green]1. Exam Script Language / Subject:[/bold green]")
    console.print("   [[bold cyan]1[/bold cyan]] [bold white]Bangla[/bold white] (Creative Questions / সৃজনশীল) [yellow]-> data/raw_pdfs/bangla[/yellow]")
    console.print("   [[bold cyan]2[/bold cyan]] [bold white]English[/bold white] (Essay / Composition Writing) [yellow]-> data/raw_pdfs/english[/yellow]")
    
    current_lang_idx = "2" if getattr(args, "lang", "bangla") == "english" else "1"
    lang_choice = Prompt.ask(
        "[bold green]   Select Language[/bold green]",
        choices=["1", "2"],
        default=current_lang_idx
    )
    args.lang = "bangla" if lang_choice == "1" else "english"

    # Automatically set separate paths and appropriate default rubric based on language
    if args.pdf_dir in ["data/raw_pdfs", "data/raw_pdfs/bangla", "data/raw_pdfs/english"]:
        args.pdf_dir = f"data/raw_pdfs/{args.lang}"
    if args.output_dir in ["outputs/runs", "outputs/runs/bangla", "outputs/runs/english"]:
        args.output_dir = f"outputs/runs/{args.lang}"
    if args.gdrive_url in [DEFAULT_GDRIVE_FOLDER_BANGLA, DEFAULT_GDRIVE_FOLDER_ENGLISH]:
        args.gdrive_url = GDRIVE_FOLDERS.get(args.lang, DEFAULT_GDRIVE_FOLDER_BANGLA)

    # 2. How many to process (--top)
    default_top_str = str(args.top) if args.top is not None else "all"
    top_input = Prompt.ask(
        f"\n[bold green]2. How many {args.lang.capitalize()} exam scripts to process?[/bold green] (number, or 'all')",
        default=default_top_str
    ).strip().lower()

    if top_input in ["all", "", "none", "0"]:
        args.top = None
    else:
        try:
            args.top = int(top_input)
        except ValueError:
            console.print(f"[yellow]Invalid number '{top_input}', defaulting to All.[/yellow]")
            args.top = None

    # 3. What rubric to use (--rubric)
    rubric_files = sorted(glob.glob("configs/rubrics/*.yaml") + glob.glob("configs/rubrics/*.yml"))
    console.print("\n[bold green]3. Available Rubrics:[/bold green]")
    default_idx = "1"
    
    # Intelligently select default rubric matching selected language
    for idx, rpath in enumerate(rubric_files, 1):
        clean_path = rpath.replace("\\", "/")
        if args.lang == "bangla" and "bangla" in clean_path:
            default_idx = str(idx)
        elif args.lang == "english" and "english" in clean_path:
            default_idx = str(idx)
        console.print(f"   [[bold cyan]{idx}[/bold cyan]] {clean_path}")
    console.print(f"   [[bold cyan]{len(rubric_files) + 1}[/bold cyan]] Custom file path...")

    rubric_choice = Prompt.ask(
        "[bold green]   Select rubric option or enter path[/bold green]",
        default=default_idx
    ).strip()

    if rubric_choice.isdigit():
        choice_num = int(rubric_choice)
        if 1 <= choice_num <= len(rubric_files):
            args.rubric = rubric_files[choice_num - 1].replace("\\", "/")
        elif choice_num == len(rubric_files) + 1:
            args.rubric = Prompt.ask("   [bold]Enter custom rubric YAML path[/bold]", default=args.rubric)
    elif rubric_choice:
        args.rubric = rubric_choice

    # 4. Model Quantization (--quant)
    default_quant = args.quant or "4bit"
    console.print("\n[bold green]4. Model Quantization:[/bold green]")
    console.print("   • [cyan]4bit[/cyan]: Recommended for RTX 5090 32GB VRAM (~18GB footprint)")
    console.print("   • [cyan]8bit[/cyan]: Higher precision (~32GB footprint)")
    console.print("   • [cyan]none[/cyan]: Full bfloat16 (multi-GPU / high VRAM)")
    args.quant = Prompt.ask(
        "[bold green]   Select Quantization[/bold green]",
        choices=["4bit", "8bit", "none"],
        default=default_quant
    )

    # 5. Execution Mode (--mock vs Real GPU)
    console.print("\n[bold green]5. Execution Engine:[/bold green]")
    console.print("   [1] Real GPU (CUDA Gemma 4 31B IT)")
    console.print("   [2] Mock Dev Mode (Fast CPU simulation without GPU/weights)")
    mode_default = "2" if args.mock else "1"
    mode_choice = Prompt.ask(
        "[bold green]   Select Execution Mode[/bold green]",
        choices=["1", "2"],
        default=mode_default
    )
    args.mock = (mode_choice == "2")

    # 6. Reasoning / Thinking Mode (--thinking)
    args.thinking = Confirm.ask(
        "\n[bold green]6. Enable Reasoning / Thinking Mode ablation?[/bold green]",
        default=args.thinking
    )

    # 7. Skip already evaluated scripts (--skip-evaluated)
    args.skip_evaluated = Confirm.ask(
        "\n[bold green]7. Skip scripts that already have completed evaluation reports?[/bold green]",
        default=args.skip_evaluated
    )

    # Review & Confirm
    console.print("\n" + "="*50)
    engine_label = "[bold yellow]Mock Dev Mode (CPU simulation)[/bold yellow]" if args.mock else "[bold green]CUDA RTX 5090 (Gemma 4 31B IT)[/bold green]"
    top_label = str(args.top) if args.top else "All available"
    thinking_label = "[green]Enabled[/green]" if args.thinking else "[dim]Disabled[/dim]"
    skip_label = "[green]Yes[/green]" if args.skip_evaluated else "[yellow]No[/yellow]"

    console.print(Panel(
        f"• [cyan]Language / Subject (--lang):[/cyan] [bold yellow]{args.lang.capitalize()}[/bold yellow]\n"
        f"• [cyan]Scripts to Process (--top):[/cyan] [bold white]{top_label}[/bold white]\n"
        f"• [cyan]Rubric (--rubric):[/cyan] [bold white]{args.rubric}[/bold white]\n"
        f"• [cyan]Quantization (--quant):[/cyan] [bold white]{args.quant}[/bold white]\n"
        f"• [cyan]Engine Mode (--mock):[/cyan] {engine_label}\n"
        f"• [cyan]Thinking Mode (--thinking):[/cyan] {thinking_label}\n"
        f"• [cyan]Skip Evaluated (--skip-evaluated):[/cyan] {skip_label}\n"
        f"• [cyan]Raw PDF Directory:[/cyan] [bold white]{args.pdf_dir}[/bold white]\n"
        f"• [cyan]Output Directory:[/cyan] [bold white]{args.output_dir}[/bold white]\n"
        f"• [cyan]GDrive Source:[/cyan] [dim]{args.gdrive_url}[/dim]",
        title="[bold green]Configuration Summary[/bold green]",
        border_style="green"
    ))

    if not Confirm.ask("[bold yellow]Proceed with processing with these settings?[/bold yellow]", default=True):
        console.print("[red]Cancelled by user.[/red]")
        sys.exit(0)

    console.print("\n[bold green]Starting evaluation pipeline...[/bold green]\n")
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Top Controller: Download & Evaluate Exam Script PDFs with Gemma 4 31B IT"
    )
    # Language selection & data source
    parser.add_argument(
        "--lang",
        "--language",
        type=str,
        choices=["bangla", "english"],
        default="bangla",
        help="Language / Subject of exam scripts: 'bangla' or 'english' (default: bangla)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Number of PDF scripts to process (e.g. --top 5). Omit to process all."
    )
    parser.add_argument(
        "--gdrive-url",
        type=str,
        default=None,
        help="Google Drive folder URL containing script PDFs (defaults to respective language folder)"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Directory to store and read raw PDF scripts (defaults to data/raw_pdfs/<lang>)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root directory for stage-by-stage evaluation reports (defaults to outputs/runs/<lang>)"
    )
    parser.add_argument(
        "--rubric",
        type=str,
        default=None,
        help="Path to rubric YAML (defaults to respective language rubric)"
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
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        default=True,
        help="Run interactive configuration wizard (default: True)"
    )
    parser.add_argument(
        "--non-interactive",
        "--batch",
        "-y",
        action="store_true",
        help="Disable interactive terminal prompts and use CLI/config values directly"
    )

    args = parser.parse_args()

    # Set language-specific defaults if not explicitly given
    if not args.pdf_dir:
        args.pdf_dir = f"data/raw_pdfs/{args.lang}"
    if not args.output_dir:
        args.output_dir = f"outputs/runs/{args.lang}"
    if not args.gdrive_url:
        args.gdrive_url = GDRIVE_FOLDERS.get(args.lang, DEFAULT_GDRIVE_FOLDER_BANGLA)
    if not args.rubric:
        args.rubric = (
            "configs/rubrics/bangla_creative_question.yaml"
            if args.lang == "bangla"
            else "configs/rubrics/english_writing.yaml"
        )

    # If running in interactive terminal and non-interactive is not requested, launch wizard
    if not args.non_interactive and sys.stdin.isatty():
        args = interactive_wizard(args)

    if not args.quant:
        args.quant = "4bit"

    console.print(Panel.fit(
        f"[bold cyan]Gemma 4 31B IT Multimodal Script Evaluation Controller[/bold cyan]\n"
        f"[green]Language / Subject:[/green] {args.lang.capitalize()}\n"
        f"[green]Google Drive Folder:[/green] {args.gdrive_url}\n"
        f"[green]Target PDF Directory:[/green] {args.pdf_dir}\n"
        f"[green]Top Limit (--top):[/green] {args.top or 'All available PDFs'}\n"
        f"[green]Rubric (--rubric):[/green] {args.rubric}\n"
        f"[green]Quantization (--quant):[/green] {args.quant}\n"
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
