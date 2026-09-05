#!/usr/bin/env python3
"""
Dedicated Extraction Runner for Multimodal Script Checking Pipeline.

Executes Stages 1 to 3:
- Stage 1: Verbatim Transcription from PDF/Image
- Stage 2: Autocorrection Verification (Audits image vs transcript & reverts silent fixes)
- Stage 3: Linguistic Error Extraction (Spelling, Grammar, Syntax, Punctuation)

Saves all artifacts per script into:
  outputs/extracted/<lang>/<script_id>/
    ├── stage1_transcription.json & stage1_raw_transcript.txt
    ├── stage2_verification.json & stage2_verified_transcript.txt
    ├── stage3_errors.json & stage3_errors.csv
    ├── extraction_result.json
    └── extraction_summary.md
"""

import os
import sys

# Configure PyTorch CUDA Allocator early to prevent memory fragmentation
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import argparse
from pathlib import Path
from typing import List


# Ensure repository root is on sys.path
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
    """Interactive terminal wizard for configuring extraction."""
    console.print(Panel.fit(
        "[bold cyan]🔍 Gemma 4 Multimodal Script Extraction - Interactive Setup[/bold cyan]\n"
        "[dim]Configuring Stages 1–3 (Verbatim Transcribe + Silent Audit + Error Cataloging)[/dim]",
        border_style="cyan"
    ))

    # 1. Language
    console.print("\n[bold green]1. Exam Script Language / Subject:[/bold green]")
    console.print("   [[bold cyan]1[/bold cyan]] [bold white]Bangla[/bold white] (Creative Questions / সৃজনশীল) [yellow]-> data/raw_pdfs/bangla[/yellow]")
    console.print("   [[bold cyan]2[/bold cyan]] [bold white]English[/bold white] (Essay / Composition Writing) [yellow]-> data/raw_pdfs/english[/yellow]")

    current_lang_idx = "2" if getattr(args, "lang", "bangla") == "english" else "1"
    lang_choice = Prompt.ask("[bold green]   Select Language[/bold green]", choices=["1", "2"], default=current_lang_idx)
    args.lang = "bangla" if lang_choice == "1" else "english"

    if args.pdf_dir in [None, "data/raw_pdfs", "data/raw_pdfs/bangla", "data/raw_pdfs/english"]:
        args.pdf_dir = f"data/raw_pdfs/{args.lang}"
    if args.output_dir in [None, "outputs/extracted", "outputs/extracted/bangla", "outputs/extracted/english"]:
        args.output_dir = f"outputs/extracted/{args.lang}"
    if args.gdrive_url in [None, DEFAULT_GDRIVE_FOLDER_BANGLA, DEFAULT_GDRIVE_FOLDER_ENGLISH]:
        args.gdrive_url = GDRIVE_FOLDERS.get(args.lang, DEFAULT_GDRIVE_FOLDER_BANGLA)

    # 2. Top limit
    default_top_str = str(args.top) if args.top is not None else "all"
    top_input = Prompt.ask(
        f"\n[bold green]2. How many {args.lang.capitalize()} scripts to extract?[/bold green] (number, or 'all')",
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

    # 3. Quantization
    default_quant = args.quant or "4bit"
    console.print("\n[bold green]3. Model Quantization:[/bold green]")
    console.print("   • [cyan]4bit[/cyan]: Recommended for RTX 5090 32GB VRAM (~18GB footprint)")
    console.print("   • [cyan]8bit[/cyan]: Higher precision (~32GB footprint)")
    console.print("   • [cyan]none[/cyan]: Full precision bfloat16")
    args.quant = Prompt.ask("[bold green]   Select Quantization[/bold green]", choices=["4bit", "8bit", "none"], default=default_quant)

    # 4. Engine Mode
    console.print("\n[bold green]4. Execution Engine:[/bold green]")
    console.print("   [1] Real GPU (CUDA Gemma 4 31B IT)")
    console.print("   [2] Mock Dev Mode (Fast CPU simulation without GPU/weights)")
    console.print("   [3] Local API Engine (LM Studio / vLLM on port 1234 - Zero Extra VRAM)")
    mode_default = "3" if getattr(args, "api", False) else ("2" if args.mock else "1")
    mode_choice = Prompt.ask("[bold green]   Select Execution Mode[/bold green]", choices=["1", "2", "3"], default=mode_default)
    args.mock = (mode_choice == "2")
    args.api = (mode_choice == "3")

    # 5. Thinking Mode
    args.thinking = Confirm.ask("\n[bold green]5. Enable Reasoning / Thinking Mode ablation?[/bold green]", default=args.thinking)

    # 6. GDrive Download action (Optional)
    console.print("\n[bold green]6. Input Script Source:[/bold green]")
    console.print("   [1] Local Scripts Only (Scan data/raw_pdfs without checking GDrive) [yellow](default)[/yellow]")
    console.print("   [2] Sync from Google Drive (Download missing scripts)")
    console.print("   [3] Force Re-download from Google Drive")

    dl_default = "3" if getattr(args, "force_download", False) else ("2" if getattr(args, "sync_drive", False) else "1")
    dl_choice = Prompt.ask("[bold green]   Select Input Source[/bold green]", choices=["1", "2", "3"], default=dl_default)
    if dl_choice == "1":
        args.sync_drive = False
        args.force_download = False
    elif dl_choice == "2":
        args.sync_drive = True
        args.force_download = False
    elif dl_choice == "3":
        args.sync_drive = True
        args.force_download = True

    # 7. Skip already extracted
    if not args.download_only:
        args.skip_extracted = Confirm.ask(
            "\n[bold green]7. Skip scripts that already have extracted artifacts in output directory?[/bold green]",
            default=args.skip_extracted
        )

    # Summary
    console.print("\n" + "="*50)
    if args.mock:
        engine_label = "[bold yellow]Mock Dev Mode (CPU simulation)[/bold yellow]"
    elif getattr(args, "api", False):
        engine_label = f"[bold green]Local API Engine ({getattr(args, 'api_url', 'http://localhost:1234/v1')})[/bold green]"
    else:
        engine_label = "[bold green]CUDA RTX 5090 (Gemma 4 31B IT)[/bold green]"
    top_label = str(args.top) if args.top else "All available"
    thinking_label = "[green]Enabled[/green]" if args.thinking else "[dim]Disabled[/dim]"
    skip_label = "[green]Yes[/green]" if args.skip_extracted else "[yellow]No[/yellow]"

    console.print(Panel(
        f"• [cyan]Language / Subject (--lang):[/cyan] [bold yellow]{args.lang.capitalize()}[/bold yellow]\n"
        f"• [cyan]Scripts to Extract (--top):[/cyan] [bold white]{top_label}[/bold white]\n"
        f"• [cyan]Quantization (--quant):[/cyan] [bold white]{args.quant}[/bold white]\n"
        f"• [cyan]Engine Mode:[/cyan] {engine_label}\n"
        f"• [cyan]Thinking Mode (--thinking):[/cyan] {thinking_label}\n"
        f"• [cyan]Skip Already Extracted:[/cyan] {skip_label}\n"
        f"• [cyan]Raw Scripts Directory:[/cyan] [bold white]{args.pdf_dir}[/bold white]\n"
        f"• [cyan]Extraction Output Directory:[/cyan] [bold white]{args.output_dir}[/bold white]",
        title="[bold green]Extraction Configuration Summary[/bold green]",
        border_style="green"
    ))

    if not Confirm.ask("[bold yellow]Proceed with extraction?[/bold yellow]", default=True):
        console.print("[red]Cancelled by user.[/red]")
        sys.exit(0)

    console.print("\n[bold green]Starting extraction pipeline...[/bold green]\n")
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Extract handwriting text and linguistic error catalogs from exam scripts using Gemma 4 31B IT"
    )
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
        help="Number of scripts to process (e.g. --top 5). Omit to extract all."
    )
    parser.add_argument(
        "--pdf-dir",
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing PDF or image scripts (defaults to data/raw_pdfs/<lang>)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single PDF or image file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root directory for saving extraction artifacts (defaults to outputs/extracted/<lang>)"
    )
    parser.add_argument(
        "--gdrive-url",
        type=str,
        default=None,
        help="Google Drive folder URL containing script PDFs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Pipeline config YAML"
    )
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
        "--sync-drive",
        action="store_true",
        default=False,
        help="Check and download missing PDFs from Google Drive before extracting (default: False; extraction is strictly local)"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download PDFs without running extraction"
    )
    parser.add_argument(
        "--skip-download",
        "--local-only",
        action="store_true",
        default=True,
        help="Use existing local scripts only (default: True, GDrive is never contacted)"
    )
    parser.add_argument(
        "--skip-extracted",
        dest="skip_extracted",
        action="store_true",
        default=True,
        help="Skip extracting scripts that already have extraction_result.json saved (default: True)"
    )
    parser.add_argument(
        "--force-extract",
        "--re-extract",
        dest="skip_extracted",
        action="store_false",
        help="Force re-extraction even if artifacts already exist"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download PDFs from Google Drive even if already present locally"
    )
    parser.add_argument(
        "--fast",
        "--skip-stage2",
        dest="fast",
        action="store_true",
        default=False,
        help="Fast single-pass extraction mode: skip Stage 2 visual verification to cut extraction time in half"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Use local OpenAI-compatible API endpoint (e.g. LM Studio on port 1234) without allocating extra VRAM"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:1234/v1",
        help="URL of OpenAI-compatible API endpoint (default: http://localhost:1234/v1)"
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
        help="Disable interactive terminal prompts and use CLI values directly"
    )

    args = parser.parse_args()

    # Defaults based on language
    if not args.pdf_dir:
        args.pdf_dir = f"data/raw_pdfs/{args.lang}"
    if not args.output_dir:
        args.output_dir = f"outputs/extracted/{args.lang}"
    if not args.gdrive_url:
        args.gdrive_url = GDRIVE_FOLDERS.get(args.lang, DEFAULT_GDRIVE_FOLDER_BANGLA)

    # Launch wizard only if interactive terminal, not bypassed, and no explicit arguments provided
    has_explicit_args = bool(args.top or args.image)
    if not args.non_interactive and not has_explicit_args and sys.stdin.isatty():
        args = interactive_wizard(args)

    if not args.quant:
        args.quant = "4bit"

    if args.mock:
        exec_mode_label = "Mock (Dev PC)"
    elif args.api:
        exec_mode_label = f"Local API ({args.api_url})"
    else:
        exec_mode_label = "CUDA RTX 5090 (Gemma 4 31B IT)"

    console.print(Panel.fit(
        f"[bold cyan]Gemma 4 31B IT Multimodal Script Extraction Controller[/bold cyan]\n"
        f"[green]Language / Subject:[/green] {args.lang.capitalize()}\n"
        f"[green]Source Directory:[/green] {args.image or args.pdf_dir}\n"
        f"[green]Top Limit (--top):[/green] {args.top or 'All available'}\n"
        f"[green]Quantization (--quant):[/green] {args.quant}\n"
        f"[green]Fast Mode (--fast):[/green] {'Enabled (single-pass)' if args.fast else 'Disabled (full 2-pass verification)'}\n"
        f"[yellow]Execution Mode:[/yellow] {exec_mode_label}\n"
        f"[yellow]Extraction Output Root:[/yellow] {args.output_dir}",
        title="Extraction Initialized"
    ))

    # 1. Discover Script Files (Strictly Local by Default)
    input_files = []
    if args.image:
        if os.path.exists(args.image):
            input_files = [args.image]
        else:
            console.print(f"[red]Specified image/PDF file does not exist: {args.image}[/red]")
            return
    elif getattr(args, "sync_drive", False) or getattr(args, "download_only", False):
        console.print("\n[bold]Step 1: Checking & Downloading Exam Script PDFs from Google Drive...[/bold]")
        input_files = download_drive_pdfs(
            gdrive_url=args.gdrive_url,
            target_dir=args.pdf_dir,
            top_limit=args.top,
            skip_existing=not args.force_download,
            skip_download=False
        )
        if getattr(args, "download_only", False):
            console.print("\n[green]--download-only flag active. Download complete![/green]")
            return
    else:
        console.print(f"\n[bold]Step 1: Discovering Local Exam Script Files in '{args.pdf_dir}'...[/bold]")
        if os.path.exists(args.pdf_dir):
            target_path = Path(args.pdf_dir)
            found = (
                list(target_path.glob("*.pdf")) +
                list(target_path.glob("*.PDF")) +
                list(target_path.glob("*.jpg")) +
                list(target_path.glob("*.jpeg")) +
                list(target_path.glob("*.png")) +
                list(target_path.glob("*.bmp"))
            )
            input_files = sorted(list(set(str(p) for p in found)))

    if not input_files:
        console.print(f"[red]No exam script files found in '{args.pdf_dir}'.[/red]")
        console.print(f"[yellow]To download scripts from Google Drive first, run:[/yellow]")
        console.print(f"  python3 scripts/download_drive_pdfs.py --lang {args.lang} --top 5\n")
        return

    # Limit to top N if requested
    if args.top and len(input_files) > args.top:
        input_files = input_files[:args.top]

    console.print(f"[green]Total scripts ready for extraction: {len(input_files)}[/green]")
    for idx, p in enumerate(input_files, 1):
        console.print(f"  [{idx}] {Path(p).name}")

    if args.download_only:
        console.print("\n[green]--download-only flag active. Download complete![/green]")
        return

    # 2. Setup Pipeline Config & Engine
    cfg = load_config(args.config)
    if args.model:
        cfg.model.model_id = args.model
    if args.quant:
        cfg.model.quantization = args.quant
    if args.thinking:
        cfg.decoding.thinking_mode = True
    cfg.pipeline.output_dir = args.output_dir

    console.print("\n[bold]Step 2: Initializing Inference Engine & Pipeline...[/bold]")
    engine = create_engine(
        cfg,
        force_mock=args.mock,
        force_api=args.api,
        api_url=args.api_url if args.api else None
    )
    pipeline = ScriptCheckingPipeline(
        engine=engine,
        config=cfg
    )

    # 3. Execute Extraction on Each Script
    console.print(f"\n[bold]Step 3: Extracting {len(input_files)} script(s) (Stages 1–3)...[/bold]")
    summary_records = []

    for idx, script_path in enumerate(input_files, 1):
        script_id = Path(script_path).stem
        script_out_dir = os.path.join(args.output_dir, script_id)
        extraction_file = os.path.join(script_out_dir, "extraction_result.json")

        if args.skip_extracted and os.path.exists(extraction_file):
            console.print(f"\n[{idx}/{len(input_files)}] Skipping already extracted: [cyan]{script_id}[/cyan]")
            continue

        console.print(f"\n{'='*60}")
        console.print(f"[{idx}/{len(input_files)}] Extracting Script: [bold cyan]{Path(script_path).name}[/bold cyan]")
        console.print(f"{'='*60}")

        try:
            extraction = pipeline.extract_script(
                input_source=script_path,
                script_id=script_id,
                thinking_mode=cfg.decoding.thinking_mode,
                output_dir=args.output_dir,
                paper=args.lang,
                skip_stage2=args.fast,
                force_extract=not args.skip_extracted
            )

            summary_records.append({
                "script_id": script_id,
                "red_ink": "Yes" if extraction.has_red_ink else "No",
                "teacher_marks": str(len(extraction.teacher_marks)),
                "words": extraction.stage1_transcription.word_count,
                "reverted": extraction.stage2_verification.total_corrections_count,
                "total_errors": extraction.stage3_errors.total_error_count,
                "status": "Success"
            })

        except Exception as e:
            console.print(f"[bold red]Failed extracting {script_path}: {e}[/bold red]")
            summary_records.append({
                "script_id": script_id,
                "red_ink": "N/A",
                "teacher_marks": "0",
                "words": 0,
                "reverted": 0,
                "total_errors": 0,
                "status": f"Error: {e}"
            })

    # 4. Extraction Summary Table
    if summary_records:
        table = Table(title=f"Extraction Phase Summary ({len(summary_records)} scripts)")
        table.add_column("Script ID", style="cyan")
        table.add_column("Red Ink (St 0)", style="bold magenta")
        table.add_column("Marks (St 0b)", style="bold yellow")
        table.add_column("Words (St 1)", style="white")
        table.add_column("Silent Reverted (St 2)", style="yellow")
        table.add_column("Errors (St 3)", style="bold red")
        table.add_column("Status", style="magenta")

        for r in summary_records:
            table.add_row(
                r["script_id"],
                r["red_ink"],
                r["teacher_marks"],
                str(r["words"]),
                str(r["reverted"]),
                str(r["total_errors"]),
                r["status"]
            )

        console.print("\n")
        console.print(table)
        console.print(f"\n[green]All extraction outputs saved inside: [bold]{args.output_dir}/<script_id>/[/bold][/green]")
        console.print(f"[green]Raw-Tier Dataset CSV saved at: [bold]{args.output_dir}/raw_tier_dataset.csv[/bold][/green]\n")


if __name__ == "__main__":
    main()
