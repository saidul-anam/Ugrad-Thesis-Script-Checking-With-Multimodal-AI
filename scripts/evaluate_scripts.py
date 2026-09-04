#!/usr/bin/env python3
"""
Dedicated Evaluation Runner for Multimodal Script Checking Pipeline.

Executes Stage 4:
- Reads pre-extracted canonical handwriting & error catalogs from extraction directory
- Applies subject-specific rubrics (Criteria weights, content scoring, linguistic deductions)
- Generates teacher-level feedback, recommendations, and complete consolidated reports

Takes either:
  1. Specific script name(s) / ID(s): --script-name <id>
  2. Count of scripts to evaluate from directory: --top <N> (or all)

Outputs per script:
  outputs/extracted/<lang>/<script_id>/  (or custom --output-dir)
    ├── stage4_evaluation.json
    ├── complete_report.json
    └── evaluation_report.md
"""

import os
import sys

# Configure PyTorch CUDA Allocator early to prevent memory fragmentation
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import argparse
from pathlib import Path

from typing import List, Optional

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
from src.utils.export_utils import load_extraction_artifacts


console = Console()


def find_extracted_scripts(extraction_dir: str) -> List[Path]:
    """Discover all extracted script folders that contain extraction artifacts."""
    candidates = []
    if not os.path.exists(extraction_dir):
        return candidates

    for entry in sorted(os.listdir(extraction_dir)):
        p = Path(extraction_dir) / entry
        if p.is_dir():
            # Check for extraction result or individual stage outputs
            has_extraction = (
                (p / "extraction_result.json").exists() or
                (p / "complete_report.json").exists() or
                ((p / "stage1_transcription.json").exists() and (p / "stage3_errors.json").exists())
            )
            if has_extraction:
                candidates.append(p)
    return candidates


def interactive_wizard(args, available_scripts: List[Path]):
    """Interactive terminal wizard for configuring evaluation."""
    console.print(Panel.fit(
        "[bold cyan]⚖️  Gemma 4 Multimodal Script Evaluation - Interactive Setup[/bold cyan]\n"
        "[dim]Configuring Stage 4 (Rubric Scoring + Pedagogical Feedback)[/dim]",
        border_style="cyan"
    ))

    # 1. Language
    console.print("\n[bold green]1. Exam Script Language / Subject:[/bold green]")
    console.print("   [[bold cyan]1[/bold cyan]] [bold white]Bangla[/bold white] (Creative Questions / সৃজনশীল)")
    console.print("   [[bold cyan]2[/bold cyan]] [bold white]English[/bold white] (Essay / Composition Writing)")

    current_lang_idx = "2" if getattr(args, "lang", "bangla") == "english" else "1"
    lang_choice = Prompt.ask("[bold green]   Select Language[/bold green]", choices=["1", "2"], default=current_lang_idx)
    args.lang = "bangla" if lang_choice == "1" else "english"

    if args.extraction_dir in [None, "outputs/extracted", "outputs/extracted/bangla", "outputs/extracted/english", "outputs/runs/bangla", "outputs/runs/english"]:
        # Check standard extraction dir first, fallback to outputs/runs if extracted is empty
        primary_dir = f"outputs/extracted/{args.lang}"
        fallback_dir = f"outputs/runs/{args.lang}"
        args.extraction_dir = primary_dir if (os.path.exists(primary_dir) and find_extracted_scripts(primary_dir)) else fallback_dir

    # Refresh available scripts
    available = find_extracted_scripts(args.extraction_dir)

    # 2. Script Selection Mode (Specific Name vs Count)
    console.print(f"\n[bold green]2. Script Selection Mode in '{args.extraction_dir}':[/bold green]")
    console.print(f"   [Found {len(available)} extracted script(s)]")
    console.print("   [[bold cyan]1[/bold cyan]] Evaluate [bold white]ALL[/bold white] extracted scripts in directory")
    console.print("   [[bold cyan]2[/bold cyan]] Evaluate [bold white]TOP N[/bold white] scripts by count")
    console.print("   [[bold cyan]3[/bold cyan]] Select [bold white]SPECIFIC SCRIPT[/bold white] by name / ID")

    default_mode = "3" if args.script_name else ("2" if args.top else "1")
    mode_choice = Prompt.ask("[bold green]   Select Mode[/bold green]", choices=["1", "2", "3"], default=default_mode)

    if mode_choice == "1":
        args.script_name = None
        args.top = None
    elif mode_choice == "2":
        top_str = Prompt.ask("   [bold]How many scripts to evaluate?[/bold]", default=str(args.top or 5))
        try:
            args.top = int(top_str)
        except ValueError:
            args.top = None
        args.script_name = None
    elif mode_choice == "3":
        if available:
            console.print("\n   Available scripts:")
            for idx, s in enumerate(available[:15], 1):
                console.print(f"     [{idx}] {s.name}")
            if len(available) > 15:
                console.print(f"     ... and {len(available) - 15} more")

        script_input = Prompt.ask("\n   [bold]Enter Script Name / ID or Number[/bold]", default=args.script_name or (available[0].name if available else ""))
        if script_input.isdigit() and available and 1 <= int(script_input) <= len(available):
            args.script_name = available[int(script_input) - 1].name
        else:
            args.script_name = script_input.strip()

    # 3. Rubric selection
    rubric_files = sorted(glob.glob("configs/rubrics/*.yaml") + glob.glob("configs/rubrics/*.yml"))
    console.print("\n[bold green]3. Available Rubrics:[/bold green]")
    default_idx = "1"
    for idx, rpath in enumerate(rubric_files, 1):
        clean_path = rpath.replace("\\", "/")
        if args.lang == "bangla" and "bangla" in clean_path:
            default_idx = str(idx)
        elif args.lang == "english" and "english" in clean_path:
            default_idx = str(idx)
        console.print(f"   [[bold cyan]{idx}[/bold cyan]] {clean_path}")
    console.print(f"   [[bold cyan]{len(rubric_files) + 1}[/bold cyan]] Custom file path...")

    rubric_choice = Prompt.ask("[bold green]   Select Rubric[/bold green]", default=default_idx).strip()
    if rubric_choice.isdigit():
        choice_num = int(rubric_choice)
        if 1 <= choice_num <= len(rubric_files):
            args.rubric = rubric_files[choice_num - 1].replace("\\", "/")
        elif choice_num == len(rubric_files) + 1:
            args.rubric = Prompt.ask("   [bold]Enter custom rubric YAML path[/bold]", default=args.rubric)
    elif rubric_choice:
        args.rubric = rubric_choice

    # 4. Engine & Quantization
    console.print("\n[bold green]4. Execution Engine:[/bold green]")
    console.print("   [1] Real GPU (CUDA Gemma 4 31B IT)")
    console.print("   [2] Mock Dev Mode (Fast CPU simulation without GPU/weights)")
    mode_default = "2" if args.mock else "1"
    args.mock = (Prompt.ask("[bold green]   Select Execution Mode[/bold green]", choices=["1", "2"], default=mode_default) == "2")

    # 5. Skip already evaluated
    args.skip_evaluated = Confirm.ask(
        "\n[bold green]5. Skip scripts that already have completed evaluation reports?[/bold green]",
        default=args.skip_evaluated
    )

    # Summary
    console.print("\n" + "="*50)
    engine_label = "[bold yellow]Mock Dev Mode (CPU simulation)[/bold yellow]" if args.mock else "[bold green]CUDA RTX 5090 (Gemma 4 31B IT)[/bold green]"
    target_label = f"Specific Script: [yellow]{args.script_name}[/yellow]" if args.script_name else (f"Top [white]{args.top}[/white] scripts" if args.top else "All available in directory")

    console.print(Panel(
        f"• [cyan]Language / Subject (--lang):[/cyan] [bold yellow]{args.lang.capitalize()}[/bold yellow]\n"
        f"• [cyan]Evaluation Target:[/cyan] [bold white]{target_label}[/bold white]\n"
        f"• [cyan]Extraction Source Directory:[/cyan] [bold white]{args.extraction_dir}[/bold white]\n"
        f"• [cyan]Rubric (--rubric):[/cyan] [bold white]{args.rubric}[/bold white]\n"
        f"• [cyan]Engine Mode (--mock):[/cyan] {engine_label}\n"
        f"• [cyan]Skip Evaluated:[/cyan] {'Yes' if args.skip_evaluated else 'No'}",
        title="[bold green]Evaluation Configuration Summary[/bold green]",
        border_style="green"
    ))

    if not Confirm.ask("[bold yellow]Proceed with evaluation?[/bold yellow]", default=True):
        console.print("[red]Cancelled by user.[/red]")
        sys.exit(0)

    console.print("\n[bold green]Starting evaluation pipeline...[/bold green]\n")
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pre-extracted exam scripts using Gemma 4 31B IT and subject rubrics (Stage 4)"
    )
    parser.add_argument(
        "--script-name",
        "--script",
        "--script-id",
        dest="script_name",
        type=str,
        default=None,
        help="Name or ID of specific script to evaluate (e.g. --script-name script_01)"
    )
    parser.add_argument(
        "--top",
        dest="top",
        type=int,
        default=None,
        help="Number of scripts to evaluate from the directory (e.g. --top 5). Omit to evaluate all."
    )
    parser.add_argument(
        "--extraction-dir",
        "--dir",
        dest="extraction_dir",
        type=str,
        default=None,
        help="Directory containing extracted scripts (defaults to outputs/extracted/<lang>)"
    )
    parser.add_argument(
        "--lang",
        "--language",
        dest="lang",
        type=str,
        choices=["bangla", "english"],
        default="bangla",
        help="Language / Subject of exam scripts: 'bangla' or 'english' (default: bangla)"
    )
    parser.add_argument(
        "--rubric",
        type=str,
        default=None,
        help="Path to rubric YAML (defaults to respective language rubric)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports (defaults to the extracted script directory)"
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
        "--skip-evaluated",
        dest="skip_evaluated",
        action="store_true",
        default=True,
        help="Skip scripts that already have completed evaluation reports (default: True)"
    )
    parser.add_argument(
        "--force-evaluate",
        "--re-evaluate",
        dest="skip_evaluated",
        action="store_false",
        help="Force re-evaluation even if completed report exists"
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
    if not args.extraction_dir:
        primary_dir = f"outputs/extracted/{args.lang}"
        fallback_dir = f"outputs/runs/{args.lang}"
        args.extraction_dir = primary_dir if (os.path.exists(primary_dir) and find_extracted_scripts(primary_dir)) else (fallback_dir if os.path.exists(fallback_dir) else primary_dir)

    if not args.rubric:
        args.rubric = (
            "configs/rubrics/bangla_creative_question.yaml"
            if args.lang == "bangla"
            else "configs/rubrics/english_writing.yaml"
        )

    # Check available scripts in directory
    discovered_scripts = find_extracted_scripts(args.extraction_dir)

    # Launch wizard if interactive terminal and not bypassed
    if not args.non_interactive and sys.stdin.isatty():
        args = interactive_wizard(args, discovered_scripts)

    # 1. Resolve Target Scripts to Evaluate
    target_script_paths = []

    if args.script_name:
        # User specified a specific script name or path
        candidate_path = Path(args.script_name)
        if candidate_path.is_dir() or candidate_path.is_file():
            target_script_paths = [candidate_path]
        else:
            # Check within extraction_dir
            match_in_dir = Path(args.extraction_dir) / args.script_name
            # Check with / without extensions
            if match_in_dir.exists():
                target_script_paths = [match_in_dir]
            else:
                # Fuzzy find by stem
                matched = [p for p in find_extracted_scripts(args.extraction_dir) if p.stem.lower() == args.script_name.lower() or p.name.lower() == args.script_name.lower()]
                if matched:
                    target_script_paths = matched
                else:
                    console.print(f"[red]Could not find script '{args.script_name}' in '{args.extraction_dir}'.[/red]")
                    return
    else:
        # Batch from directory
        target_script_paths = find_extracted_scripts(args.extraction_dir)
        if args.top and len(target_script_paths) > args.top:
            target_script_paths = target_script_paths[:args.top]

    if not target_script_paths:
        console.print(f"[red]No extracted scripts found in '{args.extraction_dir}'.[/red]")
        console.print("[yellow]Tip: Run extraction first with `python scripts/extract_scripts.py`[/yellow]")
        return

    console.print(Panel.fit(
        f"[bold cyan]Gemma 4 31B IT Multimodal Script Evaluation Controller[/bold cyan]\n"
        f"[green]Language / Subject:[/green] {args.lang.capitalize()}\n"
        f"[green]Extraction Directory:[/green] {args.extraction_dir}\n"
        f"[green]Target Count:[/green] {len(target_script_paths)} script(s)\n"
        f"[green]Rubric:[/green] {args.rubric}\n"
        f"[yellow]Execution Mode:[/yellow] {'Mock (Dev PC)' if args.mock else 'CUDA RTX 5090 (Gemma 4 31B IT)'}",
        title="Evaluation Initialized"
    ))

    console.print(f"\n[green]Scripts selected for evaluation ({len(target_script_paths)}):[/green]")
    for idx, sp in enumerate(target_script_paths, 1):
        console.print(f"  [{idx}] {sp.name}")

    # 2. Setup Pipeline Config & Engine
    cfg = load_config(args.config)
    if args.model:
        cfg.model.model_id = args.model
    if args.quant:
        cfg.model.quantization = args.quant
    if args.thinking:
        cfg.decoding.thinking_mode = True

    console.print("\n[bold]Step 1: Initializing Inference Engine & Evaluator...[/bold]")
    engine = create_engine(cfg, force_mock=args.mock)
    pipeline = ScriptCheckingPipeline(
        engine=engine,
        config=cfg,
        rubric_path=args.rubric
    )

    # 3. Evaluate Each Script
    console.print(f"\n[bold]Step 2: Evaluating {len(target_script_paths)} script(s) against rubric...[/bold]")
    summary_records = []

    for idx, script_path in enumerate(target_script_paths, 1):
        script_id = script_path.name if script_path.is_dir() else script_path.stem
        report_out_dir = args.output_dir or (str(script_path) if script_path.is_dir() else os.path.dirname(str(script_path)))
        completed_report = os.path.join(report_out_dir, "complete_report.json")

        if args.skip_evaluated and os.path.exists(completed_report):
            console.print(f"\n[{idx}/{len(target_script_paths)}] Skipping already evaluated: [cyan]{script_id}[/cyan]")
            continue

        console.print(f"\n{'='*60}")
        console.print(f"[{idx}/{len(target_script_paths)}] Evaluating Script: [bold cyan]{script_id}[/bold cyan]")
        console.print(f"{'='*60}")

        try:
            report = pipeline.evaluate_extracted_script(
                extraction_input=str(script_path),
                rubric_path=args.rubric,
                thinking_mode=cfg.decoding.thinking_mode,
                output_dir=args.output_dir
            )

            summary_records.append({
                "script_id": script_id,
                "content_score": f"{report.stage4_evaluation.content_raw_score:.2f}",
                "penalty": f"-{report.stage4_evaluation.linguistic_penalty:.2f}",
                "final_score": f"{report.stage4_evaluation.final_score:.2f} / {report.stage4_evaluation.total_max_marks:.2f}",
                "pct": f"{report.stage4_evaluation.percentage:.1f}%",
                "feedback": report.stage4_evaluation.overall_feedback[:60] + "..." if len(report.stage4_evaluation.overall_feedback) > 60 else report.stage4_evaluation.overall_feedback,
                "status": "Success"
            })

        except Exception as e:
            console.print(f"[bold red]Failed evaluating {script_id}: {e}[/bold red]")
            summary_records.append({
                "script_id": script_id,
                "content_score": "N/A",
                "penalty": "N/A",
                "final_score": "N/A",
                "pct": "0%",
                "feedback": f"Error: {e}",
                "status": f"Error: {e}"
            })

    # 4. Evaluation Summary Table
    if summary_records:
        table = Table(title=f"Evaluation Phase Summary ({len(summary_records)} scripts)")
        table.add_column("Script ID", style="cyan")
        table.add_column("Content Score", style="white")
        table.add_column("Penalty", style="red")
        table.add_column("Final Score", style="bold green")
        table.add_column("Percentage", style="bold yellow")
        table.add_column("Teacher Feedback", style="white")
        table.add_column("Status", style="magenta")

        for r in summary_records:
            table.add_row(
                r["script_id"],
                r["content_score"],
                r["penalty"],
                r["final_score"],
                r["pct"],
                r["feedback"],
                r["status"]
            )

        console.print("\n")
        console.print(table)
        console.print(f"\n[green]All evaluation reports saved in: [bold]{args.extraction_dir}/<script_id>/[/bold][/green]\n")


if __name__ == "__main__":
    main()
