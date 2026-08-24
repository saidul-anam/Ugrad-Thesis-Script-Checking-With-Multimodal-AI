#!/usr/bin/env python3
"""
CLI Runner for Gemma 4 31B IT 4-Stage Multimodal Script Evaluation Pipeline.
"""

import os
import sys
import glob
import argparse
from pathlib import Path

# Ensure repository root is on sys.path when running as a standalone script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.core.config import load_config
from src.engine.engine_factory import create_engine
from src.pipeline.orchestrator import ScriptCheckingPipeline


console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 31B IT 4-Stage Multimodal Script Evaluation Pipeline"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a script image or directory of images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline configuration YAML"
    )
    parser.add_argument(
        "--rubric",
        type=str,
        default="configs/rubrics/bangla_creative_question.yaml",
        help="Path to evaluation rubric YAML"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model ID (default: google/gemma-4-31b-it)"
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["4bit", "8bit", "none"],
        default=None,
        help="Override quantization mode (default: 4bit)"
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable reasoning/thinking mode ablation"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with Mock simulation engine (for development without GPU)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports"
    )

    args = parser.parse_args()

    # 1. Load and customize configuration
    cfg = load_config(args.config)
    if args.model:
        cfg.model.model_id = args.model
    if args.quant:
        cfg.model.quantization = args.quant
    if args.thinking:
        cfg.decoding.thinking_mode = True
    if args.output_dir:
        cfg.pipeline.output_dir = args.output_dir

    console.print(Panel.fit(
        f"[bold cyan]Gemma 4 31B IT Multimodal Script Evaluation Pipeline[/bold cyan]\n"
        f"[green]Model:[/green] {cfg.model.model_id} | [green]Quantization:[/green] {cfg.model.quantization}\n"
        f"[yellow]Thinking Mode:[/yellow] {cfg.decoding.thinking_mode} | [yellow]Temperature:[/yellow] {cfg.decoding.temperature}\n"
        f"[magenta]Mock Engine:[/magenta] {args.mock} | [magenta]Rubric:[/magenta] {args.rubric}",
        title="Pipeline Initialized"
    ))

    # 2. Instantiate Engine & Pipeline
    engine = create_engine(cfg, force_mock=args.mock)
    pipeline = ScriptCheckingPipeline(
        engine=engine,
        config=cfg,
        rubric_path=args.rubric
    )

    # 3. Collect images
    image_paths = []
    if os.path.isdir(args.image):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            image_paths.extend(glob.glob(os.path.join(args.image, ext)))
    else:
        image_paths = [args.image]

    if not image_paths:
        console.print(f"[red]No images found for path: {args.image}[/red]")
        return

    console.print(f"[bold]Processing {len(image_paths)} script image(s)...[/bold]\n")

    for img_path in image_paths:
        try:
            report = pipeline.evaluate_script(
                image_input=img_path,
                thinking_mode=cfg.decoding.thinking_mode
            )

            # Summary Table
            table = Table(title=f"Evaluation Results: {report.script_id}")
            table.add_column("Stage", style="cyan", no_wrap=True)
            table.add_column("Metric / Status", style="green")
            table.add_column("Details", style="white")

            table.add_row(
                "Stage 1 (Verbatim)",
                f"{report.stage1_transcription.word_count} words",
                f"Illegible: {report.stage1_transcription.illegible_count}, Unclear: {report.stage1_transcription.unclear_count}"
            )
            table.add_row(
                "Stage 2 (Verification)",
                f"{report.stage2_verification.total_corrections_count} silent corrections reverted",
                report.stage2_verification.verification_notes[:60] + "..." if len(report.stage2_verification.verification_notes) > 60 else report.stage2_verification.verification_notes
            )
            table.add_row(
                "Stage 3 (Errors)",
                f"{report.stage3_errors.total_error_count} errors cataloged",
                f"Spelling: {report.stage3_errors.spelling_error_count}, Grammar: {report.stage3_errors.grammar_error_count}"
            )
            table.add_row(
                "Stage 4 (Rubric Grade)",
                f"[bold magenta]{report.stage4_evaluation.final_score:.2f} / {report.stage4_evaluation.total_max_marks:.2f} ({report.stage4_evaluation.percentage:.1f}%)[/bold magenta]",
                f"Content: {report.stage4_evaluation.content_raw_score:.2f}, Penalty: -{report.stage4_evaluation.linguistic_penalty:.2f}"
            )

            console.print(table)
            console.print(f"[dim]Saved report artifacts to {cfg.pipeline.output_dir}/[/dim]\n")

        except Exception as e:
            console.print(f"[bold red]Failed evaluating {img_path}: {e}[/bold red]")


if __name__ == "__main__":
    main()
