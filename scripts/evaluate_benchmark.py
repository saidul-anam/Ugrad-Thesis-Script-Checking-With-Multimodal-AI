#!/usr/bin/env python3
"""
Ablation & Benchmark Runner:
Evaluates scripts under both Thinking Mode = ON and Thinking Mode = OFF to quantify
whether reasoning steps reduce or increase silent error correction.
"""

import os
import glob
import argparse
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.core.config import load_config
from src.engine.engine_factory import create_engine
from src.pipeline.orchestrator import ScriptCheckingPipeline


console = Console()


def run_ablation(image_dir: str, config_path: str, rubric_path: str, mock: bool, output_csv: str):
    cfg = load_config(config_path)
    engine = create_engine(cfg, force_mock=mock)
    pipeline = ScriptCheckingPipeline(engine=engine, config=cfg, rubric_path=rubric_path)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_paths:
        console.print(f"[red]No images found in {image_dir}[/red]")
        return

    results = []

    for img_path in image_paths:
        script_name = os.path.basename(img_path)
        console.print(f"\n[bold]Evaluating: {script_name}[/bold]")

        # 1. Run without thinking mode
        report_off = pipeline.evaluate_script(
            image_input=img_path,
            script_id=f"{os.path.splitext(script_name)[0]}_thinking_off",
            thinking_mode=False
        )

        # 2. Run with thinking mode
        report_on = pipeline.evaluate_script(
            image_input=img_path,
            script_id=f"{os.path.splitext(script_name)[0]}_thinking_on",
            thinking_mode=True
        )

        results.append({
            "script_name": script_name,
            "off_reverted_corrections": report_off.stage2_verification.total_corrections_count,
            "off_errors_found": report_off.stage3_errors.total_error_count,
            "off_score": report_off.stage4_evaluation.final_score,
            "off_elapsed_s": report_off.metadata.get("elapsed_seconds", 0),
            "on_reverted_corrections": report_on.stage2_verification.total_corrections_count,
            "on_errors_found": report_on.stage3_errors.total_error_count,
            "on_score": report_on.stage4_evaluation.final_score,
            "on_elapsed_s": report_on.metadata.get("elapsed_seconds", 0),
        })

    df = pd.DataFrame(results) if "pandas" in sys.modules else None
    
    table = Table(title="Thinking Mode Ablation Comparison")
    table.add_column("Script", style="cyan")
    table.add_column("Thinking OFF (Reverted / Errors / Score)", style="yellow")
    table.add_column("Thinking ON (Reverted / Errors / Score)", style="green")

    for r in results:
        table.add_row(
            r["script_name"],
            f"{r['off_reverted_corrections']} rev | {r['off_errors_found']} err | {r['off_score']:.2f} marks",
            f"{r['on_reverted_corrections']} rev | {r['on_errors_found']} err | {r['on_score']:.2f} marks"
        )

    console.print(table)


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Run thinking mode ablation benchmark")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing script images")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--rubric", type=str, default="configs/rubrics/bangla_creative_question.yaml")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output-csv", type=str, default="outputs/ablation_results.csv")
    args = parser.parse_args()

    run_ablation(args.image_dir, args.config, args.rubric, args.mock, args.output_csv)
