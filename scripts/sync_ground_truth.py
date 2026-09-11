#!/usr/bin/env python3
"""
CLI Helper to Synchronize Human Ground Truth Marks into Extracted Datasets.

Reads manual marks from gt.txt and syncs them into:
  1. outputs/extracted/<lang>/<script_id>/stage0b_teacher_marks.json
  2. outputs/extracted/<lang>/<script_id>/extraction_result.json
  3. outputs/extracted/<lang>/<script_id>/raw_tier_records.csv
  4. outputs/extracted/raw_tier_dataset.csv & outputs/extracted/<lang>/raw_tier_dataset.csv
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.utils.ground_truth import (
    parse_ground_truth_file,
    sync_ground_truth_to_extracted_artifacts
)

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Synchronize human ground-truth marks from gt.txt")
    parser.add_argument(
        "--gt-file",
        type=str,
        default="gt.txt",
        help="Path to ground truth text file (default: gt.txt)"
    )
    parser.add_argument(
        "--extracted-dir",
        type=str,
        default="outputs/extracted",
        help="Root directory containing extracted scripts (default: outputs/extracted)"
    )
    args = parser.parse_args()

    console.print(Panel.fit(
        f"[bold cyan]🎯 Ground Truth Synchronizer[/bold cyan]\n"
        f"[dim]Ground Truth File:[/dim] {args.gt_file}\n"
        f"[dim]Target Extraction Root:[/dim] {args.extracted_dir}",
        border_style="cyan"
    ))

    # Parse and display available ground truth entries
    gt_data = parse_ground_truth_file(args.gt_file)
    if not gt_data:
        console.print(f"[bold red]No valid ground-truth entries found in '{args.gt_file}'.[/bold red]")
        return

    table = Table(title="Verified Scripts in Ground Truth File")
    table.add_column("Script ID", style="cyan bold")
    table.add_column("Questions Count", justify="right")
    table.add_column("Total Human Marks", justify="right", style="green bold")
    table.add_column("Questions Summary", style="dim")

    for sid, marks in gt_data.items():
        tot = sum(marks.values())
        summary_str = ", ".join(f"Q{q}:{v}" for q, v in sorted(marks.items(), key=lambda x: (len(x[0]), x[0])))
        table.add_row(sid, str(len(marks)), f"{tot:.1f}", summary_str[:60] + ("..." if len(summary_str) > 60 else ""))

    console.print(table)

    console.print("\n[bold]Synchronizing marks into extracted folders and CSV datasets...[/bold]")
    updated_count = sync_ground_truth_to_extracted_artifacts(
        extracted_root=args.extracted_dir,
        gt_file=args.gt_file
    )

    console.print(f"\n[bold green]✅ Successfully synchronized {updated_count} script directory(ies)![/bold green]")
    console.print("[dim]Now re-run evaluation with `python3 scripts/evaluate_scripts.py --script-name <id> --api --force-evaluate -y`[/dim]\n")


if __name__ == "__main__":
    main()
