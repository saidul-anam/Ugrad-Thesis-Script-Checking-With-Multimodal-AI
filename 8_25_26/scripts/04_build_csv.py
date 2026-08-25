"""CLI: derive extraction.csv + extraction_runs.csv from transcripts (Stage 5).

Flags: --config PATH.

The per-script JSON is canonical; this CSV is a derived, regenerable view.
Runs the schema's validation assertions and prints the results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.build_csv import build_all
from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build extraction.csv from transcripts.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    cfg = load_config(args.config)

    csv_path, runs_path, assertions, n_rows = build_all(cfg)

    console.print(f"[green]wrote[/green] {csv_path}  ({n_rows} rows)")
    console.print(f"[green]wrote[/green] {runs_path}")
    console.print("\n[bold]Validation assertions[/bold]")
    failed = 0
    for line in assertions:
        if line.startswith("[FAIL]"):
            failed += 1
            console.print(f"[red]{line}[/red]")
        elif line.startswith("[WARN]"):
            console.print(f"[yellow]{line}[/yellow]")
        else:
            console.print(line)

    console.print(f"\nassertions: {len(assertions)}  failed={failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
