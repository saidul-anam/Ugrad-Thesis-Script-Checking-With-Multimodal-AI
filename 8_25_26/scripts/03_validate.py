"""CLI: validation (Stage 4).

Flags: --config PATH. (--force/--limit/--script-id accepted for interface
symmetry; validation always reads whatever transcripts exist.)

Reads data/transcripts/, flags scripts for human review, and writes
logs/validation_report.md. Reports, never fixes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.validate import validate_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate transcripts (report only).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="(accepted, no effect)")
    parser.add_argument("--limit", type=int, default=None, help="(accepted, no effect)")
    parser.add_argument("--script-id", default=None, help="(accepted, no effect)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    cfg = load_config(args.config)
    cfg.ensure_dirs()

    results, report_path = validate_all(cfg)
    if not results:
        console.print(f"[yellow]No transcripts found in {cfg.paths.transcripts}[/yellow]")
        sys.exit(1)

    table = Table(title="Stage 4 - validation")
    table.add_column("script_id")
    table.add_column("answers", justify="right")
    table.add_column("missing targets")
    table.add_column("error rate", justify="right")
    table.add_column("flags")

    for r in results:
        rate = f"{r.misspell_rate:.1%}" if r.misspell_rate is not None else "n/a"
        flags = "; ".join(r.flags) if r.flags else "-"
        color = "red" if r.needs_review else "green"
        table.add_row(
            r.script_id,
            str(r.answers_found),
            ", ".join(r.missing_targets) or "-",
            rate,
            f"[{color}]{flags}[/{color}]",
        )
    console.print(table)

    review = sum(1 for r in results if r.needs_review)
    console.print(
        f"\nevaluated={len(results)}  needs_review={review}  "
        f"report -> {report_path}"
    )


if __name__ == "__main__":
    main()
