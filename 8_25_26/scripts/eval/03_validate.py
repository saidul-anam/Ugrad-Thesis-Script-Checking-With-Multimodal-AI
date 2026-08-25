"""CLI: validate evaluation output (Stage 4).

Flags: --config PATH.

Asserts and reports, never auto-fixes. Writes logs/evaluation_report.md and
prints the assertion results + noise-floor summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.table import Table

from src.config import load_eval_config
from src.eval.validate import validate_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate evaluation output (report only).")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    cfg = load_eval_config(args.config)
    cfg.ensure_dirs()

    out = validate_all(cfg)

    table = Table(title="Stage 4 - evaluation validation")
    table.add_column("result")
    table.add_column("assertion")
    color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
    for c in out.checks:
        table.add_row(f"[{color[c.level]}]{c.level}[/{color[c.level]}]", c.message)
    console.print(table)

    if out.report_path:
        console.print(f"\nreport -> {out.report_path}")
    console.print(
        f"checks={len(out.checks)}  failed={out.failed}  "
        f"{'OK' if out.failed == 0 else 'FAILURES PRESENT'}"
    )
    sys.exit(1 if out.failed else 0)


if __name__ == "__main__":
    main()
