"""CLI: emit evaluation CSVs (Stage 3).

Flags: --config PATH.

Reads output/evaluations/*.json and writes output/evaluation.csv (long format,
one row per task_id x model x run) + output/evaluation_runs.csv (one row per API
call). Derived + regenerable; never edit by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console

from src.config import load_eval_config
from src.eval.emit_csv import emit_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emit evaluation CSVs from records.")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    cfg = load_eval_config(args.config)
    cfg.ensure_dirs()

    csv_path, runs_path, n_eval, n_runs = emit_all(cfg)
    if n_eval == 0:
        console.print("[yellow]No evaluation records found in "
                      f"{cfg.paths.evaluations}[/yellow]")
        sys.exit(1)
    console.print(f"[green]wrote[/green] {csv_path}  ({n_eval} rows)")
    console.print(f"[green]wrote[/green] {runs_path}  ({n_runs} API-call rows)")


if __name__ == "__main__":
    main()
