"""CLI: run evaluation (Stage 2).

Flags: --force, --limit N, --script-id X, --model {gemini,gemma,both}, --k N,
--config PATH.

Grades each extracted answer with the selected model(s), k runs each. Resumable.
Fails fast at startup if a needed API key is missing. Every API failure is logged
with task_id and run index; the final count reconciles
attempted = graded + cached + synthesised + failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.config import load_eval_config, require_grader_keys
from src.eval.evaluate import (
    EvalError,
    build_prompts_for_rows,
    evaluate_one,
    load_rows,
)
from src.eval.grader_client import build_grader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grade extracted answers with two models.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--force", action="store_true", help="Ignore cache / re-grade")
    p.add_argument("--limit", type=int, default=None, help="First N scripts only")
    p.add_argument("--script-id", default=None, help="Only this script id")
    p.add_argument("--model", choices=["gemini", "gemma", "both"], default="both")
    p.add_argument("--k", type=int, default=None, help="Runs per model (override config)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )

    cfg = load_eval_config(args.config)
    cfg.ensure_dirs()
    models = ["gemini", "gemma"] if args.model == "both" else [args.model]
    k = args.k if args.k is not None else cfg.k

    # Fail fast: keys for the chosen models must be present (names only, no values).
    try:
        require_grader_keys(cfg, models)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    rows = load_rows(cfg, args.script_id, args.limit)
    if not rows:
        console.print("[yellow]No matching rows in extraction.csv[/yellow]")
        sys.exit(1)

    prompts = build_prompts_for_rows(cfg, rows)
    graders = {name: build_grader(cfg, name) for name in models}

    console.print(
        f"rows={len(rows)} models={models} k={k} "
        f"thinking={cfg.thinking_level} concurrency={cfg.concurrency}  "
        f"=> {len(rows) * len(models) * k} (row x model x run) units"
    )

    # Assert byte-identical prompt across the chosen graders for each row.
    for r in rows:
        _ = prompts[r["task_id"]].prompt_bytes  # single assembled prompt reused

    results: list = []
    failures: dict[str, str] = {}
    tasks = [(r, m, run) for r in rows for m in models for run in range(k)]

    with Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("Grading", total=len(tasks))
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            futs = {
                pool.submit(
                    evaluate_one, r, prompts[r["task_id"]], m, run, cfg,
                    graders.get(m), force=args.force,
                ): (r["task_id"], m, run)
                for (r, m, run) in tasks
            }
            for fut in as_completed(futs):
                tid, m, run = futs[fut]
                try:
                    results.append(fut.result())
                except EvalError as exc:
                    failures[f"{tid}/{m}/run{run}"] = str(exc)
                    console.print(f"[red]{exc}[/red]")
                except Exception as exc:  # noqa: BLE001
                    failures[f"{tid}/{m}/run{run}"] = f"unexpected: {exc}"
                    console.print(f"[red][{tid}/{m}/run{run}] unexpected: {exc}[/red]")
                progress.advance(bar)

    graded = sum(1 for r in results if r.status == "graded")
    cached = sum(1 for r in results if r.status == "cached")
    synth = sum(1 for r in results if r.status == "synthesised")
    failed = len(failures)
    total_cost = sum(r.cost_usd for r in results)
    evid_missing = sum(1 for r in results if r.evidence_missing)

    table = Table(title="Stage 2 - evaluation")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for label, val in [
        ("(row x model x run) units", len(tasks)),
        ("graded", graded), ("cached", cached), ("synthesised", synth),
        ("failed", failed), ("records w/ evidence_not_found", evid_missing),
        ("total cost USD", f"${total_cost:.4f}"),
    ]:
        table.add_row(label, str(val))
    console.print(table)

    reconciled = graded + cached + synth + failed
    console.print(
        f"\nattempted={len(tasks)}  graded={graded}  cached={cached}  "
        f"synthesised={synth}  failed={failed}  "
        f"(sum={reconciled}, {'OK' if reconciled == len(tasks) else 'MISMATCH'})"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
