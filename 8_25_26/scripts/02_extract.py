"""CLI: extraction (Stage 3).

Flags: --force, --limit N, --script-id X, --config PATH.

Runs the vision transcription over converted scripts, writing verbatim JSON
transcripts to data/transcripts/. Resumable (content-hash cache + up-to-date
skip). Fails loudly: a script that errors is recorded FAILED, and the final
count reconciles attempted = extracted + cached + skipped + failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.config import load_config
from src.extract import ExtractionError, ExtractResult, extract_script
from src.llm_client import build_client
from src.pdf_to_images import discover_scripts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe exam scripts to JSON.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Ignore cache / re-extract")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N scripts")
    parser.add_argument("--script-id", default=None, help="Only process this script id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )

    cfg = load_config(args.config)
    cfg.ensure_dirs()

    scripts = discover_scripts(cfg)
    if args.script_id:
        scripts = [s for s in scripts if s[0] == args.script_id]
        if not scripts:
            console.print(f"[red]No script matching id '{args.script_id}'[/red]")
            sys.exit(2)
    if args.limit is not None:
        scripts = scripts[: args.limit]
    if not scripts:
        console.print(f"[yellow]No scripts found under {cfg.paths.raw_pdfs}[/yellow]")
        sys.exit(1)

    # Build the client once (validates provider + credentials up front).
    try:
        client = build_client(cfg)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Cannot build LLM client: {exc}[/red]")
        sys.exit(3)

    console.print(
        f"provider=[cyan]{cfg.provider}[/cyan] model=[cyan]{cfg.model_id}[/cyan] "
        f"thinking=[cyan]{cfg.thinking_level}[/cyan] concurrency={cfg.llm.concurrency}"
    )

    results: dict[str, ExtractResult] = {}
    failures: dict[str, str] = {}

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting", total=len(scripts))
        with ThreadPoolExecutor(max_workers=cfg.llm.concurrency) as pool:
            futures = {
                pool.submit(extract_script, sid, cfg, client, force=args.force): sid
                for sid, _ in scripts
            }
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    results[sid] = fut.result()
                except ExtractionError as exc:
                    failures[sid] = str(exc)
                    console.print(f"[red]{exc}[/red]")
                except Exception as exc:  # noqa: BLE001 - never lose a failure
                    failures[sid] = f"unexpected: {exc}"
                    console.print(f"[red][{sid}] unexpected: {exc}[/red]")
                progress.advance(task)

    table = Table(title="Stage 3 - extraction")
    table.add_column("script_id")
    table.add_column("status")
    table.add_column("answers", justify="right")
    table.add_column("not_found")
    table.add_column("cost", justify="right")
    table.add_column("wall", justify="right")

    status_color = {"extracted": "green", "cached": "cyan", "skipped": "yellow"}
    extracted = cached = skipped = 0
    total_cost = 0.0
    for sid, _ in scripts:
        if sid in failures:
            table.add_row(sid, "[red]FAILED[/red]", "-", "-", "-", "-")
            continue
        r = results[sid]
        if r.status == "extracted":
            extracted += 1
        elif r.status == "cached":
            cached += 1
        else:
            skipped += 1
        total_cost += r.total_cost_usd
        color = status_color.get(r.status, "white")
        table.add_row(
            sid,
            f"[{color}]{r.status}[/{color}]",
            str(r.answers_found),
            ",".join(r.questions_not_found) or "-",
            f"${r.total_cost_usd:.4f}",
            f"{r.wall_time_s:.1f}s",
        )

    console.print(table)

    attempted = len(scripts)
    failed = len(failures)
    reconciled = extracted + cached + skipped + failed
    console.print(
        f"\nattempted={attempted}  extracted={extracted}  cached={cached}  "
        f"skipped={skipped}  failed={failed}  total_cost=${total_cost:.4f}  "
        f"(sum={reconciled}, {'OK' if reconciled == attempted else 'MISMATCH'})"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
