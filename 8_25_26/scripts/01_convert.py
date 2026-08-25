"""CLI: PDF -> images (Stage 1).

Flags: --force, --limit N, --script-id X, --config PATH.

Converts scanned exam PDFs in data/raw_pdfs/ to deterministic per-page PNGs at
the DPI configured in config.yaml. Resumable (skips complete scripts unless
--force). Fails loudly: a script that errors is recorded FAILED, and the final
count reconciles attempted = converted + skipped + failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/01_convert.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.config import load_config
from src.pdf_to_images import (
    ConversionError,
    ScriptResult,
    convert_script,
    discover_scripts,
)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _dimensions(result: ScriptResult) -> str:
    dims = {(p.width, p.height) for p in result.pages if p.width and p.height}
    if not dims:  # skipped scripts don't reopen pages, so dims are unknown
        return "-"
    if len(dims) == 1:
        w, h = next(iter(dims))
        return f"{w}x{h}"
    return f"{len(dims)} sizes (varies)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert exam PDFs to page PNGs.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Re-render even if outputs exist")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N scripts")
    parser.add_argument("--script-id", default=None, help="Only process this script id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

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
        console.print(f"[yellow]No PDFs found in {cfg.paths.raw_pdfs}[/yellow]")
        sys.exit(1)

    table = Table(title="Stage 1 - PDF to images")
    table.add_column("script_id")
    table.add_column("pages", justify="right")
    table.add_column("dimensions")
    table.add_column("total size", justify="right")
    table.add_column("status")

    converted = skipped = failed = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting", total=len(scripts))
        for script_id, pdf_path in scripts:
            try:
                result = convert_script(script_id, pdf_path, cfg, force=args.force)
            except ConversionError as exc:
                failed += 1
                table.add_row(script_id, "-", "-", "-", "[red]FAILED[/red]")
                console.print(f"[red]{exc}[/red]")
                progress.advance(task)
                continue

            total_bytes = sum(p.size_bytes for p in result.pages)
            if result.skipped:
                skipped += 1
                status = "[yellow]skipped[/yellow]"
            else:
                converted += 1
                status = "[green]converted[/green]"
            table.add_row(
                script_id,
                str(result.page_count),
                _dimensions(result),
                _human_size(total_bytes),
                status,
            )
            progress.advance(task)

    console.print(table)

    attempted = len(scripts)
    reconciled = converted + skipped + failed
    console.print(
        f"\nattempted={attempted}  converted={converted}  "
        f"skipped={skipped}  failed={failed}  "
        f"(sum={reconciled}, {'OK' if reconciled == attempted else 'MISMATCH'})"
    )

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
