"""CLI: emit output/evaluation_avg.csv — one row per (task_id x model),
averaging the k runs (with per-run spread + evidence rate). Derived; regenerable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console

from src.config import load_eval_config
from src.eval.emit_avg import emit_averaged


def main() -> None:
    p = argparse.ArgumentParser(description="Emit averaged (per task x model) eval CSV.")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    console = Console()
    cfg = load_eval_config(args.config)
    path, n = emit_averaged(cfg)
    console.print(f"[green]wrote[/green] {path}  ({n} rows)")


if __name__ == "__main__":
    main()
