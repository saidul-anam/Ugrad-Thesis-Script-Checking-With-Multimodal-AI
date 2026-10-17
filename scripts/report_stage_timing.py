#!/usr/bin/env python3
"""
Where extraction time goes: sums metadata.stage_timing (written by the orchestrator since October 2026) over
extraction outputs and prints each stage's share of time, model calls and tokens.

Usage:
  python scripts/report_stage_timing.py outputs/extracted/english/se_10_q1
  python scripts/report_stage_timing.py outputs/benchmarks/pages/baseline_b outputs/benchmarks/pages/after
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.stage_timer import merge_timings


def collect(folder: Path):
    timings, pages, scripts = [], 0, 0
    for f in sorted(folder.rglob("extraction_result.json")):
        t = (json.load(open(f, encoding="utf-8")).get("metadata") or {}).get("stage_timing")
        if not t:
            continue
        scripts += 1
        pages += int(t.get("pages_processed", 0))
        timings.append({k: v for k, v in t.items() if isinstance(v, dict)})
    return merge_timings(timings), pages, scripts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    args = ap.parse_args()
    for folder in args.folders:
        summary, pages, scripts = collect(Path(folder))
        total = summary.get("total", {})
        print(f"\n## {folder}  ({scripts} scripts with timing, {pages} pages processed)")
        if not scripts:
            print("no stage_timing found (outputs written before the timers were added)")
            continue
        print(f"| Stage | Seconds | Share | s/page | Calls | Prompt tok | Completion tok |")
        print(f"| --- | --- | --- | --- | --- | --- | --- |")
        for stage, row in summary.items():
            share = row["seconds"] / total["seconds"] if total.get("seconds") else 0.0
            per_page = row["seconds"] / pages if pages else 0.0
            print(f"| {stage} | {row['seconds']:.0f} | {share:.0%} | {per_page:.1f} | {row['calls']:.0f} | "
                  f"{row['prompt_tokens']:.0f} | {row['completion_tokens']:.0f} |")


if __name__ == "__main__":
    main()
