#!/usr/bin/env python3
"""
Repair LaTeX backslash escapes and formatting glitches in checkpoint JSON files.
Audits historical extractions in outputs/extracted/ to ensure LaTeX symbols
such as $\rightarrow$ were not corrupted into 'ightarrow' or '\\r' carriage returns.

Usage:
  python scripts/repair_checkpoints.py
  python scripts/repair_checkpoints.py --dry-run
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Tuple


# Regex for corrupted rightarrow variants:
# 1. Literal carriage return '\r' followed by 'ightarrow'
# 2. 'ightarrow' preceded by start of line, whitespace, or math delimiter without 'r'
# 3. '$\rightarrow$' missing backslash like '$rightarrow$'
_CORRUPT_PATTERNS = [
    (re.compile(r"\r\s*ightarrow"), r"$\\rightarrow$"),
    (re.compile(r"(?<![a-zA-Z0-9\$\\])ightarrow\b"), r"$\\rightarrow$"),
    (re.compile(r"\$rightarrow\$"), r"$\\rightarrow$"),
]


def repair_text(text: str) -> Tuple[str, int]:
    """Inspect and repair corrupted LaTeX sequences in a string."""
    if not isinstance(text, str):
        return text, 0
    total_fixes = 0
    current = text
    for pattern, repl in _CORRUPT_PATTERNS:
        matches = len(pattern.findall(current))
        if matches > 0:
            current = pattern.sub(repl, current)
            total_fixes += matches
    return current, total_fixes


def repair_obj(obj: Any) -> Tuple[Any, int]:
    """Recursively repair strings within JSON structures."""
    total_fixes = 0
    if isinstance(obj, str):
        return repair_text(obj)
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            fixed_v, count = repair_obj(v)
            new_dict[k] = fixed_v
            total_fixes += count
        return new_dict, total_fixes
    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            fixed_item, count = repair_obj(item)
            new_list.append(fixed_item)
            total_fixes += count
        return new_list, total_fixes
    return obj, 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair LaTeX math escape glitches in checkpoints.")
    ap.add_argument("--dir", default="outputs/extracted", help="Root extracted directory to scan")
    ap.add_argument("--dry-run", action="store_true", help="Audit only without writing changes")
    args = ap.parse_args()

    root_dir = Path(args.dir)
    if not root_dir.exists():
        print(f"Directory {root_dir} does not exist.")
        sys.exit(1)

    json_files = sorted(root_dir.glob("**/*.json"))
    scanned = 0
    repaired_files = 0
    total_fixes = 0

    print(f"Scanning {len(json_files)} JSON files in {root_dir}...")
    for p in json_files:
        scanned += 1
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] Could not parse {p}: {e}")
            continue

        fixed_data, count = repair_obj(data)
        if count > 0:
            repaired_files += 1
            total_fixes += count
            print(f"  [REPAIRED] {p.relative_to(root_dir)}: {count} glitch(es) fixed")
            if not args.dry_run:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(fixed_data, f, ensure_ascii=False, indent=2)

    print("\n=== Checkpoint LaTeX Repair Summary ===")
    print(f"  Total JSON files scanned:  {scanned}")
    print(f"  Files requiring repairs:   {repaired_files}")
    print(f"  Total escape glitches fixed: {total_fixes}")
    if args.dry_run:
        print("  (Dry-run mode: no files were modified on disk)")


if __name__ == "__main__":
    main()
