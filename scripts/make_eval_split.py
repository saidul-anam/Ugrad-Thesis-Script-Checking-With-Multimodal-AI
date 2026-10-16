#!/usr/bin/env python3
"""
Fixed dev/test split of evaluated scripts, by script and per paper.

Calibration choices for Stage 4 (answer-key conventions, rearrangement scoring, caps) are made on the
dev scripts only; the test scripts are scored once at the end. Students are the unit, so a script is
never split across dev and test.

The split is seeded and written to configs/splits/eval_split.csv (tracked in git). Re-running with the
same seed and the same teacher-mark files reproduces it exactly; existing assignments are never
reshuffled unless --overwrite is given.

Usage:
  python scripts/make_eval_split.py                  # 70% dev / 30% test per paper, seed 20261016
  python scripts/make_eval_split.py --test-frac 0.3 --seed 20261016 --overwrite
"""

import argparse
import csv
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.ground_truth import load_all_teacher_marks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="configs/splits/eval_split.csv")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        print(f"{args.out} exists; not reshuffling (use --overwrite).")
        return

    by_paper = defaultdict(list)
    for sid in sorted(load_all_teacher_marks()):
        m = re.match(r"^([A-Z]+_\d+_Q\d+)_\d+$", sid)
        if m:
            by_paper[m.group(1)].append(sid)

    rows = []
    for paper in sorted(by_paper):
        sids = sorted(by_paper[paper])
        rng = random.Random(f"{args.seed}:{paper}")
        shuffled = sids[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(sids) * args.test_frac)) if len(sids) >= 4 else 0
        test = set(shuffled[:n_test])
        for sid in sids:
            rows.append({"script_id": sid, "paper": paper, "split": "test" if sid in test else "dev"})
        print(f"{paper}: {len(sids) - n_test} dev / {n_test} test")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["script_id", "paper", "split"])
        w.writeheader()
        w.writerows(rows)
    print(f"Written {args.out} ({len(rows)} scripts, seed {args.seed})")


if __name__ == "__main__":
    main()
