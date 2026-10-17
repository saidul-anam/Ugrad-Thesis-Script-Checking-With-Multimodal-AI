#!/usr/bin/env python3
"""
Check that each paper's question paper, rubric, answer key and penalty profile agree.

For every paper it resolves the same files Stage 4 would use and reports:
  ERROR    Stage 4 would mis-score or cannot score (marks that do not add up, a band table with a gap,
           a Mode A question without a key, key items that do not match the rubric, ...)
  WARNING  worth a human look (unverified key, Mode B without reference answers, printed marks that
           differ from the parsed paper, task types named differently, ...)

Exit code 1 when any paper has an ERROR. The checks themselves are in src/utils/rubric_lint.py.

Usage:
  python scripts/lint_rubrics.py                      # every paper with an answer key
  python scripts/lint_rubrics.py --papers SE_07_Q1 SE_10_Q1
  python scripts/lint_rubrics.py --errors-only
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.rubric_lint import ERROR, lint_repo_paper


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", nargs="*", help="paper IDs (default: every configs/answer_keys/SE_*.yaml)")
    ap.add_argument("--errors-only", action="store_true")
    args = ap.parse_args()

    papers = args.papers or sorted(p.stem for p in Path("configs/answer_keys").glob("SE_*.yaml"))
    n_errors = 0
    for paper in papers:
        issues = lint_repo_paper(paper)
        errors = [i for i in issues if i.level == ERROR]
        n_errors += len(errors)
        shown = errors if args.errors_only else issues
        print(f"{paper}: {len(errors)} error(s), {len(issues) - len(errors)} warning(s)")
        for i in shown:
            q = f"Q{i.question:6}" if i.question else " " * 7
            print(f"  {i.level:7} {q} {i.message}")
    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    main()
