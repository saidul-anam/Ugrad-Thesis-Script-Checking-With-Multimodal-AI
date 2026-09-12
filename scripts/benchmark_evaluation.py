#!/usr/bin/env python3
"""
Cross-script Stage 4 benchmark against human examiner marks (gt.txt).

Reads every outputs/evaluated/<lang>/<script>/stage4_evaluation.json (or a custom --eval-dir),
joins the per-question awarded marks with the ground truth and reports:

  - MAE over scored questions, MAE counting missing/unscored questions as 0
  - MAE per scoring mode (A_ITEM / B_POINT / C_BAND / generic) and per question number
  - Quadratic weighted kappa and Pearson r on half-mark categories
  - Baseline: predict the per-question median of the OTHER scripts (leave-one-script-out)
  - Bootstrap 95% CI of the MAE (resampling scripts)
  - Counts of missing / unscored questions and of hard caps applied

Two evaluation directories can be compared side by side (--compare), e.g. modular vs monolithic,
or evidence-gate vs legacy-gate extractions.

Usage:
  python scripts/benchmark_evaluation.py --lang english
  python scripts/benchmark_evaluation.py --eval-dir outputs/evaluated/english --compare outputs/evaluated_legacy/english --tag gate_ablation
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.ground_truth import parse_ground_truth_file, canonicalize_question_key


def load_rows(eval_dir: Path, gt_all: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s4_path in sorted(eval_dir.glob("*/stage4_evaluation.json")):
        script_id = s4_path.parent.name
        try:
            data = json.load(open(s4_path, "r", encoding="utf-8"))
        except Exception:
            continue
        gt = {canonicalize_question_key(k): float(v) for k, v in (gt_all.get(script_id.upper()) or {}).items()}
        for qe in data.get("question_evaluations", []):
            q = canonicalize_question_key(str(qe.get("q_no", "")))
            rows.append({
                "script_id": script_id, "q_no": q, "mode": qe.get("task_mode") or "generic",
                "status": qe.get("scoring_status", "scored"), "awarded": float(qe.get("awarded_marks", 0.0)),
                "max": float(qe.get("max_marks", 0.0)), "gt": gt.get(q), "cap": bool(qe.get("cap_applied", False)),
                "eval_mode": data.get("eval_mode", "modular"),
            })
    return rows


def mae(pairs: List[Tuple[float, float]]) -> Optional[float]:
    return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else None


def pearson(pairs: List[Tuple[float, float]]) -> Optional[float]:
    n = len(pairs)
    if n < 3:
        return None
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def quadratic_weighted_kappa(pairs: List[Tuple[float, float]], step: float = 0.5) -> Optional[float]:
    """QWK on categories = round(mark / step); works across questions with different maxima."""
    if len(pairs) < 3:
        return None
    a = [int(round(p / step)) for p, _ in pairs]
    b = [int(round(g / step)) for _, g in pairs]
    k = max(max(a), max(b)) + 1
    n = len(a)
    O = [[0] * k for _ in range(k)]
    for i, j in zip(a, b):
        O[i][j] += 1
    ha = [sum(1 for x in a if x == i) for i in range(k)]
    hb = [sum(1 for x in b if x == j) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0.0
            num += w * O[i][j]
            den += w * ha[i] * hb[j] / n
    return 1.0 - num / den if den > 0 else None


def summarize(rows: List[Dict[str, Any]], seed: int = 0) -> Dict[str, Any]:
    with_gt = [r for r in rows if r["gt"] is not None]
    scored = [r for r in with_gt if r["status"] == "scored"]
    pairs_scored = [(r["awarded"], r["gt"]) for r in scored]
    pairs_all = [(r["awarded"] if r["status"] == "scored" else 0.0, r["gt"]) for r in with_gt]
    scripts = sorted({r["script_id"] for r in rows})

    per_mode: Dict[str, Any] = {}
    for m in sorted({r["mode"] for r in scored}):
        prs = [(r["awarded"], r["gt"]) for r in scored if r["mode"] == m]
        per_mode[m] = {"n": len(prs), "mae": mae(prs)}
    per_q: Dict[str, Any] = {}
    for q in sorted({r["q_no"] for r in scored}, key=lambda x: (len(x), x)):
        prs = [(r["awarded"], r["gt"]) for r in scored if r["q_no"] == q]
        bias = sum(a - g for a, g in prs) / len(prs) if prs else None
        per_q[q] = {"n": len(prs), "mae": mae(prs), "bias_ai_minus_human": bias}

    # leave-one-script-out median baseline
    base_pairs = []
    for r in scored:
        others = [x["gt"] for x in with_gt if x["q_no"] == r["q_no"] and x["script_id"] != r["script_id"]]
        if others:
            others.sort()
            med = others[len(others) // 2] if len(others) % 2 else (others[len(others) // 2 - 1] + others[len(others) // 2]) / 2
            base_pairs.append((med, r["gt"]))
    # bootstrap over scripts
    rng = random.Random(seed)
    boots = []
    by_script = defaultdict(list)
    for r in scored:
        by_script[r["script_id"]].append((r["awarded"], r["gt"]))
    keys = list(by_script)
    if len(keys) >= 2:
        for _ in range(1000):
            sample = [p for k in (rng.choice(keys) for _ in keys) for p in by_script[k]]
            m = mae(sample)
            if m is not None:
                boots.append(m)
        boots.sort()
    ci = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if len(boots) >= 40 else None

    return {
        "scripts": scripts, "n_scripts": len(scripts), "n_questions": len(rows), "n_with_gt": len(with_gt),
        "n_scored_with_gt": len(scored),
        "n_missing": sum(1 for r in rows if r["status"] == "missing"),
        "n_unscored": sum(1 for r in rows if r["status"] == "unscored"),
        "n_caps_applied": sum(1 for r in rows if r["cap"]),
        "mae_scored": mae(pairs_scored), "mae_including_missing": mae(pairs_all),
        "mae_ci95_bootstrap_scripts": ci,
        "pearson_r": pearson(pairs_scored), "qwk_half_marks": quadratic_weighted_kappa(pairs_scored),
        "exact_match_rate": (sum(1 for a, g in pairs_scored if abs(a - g) < 0.01) / len(pairs_scored)) if pairs_scored else None,
        "within_1_mark_rate": (sum(1 for a, g in pairs_scored if abs(a - g) <= 1.0) / len(pairs_scored)) if pairs_scored else None,
        "baseline_loso_median_mae": mae(base_pairs), "baseline_n": len(base_pairs),
        "total_awarded_vs_gt": [(s, sum(r["awarded"] for r in scored if r["script_id"] == s), sum(r["gt"] for r in with_gt if r["script_id"] == s)) for s in scripts],
        "per_mode": per_mode, "per_question": per_q,
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, tuple):
        return f"[{v[0]:.3f}, {v[1]:.3f}]"
    return str(v)


def render_md(title: str, s: Dict[str, Any]) -> List[str]:
    md = [f"## {title}", "",
          f"- Scripts: {s['n_scripts']} ({', '.join(s['scripts'])}) | question rows: {s['n_questions']} | with GT: {s['n_with_gt']} | scored with GT: {s['n_scored_with_gt']}",
          f"- Missing segments: {s['n_missing']} | unscored: {s['n_unscored']} | hard caps applied: {s['n_caps_applied']}",
          "", "| Metric | Value |", "| --- | --- |",
          f"| MAE (scored) | {_fmt(s['mae_scored'])} |",
          f"| MAE incl. missing/unscored as 0 | {_fmt(s['mae_including_missing'])} |",
          f"| MAE 95% CI (bootstrap over scripts) | {_fmt(s['mae_ci95_bootstrap_scripts'])} |",
          f"| Baseline MAE (leave-one-script-out median, n={s['baseline_n']}) | {_fmt(s['baseline_loso_median_mae'])} |",
          f"| Pearson r | {_fmt(s['pearson_r'])} |",
          f"| Quadratic weighted kappa (0.5-mark categories) | {_fmt(s['qwk_half_marks'])} |",
          f"| Exact match rate | {_fmt(s['exact_match_rate'])} |",
          f"| Within 1 mark rate | {_fmt(s['within_1_mark_rate'])} |",
          "", "### Per mode", "| Mode | n | MAE |", "| --- | --- | --- |"]
    for m, v in s["per_mode"].items():
        md.append(f"| {m} | {v['n']} | {_fmt(v['mae'])} |")
    md += ["", "### Per question", "| Q | n | MAE | Bias (AI − human) |", "| --- | --- | --- | --- |"]
    for q, v in s["per_question"].items():
        md.append(f"| {q} | {v['n']} | {_fmt(v['mae'])} | {_fmt(v['bias_ai_minus_human'])} |")
    md += ["", "### Script totals (scored questions)", "| Script | AI total | Human total (same questions) |", "| --- | --- | --- |"]
    for sid, ai, hu in s["total_awarded_vs_gt"]:
        md.append(f"| {sid} | {ai:.1f} | {hu:.1f} |")
    md.append("")
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-script Stage 4 benchmark vs gt.txt")
    ap.add_argument("--lang", default="english")
    ap.add_argument("--eval-dir", default=None, help="default: outputs/evaluated/<lang>")
    ap.add_argument("--compare", default=None, help="second evaluation dir to report side by side")
    ap.add_argument("--gt", default="gt.txt")
    ap.add_argument("--out-dir", default="outputs/benchmarks")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gt_all = parse_ground_truth_file(args.gt)
    if not gt_all:
        print(f"No ground truth parsed from {args.gt}")
        sys.exit(1)
    eval_dir = Path(args.eval_dir or f"outputs/evaluated/{args.lang}")
    rows = load_rows(eval_dir, gt_all)
    if not rows:
        print(f"No stage4_evaluation.json under {eval_dir}. Run scripts/evaluate_scripts.py first.")
        sys.exit(1)
    summary = summarize(rows)
    md = [f"# Stage 4 Benchmark vs Human Examiner ({args.lang}{(' / ' + args.tag) if args.tag else ''})",
          f"Generated {datetime.now().isoformat(timespec='seconds')} | GT scripts available: {len(gt_all)}", ""]
    md += render_md(f"Run A: {eval_dir}", summary)
    out = {"run_a": {"dir": str(eval_dir), "summary": summary, "rows": rows}}
    if args.compare:
        rows_b = load_rows(Path(args.compare), gt_all)
        if rows_b:
            summary_b = summarize(rows_b)
            md += render_md(f"Run B: {args.compare}", summary_b)
            out["run_b"] = {"dir": args.compare, "summary": summary_b, "rows": rows_b}
            md += ["## A vs B", "| Metric | A | B |", "| --- | --- | --- |"]
            for k in ("mae_scored", "mae_including_missing", "pearson_r", "qwk_half_marks", "within_1_mark_rate", "n_missing", "n_unscored"):
                md.append(f"| {k} | {_fmt(summary[k])} | {_fmt(summary_b[k])} |")
            md.append("")

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.out_dir) / f"evaluation_{args.lang}{('_' + args.tag) if args.tag else ''}_{stamp}"
    with open(str(base) + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    with open(str(base) + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"Written: {base}.md / .json")


if __name__ == "__main__":
    main()
