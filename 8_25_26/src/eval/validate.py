"""Evaluation validation — assert and report, never auto-fix.

Validates the JSON records in output/evaluations/ (richer than the CSV) plus a
max_mark/extraction_status cross-check against extraction.csv. Reports the
within-model score range (the noise floor) so a cross-model gap can be judged
significant, the evidence_not_found rate per model, per-question means, and where
the two graders disagree most. Writes logs/evaluation_report.md.
"""

from __future__ import annotations

import csv
import itertools
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from src.config import EvalConfig
from src.eval.models import CAP_VALUES, FOUR_CRITERIA, SUBSCORE_CEILINGS, band_for

TOL = 1e-6


@dataclass
class Check:
    level: str   # PASS | FAIL | WARN
    message: str


@dataclass
class ValidationOutput:
    checks: list[Check] = field(default_factory=list)
    report_path: Path | None = None

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.level == "FAIL")


def _add(checks: list[Check], ok: bool, msg: str, warn: bool = False) -> None:
    checks.append(Check("PASS" if ok else ("WARN" if warn else "FAIL"), msg))


def load_records(cfg: EvalConfig) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(cfg.paths.evaluations.glob("*.json"))]


def load_extraction_map(cfg: EvalConfig) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with cfg.paths.extraction_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[r["task_id"]] = r
    return out


def _total(rec: dict) -> float:
    return float(rec["score_breakdown"]["total_score"])


def _key(rec: dict) -> tuple[str, str, int]:
    m = rec["_metadata"]
    return (m["task_id"], m["model"], int(m["run_index"]))


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
def run_assertions(records: list[dict], extraction: dict[str, dict]) -> list[Check]:
    checks: list[Check] = []

    keys = [_key(r) for r in records]
    _add(checks, len(keys) == len(set(keys)),
         f"every (task_id, model, run) present exactly once ({len(keys)} records)")

    task_ids = sorted({r["_metadata"]["task_id"] for r in records})
    models = sorted({r["_metadata"]["model"] for r in records})
    runs = sorted({int(r["_metadata"]["run_index"]) for r in records})
    expected = len(task_ids) * len(models) * len(runs)
    grid = set(itertools.product(task_ids, models, runs))
    missing = grid - set(keys)
    _add(checks, len(records) == expected and not missing,
         f"row count == tasks({len(task_ids)}) x models({len(models)}) x k({len(runs)}) "
         f"= {expected}; got {len(records)}"
         + (f"; MISSING {sorted(missing)[:5]}" if missing else ""))

    # per-row structural checks
    sum_ok = ceil_ok = total_ok = band_ok = maxmark_ok = True
    for r in records:
        m = r["_metadata"]
        qtype = r["task_type"]
        raw = r["raw_subscores"]
        sb = r["score_breakdown"]
        max_mark = int(m["max_mark"])

        raw_sum = sum(float(raw[c]) for c in FOUR_CRITERIA)
        if abs(raw_sum - float(raw["raw_total"])) > TOL:
            sum_ok = False
        ceilings = SUBSCORE_CEILINGS.get(qtype, {})
        for c in FOUR_CRITERIA:
            cap = ceilings.get(c, 99)
            if float(raw[c]) > cap + TOL or float(sb[c]) > cap + TOL:
                ceil_ok = False
        total = float(sb["total_score"])
        cap_val = CAP_VALUES.get(r["structural_audit"]["applied_cap_reason"])
        limit = min(max_mark, cap_val) if cap_val is not None else max_mark
        if total > limit + TOL:
            total_ok = False
        if r["performance_band"] != band_for(round(total), max_mark):
            band_ok = False
        if int(r["max_mark_applied"]) != max_mark:
            maxmark_ok = False

    _add(checks, sum_ok, "raw_subscores sum == raw_total on every row")
    _add(checks, ceil_ok, "every sub-score within its per-task ceiling")
    _add(checks, total_ok, "total_score <= max_mark and <= applied cap")
    _add(checks, band_ok, "performance_band contains total_score for its max_mark")
    _add(checks, maxmark_ok, "max_mark_applied == extraction.csv max_mark")

    # cross-check extraction.csv max_mark directly
    xmark_ok = all(
        tid in extraction and int(extraction[tid]["max_mark"]) == int(r["_metadata"]["max_mark"])
        for r in records for tid in [r["_metadata"]["task_id"]]
    )
    _add(checks, xmark_ok, "metadata max_mark matches extraction.csv per task")

    # global-identical fields
    rubric_hashes = {r["_metadata"]["rubric_hash"] for r in records}
    think = {r["_metadata"]["thinking_level"] for r in records}
    _add(checks, len(rubric_hashes) == 1, f"rubric_hash identical across all rows ({len(rubric_hashes)} distinct)")
    _add(checks, len(think) == 1, f"thinking_level identical across all rows ({sorted(think)})")

    # prompt_hash identical across models for same (task_id, run)
    by_tr: dict[tuple[str, int], set[str]] = {}
    for r in records:
        m = r["_metadata"]
        by_tr.setdefault((m["task_id"], int(m["run_index"])), set()).add(m["prompt_hash"])
    prompt_ok = all(len(v) == 1 for v in by_tr.values())
    _add(checks, prompt_ok, "prompt_hash identical across models for same (task_id, run)")

    # synthesised rows
    synth = [r for r in records if r["_metadata"].get("synthesised")]
    synth_ok = all(
        _total(r) == 0 and r["performance_band"] == "Band 0"
        and float(r["_metadata"].get("cost_usd", 0)) == 0
        and not r["_metadata"].get("calls")
        for r in synth
    )
    _add(checks, synth_ok, f"synthesised rows: total 0, Band 0, no API cost ({len(synth)} rows)")

    return checks


# ---------------------------------------------------------------------------
# Analysis (noise floor, evidence rate, means, disagreement)
# ---------------------------------------------------------------------------
def _group_scores(records: list[dict]) -> dict[tuple[str, str], list[float]]:
    """(task_id, model) -> list of total_score across runs."""
    g: dict[tuple[str, str], list[float]] = {}
    for r in records:
        m = r["_metadata"]
        g.setdefault((m["task_id"], m["model"]), []).append(_total(r))
    return g


def analyse(records: list[dict]) -> dict:
    models = sorted({r["_metadata"]["model"] for r in records})
    grouped = _group_scores(records)

    # within-model range (noise floor)
    ranges: dict[str, list[float]] = {mdl: [] for mdl in models}
    per_task_model_mean: dict[str, dict[str, float]] = {}
    for (tid, mdl), scores in grouped.items():
        ranges[mdl].append(max(scores) - min(scores))
        per_task_model_mean.setdefault(tid, {})[mdl] = statistics.mean(scores)

    # evidence_not_found rate per model (graded rows only)
    evid: dict[str, list[int]] = {mdl: [] for mdl in models}
    for r in records:
        m = r["_metadata"]
        if not m.get("synthesised"):
            evid[m["model"]].append(1 if m.get("evidence_not_found") else 0)

    # per-question mean by model
    per_q: dict[str, dict[str, list[float]]] = {}
    for r in records:
        m = r["_metadata"]
        per_q.setdefault(m["question_no"], {}).setdefault(m["model"], []).append(_total(r))

    # cross-model gap per task vs within-model noise
    disagreement = []
    for tid, means in per_task_model_mean.items():
        if len(means) >= 2:
            vals = list(means.values())
            gap = max(vals) - min(vals)
            noise = max((max(grouped[(tid, mdl)]) - min(grouped[(tid, mdl)]))
                        for mdl in means)
            disagreement.append((tid, gap, noise, means))
    disagreement.sort(key=lambda x: -x[1])

    return {
        "models": models,
        "ranges": ranges,
        "evid": evid,
        "per_q": per_q,
        "disagreement": disagreement,
    }


def build_report(checks: list[Check], analysis: dict, records: list[dict]) -> str:
    L: list[str] = ["# Evaluation Validation Report\n"]
    failed = sum(1 for c in checks if c.level == "FAIL")
    warned = sum(1 for c in checks if c.level == "WARN")
    L.append(f"Records: **{len(records)}**  ·  Checks failed: **{failed}**  ·  warnings: **{warned}**\n")

    L.append("## Assertions\n")
    for c in checks:
        mark = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[c.level]
        L.append(f"- {mark} {c.message}")

    models = analysis["models"]

    L.append("\n## Noise floor — within-model score range across runs\n")
    L.append("| model | mean range | max range | tasks |")
    L.append("|---|---|---|---|")
    for mdl in models:
        rs = analysis["ranges"][mdl]
        mean_r = round(statistics.mean(rs), 3) if rs else 0
        max_r = max(rs) if rs else 0
        L.append(f"| {mdl} | {mean_r} | {max_r} | {len(rs)} |")
    L.append("\n_If within-model range routinely exceeds the cross-model gap below, "
             "raise k before making any divergence claim._")

    L.append("\n## Evidence-not-found rate (hallucination indicator)\n")
    L.append("| model | rate | graded rows |")
    L.append("|---|---|---|")
    for mdl in models:
        e = analysis["evid"][mdl]
        rate = f"{sum(e)}/{len(e)}" if e else "0/0"
        pct = f"{(sum(e)/len(e)*100):.0f}%" if e else "n/a"
        L.append(f"| {mdl} | {pct} ({rate}) | {len(e)} |")

    L.append("\n## Per-question mean score by model\n")
    header = "| question | " + " | ".join(models) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(models) + 1))
    for q in sorted(analysis["per_q"], key=lambda x: (len(x), x)):
        cells = []
        for mdl in models:
            vals = analysis["per_q"][q].get(mdl, [])
            cells.append(f"{statistics.mean(vals):.2f}" if vals else "-")
        L.append(f"| Q{q} | " + " | ".join(cells) + " |")

    L.append("\n## Where the two graders disagree most (cross-model gap vs noise)\n")
    L.append("| task_id | cross-model gap | within-model noise | means |")
    L.append("|---|---|---|---|")
    for tid, gap, noise, means in analysis["disagreement"]:
        flag = "  ⚠️ noise≥gap" if noise >= gap and gap > 0 else ""
        means_s = ", ".join(f"{k}={v:.2f}" for k, v in means.items())
        L.append(f"| {tid} | {gap:.2f} | {noise:.2f} | {means_s}{flag} |")

    return "\n".join(L) + "\n"


def validate_all(cfg: EvalConfig) -> ValidationOutput:
    records = load_records(cfg)
    if not records:
        out = ValidationOutput()
        out.checks.append(Check("FAIL", "no evaluation records found"))
        return out
    extraction = load_extraction_map(cfg)
    checks = run_assertions(records, extraction)
    analysis = analyse(records)
    report = build_report(checks, analysis, records)
    report_path = cfg.paths.logs / "evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    out = ValidationOutput(checks=checks, report_path=report_path)
    return out
