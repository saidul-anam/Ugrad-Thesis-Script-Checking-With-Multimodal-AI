#!/usr/bin/env python3
"""
Evaluate and (optionally) fit the Stage 3b ambiguity gate against human labels. No GPU needed:
every record stores its raw signals, so re-fusion with different weights is instantaneous.

Reports:
  - confusion of verdict vs label, precision/recall/F1 for GENUINE and AMBIGUITY, coverage of UNCERTAIN
  - Expected Calibration Error of ambiguity_score vs P(label == A)
  - per-signal ablation (each signal zeroed in turn)
  - leave-one-script-out: weights fitted on N-1 writers, tested on the held-out writer
  - with --fit: fitted weights + thresholds printed as a YAML snippet for configs/pipeline_config.yaml

Labels: G = genuine, A = ambiguity, U = cannot tell (U rows are excluded from fitting and metrics).

Usage:
  python scripts/evaluate_arbitration.py --labels data/labels/arbitration_labels.csv
  python scripts/evaluate_arbitration.py --labels data/labels/arbitration_labels.csv --fit --write-config
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_config, ArbitrationWeights
from src.pipeline.arbitration.fusion import (
    fuse_signals, quantize, fit_weights, choose_thresholds, SIGNAL_NAMES, GENUINE, AMBIGUITY, UNCERTAIN,
)


def _f(v) -> Optional[float]:
    try:
        if v is None or v == "" or str(v).lower() == "none":
            return None
        return float(v)
    except Exception:
        return None


def load_rows(labels_csv: str, extracted_dir: Optional[str]) -> List[Dict[str, Any]]:
    """Join labels with the stored raw signals (prefer the JSON records; CSV signals as fallback)."""
    rows: List[Dict[str, Any]] = []
    signal_cache: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    with open(labels_csv, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            lab = (r.get("human_label") or "").strip().upper()[:1]
            if lab not in ("G", "A", "U"):
                continue
            sid, cid = r["script_id"], r["candidate_id"]
            sig: Optional[Dict[str, Optional[float]]] = None
            if extracted_dir:
                if sid not in signal_cache:
                    signal_cache[sid] = {}
                    p = Path(extracted_dir) / sid / "stage3b_arbitration.json"
                    if p.exists():
                        for rec in json.load(open(p, "r", encoding="utf-8")).get("records", []):
                            e = rec["evidence"]
                            signal_cache[sid][rec["candidate"]["candidate_id"]] = {
                                "phonetic": e.get("phonetic_signal"), "writer": e.get("writer_prior"),
                                "consensus": e.get("consensus_signal"), "forced_choice": e.get("forced_choice_signal"),
                            }
                sig = signal_cache[sid].get(cid)
            if sig is None:
                sig = {k: _f(r.get(k)) for k in SIGNAL_NAMES}
            rows.append({"script_id": sid, "candidate_id": cid, "label": lab, "signals": sig,
                         "read": r.get("read_token"), "intended": r.get("intended_token")})
    return rows


def metrics(rows: List[Dict[str, Any]], w: ArbitrationWeights, thr_amb: float, thr_gen: float) -> Dict[str, Any]:
    conf = defaultdict(int)
    scores, labels = [], []
    for r in rows:
        if r["label"] == "U":
            continue
        s = fuse_signals(r["signals"], w)
        v = quantize(s, thr_amb, thr_gen)
        conf[(r["label"], v)] += 1
        scores.append(s)
        labels.append(1 if r["label"] == "A" else 0)
    n = len(scores)

    def prf(verdict: str, pos_label: str) -> Tuple[float, float, float]:
        tp = conf[(pos_label, verdict)]
        fp = sum(conf[(l, verdict)] for l in ("G", "A") if l != pos_label)
        fn = sum(conf[(pos_label, v)] for v in (GENUINE, AMBIGUITY, UNCERTAIN) if v != verdict)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        return p, rc, f1

    # ECE
    bins = [[] for _ in range(10)]
    for s, l in zip(scores, labels):
        bins[min(9, int(s * 10))].append((s, l))
    ece = 0.0
    for b in bins:
        if b:
            conf_mean = sum(s for s, _ in b) / len(b)
            acc = sum(l for _, l in b) / len(b)
            ece += len(b) / max(1, n) * abs(conf_mean - acc)
    wrong_deductions = conf[("A", GENUINE)]           # ambiguity penalised (the failure we care about most)
    missed_genuine = conf[("G", AMBIGUITY)]           # genuine error forgiven
    return {
        "n": n,
        "genuine_precision_recall_f1": prf(GENUINE, "G"),
        "ambiguity_precision_recall_f1": prf(AMBIGUITY, "A"),
        "uncertain_share": (sum(conf[(l, UNCERTAIN)] for l in ("G", "A")) / n) if n else 0.0,
        "ambiguity_penalised": wrong_deductions,
        "genuine_forgiven": missed_genuine,
        "ece": ece,
        "confusion": {f"{l}->{v}": c for (l, v), c in sorted(conf.items())},
    }


def _print_metrics(title: str, m: Dict[str, Any]) -> None:
    gp, gr, gf = m["genuine_precision_recall_f1"]
    ap_, ar, af = m["ambiguity_precision_recall_f1"]
    print(f"\n== {title} (n={m['n']}) ==")
    print(f"  GENUINE   P={gp:.2f} R={gr:.2f} F1={gf:.2f}")
    print(f"  AMBIGUITY P={ap_:.2f} R={ar:.2f} F1={af:.2f}")
    print(f"  UNCERTAIN share={m['uncertain_share']:.2f} | ambiguity penalised={m['ambiguity_penalised']} | genuine forgiven={m['genuine_forgiven']} | ECE={m['ece']:.3f}")
    print(f"  confusion: {m['confusion']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate / fit the Stage 3b gate on human labels")
    ap.add_argument("--labels", default="data/labels/arbitration_labels.csv")
    ap.add_argument("--extracted-dir", default="outputs/extracted/english")
    ap.add_argument("--config", default="configs/pipeline_config.yaml")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--write-config", action="store_true", help="with --fit: write fitted weights/thresholds into the config YAML")
    ap.add_argument("--target-precision", type=float, default=0.9)
    args = ap.parse_args()

    cfg = load_config(args.config).arbitration
    rows = load_rows(args.labels, args.extracted_dir)
    usable = [r for r in rows if r["label"] in ("G", "A")]
    if len(usable) < 5:
        print(f"Only {len(usable)} labelled G/A rows in {args.labels}; label more candidates first.")
        sys.exit(1)
    print(f"Labelled rows: {len(rows)} (G={sum(r['label']=='G' for r in rows)}, A={sum(r['label']=='A' for r in rows)}, U={sum(r['label']=='U' for r in rows)})"
          f" across {len({r['script_id'] for r in rows})} scripts")

    w0, ta0, tg0 = cfg.weights, cfg.threshold_ambiguity, cfg.threshold_genuine
    _print_metrics("Current config weights", metrics(usable, w0, ta0, tg0))

    # per-signal ablation with current weights
    print("\n== Per-signal ablation (current weights, signal zeroed) ==")
    for name in SIGNAL_NAMES:
        w_abl = ArbitrationWeights(**{**w0.model_dump(), name: 0.0})
        m = metrics(usable, w_abl, ta0, tg0)
        print(f"  -{name:13s} GENUINE F1={m['genuine_precision_recall_f1'][2]:.2f} AMBIGUITY F1={m['ambiguity_precision_recall_f1'][2]:.2f} "
              f"penalised={m['ambiguity_penalised']} forgiven={m['genuine_forgiven']}")

    # leave-one-script-out
    scripts = sorted({r["script_id"] for r in usable})
    if len(scripts) >= 2:
        print("\n== Leave-one-script-out (fit on other writers, test on held-out writer) ==")
        agg = defaultdict(int)
        for held in scripts:
            train = [r for r in usable if r["script_id"] != held]
            test = [r for r in usable if r["script_id"] == held]
            if len(train) < 5 or not test:
                continue
            w_l = fit_weights([r["signals"] for r in train], [1 if r["label"] == "A" else 0 for r in train], init=w0)
            ta_l, tg_l = choose_thresholds([fuse_signals(r["signals"], w_l) for r in train],
                                           [1 if r["label"] == "A" else 0 for r in train],
                                           args.target_precision, args.target_precision)
            m = metrics(test, w_l, ta_l, tg_l)
            for k, v in m["confusion"].items():
                agg[k] += v
            print(f"  held-out {held}: n={m['n']} GENUINE F1={m['genuine_precision_recall_f1'][2]:.2f} "
                  f"AMBIGUITY F1={m['ambiguity_precision_recall_f1'][2]:.2f} penalised={m['ambiguity_penalised']} forgiven={m['genuine_forgiven']} uncertain={m['uncertain_share']:.2f}")
        print(f"  pooled held-out confusion: {dict(agg)}")

    if args.fit:
        w_fit = fit_weights([r["signals"] for r in usable], [1 if r["label"] == "A" else 0 for r in usable], init=w0)
        ta, tg = choose_thresholds([fuse_signals(r["signals"], w_fit) for r in usable],
                                   [1 if r["label"] == "A" else 0 for r in usable],
                                   args.target_precision, args.target_precision)
        _print_metrics("Fitted weights (in-sample)", metrics(usable, w_fit, ta, tg))
        snippet = (
            "arbitration:\n"
            f"  threshold_ambiguity: {ta:.2f}\n"
            f"  threshold_genuine: {tg:.2f}\n"
            "  weights:\n"
            f"    bias: {w_fit.bias:.3f}\n"
            f"    phonetic: {w_fit.phonetic:.3f}\n"
            f"    writer: {w_fit.writer:.3f}\n"
            f"    consensus: {w_fit.consensus:.3f}\n"
            f"    forced_choice: {w_fit.forced_choice:.3f}\n"
        )
        print("\nPaste into configs/pipeline_config.yaml:\n" + snippet)
        if args.write_config:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            data.setdefault("arbitration", {})
            data["arbitration"]["threshold_ambiguity"] = round(ta, 2)
            data["arbitration"]["threshold_genuine"] = round(tg, 2)
            data["arbitration"]["weights"] = {k: round(float(getattr(w_fit, k)), 3) for k in ("bias",) + SIGNAL_NAMES}
            with open(args.config, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
            print(f"Config updated: {args.config} (comments were dropped by the YAML writer; see git diff)")


if __name__ == "__main__":
    main()
