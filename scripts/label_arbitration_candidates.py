#!/usr/bin/env python3
"""
Build a human labelling sheet for the Stage 3b ambiguity gate.

Walks outputs/extracted/<lang>/*/stage3b_arbitration.json, samples candidates STRATIFIED PER SCRIPT
(so fitted weights generalise across writers), and writes:

    data/labels/arbitration_labels.csv      one row per candidate; fill `human_label` with
                                             G = genuine student error, A = handwriting ambiguity, U = cannot tell
    data/labels/contact_sheets/<script>.png  the line crops with their ids, for fast labelling

Existing labels in the CSV are preserved (rows are merged on candidate key).

Usage:
  python scripts/label_arbitration_candidates.py --lang english --per-script 20
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIELDS = ["key", "script_id", "candidate_id", "q_no", "error_type", "read_token", "intended_token", "context",
          "crop_path", "ambiguity_score", "verdict", "phonetic", "writer", "consensus", "forced_choice",
          "human_label", "note"]


def _key(script_id: str, cid: str) -> str:
    return f"{script_id}|{cid}"


def _contact_sheet(script_id: str, rows: List[Dict[str, Any]], out_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    crops = []
    for r in rows:
        p = r.get("crop_path")
        if p and os.path.exists(p):
            try:
                crops.append((r, Image.open(p).convert("RGB")))
            except Exception:
                pass
    if not crops:
        return
    width = 1200
    rows_h = []
    for r, im in crops:
        scale = min(1.0, (width - 20) / im.size[0])
        im2 = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)))
        rows_h.append((r, im2))
    total_h = sum(im.size[1] + 34 for _, im in rows_h) + 10
    sheet = Image.new("RGB", (width, total_h), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    y = 5
    for r, im in rows_h:
        d.text((10, y), f"[{r['candidate_id']}] read='{r['read_token']}' intended='{r['intended_token']}' "
                        f"score={r['ambiguity_score']} {r['verdict']}", fill=(180, 0, 0))
        y += 16
        sheet.paste(im, (10, y))
        y += im.size[1] + 18
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create/merge the arbitration labelling sheet")
    ap.add_argument("--lang", default="english")
    ap.add_argument("--extracted-dir", default=None)
    ap.add_argument("--out", default="data/labels/arbitration_labels.csv")
    ap.add_argument("--per-script", type=int, default=25, help="max candidates sampled per script")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    ext_dir = Path(args.extracted_dir or f"outputs/extracted/{args.lang}")
    out_path = Path(args.out)

    existing: Dict[str, Dict[str, str]] = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                existing[row["key"]] = row

    new_rows: Dict[str, Dict[str, Any]] = {}
    per_script_counts: Dict[str, int] = {}
    for arb_file in sorted(ext_dir.glob("*/stage3b_arbitration.json")):
        script_id = arb_file.parent.name
        data = json.load(open(arb_file, "r", encoding="utf-8"))
        recs = data.get("records", [])
        rng.shuffle(recs)
        # stratify by verdict so every bin is represented
        by_verdict: Dict[str, List[Any]] = {}
        for r in recs:
            by_verdict.setdefault(r.get("verdict", "UNCERTAIN"), []).append(r)
        chosen: List[Any] = []
        while len(chosen) < args.per_script and any(by_verdict.values()):
            for v in list(by_verdict):
                if by_verdict[v] and len(chosen) < args.per_script:
                    chosen.append(by_verdict[v].pop())
        for r in chosen:
            c, e = r["candidate"], r["evidence"]
            key = _key(script_id, c["candidate_id"])
            row = {
                "key": key, "script_id": script_id, "candidate_id": c["candidate_id"], "q_no": c.get("question_no") or "",
                "error_type": c.get("error_type", ""), "read_token": c["candidate_token"], "intended_token": c["intended_token"],
                "context": (c.get("context_sentence") or "")[:160], "crop_path": e.get("crop_path") or "",
                "ambiguity_score": r.get("ambiguity_score", ""), "verdict": r.get("verdict", ""),
                "phonetic": e.get("phonetic_signal"), "writer": e.get("writer_prior"),
                "consensus": e.get("consensus_signal"), "forced_choice": e.get("forced_choice_signal"),
                "human_label": existing.get(key, {}).get("human_label", ""), "note": existing.get(key, {}).get("note", ""),
            }
            new_rows[key] = row
        per_script_counts[script_id] = len(chosen)
        _contact_sheet(script_id, [new_rows[_key(script_id, r["candidate"]["candidate_id"])] for r in chosen],
                       out_path.parent / "contact_sheets" / f"{script_id}.png")

    # keep previously labelled rows that were not re-sampled
    for key, row in existing.items():
        if key not in new_rows and row.get("human_label"):
            new_rows[key] = {k: row.get(k, "") for k in FIELDS}

    if not new_rows:
        print(f"No stage3b_arbitration.json found under {ext_dir}. Run extraction with arbitration.mode: evidence first.")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for key in sorted(new_rows):
            w.writerow({k: new_rows[key].get(k, "") for k in FIELDS})

    labelled = sum(1 for r in new_rows.values() if r.get("human_label"))
    print(f"Sheet: {out_path} | rows: {len(new_rows)} | already labelled: {labelled}")
    for s, n in per_script_counts.items():
        print(f"  {s}: {n} candidates (contact sheet: {out_path.parent / 'contact_sheets' / (s + '.png')})")
    print("Fill human_label with G (genuine error), A (handwriting ambiguity) or U (cannot tell), then run\n"
          f"  python scripts/evaluate_arbitration.py --labels {out_path} --fit")


if __name__ == "__main__":
    main()
