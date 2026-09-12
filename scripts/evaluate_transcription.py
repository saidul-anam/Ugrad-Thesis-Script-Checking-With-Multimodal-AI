#!/usr/bin/env python3
"""
Score pipeline transcripts against human-corrected page transcripts (CER / WER / silent-correction rate).

Reads:  data/ground_truth/transcripts/<lang>/<script_id>/page_<n>.txt   (human-corrected reference)
        outputs/extracted/<lang>/<script_id>/checkpoints/page_<n>.json   (Stage 1 raw, Stage 2 verified)
Writes: outputs/benchmarks/transcription_<lang>_<timestamp>.json and .md

Pages whose .meta.json still says DRAFT are skipped unless --include-drafts is given (they would
score the machine against itself).

Usage:
  python scripts/evaluate_transcription.py --lang english
  python scripts/evaluate_transcription.py --lang english --extracted-dir outputs/extracted/english_ablation_no_stage2 --tag no_stage2
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.text_metrics import score_transcription
from src.utils.linguistic_sanitizer import get_english_lexicon


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcription CER/WER benchmark")
    ap.add_argument("--lang", default="english", choices=["english", "bangla"])
    ap.add_argument("--gt-dir", default=None, help="default: data/ground_truth/transcripts/<lang>")
    ap.add_argument("--extracted-dir", default=None, help="default: outputs/extracted/<lang>")
    ap.add_argument("--out-dir", default="outputs/benchmarks")
    ap.add_argument("--tag", default="", help="label for this run (e.g. ablation name)")
    ap.add_argument("--include-drafts", action="store_true")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir or f"data/ground_truth/transcripts/{args.lang}")
    ext_dir = Path(args.extracted_dir or f"outputs/extracted/{args.lang}")
    lexicon = get_english_lexicon() if args.lang == "english" else set()

    if not gt_dir.exists():
        print(f"No ground truth at {gt_dir}. Run scripts/make_transcription_gt.py and correct the drafts first.")
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    skipped_drafts = 0
    for script_dir in sorted(p for p in gt_dir.iterdir() if p.is_dir()):
        script_id = script_dir.name
        for txt in sorted(script_dir.glob("page_*.txt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0])):
            n = int(re.findall(r"\d+", txt.stem)[0])
            meta_path = script_dir / f"page_{n}.meta.json"
            if meta_path.exists() and not args.include_drafts:
                try:
                    meta = json.load(open(meta_path, "r", encoding="utf-8"))
                    if str(meta.get("status", "")).upper().startswith("DRAFT"):
                        skipped_drafts += 1
                        continue
                except Exception:
                    pass
            ckpt = ext_dir / script_id / "checkpoints" / f"page_{n}.json"
            if not ckpt.exists():
                print(f"  [{script_id}] page {n}: no checkpoint in {ext_dir}, skipped")
                continue
            reference = open(txt, "r", encoding="utf-8").read()
            data = json.load(open(ckpt, "r", encoding="utf-8"))
            raw = (data.get("stage1_transcription") or {}).get("raw_transcript") or ""
            verified = (data.get("stage2_verification") or {}).get("verified_transcript") or ""
            s1 = score_transcription(reference, raw, lexicon)
            s2 = score_transcription(reference, verified, lexicon)
            rows.append({"script_id": script_id, "page_no": n, "stage1": s1.as_dict(), "stage1+2": s2.as_dict()})
            print(f"  [{script_id}] page {n:>2}: CER s1={s1.cer:.3f} s1+2={s2.cer:.3f} | WER s1={s1.wer:.3f} s1+2={s2.wer:.3f} "
                  f"| silent-corr s1={s1.silent_correction_rate} s1+2={s2.silent_correction_rate}")

    if not rows:
        print(f"No scorable pages (drafts skipped: {skipped_drafts}). Mark corrected pages by changing 'status' in page_<n>.meta.json "
              f"to anything not starting with DRAFT, e.g. \"CORRECTED\".")
        sys.exit(1)

    summary = {}
    for stage in ("stage1", "stage1+2"):
        cer_micro = sum(r[stage]["cer"] * r[stage]["ref_chars"] for r in rows) / max(1, sum(r[stage]["ref_chars"] for r in rows))
        wer_micro = sum(r[stage]["wer"] * r[stage]["ref_words"] for r in rows) / max(1, sum(r[stage]["ref_words"] for r in rows))
        nonwords = sum(r[stage]["student_nonwords"] for r in rows)
        preserved = sum(r[stage]["nonwords_preserved"] for r in rows)
        summary[stage] = {
            "cer_macro": round(_mean([r[stage]["cer"] for r in rows]), 4),
            "cer_micro": round(cer_micro, 4),
            "wer_macro": round(_mean([r[stage]["wer"] for r in rows]), 4),
            "wer_micro": round(wer_micro, 4),
            "student_nonwords": nonwords,
            "nonwords_preserved": preserved,
            "silent_correction_rate": round(1 - preserved / nonwords, 4) if nonwords else None,
        }

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    base = Path(args.out_dir) / f"transcription_{args.lang}{tag}_{stamp}"
    with open(str(base) + ".json", "w", encoding="utf-8") as f:
        json.dump({"lang": args.lang, "tag": args.tag, "extracted_dir": str(ext_dir), "gt_dir": str(gt_dir),
                   "pages": rows, "summary": summary, "skipped_drafts": skipped_drafts}, f, ensure_ascii=False, indent=2)

    md = [f"# Transcription Benchmark ({args.lang}{tag})", f"Generated {stamp} | pages scored: {len(rows)} | drafts skipped: {skipped_drafts}", "",
          "| Stage | CER (macro) | CER (micro) | WER (macro) | WER (micro) | student non-words | preserved | silent-correction rate |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for stage, s in summary.items():
        md.append(f"| {stage} | {s['cer_macro']:.4f} | {s['cer_micro']:.4f} | {s['wer_macro']:.4f} | {s['wer_micro']:.4f} | "
                  f"{s['student_nonwords']} | {s['nonwords_preserved']} | {s['silent_correction_rate']} |")
    md += ["", "## Per page", "", "| Script | Page | CER s1 | CER s1+2 | WER s1 | WER s1+2 | non-words lost (s1+2) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(f"| {r['script_id']} | {r['page_no']} | {r['stage1']['cer']:.3f} | {r['stage1+2']['cer']:.3f} | "
                  f"{r['stage1']['wer']:.3f} | {r['stage1+2']['wer']:.3f} | {', '.join(r['stage1+2']['nonwords_lost'][:8])} |")
    with open(str(base) + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n== Summary ==")
    for stage, s in summary.items():
        print(f"  {stage:9s} CER macro {s['cer_macro']:.4f} | micro {s['cer_micro']:.4f} | WER macro {s['wer_macro']:.4f} | "
              f"silent-correction {s['silent_correction_rate']}")
    print(f"Written: {base}.json / .md")


if __name__ == "__main__":
    main()
