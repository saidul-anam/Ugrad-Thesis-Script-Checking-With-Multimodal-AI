#!/usr/bin/env python3
"""
Create a page-level transcription ground-truth workspace for CER/WER measurement.

For each selected page this writes:
    data/ground_truth/transcripts/<lang>/<script_id>/page_<n>.txt   (pre-filled with the Stage 2 verified
                                                                     transcript from the checkpoint; CORRECT IT)
    data/ground_truth/transcripts/<lang>/<script_id>/page_<n>.png   (copy of the page image, for reference)
    data/ground_truth/transcripts/<lang>/<script_id>/page_<n>.meta.json

Correcting a machine draft is much faster than transcribing from scratch. Rules for the human:
  - keep the student's real mistakes exactly as written (do NOT fix spelling/grammar);
  - use [illegible] for unreadable words, [struck: text] for crossed-out text;
  - keep line breaks as on the page; ignore red teacher ink.

Selection (default): every page that contains a continuous-writing question (Q3, Q7-Q11) across all
extracted scripts, capped by --max-pages, spread evenly across scripts.

Usage:
  python scripts/make_transcription_gt.py --lang english
  python scripts/make_transcription_gt.py --lang english --pages SE_11_Q1_0002:3,12,13 SE_11_Q1_0010:5,6
  python scripts/make_transcription_gt.py --lang english --max-pages 20 --force
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONTINUOUS_Q = {"3", "7", "8", "9", "10", "11"}
_HEADER_RE = re.compile(r"(?i)\bq(?:uestion)?\.?\s*(?:no\.?)?\s*-?\s*0?(\d{1,2})")


def _page_question_hint(transcript: str) -> List[str]:
    return sorted(set(_HEADER_RE.findall(transcript or "")))


def _load_checkpoints(script_dir: Path) -> List[Tuple[int, dict]]:
    out = []
    ck = script_dir / "checkpoints"
    if not ck.exists():
        return out
    for p in sorted(ck.glob("page_*.json"), key=lambda x: int(re.findall(r"\d+", x.stem)[0])):
        try:
            with open(p, "r", encoding="utf-8") as f:
                out.append((int(re.findall(r"\d+", p.stem)[0]), json.load(f)))
        except Exception:
            continue
    return out


def _default_selection(extracted_dir: Path, max_pages: int) -> Dict[str, List[int]]:
    """Pages holding continuous-writing questions, spread evenly across scripts."""
    per_script: Dict[str, List[int]] = {}
    for script_dir in sorted(p for p in extracted_dir.iterdir() if p.is_dir()):
        pages = _load_checkpoints(script_dir)
        if not pages:
            continue
        current_q = None
        chosen = []
        for page_no, data in pages:
            txt = (data.get("stage2_verification") or {}).get("verified_transcript") or \
                  (data.get("stage1_transcription") or {}).get("raw_transcript") or ""
            hints = _page_question_hint(txt)
            if hints:
                current_q = hints[-1]
            if current_q in CONTINUOUS_Q and len(txt.split()) >= 40:
                chosen.append(page_no)
        if chosen:
            per_script[script_dir.name] = chosen
    # round-robin cap
    selection: Dict[str, List[int]] = {s: [] for s in per_script}
    total = 0
    idx = 0
    while total < max_pages and any(len(per_script[s]) > idx for s in per_script):
        for s in per_script:
            if len(per_script[s]) > idx and total < max_pages:
                selection[s].append(per_script[s][idx])
                total += 1
        idx += 1
    return {s: sorted(v) for s, v in selection.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build transcription ground-truth workspace")
    ap.add_argument("--lang", default="english", choices=["english", "bangla"])
    ap.add_argument("--extracted-dir", default=None, help="default: outputs/extracted/<lang>")
    ap.add_argument("--samples-dir", default="data/samples", help="rendered page PNGs root")
    ap.add_argument("--out-dir", default=None, help="default: data/ground_truth/transcripts/<lang>")
    ap.add_argument("--pages", nargs="*", default=None, help="explicit SCRIPT:1,2,3 selections")
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--force", action="store_true", help="overwrite existing .txt drafts (destroys corrections!)")
    args = ap.parse_args()

    extracted_dir = Path(args.extracted_dir or f"outputs/extracted/{args.lang}")
    out_dir = Path(args.out_dir or f"data/ground_truth/transcripts/{args.lang}")
    if not extracted_dir.exists():
        print(f"No extracted outputs at {extracted_dir}. Run scripts/extract_scripts.py first.")
        sys.exit(1)

    if args.pages:
        selection: Dict[str, List[int]] = {}
        for spec in args.pages:
            sid, _, nums = spec.partition(":")
            selection[sid] = sorted({int(n) for n in nums.split(",") if n.strip()})
    else:
        selection = _default_selection(extracted_dir, args.max_pages)

    if not selection:
        print("Nothing selected (no checkpoints with continuous-writing pages found).")
        sys.exit(1)

    written, skipped = 0, 0
    for script_id, page_nos in selection.items():
        script_dir = extracted_dir / script_id
        pages = dict(_load_checkpoints(script_dir))
        target = out_dir / script_id
        target.mkdir(parents=True, exist_ok=True)
        for n in page_nos:
            data = pages.get(n)
            if data is None:
                print(f"  [{script_id}] page {n}: no checkpoint, skipped")
                continue
            txt_path = target / f"page_{n}.txt"
            if txt_path.exists() and not args.force:
                skipped += 1
                continue
            draft = (data.get("stage2_verification") or {}).get("verified_transcript") or \
                    (data.get("stage1_transcription") or {}).get("raw_transcript") or ""
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(draft.rstrip() + "\n")
            # page image
            img_src = data.get("image_path") or ""
            cands = [Path(img_src)] if img_src else []
            cands += [Path(args.samples_dir) / script_id / f"{script_id}_page_{n:02d}.png",
                      Path(args.samples_dir) / script_id / f"{script_id}_page_{n}.png"]
            for c in cands:
                if c.exists():
                    shutil.copyfile(c, target / f"page_{n}.png")
                    break
            with open(target / f"page_{n}.meta.json", "w", encoding="utf-8") as f:
                json.dump({"script_id": script_id, "page_no": n, "source_checkpoint": str(script_dir / "checkpoints" / f"page_{n}.json"),
                           "draft_origin": "stage2_verified", "status": "DRAFT - needs human correction"}, f, indent=2)
            written += 1
            print(f"  [{script_id}] page {n}: draft written -> {txt_path}")

    print(f"\nDrafts written: {written} | untouched existing: {skipped}")
    print(f"Now correct the .txt files under {out_dir} to the exact handwriting, then run:\n"
          f"  python scripts/evaluate_transcription.py --lang {args.lang}")


if __name__ == "__main__":
    main()
