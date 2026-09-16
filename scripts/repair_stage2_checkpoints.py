#!/usr/bin/env python3
"""
Repair Stage 2 Checkpoints: Fix LaTeX arrow corruptions and restore dropped [struck: ...] tags.

Processes: outputs/extracted/<lang>/<script_id>/checkpoints/page_<n>.json
Repairs:
  1. Corrupted LaTeX arrows: converts \r + ightarrow, \right\rarrow, \right\tarrow -> $\rightarrow$
  2. Dropped [struck: ...] tags: restores any struck tags present in Stage 1 that were omitted by Stage 2
  3. Regenerates stage2_verified_transcript.txt and extraction_result.json

Usage:
  python scripts/repair_stage2_checkpoints.py --lang english
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def repair_text(verified: str, raw: str) -> Tuple[str, int, int]:
    """Returns (repaired_text, arrow_fixes, tag_fixes)."""
    text = verified
    arrow_fixes = 0
    tag_fixes = 0

    # 1. LaTeX arrow repair
    arrow_patterns = [
        r'\$\s*[\r\n\t\x0b\x0c]?\\?r?ight[\r\n\t\x0b\x0c]?arrow\s*\$',
        r'[\r\n\t\x0b\x0c]?\\?right[\r\n\t\x0b\x0c]?arrow',
        r'[\r\n\t\x0b\x0c]ightarrow',
    ]
    for pat in arrow_patterns:
        matches = len(re.findall(pat, text))
        if matches:
            arrow_fixes += matches
            text = re.sub(pat, r'$\\rightarrow$', text)

    # Clean double dollar signs if created
    text = text.replace(r'$$\rightarrow$$', r'$\rightarrow$')
    text = text.replace(r'$$\rightarrow', r'$\rightarrow$')
    text = text.replace(r'\rightarrow$$', r'$\rightarrow$')
    text = text.replace(r'$$\rightarrow$', r'$\rightarrow$')
    text = text.replace(r'$\rightarrow$$', r'$\rightarrow$')

    # If raw had $\rightarrow$ and text has raw \rightarrow without $, wrap it
    if r'$\rightarrow$' in raw:
        text = re.sub(r'(?<!\$)\\rightarrow(?!\$)', r'$\\rightarrow$', text)

    # 2. Restore dropped [struck: ...] tags
    struck_tags = re.findall(r'\[struck:[^\]]+\]', raw)
    for tag in struck_tags:
        if tag not in text:
            inner = tag[len("[struck:"): -1].strip()
            if inner and inner in text:
                text = re.sub(rf'\b{re.escape(inner)}\b', tag, text, count=1)
                tag_fixes += 1

    return text, arrow_fixes, tag_fixes


def repair_script_checkpoints(script_dir: Path) -> Dict[str, int]:
    ck_dir = script_dir / "checkpoints"
    if not ck_dir.exists():
        return {"pages": 0, "arrows": 0, "tags": 0}

    stats = {"pages": 0, "arrows": 0, "tags": 0}
    pages_data = []

    for ck_path in sorted(ck_dir.glob("page_*.json"), key=lambda p: int(re.findall(r'\d+', p.stem)[0])):
        stats["pages"] += 1
        with open(ck_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw = (data.get("stage1_transcription") or {}).get("raw_transcript") or ""
        s2 = data.get("stage2_verification") or {}
        verified = s2.get("verified_transcript") or raw

        repaired, a_fixes, t_fixes = repair_text(verified, raw)
        stats["arrows"] += a_fixes
        stats["tags"] += t_fixes

        if a_fixes > 0 or t_fixes > 0:
            s2["verified_transcript"] = repaired
            data["stage2_verification"] = s2
            with open(ck_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        page_no = data.get("page_no", int(re.findall(r'\d+', ck_path.stem)[0]))
        pages_data.append((page_no, repaired))

    # Regenerate stage2_verified_transcript.txt
    pages_data.sort(key=lambda x: x[0])
    combined_verified = "\n\n--- Page Break ---\n\n".join(t for _, t in pages_data)
    with open(script_dir / "stage2_verified_transcript.txt", "w", encoding="utf-8") as f:
        f.write(combined_verified)

    # Update extraction_result.json if present
    res_path = script_dir / "extraction_result.json"
    if res_path.exists():
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                res_data = json.load(f)
            if "stage2_verification" in res_data:
                res_data["stage2_verification"]["verified_transcript"] = combined_verified
            if "full_transcript" in res_data:
                res_data["full_transcript"] = combined_verified
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(res_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Warning: could not update {res_path}: {e}")

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair Stage 2 checkpoints (LaTeX arrows & struck tags)")
    ap.add_argument("--lang", default="english", choices=["english", "bangla"])
    ap.add_argument("--extracted-dir", default=None)
    args = ap.parse_args()

    ext_dir = Path(args.extracted_dir or f"outputs/extracted/{args.lang}")
    if not ext_dir.exists():
        print(f"Directory {ext_dir} does not exist.")
        sys.exit(1)

    total_stats = {"scripts": 0, "pages": 0, "arrows": 0, "tags": 0}
    for script_dir in sorted(p for p in ext_dir.iterdir() if p.is_dir()):
        script_id = script_dir.name
        s = repair_script_checkpoints(script_dir)
        total_stats["scripts"] += 1
        total_stats["pages"] += s["pages"]
        total_stats["arrows"] += s["arrows"]
        total_stats["tags"] += s["tags"]
        print(f"[{script_id}] {s['pages']} pages repaired: {s['arrows']} corrupted arrows fixed, {s['tags']} struck tags restored.")

    print(f"\n--- Total Summary ---")
    print(f"Scripts: {total_stats['scripts']}")
    print(f"Pages: {total_stats['pages']}")
    print(f"LaTeX arrows repaired: {total_stats['arrows']}")
    print(f"Struck tags restored: {total_stats['tags']}")


if __name__ == "__main__":
    main()
