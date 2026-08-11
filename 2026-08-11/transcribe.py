"""Stage 1: render scan PDFs to 300 DPI page images and transcribe each page.

Usage:
    python transcribe.py --backend gemini
    python transcribe.py --backend mistral_ocr
    python transcribe.py --backend mistral_ocr --max-pages 1   # smoke test

Resumable: pages whose transcript file already exists are skipped.
Failures: one retry, then logged to logs/api_failures.csv and skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pymupdf
from dotenv import load_dotenv

from common import (
    PAGE_IMAGES_DIR,
    RENDER_DPI,
    SCANS_DIR,
    TRANSCRIPTS_RAW_DIR,
    append_run_log,
    log_api_failure,
    page_image_path,
    page_txt_path,
    raw_response_path,
    utc_now,
)
from backends import get_backend


def render_pages(pdf_path) -> list[tuple[int, object]]:
    """Render every page to page_images/ at 300 DPI (skip existing). Returns
    [(page_no, image_path)] for all pages."""
    script_id = pdf_path.stem
    out = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            img_path = page_image_path(script_id, i)
            if not img_path.exists():
                img_path.parent.mkdir(parents=True, exist_ok=True)
                pix = page.get_pixmap(dpi=RENDER_DPI)
                pix.save(img_path)
            out.append((i, img_path))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["gemini", "mistral_ocr"])
    parser.add_argument("--model", help="override the backend's default model")
    parser.add_argument("--scans", nargs="*", help="specific PDF paths (default: all of scans/*.pdf)")
    parser.add_argument("--max-pages", type=int, help="cap pages per script (smoke testing only)")
    args = parser.parse_args()

    load_dotenv()
    backend = get_backend(args.backend, args.model)

    if args.scans:
        pdfs = [Path(p) if Path(p).is_absolute() else SCANS_DIR / p for p in args.scans]
    else:
        pdfs = sorted(SCANS_DIR.glob("*.pdf"))
    if not pdfs:
        print("no PDFs found in scans/", file=sys.stderr)
        return 1

    totals = {"pages_done": 0, "pages_skipped": 0, "pages_failed": 0}
    usage_totals: dict[str, float] = {}
    t_start = time.time()

    for pdf_path in pdfs:
        script_id = pdf_path.stem
        pages = render_pages(pdf_path)
        if args.max_pages:
            pages = pages[: args.max_pages]
        print(f"\n=== {script_id}: {len(pages)} pages [{backend.name}] ===")

        for page_no, img_path in pages:
            txt_path = page_txt_path(backend.name, script_id, page_no)
            if txt_path.exists():
                totals["pages_skipped"] += 1
                print(f"  p{page_no:02d}: exists, skipped")
                continue

            result = None
            for attempt in (1, 2):
                result = backend.transcribe_page(img_path)
                if result.error is None:
                    break
                log_api_failure(script_id, page_no, backend.name, attempt, result.error)
                print(f"  p{page_no:02d}: attempt {attempt} FAILED: {result.error[:200]}")

            # Archive the raw response of the final attempt regardless of outcome.
            if result.raw_response is not None:
                rp = raw_response_path(backend.name, script_id, page_no)
                rp.parent.mkdir(parents=True, exist_ok=True)
                rp.write_text(json.dumps(result.raw_response, ensure_ascii=False, indent=1, default=str))

            append_run_log(backend.name, {
                "timestamp": utc_now(),
                "script_id": script_id,
                "page": page_no,
                "backend": backend.name,
                "backend_version": backend.version,
                "settings": backend.settings(),
                "usage": result.usage,
                "duration_s": round(result.duration_s, 2),
                "error": result.error,
            })

            if result.error is not None:
                totals["pages_failed"] += 1
                continue

            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(result.text)
            totals["pages_done"] += 1
            for k, v in result.usage.items():
                if isinstance(v, (int, float)):
                    usage_totals[k] = usage_totals.get(k, 0) + v
            running = ", ".join(f"{k}={int(v)}" for k, v in usage_totals.items())
            print(f"  p{page_no:02d}: ok ({result.duration_s:.1f}s, {len(result.text)} chars) | running: {running}")

    elapsed = time.time() - t_start
    print(f"\n[{backend.name} / {backend.version}] done in {elapsed:.0f}s: "
          f"{totals['pages_done']} transcribed, {totals['pages_skipped']} skipped, "
          f"{totals['pages_failed']} failed")
    print(f"usage totals: {usage_totals}")
    return 0 if totals["pages_failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
