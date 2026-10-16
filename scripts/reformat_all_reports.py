#!/usr/bin/env python3
"""
Reformat all evaluation_report.md files across outputs/evaluated/ using the unified layout.
Pure CPU text re-rendering from complete_report.json. Zero GPU, zero model inference.
"""

import os
import sys
import glob
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.schemas import CompleteEvaluationReport
from src.utils.export_utils import export_report_markdown


def reformat_all(base_dir: str = "outputs/evaluated"):
    target_pattern = os.path.join(PROJECT_ROOT, base_dir, "**", "complete_report.json")
    json_files = sorted(glob.glob(target_pattern, recursive=True))

    print(f"Found {len(json_files)} complete_report.json files in {base_dir}")
    if not json_files:
        print("No evaluation reports found.")
        return

    success_count = 0
    error_count = 0
    start_time = time.time()

    by_set = {}

    for i, json_path in enumerate(json_files, 1):
        try:
            rel_path = os.path.relpath(json_path, PROJECT_ROOT)
            parts = rel_path.split(os.sep)
            exam_set = parts[2] + "/" + parts[3] if len(parts) > 3 else (parts[2] if len(parts) > 2 else "unknown")
            script_id = parts[4] if len(parts) > 4 else os.path.basename(os.path.dirname(json_path))

            with open(json_path, "r", encoding="utf-8") as f:
                report = CompleteEvaluationReport.model_validate_json(f.read())

            md_path = os.path.join(os.path.dirname(json_path), "evaluation_report.md")
            export_report_markdown(report, md_path)

            success_count += 1
            by_set[exam_set] = by_set.get(exam_set, 0) + 1

            if i % 25 == 0 or i == len(json_files):
                print(f"[{i}/{len(json_files)}] Rendered {rel_path} -> evaluation_report.md")

        except Exception as e:
            error_count += 1
            print(f"❌ Error processing {json_path}: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Re-rendering Summary:")
    for es, cnt in sorted(by_set.items()):
        print(f"  - {es}: {cnt} reports updated")
    print(f"Total Successful: {success_count} / {len(json_files)}")
    print(f"Total Errors:     {error_count}")
    print(f"Elapsed Time:     {elapsed:.2f} seconds ({elapsed / max(1, success_count) * 1000:.1f} ms/report)")
    print("=" * 60)


if __name__ == "__main__":
    reformat_all()
