"""
Threshold Routing Analysis Script.
Evaluates candidate thresholds (0.50 to 0.95) for routing across 4 quadrants.
Usage:
    python scripts/threshold_analysis.py --benchmark_csv outputs/reports/ocr_benchmark.csv
"""

import sys
import argparse
import yaml
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.evaluation.threshold import evaluate_routing_thresholds


def main():
    parser = argparse.ArgumentParser(description="Evaluate Candidate Confidence Thresholds for Multimodal Routing")
    parser.add_argument("--benchmark_csv", default="outputs/reports/ocr_benchmark.csv", help="Benchmark results CSV")
    parser.add_argument("--config", default="configs/confidence.yaml", help="Confidence YAML config")
    parser.add_argument("--output_csv", default="outputs/confidence/threshold_analysis.csv", help="Output threshold CSV")
    args = parser.parse_args()

    csv_path = Path(args.benchmark_csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run benchmark_ocr.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    valid_df = df.dropna(subset=["confidence"]).copy()

    if len(valid_df) == 0:
        print("No samples with valid confidence scores.")
        sys.exit(0)

    # Load candidate thresholds
    thresh_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            thresh_list = cfg.get("evaluation", {}).get("candidate_thresholds", thresh_list)

    thresh_df, meta = evaluate_routing_thresholds(
        confidences=valid_df["confidence"].tolist(),
        is_acceptable_list=valid_df["is_acceptable"].tolist(),
        cers=valid_df["normalized_cer"].tolist(),
        wers=valid_df["normalized_wer"].tolist(),
        candidate_thresholds=thresh_list
    )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    thresh_df.to_csv(args.output_csv, index=False)

    print("\n" + "=" * 80)
    print("ROUTING THRESHOLD & 4-QUADRANT EVALUATION")
    print("=" * 80)
    cols_to_print = [
        "threshold", "pct_high_confidence", "high_conf_cer", "high_conf_acceptable_rate",
        "low_conf_cer", "low_conf_acceptable_rate", "high_incorrect_count"
    ]
    print(thresh_df[cols_to_print].to_string(index=False))
    print("=" * 80)
    print("Quadrant Summary:")
    print("  - high_correct_count:   Correctly routed to fast text-only grading")
    print("  - high_incorrect_count: CRITICAL RISK (High confidence but unacceptable OCR)")
    print("  - low_correct_count:    Unnecessary multimodal inference cost")
    print("  - low_incorrect_count:  Correctly routed to multimodal visual fallback")
    print("=" * 80)
    print(f"Full analysis saved to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
