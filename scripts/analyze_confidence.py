"""
Confidence Analysis and Calibration Metrics Script.
Calculates Confidence Bins, ECE, Brier Score, and generates analysis CSVs.
Usage:
    python scripts/analyze_confidence.py --benchmark_csv outputs/reports/ocr_benchmark.csv
"""

import sys
import json
import argparse
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.confidence.analysis import compute_confidence_bins, compute_calibration_metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze OCR Confidence vs Ground-Truth Correctness")
    parser.add_argument("--benchmark_csv", default="outputs/reports/ocr_benchmark.csv", help="Benchmark results CSV")
    parser.add_argument("--output_dir", default="outputs/confidence", help="Output directory for confidence analysis")
    parser.add_argument("--bins", type=int, default=10, help="Number of confidence bins")
    args = parser.parse_args()

    csv_path = Path(args.benchmark_csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run benchmark_ocr.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # Filter rows with confidence
    valid_df = df.dropna(subset=["confidence"]).copy()

    if len(valid_df) == 0:
        print("No samples with valid confidence scores found in benchmark results.")
        sys.exit(0)

    confidences = valid_df["confidence"].tolist()
    labels = valid_df["is_acceptable"].astype(int).tolist()
    cers = valid_df["normalized_cer"].tolist()
    wers = valid_df["normalized_wer"].tolist()

    # 1. Compute Confidence Bins
    bins_df = compute_confidence_bins(
        confidences=confidences,
        labels=labels,
        cers=cers,
        wers=wers,
        num_bins=args.bins
    )

    # 2. Compute Calibration Metrics
    calib_metrics = compute_calibration_metrics(
        confidences=confidences,
        labels=labels,
        num_bins=args.bins
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bins_csv_path = out_dir / "confidence_bins.csv"
    bins_df.to_csv(bins_csv_path, index=False)

    calib_json_path = out_dir / "calibration_metrics.json"
    with open(calib_json_path, "w", encoding="utf-8") as f:
        json.dump(calib_metrics, f, indent=2)

    print("\n" + "=" * 50)
    print("CONFIDENCE VS CORRECTNESS BIN ANALYSIS")
    print("=" * 50)
    print(bins_df.to_string(index=False))
    print("=" * 50)
    print(f"Expected Calibration Error (ECE): {calib_metrics['ece']:.4f}")
    print(f"Maximum Calibration Error (MCE):  {calib_metrics['mce']:.4f}")
    print(f"Brier Score:                     {calib_metrics['brier_score']:.4f}")
    print("=" * 50)
    print(f"Confidence bins saved to:        {bins_csv_path}")
    print(f"Calibration metrics saved to:    {calib_json_path}\n")


if __name__ == "__main__":
    main()
