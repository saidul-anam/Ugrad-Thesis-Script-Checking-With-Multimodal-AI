"""
Confidence Calibration CLI Script.
Fits Isotonic Regression or Platt Scaling on a calibration split.
Usage:
    python scripts/calibrate_confidence.py --benchmark_csv outputs/reports/ocr_benchmark.csv --method platt_scaling
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.confidence.calibration import ConfidenceCalibrator


def main():
    parser = argparse.ArgumentParser(description="Fit Confidence Calibrator on Validation/Calibration Data")
    parser.add_argument("--benchmark_csv", default="outputs/reports/ocr_benchmark.csv", help="Calibration dataset CSV")
    parser.add_argument("--method", default="platt_scaling", choices=["platt_scaling", "isotonic_regression"])
    parser.add_argument("--output_model", default="outputs/confidence/calibrator.joblib", help="Output model path")
    args = parser.parse_args()

    csv_path = Path(args.benchmark_csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path).dropna(subset=["confidence"])
    if len(df) < 5:
        print("Need at least 5 samples with confidence scores to calibrate.")
        sys.exit(0)

    calibrator = ConfidenceCalibrator(method=args.method)
    calibrator.fit(
        uncalibrated_confidences=df["confidence"].tolist(),
        labels=df["is_acceptable"].astype(int).tolist()
    )

    calibrator.save(args.output_model)
    print(f"\nCalibration model ({args.method}) successfully fitted and saved to: {args.output_model}\n")


if __name__ == "__main__":
    main()
