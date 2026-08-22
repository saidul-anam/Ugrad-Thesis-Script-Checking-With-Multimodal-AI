"""
Confidence Analysis and Calibration Metrics.
Implements Confidence Bins, Expected Calibration Error (ECE), Brier Score, and Accuracy vs Confidence Curves.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def compute_confidence_bins(
    confidences: List[float],
    labels: List[int],
    cers: List[float],
    wers: List[float],
    num_bins: int = 10
) -> pd.DataFrame:
    """
    Partition samples into uniform confidence intervals (e.g. 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
    and compute empirical statistics per bin.
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_data = []

    conf_arr = np.array(confidences)
    label_arr = np.array(labels)
    cer_arr = np.array(cers)
    wer_arr = np.array(wers)

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        # Include upper boundary on last bin
        if i == num_bins - 1:
            idx = np.where((conf_arr >= bin_lower) & (conf_arr <= bin_upper))[0]
        else:
            idx = np.where((conf_arr >= bin_lower) & (conf_arr < bin_upper))[0]

        count = len(idx)
        if count > 0:
            avg_conf = float(np.mean(conf_arr[idx]))
            avg_cer = float(np.mean(cer_arr[idx]))
            avg_wer = float(np.mean(wer_arr[idx]))
            acc_rate = float(np.mean(label_arr[idx]))
        else:
            avg_conf = (bin_lower + bin_upper) / 2.0
            avg_cer = 0.0
            avg_wer = 0.0
            acc_rate = 0.0

        bin_data.append({
            "bin_index": i + 1,
            "bin_range": f"{bin_lower:.1f}-{bin_upper:.1f}",
            "sample_count": count,
            "avg_confidence": round(avg_conf, 4),
            "avg_cer": round(avg_cer, 4),
            "avg_wer": round(avg_wer, 4),
            "acceptable_ocr_percentage": round(acc_rate * 100.0, 2)
        })

    return pd.DataFrame(bin_data)


def compute_calibration_metrics(
    confidences: List[float],
    labels: List[int],
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Calculate Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score.
    """
    if not confidences or not labels:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0}

    conf_arr = np.array(confidences)
    label_arr = np.array(labels)
    total_samples = len(conf_arr)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        if i == num_bins - 1:
            idx = np.where((conf_arr >= bin_lower) & (conf_arr <= bin_upper))[0]
        else:
            idx = np.where((conf_arr >= bin_lower) & (conf_arr < bin_upper))[0]

        count = len(idx)
        if count > 0:
            avg_conf = float(np.mean(conf_arr[idx]))
            empirical_acc = float(np.mean(label_arr[idx]))
            gap = abs(empirical_acc - avg_conf)
            ece += (count / total_samples) * gap
            mce = max(mce, gap)

    # Brier Score = mean squared error between probability and binary label
    brier_score = float(np.mean((conf_arr - label_arr) ** 2))

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier_score, 4)
    }
