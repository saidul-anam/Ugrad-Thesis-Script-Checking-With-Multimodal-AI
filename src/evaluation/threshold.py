"""
Threshold Evaluation and Routing Quadrant Analysis for Downstream Multimodal Routing.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np


def evaluate_routing_thresholds(
    confidences: List[float],
    is_acceptable_list: List[bool],
    cers: List[float],
    wers: List[float],
    candidate_thresholds: List[float]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate candidate thresholds across 4 routing quadrants:
      - Q1: High Confidence + Correct (True Positive: Fast text-only routing)
      - Q2: High Confidence + Incorrect (False Positive: Critical Risk of erroneous text-only grading)
      - Q3: Low Confidence + Correct (False Negative: Unnecessary multimodal inference cost)
      - Q4: Low Confidence + Incorrect (True Negative: Correctly routed to multimodal inspection)
    """
    results = []
    conf_arr = np.array(confidences)
    acc_arr = np.array(is_acceptable_list)
    cer_arr = np.array(cers)
    wer_arr = np.array(wers)
    total_samples = len(conf_arr)

    if total_samples == 0:
        return pd.DataFrame(), {}

    for thresh in candidate_thresholds:
        # Rule: confidence >= threshold -> High Confidence
        high_mask = conf_arr >= thresh
        low_mask = ~high_mask

        n_high = int(np.sum(high_mask))
        n_low = int(np.sum(low_mask))

        # Quadrants
        q1_high_correct = int(np.sum(high_mask & acc_arr))
        q2_high_incorrect = int(np.sum(high_mask & ~acc_arr))
        q3_low_correct = int(np.sum(low_mask & acc_arr))
        q4_low_incorrect = int(np.sum(low_mask & ~acc_arr))

        # Metrics for High-Confidence Group
        high_cer = float(np.mean(cer_arr[high_mask])) if n_high > 0 else 0.0
        high_wer = float(np.mean(wer_arr[high_mask])) if n_high > 0 else 0.0
        high_acc_rate = float(q1_high_correct / n_high) if n_high > 0 else 0.0

        # Metrics for Low-Confidence Group
        low_cer = float(np.mean(cer_arr[low_mask])) if n_low > 0 else 0.0
        low_wer = float(np.mean(wer_arr[low_mask])) if n_low > 0 else 0.0
        low_acc_rate = float(q3_low_correct / n_low) if n_low > 0 else 0.0

        results.append({
            "threshold": thresh,
            "pct_high_confidence": round((n_high / total_samples) * 100.0, 2),
            "pct_low_confidence": round((n_low / total_samples) * 100.0, 2),
            "high_conf_cer": round(high_cer, 4),
            "high_conf_wer": round(high_wer, 4),
            "high_conf_acceptable_rate": round(high_acc_rate * 100.0, 2),
            "low_conf_cer": round(low_cer, 4),
            "low_conf_wer": round(low_wer, 4),
            "low_conf_acceptable_rate": round(low_acc_rate * 100.0, 2),
            "high_correct_count": q1_high_correct,
            "high_incorrect_count": q2_high_incorrect,
            "low_correct_count": q3_low_correct,
            "low_incorrect_count": q4_low_incorrect
        })

    df = pd.DataFrame(results)
    return df, {"total_samples": total_samples}
