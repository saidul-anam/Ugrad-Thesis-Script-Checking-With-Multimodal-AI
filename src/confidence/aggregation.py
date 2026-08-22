"""
Confidence Aggregation Algorithms for Token- and Line-Level OCR Confidences.
"""

from typing import List, Optional, Union
import math
import numpy as np


def aggregate_confidence(
    confidences: List[float],
    weights: Optional[List[float]] = None,
    method: str = "length_weighted_mean"
) -> Optional[float]:
    """
    Aggregate a list of segment/token confidences into a single scalar confidence [0.0, 1.0].

    Supported methods:
        - 'mean': Simple arithmetic mean
        - 'minimum': Conservative minimum confidence
        - 'length_weighted_mean': Arithmetic mean weighted by token/segment lengths
        - 'geometric_mean': Geometric mean (penalizes low-confidence outliers)
    """
    if not confidences:
        return None

    # Filter invalid values and clamp to [0.0, 1.0]
    valid_pairs = []
    for i, c in enumerate(confidences):
        if c is not None and not math.isnan(c):
            clamped_c = max(0.0, min(1.0, float(c)))
            w = float(weights[i]) if (weights is not None and i < len(weights) and weights[i] is not None) else 1.0
            valid_pairs.append((clamped_c, max(0.001, w)))

    if not valid_pairs:
        return None

    scores = [p[0] for p in valid_pairs]
    w_list = [p[1] for p in valid_pairs]

    method = method.lower().strip()

    if method == "mean":
        return float(np.mean(scores))

    elif method == "minimum":
        return float(np.min(scores))

    elif method == "length_weighted_mean":
        total_weight = sum(w_list)
        if total_weight <= 0:
            return float(np.mean(scores))
        weighted_sum = sum(s * w for s, w in zip(scores, w_list))
        return float(weighted_sum / total_weight)

    elif method == "geometric_mean":
        # Add epsilon to prevent log(0)
        eps = 1e-6
        log_scores = [math.log(max(eps, s)) for s in scores]
        return float(math.exp(np.mean(log_scores)))

    else:
        # Default fallback to length-weighted mean
        total_weight = sum(w_list)
        return float(sum(s * w for s, w in zip(scores, w_list)) / total_weight)
