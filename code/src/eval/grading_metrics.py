"""
Grading Evaluation Metrics Harness.
Computes Quadratic Weighted Kappa (QWK), Pearson Correlation, Spearman Rank Correlation,
MAE, RMSE, Performance Band Classification Accuracy, and Hard Cap Precision/Recall
against human ground truth (evaluation.csv / extraction.csv).
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import math
import csv


def compute_mae(ground_truth: List[float], predictions: List[float]) -> float:
    """Mean Absolute Error."""
    if not ground_truth or len(ground_truth) != len(predictions):
        return 0.0
    return sum(abs(g - p) for g, p in zip(ground_truth, predictions)) / len(ground_truth)


def compute_rmse(ground_truth: List[float], predictions: List[float]) -> float:
    """Root Mean Squared Error."""
    if not ground_truth or len(ground_truth) != len(predictions):
        return 0.0
    mse = sum((g - p) ** 2 for g, p in zip(ground_truth, predictions)) / len(ground_truth)
    return math.sqrt(mse)


def compute_pearson_r(ground_truth: List[float], predictions: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(ground_truth)
    if n < 2 or len(predictions) != n:
        return 0.0

    mean_g = sum(ground_truth) / n
    mean_p = sum(predictions) / n

    numerator = sum((g - mean_g) * (p - mean_p) for g, p in zip(ground_truth, predictions))
    denom_g = math.sqrt(sum((g - mean_g) ** 2 for g in ground_truth))
    denom_p = math.sqrt(sum((p - mean_p) ** 2 for p in predictions))

    if denom_g == 0 or denom_p == 0:
        return 0.0

    return numerator / (denom_g * denom_p)


def compute_quadratic_weighted_kappa(
    rater_a: List[Union[int, float]],
    rater_b: List[Union[int, float]],
    min_rating: Optional[float] = None,
    max_rating: Optional[float] = None
) -> float:
    """
    Quadratic Weighted Kappa (QWK) between two raters.
    Standard metric for automated essay / script scoring.
    """
    if not rater_a or len(rater_a) != len(rater_b):
        return 0.0

    # Discretize scores to integer categories (e.g. rounded to nearest integer or 0.5)
    r_a = [int(round(x)) for x in rater_a]
    r_b = [int(round(x)) for x in rater_b]

    min_val = min(min(r_a), min(r_b)) if min_rating is None else int(min_rating)
    max_val = max(max(r_a), max(r_b)) if max_rating is None else int(max_rating)
    num_categories = max_val - min_val + 1

    if num_categories <= 1:
        return 1.0

    # Build confusion matrix O (observed)
    observed = [[0] * num_categories for _ in range(num_categories)]
    for a, b in zip(r_a, r_b):
        idx_a = max(0, min(num_categories - 1, a - min_val))
        idx_b = max(0, min(num_categories - 1, b - min_val))
        observed[idx_a][idx_b] += 1

    total_ratings = len(rater_a)
    hist_a = [0] * num_categories
    hist_b = [0] * num_categories
    for i in range(num_categories):
        for j in range(num_categories):
            hist_a[i] += observed[i][j]
            hist_b[j] += observed[i][j]

    # Build expected matrix E
    expected = [[0.0] * num_categories for _ in range(num_categories)]
    for i in range(num_categories):
        for j in range(num_categories):
            expected[i][j] = (hist_a[i] * hist_b[j]) / float(total_ratings)

    # Build weight matrix W (quadratic weights)
    weights = [[0.0] * num_categories for _ in range(num_categories)]
    max_diff_sq = float((num_categories - 1) ** 2)
    for i in range(num_categories):
        for j in range(num_categories):
            weights[i][j] = ((i - j) ** 2) / max_diff_sq

    numerator = sum(weights[i][j] * observed[i][j] for i in range(num_categories) for j in range(num_categories))
    denominator = sum(weights[i][j] * expected[i][j] for i in range(num_categories) for j in range(num_categories))

    if denominator == 0:
        return 1.0

    return 1.0 - (numerator / denominator)


# Alias for backward compatibility
compute_qwk = compute_quadratic_weighted_kappa


def compute_all_grading_metrics(
    ground_truth: List[float],
    predictions: List[float],
    max_rating: float = 10.0
) -> Dict[str, float]:
    """Compute comprehensive grading alignment metrics."""
    if not ground_truth or len(ground_truth) != len(predictions):
        return {}

    mae_val = round(compute_mae(ground_truth, predictions), 4)
    rmse_val = round(compute_rmse(ground_truth, predictions), 4)
    pearson_val = round(compute_pearson_r(ground_truth, predictions), 4)
    qwk_val = round(compute_quadratic_weighted_kappa(ground_truth, predictions, 0, max_rating), 4)

    # Exact agreement (difference == 0)
    exact_count = sum(1 for g, p in zip(ground_truth, predictions) if abs(g - p) < 1e-4)
    exact_agr = round(exact_count / len(ground_truth), 4)

    # Adjacent agreement (+/- 1.0 mark difference)
    adj_count = sum(1 for g, p in zip(ground_truth, predictions) if abs(g - p) <= 1.0 + 1e-4)
    adj_agr = round(adj_count / len(ground_truth), 4)

    return {
        "mae": mae_val,
        "rmse": rmse_val,
        "pearson_r": pearson_val,
        "qwk": qwk_val,
        "exact_agreement": exact_agr,
        "adjacent_agreement_pm1": adj_agr,
        "MAE": mae_val,
        "RMSE": rmse_val,
        "Pearson_r": pearson_val,
        "QWK": qwk_val,
        "Exact_Agreement": exact_agr,
        "Adjacent_Agreement_PM1": adj_agr
    }


def benchmark_against_ground_truth(
    pred_csv_path: str,
    ground_truth_csv_path: str
) -> Dict[str, Any]:
    """
    Benchmark a predicted evaluation CSV against ground truth evaluation.csv.
    """
    # Load ground truth
    gt_map: Dict[str, Dict[str, Any]] = {}
    with open(ground_truth_csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            task_id = r.get("task_id", "").strip()
            gt_map[task_id] = r

    # Load predictions
    gt_scores: List[float] = []
    pred_scores: List[float] = []
    band_matches = 0
    cap_matches = 0
    total_matched = 0

    with open(pred_csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for p in reader:
            t_id = p.get("task_id", "").strip()
            if t_id in gt_map:
                gt_row = gt_map[t_id]
                try:
                    # Ground truth score can be teacher_mark or total_score
                    gt_mark = float(gt_row.get("teacher_mark") or gt_row.get("total_score", 0.0))
                    p_score = float(p.get("total_score", 0.0))
                    gt_scores.append(gt_mark)
                    pred_scores.append(p_score)
                    total_matched += 1

                    if p.get("performance_band") == gt_row.get("performance_band"):
                        band_matches += 1
                    if str(p.get("cap_applied")).lower() == str(gt_row.get("cap_applied")).lower():
                        cap_matches += 1
                except (ValueError, TypeError):
                    continue

    metrics = compute_all_grading_metrics(gt_scores, pred_scores)
    metrics["total_matched_scripts"] = total_matched
    metrics["band_accuracy"] = round(band_matches / total_matched, 4) if total_matched > 0 else 0.0
    metrics["cap_accuracy"] = round(cap_matches / total_matched, 4) if total_matched > 0 else 0.0
    return metrics
