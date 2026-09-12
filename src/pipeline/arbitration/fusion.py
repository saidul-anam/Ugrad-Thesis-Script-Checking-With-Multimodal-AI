"""
Evidence fusion: logistic combination of the per-candidate signals into an ambiguity score,
quantization into three verdicts, and fitting of the weights from human labels.

Signals (each in [0,1], 1 = perceptual ambiguity, None = unavailable):
    phonetic      = 1 - phonetic_plausibility
    writer        = writer_prior
    consensus     = consensus_signal
    forced_choice = forced_choice_signal

score = sigmoid(bias + sum_i w_i * (2*s_i - 1)) over available signals.
Missing signals contribute nothing (their centred value is 0), so a candidate with no crop
evidence is scored from the model-free signals alone.
"""

import math
from typing import Dict, List, Optional, Tuple, Iterable

from src.core.schemas import ArbitrationEvidence
from src.core.config import ArbitrationWeights

SIGNAL_NAMES = ("phonetic", "writer", "consensus", "forced_choice")

GENUINE = "GENUINE_ERROR"
AMBIGUITY = "HANDWRITING_AMBIGUITY"
UNCERTAIN = "UNCERTAIN"


def signals_from_evidence(ev: ArbitrationEvidence) -> Dict[str, Optional[float]]:
    return {
        "phonetic": ev.phonetic_signal,
        "writer": ev.writer_prior,
        "consensus": ev.consensus_signal,
        "forced_choice": ev.forced_choice_signal,
    }


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fuse_signals(signals: Dict[str, Optional[float]], w: ArbitrationWeights) -> float:
    z = w.bias
    for name in SIGNAL_NAMES:
        s = signals.get(name)
        if s is None:
            continue
        s = min(1.0, max(0.0, float(s)))
        z += getattr(w, name) * (2.0 * s - 1.0)
    return _sigmoid(z)


def fuse(ev: ArbitrationEvidence, w: ArbitrationWeights) -> float:
    return fuse_signals(signals_from_evidence(ev), w)


def quantize(score: float, threshold_ambiguity: float = 0.65, threshold_genuine: float = 0.35) -> str:
    if score >= threshold_ambiguity:
        return AMBIGUITY
    if score <= threshold_genuine:
        return GENUINE
    return UNCERTAIN


# ---------------------------------------------------------------------------
# Fitting (numpy logistic regression, no sklearn dependency)
# ---------------------------------------------------------------------------
def _design_row(signals: Dict[str, Optional[float]]) -> List[float]:
    row = []
    for name in SIGNAL_NAMES:
        s = signals.get(name)
        row.append(0.0 if s is None else 2.0 * min(1.0, max(0.0, float(s))) - 1.0)
    return row


def fit_weights(
    signal_rows: Iterable[Dict[str, Optional[float]]],
    labels: Iterable[int],
    l2: float = 0.05,
    epochs: int = 3000,
    lr: float = 0.1,
    init: Optional[ArbitrationWeights] = None,
) -> ArbitrationWeights:
    """
    Fit logistic weights so that score ~= P(label == 1), where label 1 = HANDWRITING_AMBIGUITY.
    Weights are constrained to be non-negative (every signal is defined so that higher = more
    ambiguous), which keeps the fitted model interpretable.
    """
    import numpy as np

    X = np.array([_design_row(s) for s in signal_rows], dtype=float)
    y = np.array(list(labels), dtype=float)
    if X.size == 0:
        return init or ArbitrationWeights()
    n, k = X.shape
    w0 = init or ArbitrationWeights()
    w = np.array([getattr(w0, name) for name in SIGNAL_NAMES], dtype=float)
    b = float(w0.bias)
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        grad_w = X.T @ g / n + l2 * w
        grad_b = float(g.mean())
        w -= lr * grad_w
        b -= lr * grad_b
        w = np.maximum(w, 0.0)
    return ArbitrationWeights(bias=float(b), **{name: float(w[i]) for i, name in enumerate(SIGNAL_NAMES)})


def choose_thresholds(
    scores: List[float],
    labels: List[int],
    target_genuine_precision: float = 0.9,
    target_ambiguity_precision: float = 0.9,
) -> Tuple[float, float]:
    """
    Pick (threshold_ambiguity, threshold_genuine) on labelled data.
    threshold_genuine: largest t such that among candidates with score <= t, >= target share are
    genuine (label 0). threshold_ambiguity: smallest t such that among candidates with score >= t,
    >= target share are ambiguous (label 1). Falls back to (0.65, 0.35) when data are too thin.
    """
    pairs = sorted(zip(scores, labels))
    if len(pairs) < 10:
        return 0.65, 0.35
    grid = [i / 100 for i in range(1, 100)]
    thr_gen = 0.35
    for t in grid:
        sel = [l for s, l in pairs if s <= t]
        if len(sel) >= 3 and (sel.count(0) / len(sel)) >= target_genuine_precision:
            thr_gen = t
    thr_amb = 0.65
    for t in reversed(grid):
        sel = [l for s, l in pairs if s >= t]
        if len(sel) >= 3 and (sel.count(1) / len(sel)) >= target_ambiguity_precision:
            thr_amb = t
    if thr_amb <= thr_gen:
        mid = (thr_amb + thr_gen) / 2
        thr_amb, thr_gen = min(0.99, mid + 0.05), max(0.01, mid - 0.05)
    return thr_amb, thr_gen
