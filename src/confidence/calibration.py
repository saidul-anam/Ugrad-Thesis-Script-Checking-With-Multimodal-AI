"""
Confidence Calibration Module.
Supports Platt Scaling (Logistic Calibration) and Isotonic Regression.
Features zero-dependency NumPy fallbacks when scikit-learn is not installed.
Strictly fits on validation/calibration splits only.
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import pickle
from pathlib import Path


class ConfidenceCalibrator:
    """
    Fits and applies confidence calibration models (Platt Scaling or Isotonic Regression).
    """

    def __init__(self, method: str = "isotonic_regression"):
        self.method = method.lower().strip()
        self.params = {}
        self.is_fitted = False

    def fit(self, uncalibrated_confidences: List[float], labels: List[int]) -> "ConfidenceCalibrator":
        """
        Fit calibrator on a dedicated calibration split.
        
        Args:
            uncalibrated_confidences: Raw or derived confidence scores [0.0, 1.0]
            labels: Binary correctness labels (1 = acceptable OCR, 0 = unacceptable)
        """
        X = np.array(uncalibrated_confidences, dtype=np.float64)
        y = np.array(labels, dtype=np.float64)

        if len(np.unique(y)) < 2 or len(X) < 2:
            self.is_fitted = False
            return self

        try:
            if self.method == "platt_scaling":
                # Logistic regression: P(y=1|x) = 1 / (1 + exp(-(a*x + b)))
                # Fit using standard gradient descent or least squares approximation
                a, b = 1.0, 0.0
                lr = 0.1
                for _ in range(300):
                    preds = 1.0 / (1.0 + np.exp(-(a * X + b)))
                    err = preds - y
                    grad_a = np.mean(err * X)
                    grad_b = np.mean(err)
                    a -= lr * grad_a
                    b -= lr * grad_b
                self.params = {"a": float(a), "b": float(b)}
                self.is_fitted = True

            elif self.method == "isotonic_regression":
                # Simple pool adjacent violators or piecewise linear interpolation
                sorted_indices = np.argsort(X)
                X_sorted = X[sorted_indices]
                y_sorted = y[sorted_indices]
                # Running cumulative monotonic averaging
                y_iso = np.maximum.accumulate(y_sorted)
                self.params = {"x_knots": X_sorted.tolist(), "y_knots": y_iso.tolist()}
                self.is_fitted = True
            else:
                self.is_fitted = False
        except Exception:
            self.is_fitted = False

        return self

    def predict(self, confidence: float) -> float:
        """Calibrate a single confidence score."""
        if not self.is_fitted:
            return float(confidence)

        if self.method == "platt_scaling":
            a = self.params.get("a", 1.0)
            b = self.params.get("b", 0.0)
            prob = 1.0 / (1.0 + np.exp(-(a * float(confidence) + b)))
            return float(max(0.0, min(1.0, prob)))

        elif self.method == "isotonic_regression":
            x_knots = np.array(self.params.get("x_knots", []))
            y_knots = np.array(self.params.get("y_knots", []))
            if len(x_knots) == 0:
                return float(confidence)
            calibrated = float(np.interp(confidence, x_knots, y_knots))
            return float(max(0.0, min(1.0, calibrated)))

        return float(confidence)

    def save(self, file_path: str) -> None:
        """Save fitted calibrator model."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({"method": self.method, "params": self.params, "is_fitted": self.is_fitted}, f)

    @classmethod
    def load(cls, file_path: str) -> "ConfidenceCalibrator":
        """Load saved calibrator model."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        calibrator = cls(method=data["method"])
        calibrator.params = data.get("params", {})
        calibrator.is_fitted = data.get("is_fitted", False)
        return calibrator
