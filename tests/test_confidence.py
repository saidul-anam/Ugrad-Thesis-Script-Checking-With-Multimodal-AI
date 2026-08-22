"""
Unit tests for Confidence Aggregation, Estimation, and Calibration.
"""

import unittest
import numpy as np
from src.confidence.aggregation import aggregate_confidence
from src.confidence.estimator import ConfidenceEstimator
from src.confidence.calibration import ConfidenceCalibrator


class TestConfidence(unittest.TestCase):

    def test_aggregation_mean(self):
        scores = [0.8, 0.9, 1.0]
        agg = aggregate_confidence(scores, method="mean")
        self.assertAlmostEqual(agg, 0.9, places=4)

    def test_aggregation_minimum(self):
        scores = [0.8, 0.9, 0.6]
        agg = aggregate_confidence(scores, method="minimum")
        self.assertAlmostEqual(agg, 0.6, places=4)

    def test_aggregation_length_weighted(self):
        scores = [0.5, 1.0]
        weights = [10.0, 90.0]
        agg = aggregate_confidence(scores, weights=weights, method="length_weighted_mean")
        # (0.5 * 10 + 1.0 * 90) / 100 = 95 / 100 = 0.95
        self.assertAlmostEqual(agg, 0.95, places=4)

    def test_aggregation_geometric_mean(self):
        scores = [0.5, 0.8]
        agg = aggregate_confidence(scores, method="geometric_mean")
        expected = np.exp(np.mean([np.log(0.5), np.log(0.8)]))
        self.assertAlmostEqual(agg, float(expected), places=4)

    def test_aggregation_empty_list(self):
        self.assertIsNone(aggregate_confidence([]))

    def test_derived_confidence_estimator(self):
        clean_english = "The student writes a detailed answer explaining the significance of renewable energy."
        conf = ConfidenceEstimator.estimate_derived_confidence(clean_english)
        self.assertIsNotNone(conf)
        self.assertGreater(conf, 0.6)

        noisy_gibberish = "^~ ||| 1'[[ %%%%% }}}{"
        conf_noisy = ConfidenceEstimator.estimate_derived_confidence(noisy_gibberish)
        self.assertIsNotNone(conf_noisy)
        self.assertLess(conf_noisy, conf)

    def test_calibration_platt_scaling(self):
        X_train = [0.2, 0.4, 0.6, 0.8, 0.9]
        y_train = [0, 0, 1, 1, 1]
        calib = ConfidenceCalibrator(method="platt_scaling")
        calib.fit(X_train, y_train)
        pred = calib.predict(0.85)
        self.assertGreater(pred, 0.5)


if __name__ == "__main__":
    unittest.main()
