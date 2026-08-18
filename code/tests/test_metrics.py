"""
Unit tests for OCR metrics (CER/WER) and Grading metrics (QWK, MAE, RMSE).
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.ocr_metrics import compute_cer, compute_wer, normalize_bangla_text, evaluate_ocr_batch
from eval.grading_metrics import compute_mae, compute_rmse, compute_qwk, compute_all_grading_metrics


class TestMetrics(unittest.TestCase):

    def test_bangla_unicode_normalization(self):
        text1 = "গতিশক্তি"
        text2 = "গতিশক্তি "
        self.assertEqual(normalize_bangla_text(text1), normalize_bangla_text(text2))

    def test_cer_wer_exact_match(self):
        ref = "গতিশক্তি হলো বস্তুর গতির জন্য কাজ করার সামর্থ্য"
        hyp = "গতিশক্তি হলো বস্তুর গতির জন্য কাজ করার সামর্থ্য"
        self.assertAlmostEqual(compute_cer(ref, hyp), 0.0)
        self.assertAlmostEqual(compute_wer(ref, hyp), 0.0)

    def test_cer_wer_mismatch(self):
        ref = "kinetic energy is work"
        hyp = "kinetic energy is power"
        cer = compute_cer(ref, hyp)
        wer = compute_wer(ref, hyp)
        self.assertGreater(cer, 0.0)
        self.assertAlmostEqual(wer, 0.25)  # 1 word wrong out of 4

    def test_grading_metrics_exact(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(compute_mae(y_true, y_pred), 0.0)
        self.assertAlmostEqual(compute_rmse(y_true, y_pred), 0.0)
        self.assertAlmostEqual(compute_qwk(y_true, y_pred), 1.0)

    def test_grading_metrics_divergent(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [2.0, 2.0, 4.0, 3.0, 5.0]
        metrics = compute_all_grading_metrics(y_true, y_pred)
        self.assertGreater(metrics["MAE"], 0.0)
        self.assertLessEqual(metrics["QWK"], 1.0)
        self.assertGreater(metrics["Adjacent_Agreement_PM1"], 0.99)


if __name__ == "__main__":
    unittest.main()
