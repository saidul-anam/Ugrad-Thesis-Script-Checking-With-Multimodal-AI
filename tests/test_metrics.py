"""
Unit tests for Character Error Rate (CER) and Word Error Rate (WER) Metrics.
"""

import unittest
from src.evaluation.cer import compute_cer
from src.evaluation.wer import compute_wer


class TestMetrics(unittest.TestCase):

    def test_cer_identical_strings(self):
        self.assertEqual(compute_cer("hello world", "hello world"), 0.0)

    def test_cer_substitution(self):
        # 1 substitution in length 5 ('c' -> 'h') -> 1/5 = 0.20
        self.assertAlmostEqual(compute_cer("cat", "hat"), 1.0 / 3.0, places=4)

    def test_cer_normalized(self):
        # Case insensitive when normalized
        self.assertEqual(compute_cer("HELLO WORLD", "hello world", normalize=True), 0.0)

    def test_wer_identical_strings(self):
        self.assertEqual(compute_wer("the quick brown fox", "the quick brown fox"), 0.0)

    def test_wer_single_word_substitution(self):
        # 1 word change out of 4 words -> 1/4 = 0.25
        self.assertAlmostEqual(compute_wer("the fast brown fox", "the quick brown fox"), 0.25, places=4)

    def test_wer_punctuation_normalization(self):
        # Normalization strips punctuation
        hyp = "Hello, world! It is a test."
        ref = "hello world it is a test"
        self.assertEqual(compute_wer(hyp, ref, normalize=True), 0.0)

    def test_empty_reference(self):
        self.assertEqual(compute_cer("", ""), 0.0)
        self.assertEqual(compute_wer("", ""), 0.0)


if __name__ == "__main__":
    unittest.main()
