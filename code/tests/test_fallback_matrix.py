"""
Exhaustive test suite for the Stage Toggling & Fallback Matrix.
Tests all 2^6 = 64 stage combinations with the Mock client.
"""

import unittest
import sys
import os
import itertools
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schemas import PipelineConfig, EnabledStagesConfig
from orchestrator import run_pipeline, resolve_final_ocr, resolve_final_marks
from model_client.mock_client import MockClient


class TestFallbackMatrix(unittest.TestCase):

    def setUp(self):
        self.mock_client = MockClient()
        self.test_image = Image.new("RGB", (100, 100), color=(255, 255, 255))
        self.rubric = "Rubric v2 Specification: 10 marks total."
        self.reference_solution = "Reference Context."

    def test_all_64_stage_combinations(self):
        """Test every possible on/off combination of the 6 stages."""
        stage_names = [
            "extractor_a",
            "rubric_aligner",
            "extractor_b",
            "ocr_supervisor",
            "examiner",
            "compressor"
        ]

        comb_count = 0
        for toggle_tuple in itertools.product([True, False], repeat=6):
            enabled_dict = dict(zip(stage_names, toggle_tuple))
            config = PipelineConfig(
                enabled_stages=EnabledStagesConfig.from_dict(enabled_dict)
            )
            config.logging.save_per_stage_json = False

            result = run_pipeline(
                pages=[self.test_image],
                rubric=self.rubric,
                reference_solution=self.reference_solution,
                config=config,
                task_type="Graph_Chart",
                max_mark=10.0,
                task_id="CHART_001",
                student_id="test_student",
                model_client=self.mock_client,
                verbose=False
            )

            # Assert result is well-formed
            self.assertIsNotNone(result)
            self.assertIsInstance(result.total_score, (int, float))
            self.assertIsInstance(result.performance_band, str)
            comb_count += 1

        self.assertEqual(comb_count, 64)

    def test_stage2_disabled_fallback(self):
        """When Stage 2 is disabled, operative_rubric must equal original rubric verbatim."""
        config = PipelineConfig(
            enabled_stages=EnabledStagesConfig(
                extractor_a=True, rubric_aligner=False, extractor_b=True,
                ocr_supervisor=True, examiner=True, compressor=True
            )
        )
        config.logging.save_per_stage_json = False

        res = run_pipeline(
            pages=[self.test_image],
            rubric=self.rubric,
            reference_solution=self.reference_solution,
            config=config,
            task_type="Graph_Chart",
            max_mark=10.0,
            model_client=self.mock_client
        )

        self.assertIsNotNone(res.stage2_output)
        self.assertEqual(res.stage2_output.operative_rubric, self.rubric)
        self.assertIsNone(res.stage2_output.examiner_note)

    def test_stages_3_and_4_disabled_fallback(self):
        """When Stages 3 & 4 are disabled, final OCR must come directly from Stage 1 (Candidate A)."""
        config = PipelineConfig(
            enabled_stages=EnabledStagesConfig(
                extractor_a=True, rubric_aligner=True, extractor_b=False,
                ocr_supervisor=False, examiner=True, compressor=True
            )
        )
        config.logging.save_per_stage_json = False

        res = run_pipeline(
            pages=[self.test_image],
            rubric=self.rubric,
            reference_solution=self.reference_solution,
            config=config,
            task_type="Graph_Chart",
            max_mark=10.0,
            model_client=self.mock_client
        )

        self.assertIsNotNone(res.stage1_output)
        self.assertEqual(res.final_ocr_answer, res.stage1_output.STUDENT_ANSWER)

    def test_stage_6_disabled_fallback(self):
        """When Stage 6 is disabled, final_marks must be parsed from Stage 5 with 'not verified' flag."""
        config = PipelineConfig(
            enabled_stages=EnabledStagesConfig(
                extractor_a=True, rubric_aligner=True, extractor_b=True,
                ocr_supervisor=True, examiner=True, compressor=False
            )
        )
        config.logging.save_per_stage_json = False

        res = run_pipeline(
            pages=[self.test_image],
            rubric=self.rubric,
            reference_solution=self.reference_solution,
            config=config,
            task_type="Graph_Chart",
            max_mark=10.0,
            model_client=self.mock_client
        )

        self.assertIsNone(res.stage6_output)
        self.assertEqual(res.total_score, 5.0)
        self.assertIn("not verified", str(res.error_detection))


if __name__ == "__main__":
    unittest.main()
