"""
Unit tests for PageRouter, rubric_v2 hard cap constraints, and performance band logic.
"""

import unittest
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schemas import (
    Stage5Output,
    ScoreBreakdown,
    StructuralAudit,
    PerformanceBand,
    CapReason,
    TASK_MAX_MARKS
)
from ingestion.page_router import parse_page_range, load_manifest_from_csv, PageRouter
from stages.stage6_compressor import determine_performance_band, get_hard_cap_limit, run_stage6_compressor


class TestRubricV2AndRouter(unittest.TestCase):

    def test_page_range_parser(self):
        self.assertEqual(parse_page_range("3-4"), [2, 3])
        self.assertEqual(parse_page_range("15"), [14])
        self.assertEqual(parse_page_range("1-3"), [0, 1, 2])
        self.assertEqual(parse_page_range("7, 9"), [6, 8])
        self.assertEqual(parse_page_range(""), [])
        self.assertEqual(parse_page_range(None), [])

    def test_performance_band_assignment(self):
        # Max 10: Band 4 (8-10), Band 3 (6-7), Band 2 (4-5), Band 1 (1-3), Band 0 (0)
        self.assertEqual(determine_performance_band(10.0, 10.0), "Band 4")
        self.assertEqual(determine_performance_band(8.0, 10.0), "Band 4")
        self.assertEqual(determine_performance_band(6.5, 10.0), "Band 3")
        self.assertEqual(determine_performance_band(5.0, 10.0), "Band 2")
        self.assertEqual(determine_performance_band(2.0, 10.0), "Band 1")
        self.assertEqual(determine_performance_band(0.0, 10.0), "Band 0")

        # Max 5: Band 4 (4-5), Band 3 (3), Band 2 (2), Band 1 (1), Band 0 (0)
        self.assertEqual(determine_performance_band(4.5, 5.0), "Band 4")
        self.assertEqual(determine_performance_band(3.0, 5.0), "Band 3")
        self.assertEqual(determine_performance_band(2.0, 5.0), "Band 2")
        self.assertEqual(determine_performance_band(1.0, 5.0), "Band 1")

    def test_hard_cap_limits(self):
        self.assertEqual(get_hard_cap_limit("Graph_Chart", 10.0, "Graph_External_Facts"), 6.0)
        self.assertEqual(get_hard_cap_limit("Paragraph", 10.0, "Paragraph_Subdivisions"), 5.0)
        self.assertEqual(get_hard_cap_limit("Summary", 10.0, "Summary_Verbatim_Length"), 5.0)
        self.assertEqual(get_hard_cap_limit("Theme", 8.0, "Theme_Verbatim_Copy"), 4.0)
        self.assertEqual(get_hard_cap_limit("Letter_Email", 5.0, "Letter_Missing_Layout"), 2.0)
        self.assertIsNone(get_hard_cap_limit("Paragraph", 10.0, "None"))

    def test_stage6_hard_cap_enforcement(self):
        # Student scored 8/10 on Graph, but included external opinions -> capped at 6.0
        s5 = Stage5Output(
            task_type="Graph_Chart",
            max_mark_applied=10.0,
            score_breakdown=ScoreBreakdown(
                context_content_data=3.0,
                structure_format_brevity=2.0,
                language_mechanics=1.5,
                originality_comparisons_paraphrase=1.5,
                total_score=8.0
            ),
            structural_audit=StructuralAudit(
                cap_applied=True,
                applied_cap_reason="Graph_External_Facts"
            ),
            performance_band="Band 4",
            stated_total=8.0
        )

        s6 = run_stage6_compressor(s5)
        self.assertEqual(s6.final_marks, 6.0)
        self.assertEqual(s6.performance_band, "Band 3")
        self.assertIn("Hard Cap Enforced", s6.error_detection)

    def test_manifest_loading(self):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "extraction.csv")
        if os.path.exists(csv_path):
            manifests = load_manifest_from_csv(csv_path)
            self.assertIn("SE_11_Q1_0001", manifests)
            m1 = manifests["SE_11_Q1_0001"]
            self.assertIn("CHART_001", m1.questions)
            self.assertEqual(m1.questions["CHART_001"].page_indices, [2, 3])  # Pages 3-4 -> [2, 3]


if __name__ == "__main__":
    unittest.main()
