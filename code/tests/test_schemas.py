"""
Unit tests for schemas and configurations matching rubric_v2.txt and evaluation.csv.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schemas import (
    Stage1Output,
    Stage2Output,
    Stage3Output,
    Stage4Output,
    Stage5Output,
    Stage6Output,
    ScoreBreakdown,
    StructuralAudit,
    AttemptStatus,
    ErrorAnalysis,
    QuestionSegment,
    ScriptManifest,
    UncertaintyArea,
    Stage2Decision,
    PipelineConfig,
    EnabledStagesConfig,
    PipelineResult
)


class TestSchemas(unittest.TestCase):

    def test_stage1_serialization(self):
        s1 = Stage1Output(
            QUESTION_TEXT="Ans to Q. No. 8",
            STUDENT_ANSWER="The sources of the USA electricity...",
            UNCERTAINTY_AREAS=[UncertaintyArea(location="line 1", text_guess="pie-chart", reason="faint stroke")],
            extracted_text_raw="The sources... [struck: 24%]",
            struck_tokens=["24%"],
            word_count=120
        )
        d = s1.to_dict()
        s1_back = Stage1Output.from_dict(d)
        self.assertEqual(s1.QUESTION_TEXT, s1_back.QUESTION_TEXT)
        self.assertEqual(s1.STUDENT_ANSWER, s1_back.STUDENT_ANSWER)
        self.assertEqual(len(s1_back.UNCERTAINTY_AREAS), 1)
        self.assertEqual(s1_back.UNCERTAINTY_AREAS[0].text_guess, "pie-chart")
        self.assertEqual(s1_back.struck_tokens, ["24%"])

    def test_stage2_serialization(self):
        s2 = Stage2Output(
            operative_rubric="Rubric v2 Specification",
            examiner_note="Standard alignment",
            shadow_solution=None,
            decision=Stage2Decision.KEEP
        )
        d = s2.to_dict()
        s2_back = Stage2Output.from_dict(d)
        self.assertEqual(s2_back.decision, Stage2Decision.KEEP)
        self.assertEqual(s2_back.operative_rubric, "Rubric v2 Specification")

    def test_stage5_and_6_serialization(self):
        bd = ScoreBreakdown(
            context_content_data=2.0,
            structure_format_brevity=1.0,
            language_mechanics=1.0,
            originality_comparisons_paraphrase=1.0,
            total_score=5.0
        )
        audit = StructuralAudit(cap_applied=True, applied_cap_reason="Graph_External_Facts")
        errs = ErrorAnalysis(
            frequent_errors=["External moral opinions"],
            positive_aspects=["Accurate percentages"]
        )

        s5 = Stage5Output(
            task_type="Graph_Chart",
            max_mark_applied=10.0,
            score_breakdown=bd,
            structural_audit=audit,
            performance_band="Band 2",
            error_analysis=errs,
            feedback_summary="Good attempt with score cap.",
            stated_total=5.0
        )

        s6 = Stage6Output(
            task_type="Graph_Chart",
            max_mark_applied=10.0,
            final_marks=5.0,
            performance_band="Band 2",
            score_breakdown=bd,
            structural_audit=audit,
            error_analysis=errs,
            feedback_summary="Good attempt with score cap.",
            sum_check_passed=True,
            band_check_passed=True
        )

        d5 = s5.to_dict()
        s5_back = Stage5Output.from_dict(d5)
        self.assertEqual(s5_back.stated_total, 5.0)
        self.assertEqual(s5_back.score_breakdown.context_content_data, 2.0)
        self.assertTrue(s5_back.structural_audit.cap_applied)

        d6 = s6.to_dict()
        s6_back = Stage6Output.from_dict(d6)
        self.assertEqual(s6_back.final_marks, 5.0)
        self.assertEqual(s6_back.performance_band, "Band 2")

    def test_pipeline_result_to_csv_row(self):
        res = PipelineResult(
            task_id="CHART_001",
            script_id="SE_11_Q1_0001",
            max_mark=10.0,
            teacher_mark=4.0,
            total_score=5.0,
            performance_band="Band 2",
            context_content_data=2.0,
            structure_format_brevity=1.0,
            language_mechanics=1.0,
            originality_comparisons_paraphrase=1.0,
            cap_applied=True,
            cap_reason="Graph_External_Facts",
            feedback_summary="Capped test summary."
        )
        row = res.to_evaluation_csv_row()
        self.assertEqual(row["task_id"], "CHART_001")
        self.assertEqual(row["script_id"], "SE_11_Q1_0001")
        self.assertEqual(row["total_score"], 5)
        self.assertEqual(row["cap_reason"], "Graph_External_Facts")


if __name__ == "__main__":
    unittest.main()
