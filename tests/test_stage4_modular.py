import os
import tempfile
import pytest
from src.core.config import PipelineConfig
from src.core.schemas import (
    AlignedAnswerItem,
    ExtractionResult,
    PageExtractionResult,
    Stage1TranscriptionResult,
    Stage2VerificationResult,
    Stage3ErrorResult,
    LinguisticErrorItem,
    Stage4EvaluationResult,
    ExtractedQuestion
)
from src.engine.mock_engine import MockGemmaEngine
from src.pipeline.answer_segmenter import (
    normalize_header_text,
    extract_header_qno,
    segment_script_into_questions
)
from src.prompts.stage4_modular import build_modular_question_prompt
from src.pipeline.stage4_evaluator import Stage4Evaluator
from src.pipeline.orchestrator import ScriptCheckingPipeline
from src.utils.export_utils import export_report_markdown


def test_header_normalization_and_extraction():
    # Test OCR variants normalization
    assert "No. 7" in normalize_header_text("Ans to the Q No. Z")
    assert "Ans: to the Q. No. 1(B)" in normalize_header_text("Dans to the Q")

    # Test Q extraction
    assert extract_header_qno("Ans to the Question No 1(A)\nHere is answer") == "1(A)"
    assert extract_header_qno("Ans to the Q No 1(B)\nHere is answer") == "1(B)"
    assert extract_header_qno("Ans to the Q No. 02\nHere is answer") == "2"
    assert extract_header_qno("Ans to the Q. No. 11\nTheme: The poem is about dreams") == "11"
    assert extract_header_qno("Ans to the Q No 7\nArtificial intelligence is changing the world") == "7"


def test_answer_segmentation():
    # Create synthetic multi-page extraction result
    p1_s1 = Stage1TranscriptionResult(raw_transcript="Ans to the Question No 1(A)\n(a) False.\n(b) True.", word_count=10)
    p1_s2 = Stage2VerificationResult(verified_transcript="Ans to the Question No 1(A)\n(a) False.\n(b) True.", total_corrections_count=0, silent_corrections_fixed=[])
    err = LinguisticErrorItem(
        error_type="Spelling",
        erroneous_text="liberats",
        suggested_correction="liberates",
        context_sentence="Education liberats our mind.",
        explanation="Misspelled word",
        page_number=1,
        severity="Minor"
    )
    p1_s3 = Stage3ErrorResult(total_error_count=1, spelling_error_count=1, grammar_error_count=0, syntax_error_count=0, punctuation_error_count=0, errors=[err], linguistic_summary="1 error")

    page1 = PageExtractionResult(
        page_no=1,
        image_path="/tmp/p1.png",
        stage1_transcription=p1_s1,
        stage2_verification=p1_s2,
        stage3_errors=p1_s3
    )

    p2_s1 = Stage1TranscriptionResult(raw_transcript="Ans to the Question No 2\nNelson Mandela was a South African leader.", word_count=10)
    p2_s2 = Stage2VerificationResult(verified_transcript="Ans to the Question No 2\nNelson Mandela was a South African leader.", total_corrections_count=0, silent_corrections_fixed=[])
    p2_s3 = Stage3ErrorResult(total_error_count=0, spelling_error_count=0, grammar_error_count=0, syntax_error_count=0, punctuation_error_count=0, errors=[], linguistic_summary="")

    page2 = PageExtractionResult(
        page_no=2,
        image_path="/tmp/p2.png",
        stage1_transcription=p2_s1,
        stage2_verification=p2_s2,
        stage3_errors=p2_s3
    )

    extraction = ExtractionResult(
        script_id="TEST_001",
        image_path="/tmp/test.pdf",
        timestamp="2026-09-12T00:00:00",
        pages=[page1, page2],
        stage1_transcription=Stage1TranscriptionResult(raw_transcript=p1_s1.raw_transcript + "\n\n" + p2_s1.raw_transcript, word_count=20),
        stage2_verification=Stage2VerificationResult(verified_transcript=p1_s2.verified_transcript + "\n\n" + p2_s2.verified_transcript, total_corrections_count=0, silent_corrections_fixed=[]),
        stage3_errors=Stage3ErrorResult(total_error_count=1, spelling_error_count=1, grammar_error_count=0, syntax_error_count=0, punctuation_error_count=0, errors=[err], linguistic_summary="1 spelling error")
    )

    question_obj = ExtractedQuestion(
        question_id="SE_11_Q1",
        subject="English",
        paper="1st Paper",
        total_marks=100.0,
        question_text="Full question text",
        sub_questions=[
            {"q_no": "1(A)", "marks": 5.0, "title": "Multiple Choice Questions"},
            {"q_no": "2", "marks": 10.0, "title": "Open-ended Comprehension Questions"}
        ]
    )

    answers = segment_script_into_questions(extraction, question_obj)
    assert len(answers) == 2

    q1 = answers[0]
    assert q1.q_no == "1(A)"
    assert q1.page_numbers == [1]
    assert len(q1.errors) == 1
    assert q1.errors[0]["erroneous_text"] == "liberats"

    q2 = answers[1]
    assert q2.q_no == "2"
    assert q2.page_numbers == [2]
    assert len(q2.errors) == 0


def test_build_modular_prompt():
    ans = AlignedAnswerItem(
        q_no="2",
        q_name="Open-ended Questions",
        answer_text="Nelson Mandela fought against racial oppression.",
        page_numbers=[2],
        errors=[],
        word_count=7,
        character_count=48
    )

    prompt = build_modular_question_prompt(
        answer=ans,
        question_prompt_text="Answer the following questions: Why is Nelson Mandela famous?",
        max_marks=10.0,
        subject="English",
        rubric_penalties={"spelling": 0.5}
    )

    assert "Question Number: 2" in prompt
    assert "Maximum Marks: 10.0" in prompt
    assert "Nelson Mandela fought against racial oppression." in prompt


def test_stage4_modular_evaluator():
    engine = MockGemmaEngine()
    evaluator = Stage4Evaluator(engine=engine)

    ans = AlignedAnswerItem(
        q_no="2",
        q_name="Open-ended Questions",
        answer_text="Nelson Mandela fought against racial oppression.",
        page_numbers=[2],
        errors=[],
        word_count=7,
        character_count=48
    )

    question_obj = ExtractedQuestion(
        question_id="SE_11_Q1",
        subject="English",
        paper="1st Paper",
        total_marks=100.0,
        question_text="Full question text",
        sub_questions=[{"q_no": "2", "marks": 10.0, "title": "Questions"}]
    )

    rubric_data = {
        "subject": "English",
        "penalties": {"spelling": 0.5, "grammar": 0.5}
    }

    gt_dict = {"2": 8.0}

    # Evaluate modular
    res = evaluator.evaluate_modular(
        answers=[ans],
        question_obj=question_obj,
        rubric_data=rubric_data,
        ground_truth_marks=gt_dict
    )

    assert res.eval_mode == "modular"
    assert len(res.question_evaluations) == 1
    q_eval = res.question_evaluations[0]
    assert q_eval.q_no == "2"
    assert q_eval.human_gt == 8.0
    assert q_eval.delta is not None
    assert res.mae_vs_human is not None
    assert 0.0 <= q_eval.awarded_marks <= 10.0


def test_orchestrator_modular_vs_monolithic():
    engine = MockGemmaEngine()
    config = PipelineConfig()
    pipeline = ScriptCheckingPipeline(config=config, engine=engine)

    p1_s1 = Stage1TranscriptionResult(raw_transcript="Ans to the Question No 1(A)\n(a) False.\n(b) True.", word_count=10)
    p1_s2 = Stage2VerificationResult(verified_transcript="Ans to the Question No 1(A)\n(a) False.\n(b) True.", total_corrections_count=0, silent_corrections_fixed=[])
    p1_s3 = Stage3ErrorResult(total_error_count=0, spelling_error_count=0, grammar_error_count=0, syntax_error_count=0, punctuation_error_count=0, errors=[], linguistic_summary="No errors")

    page1 = PageExtractionResult(
        page_no=1,
        image_path="/tmp/p1.png",
        stage1_transcription=p1_s1,
        stage2_verification=p1_s2,
        stage3_errors=p1_s3
    )

    extraction = ExtractionResult(
        script_id="TEST_MOCK_01",
        image_path="/tmp/test.pdf",
        timestamp="2026-09-12T00:00:00",
        pages=[page1],
        stage1_transcription=Stage1TranscriptionResult(raw_transcript=page1.stage1_transcription.raw_transcript, word_count=10),
        stage2_verification=Stage2VerificationResult(verified_transcript=page1.stage2_verification.verified_transcript, total_corrections_count=0, silent_corrections_fixed=[]),
        stage3_errors=Stage3ErrorResult(total_error_count=0, spelling_error_count=0, grammar_error_count=0, syntax_error_count=0, punctuation_error_count=0, errors=[], linguistic_summary="No errors"),
        metadata={"ground_truth_marks": {"1(A)": 5.0}}
    )

    question_obj = ExtractedQuestion(
        question_id="SE_11_Q1",
        subject="English",
        paper="1st Paper",
        total_marks=100.0,
        question_text="Q1: Choose the right answer",
        sub_questions=[{"q_no": "1(A)", "marks": 5.0, "title": "MCQ"}]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test modular mode
        rep_modular = pipeline.evaluate_extracted_script(
            extraction_input=extraction,
            rubric_path="configs/rubrics/english_writing.yaml",
            question_input=question_obj,
            output_dir=tmpdir,
            eval_mode="modular"
        )
        assert rep_modular.stage4_evaluation.eval_mode == "modular"
        assert len(rep_modular.stage4_evaluation.question_evaluations) > 0
        assert os.path.exists(os.path.join(tmpdir, "TEST_MOCK_01", "evaluation_report.md"))

        # Test monolithic mode
        rep_monolithic = pipeline.evaluate_extracted_script(
            extraction_input=extraction,
            rubric_path="configs/rubrics/english_writing.yaml",
            question_input=question_obj,
            output_dir=tmpdir,
            eval_mode="monolithic"
        )
        assert rep_monolithic.stage4_evaluation.eval_mode == "monolithic"
        assert len(rep_monolithic.stage4_evaluation.criteria_scores) > 0
