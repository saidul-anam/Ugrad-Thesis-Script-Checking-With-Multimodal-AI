import pytest
from src.core.schemas import (
    Stage1TranscriptionResult,
    Stage2VerificationResult,
    AutocorrectionDiffItem,
    Stage3ErrorResult,
    LinguisticErrorItem,
    Stage4EvaluationResult,
    CriterionScore,
    CompleteEvaluationReport
)


def test_stage1_schema():
    res = Stage1TranscriptionResult(
        raw_transcript="Test transcript [illegible] [unclear: reading]",
        illegible_count=1,
        unclear_count=1,
        character_count=45,
        word_count=5,
        detected_script="English"
    )
    assert res.illegible_count == 1
    assert res.unclear_count == 1
    assert res.word_count == 5


def test_stage2_schema():
    diff = AutocorrectionDiffItem(
        stage1_output="correct",
        actual_handwritten="corect",
        reason="Model corrected student spelling error",
        context_snippet="this is corect"
    )
    res = Stage2VerificationResult(
        verified_transcript="this is corect",
        silent_corrections_fixed=[diff],
        total_corrections_count=1,
        verification_notes="Reverted 1 autocorrection"
    )
    assert res.total_corrections_count == 1
    assert res.silent_corrections_fixed[0].actual_handwritten == "corect"


def test_stage3_schema():
    err = LinguisticErrorItem(
        error_type="spelling",
        erroneous_text="corect",
        suggested_correction="correct",
        context_sentence="this is corect",
        explanation="Missing 'r'"
    )
    res = Stage3ErrorResult(
        errors=[err],
        spelling_error_count=1,
        grammar_error_count=0,
        syntax_error_count=0,
        punctuation_error_count=0,
        total_error_count=1,
        linguistic_summary="Single spelling mistake."
    )
    assert res.total_error_count == 1


def test_stage4_schema():
    crit = CriterionScore(
        criterion_id="c1",
        criterion_name="Content",
        max_marks=5.0,
        awarded_marks=4.0,
        justification="Good explanation"
    )
    res = Stage4EvaluationResult(
        subject="English",
        question_type="Essay",
        criteria_scores=[crit],
        content_raw_score=4.0,
        linguistic_penalty=0.5,
        final_score=3.5,
        total_max_marks=5.0,
        percentage=70.0,
        overall_feedback="Well written.",
        actionable_recommendations=["Improve spelling."]
    )
    assert res.final_score == 3.5
    assert res.percentage == 70.0


def test_complete_report_serialization():
    s1 = Stage1TranscriptionResult(raw_transcript="raw", illegible_count=0, unclear_count=0, character_count=3, word_count=1, detected_script="English")
    s2 = Stage2VerificationResult(verified_transcript="raw", silent_corrections_fixed=[], total_corrections_count=0, verification_notes="")
    s3 = Stage3ErrorResult(errors=[], spelling_error_count=0, grammar_error_count=0, syntax_error_count=0, punctuation_error_count=0, total_error_count=0, linguistic_summary="")
    s4 = Stage4EvaluationResult(subject="Eng", question_type="Q", criteria_scores=[], content_raw_score=5.0, linguistic_penalty=0.0, final_score=5.0, total_max_marks=10.0, percentage=50.0, overall_feedback="", actionable_recommendations=[])

    report = CompleteEvaluationReport(
        script_id="test_001",
        image_path="test.jpg",
        model_id="google/gemma-4-31b-it",
        timestamp="2026-08-25T00:00:00",
        stage1_transcription=s1,
        stage2_verification=s2,
        stage3_errors=s3,
        stage4_evaluation=s4
    )

    data = report.model_dump()
    assert data["script_id"] == "test_001"
    assert data["model_id"] == "google/gemma-4-31b-it"
