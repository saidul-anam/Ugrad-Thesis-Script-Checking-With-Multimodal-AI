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
    assert "Ans to the Q" in normalize_header_text("Dans to the Q")

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


def test_stage4_dynamic_subpart_prompt_slicing():
    from src.pipeline.stage4_evaluator import get_sub_question_prompt_and_marks
    from src.core.schemas import ExtractedQuestion

    question_text = """
1. Read the passage and answer questions A, B and C.
Passage text about technology and climate change...

A. Choose the correct answer from the following alternatives:
(a) What does climate mean?

B. Answer the following questions:
(a) Explain the effects of global warming.

C. Fill in the blanks with suitable words:
(a) The ozone layer protects us.

2. Read the following passage...
"""
    q_obj = ExtractedQuestion(
        question_id="TEST_Q",
        question_text=question_text,
        sub_questions=[
            {"q_no": "1(A)", "marks": 5.0},
            {"q_no": "1(B)", "marks": 10.0},
            {"q_no": "1(C)", "marks": 5.0}
        ]
    )

    prompt_a, marks_a = get_sub_question_prompt_and_marks("1(A)", q_obj)
    assert "Instructions (Part A):" in prompt_a
    assert "Choose the correct answer" in prompt_a
    assert marks_a == 5.0

    prompt_b, marks_b = get_sub_question_prompt_and_marks("1(B)", q_obj)
    assert "Instructions (Part B):" in prompt_b
    assert "Answer the following questions" in prompt_b
    assert marks_b == 10.0

    prompt_c, marks_c = get_sub_question_prompt_and_marks("1(C)", q_obj)
    assert "Instructions (Part C):" in prompt_c
    assert "Fill in the blanks" in prompt_c
    assert marks_c == 5.0


def test_evaluation_report_mentions_teacher_marks_at_end_of_table():
    from src.core.schemas import CompleteEvaluationReport, QuestionEvaluationItem, Stage4EvaluationResult
    from src.utils.export_utils import export_report_markdown

    qe1 = QuestionEvaluationItem(
        q_no="1",
        q_name="Reading Comprehension",
        max_marks=11.0,
        awarded_marks=11.0,
        human_ground_truth=11.0,
        scoring_status="scored"
    )
    qe2 = QuestionEvaluationItem(
        q_no="2",
        q_name="Comprehension Questions",
        max_marks=10.0,
        awarded_marks=9.0,
        human_ground_truth=10.0,
        scoring_status="scored"
    )

    s4 = Stage4EvaluationResult(
        subject="English",
        question_type="Junior_Secondary",
        question_id="SE_11_Q1",
        eval_mode="modular",
        question_evaluations=[qe1, qe2],
        content_raw_score=20.0,
        linguistic_penalty=0.0,
        final_score=20.0,
        total_max_marks=21.0,
        percentage=95.2,
        overall_feedback="Good work.",
        actionable_recommendations=["Keep practicing."]
    )

    p1_s1 = Stage1TranscriptionResult(raw_transcript="Ans to Q1", word_count=5)
    p1_s2 = Stage2VerificationResult(verified_transcript="Ans to Q1", total_corrections_count=0, silent_corrections_fixed=[])
    p1_s3 = Stage3ErrorResult(total_error_count=0, spelling_error_count=0, grammar_error_count=0, syntax_error_count=0, punctuation_error_count=0, errors=[], linguistic_summary="")

    # Test with script SE_11_Q1_0002 which exists in data/ground_truth/transcripts/teacher_marks/SE_11_Q1_teacher_marks.csv
    report = CompleteEvaluationReport(
        script_id="SE_11_Q1_0002",
        image_path="/tmp/test.pdf",
        timestamp="2026-10-09T22:00:00",
        stage1_transcription=p1_s1,
        stage2_verification=p1_s2,
        stage3_errors=p1_s3,
        stage4_evaluation=s4,
        metadata={
            "ground_truth_marks": {"1": 11.0, "2": 10.0, "total_marks": 70.0},
            "ground_truth_source": "data/ground_truth/transcripts/teacher_marks/SE_11_Q1_teacher_marks.csv"
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "evaluation_report.md")
        export_report_markdown(report, md_path)

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Official Ground Truth is cleanly separated from Stage 0b
        assert "SE_11_Q1_teacher_marks.csv" in content
        assert "Model-Detected Red-Ink Marks (Stage 0b)" in content
        assert "autonomous VLM, unverified" in content

        # 2. Final row in Stage 4 table has Human GT and delta
        assert "| **TOTAL** | **All Questions** | - | - | - | **21.0** | **20.0** | - | - | **20.0** | **70.0** | **-50.0** | - |" in content

        # 3. Final Given Marks Summary Table has Teacher Marks (GT) row at the end
        assert "| **Teacher Marks (GT)** | **Verified Examiner (`SE_11_Q1_teacher_marks.csv`)" in content

        # 4. Teacher Ground Truth Marks Comparison callout is present below the table
        assert "> 🧑‍🏫 **Teacher Ground Truth Marks Comparison (`SE_11_Q1_teacher_marks.csv`)" in content
        assert "- **Teacher Total Score**: **70.0 / 21.0**" in content
        assert "- **AI Evaluation Score**: **20.0 / 21.0**" in content
        assert "- **Score Discrepancy (Δ AI - Teacher)**: **-50.0 marks**" in content


def test_evaluation_report_without_teacher_marks():
    from src.core.schemas import CompleteEvaluationReport, QuestionEvaluationItem, Stage4EvaluationResult
    from src.utils.export_utils import export_report_markdown

    qe1 = QuestionEvaluationItem(
        q_no="1",
        q_name="Writing Task",
        max_marks=10.0,
        awarded_marks=8.0,
        scoring_status="scored"
    )

    s4 = Stage4EvaluationResult(
        subject="English",
        question_type="Junior_Secondary",
        eval_mode="modular",
        question_evaluations=[qe1],
        content_raw_score=8.0,
        linguistic_penalty=0.0,
        final_score=8.0,
        total_max_marks=10.0,
        percentage=80.0,
        overall_feedback="Well done.",
        actionable_recommendations=[]
    )

    p1_s1 = Stage1TranscriptionResult(raw_transcript="Ans to Q1", word_count=5)
    p1_s2 = Stage2VerificationResult(verified_transcript="Ans to Q1", total_corrections_count=0, silent_corrections_fixed=[])
    p1_s3 = Stage3ErrorResult(total_error_count=0, spelling_error_count=0, grammar_error_count=0, syntax_error_count=0, punctuation_error_count=0, errors=[], linguistic_summary="")

    # Completely unmapped dummy script
    report = CompleteEvaluationReport(
        script_id="UNMAPPED_STUDENT_9999",
        image_path="/tmp/test.pdf",
        timestamp="2026-10-09T22:00:00",
        stage1_transcription=p1_s1,
        stage2_verification=p1_s2,
        stage3_errors=p1_s3,
        stage4_evaluation=s4,
        metadata={}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "evaluation_report.md")
        export_report_markdown(report, md_path)

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should NOT contain Teacher Marks (GT) row or callout
        assert "Teacher Marks (GT)" not in content
        assert "> 🧑‍🏫 **Teacher Ground Truth Marks Comparison" not in content
        assert "| **TOTAL** | **All Questions** | - | - | - | **10.0** | **8.0** | - | - | **8.0** | - | - | - |" in content


def test_evaluation_report_segmented_writing():
    from src.core.schemas import (
        CompleteEvaluationReport, QuestionEvaluationItem, Stage4EvaluationResult,
        Stage1TranscriptionResult, Stage2VerificationResult, Stage3ErrorResult
    )
    from src.utils.export_utils import export_report_markdown

    qe1 = QuestionEvaluationItem(
        q_no="1",
        q_name="Multiple Choice Questions",
        page_numbers=[1],
        max_marks=7.0,
        awarded_marks=7.0,
        scoring_status="scored",
        student_writing="Answers to the Question no. 1\n(a) -> (iv)\n(b) -> (ii)"
    )
    qe4 = QuestionEvaluationItem(
        q_no="4",
        q_name="Synonyms and Antonyms",
        page_numbers=[3],
        max_marks=5.0,
        awarded_marks=5.0,
        scoring_status="scored",
        student_writing="Answers to the Question no.4\nwords | Synonym | Antonym\nTraditional | classy, cultural |"
    )
    qe5 = QuestionEvaluationItem(
        q_no="5",
        q_name="Cloze Test with Clues",
        page_numbers=[],
        max_marks=5.0,
        awarded_marks=0.0,
        scoring_status="missing",
        student_writing=None
    )

    s4 = Stage4EvaluationResult(
        subject="English",
        question_type="Junior_Secondary",
        eval_mode="modular",
        question_evaluations=[qe1, qe4, qe5],
        content_raw_score=12.0,
        linguistic_penalty=0.0,
        final_score=12.0,
        total_max_marks=17.0,
        percentage=70.6,
        overall_feedback="Good performance.",
        actionable_recommendations=[]
    )

    full_transcript = "Answers to the Question no. 1\n(a) -> (iv)\n(b) -> (ii)\n\nAnswers to the Question no.4\nwords | Synonym | Antonym\nTraditional | classy, cultural |"
    p1_s1 = Stage1TranscriptionResult(raw_transcript=full_transcript, word_count=20)
    p1_s2 = Stage2VerificationResult(verified_transcript=full_transcript, total_corrections_count=0, silent_corrections_fixed=[])
    p1_s3 = Stage3ErrorResult(total_error_count=0, spelling_error_count=0, grammar_error_count=0, syntax_error_count=0, punctuation_error_count=0, errors=[], linguistic_summary="")

    report = CompleteEvaluationReport(
        script_id="TEST_SEGMENT_01",
        image_path="/tmp/test.pdf",
        timestamp="2026-10-10T04:00:00",
        stage1_transcription=p1_s1,
        stage2_verification=p1_s2,
        stage3_errors=p1_s3,
        stage4_evaluation=s4,
        metadata={}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "evaluation_report.md")
        export_report_markdown(report, md_path)

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Heading reflects segmented writing
        assert "## Stage 2: Segmented Student Answers by Question" in content
        assert "The handwritten script has been segmented into question-aligned blocks" in content

        # 2. Q1 segmented writing and metadata
        assert "### Question 1: Multiple Choice Questions" in content
        assert "Page 1" in content
        assert "(a) -> (iv)" in content

        # 3. Q4 segmented writing and metadata
        assert "### Question 4: Synonyms and Antonyms" in content
        assert "Page 3" in content
        assert "Traditional | classy, cultural |" in content

        # 4. Missing question notice
        assert "### Question 5: Cloze Test with Clues" in content
        assert "No student answer segment identified for this question (unattempted)" in content

        # 5. Collapsible reference transcript is present
        assert "<details>" in content
        assert "Click to expand full unsegmented canonical transcript (Reference)" in content




