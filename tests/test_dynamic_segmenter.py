import pytest
from src.core.schemas import (
    ExtractedQuestion,
    ExtractionResult,
    PageExtractionResult,
    Stage1TranscriptionResult,
    Stage2VerificationResult,
    Stage3ErrorResult
)
from src.pipeline.answer_segmenter import (
    to_arabic_digits,
    extract_header_qno,
    segment_script_into_questions
)


def test_to_arabic_digits():
    assert to_arabic_digits("১ নং প্রশ্নের উত্তর (ক)") == "1 নং প্রশ্নের উত্তর (ক)"
    assert to_arabic_digits("প্রশ্ন নং ১০") == "প্রশ্ন নং 10"
    assert to_arabic_digits("English text 123") == "English text 123"


def test_bengali_header_extraction():
    # Bengali digit with subpart in parentheses
    assert extract_header_qno("১ নং প্রশ্নের উত্তর (ক)\nএখানে উত্তর লেখা হলো।") == "1(A)"
    assert extract_header_qno("১(ক) নং প্রশ্নের উত্তর\nউত্তর শুরু...") == "1(A)"
    # Bengali subpart (খ) following (ক)
    assert extract_header_qno("(খ)\nউত্তর...", current_parent="1(A)") == "1(B)"
    # Standard Bengali question number
    assert extract_header_qno("৭ নং প্রশ্নের উত্তর\nসারাংশ লেখা হলো...") == "7"
    assert extract_header_qno("১০ নং প্রশ্নের উত্তর\nভাবসম্প্রসারণ...") == "10"


def test_english_standard_and_subpart_extraction():
    assert extract_header_qno("Ans: to the Q. No. 1\n\n(A)\na) -> iii") == "1(A)"
    assert extract_header_qno("(B)\nAns: The writer cannot write...", current_parent="1(A)") == "1(B)"
    assert extract_header_qno("Ans to the Question No-02\nAnswer starts here") == "2"
    assert extract_header_qno("Ans to the Question No -06\nAnswer text") == "6"
    assert extract_header_qno("Ans to the question no-8\nA Lion was sleeping...") == "8"


def test_dynamic_question_schema_keyword_matching():
    # Schema with custom sub-question topic
    q_schema = ExtractedQuestion(
        question_id="PHY_11_Q1",
        language="english",
        question_text="Physics Final Exam 2026",
        sub_questions=[
            {"q_no": "3", "name": "Thermodynamics and Heat Engine Carnot Cycle", "marks": 10.0},
            {"q_no": "5", "name": "Electromagnetic Induction and Faraday Laws", "marks": 10.0}
        ]
    )
    # Student wrote topic title instead of clear question number
    sec1 = "Ans to Q:\nCarnot cycle thermodynamics heat engine principles.\nWork done..."
    sec2 = "Ans: Faraday electromagnetic induction laws and magnetic flux."

    assert extract_header_qno(sec1, question_obj=q_schema) == "3"
    assert extract_header_qno(sec2, question_obj=q_schema) == "5"


def test_multi_page_continuation_segmentation():
    """Verify that multi-page answers (e.g. story starting on Page 10 and continuing on Page 11) are unified."""
    p10 = PageExtractionResult(
        page_no=10,
        image_path="page_10.png",
        has_red_ink=True,
        red_pixel_count=1000,
        stage1_transcription=Stage1TranscriptionResult(raw_transcript="Ans to the question no-8\nA Lion and a Mouse lived in a forest...", word_count=20),
        stage2_verification=Stage2VerificationResult(verified_transcript="Ans to the question no-8\nA Lion and a Mouse lived in a forest...", total_corrections_count=0),
        stage3_errors=Stage3ErrorResult(errors=[])
    )
    # Continuation page: no question header, just student continuing the story
    p11 = PageExtractionResult(
        page_no=11,
        image_path="page_11.png",
        has_red_ink=False,
        red_pixel_count=0,
        stage1_transcription=Stage1TranscriptionResult(raw_transcript="The mouse cut the net with its sharp teeth and freed the lion.", word_count=14),
        stage2_verification=Stage2VerificationResult(verified_transcript="The mouse cut the net with its sharp teeth and freed the lion.", total_corrections_count=0),
        stage3_errors=Stage3ErrorResult(errors=[])
    )
    # Next question on Page 12
    p12 = PageExtractionResult(
        page_no=12,
        image_path="page_12.png",
        has_red_ink=True,
        red_pixel_count=800,
        stage1_transcription=Stage1TranscriptionResult(raw_transcript="Ans to the Question No-9\nDear Friend, I received your email...", word_count=15),
        stage2_verification=Stage2VerificationResult(verified_transcript="Ans to the Question No-9\nDear Friend, I received your email...", total_corrections_count=0),
        stage3_errors=Stage3ErrorResult(errors=[])
    )

    extraction = ExtractionResult(
        script_id="TEST_SCRIPT",
        image_path="test_script.pdf",
        timestamp="2026-09-12T00:00:00",
        pages=[p10, p11, p12],
        stage1_transcription=Stage1TranscriptionResult(raw_transcript="", word_count=0),
        stage2_verification=Stage2VerificationResult(verified_transcript="", total_corrections_count=0),
        stage3_errors=Stage3ErrorResult(errors=[])
    )

    aligned = segment_script_into_questions(extraction)

    q_dict = {item.q_no: item for item in aligned}
    assert "8" in q_dict
    assert "9" in q_dict
    # Question 8 must span both Page 10 and Page 11
    assert q_dict["8"].page_numbers == [10, 11]
    assert "cut the net" in q_dict["8"].answer_text
    # Question 9 on Page 12
    assert q_dict["9"].page_numbers == [12]


def test_question_aware_error_attribution_and_markdown():
    from src.utils.export_utils import export_extraction_summary_markdown
    from src.core.schemas import LinguisticErrorItem

    letter_err = LinguisticErrorItem(
        error_type="spelling",
        erroneous_text="familyes",
        suggested_correction="families",
        context_sentence="Take care of your familyes.",
        explanation="Incorrect plural form",
        question_no="10"
    )

    p1 = PageExtractionResult(
        page_no=1,
        image_path="page_1.png",
        has_red_ink=False,
        red_pixel_count=0,
        stage1_transcription=Stage1TranscriptionResult(
            raw_transcript="Ans to the Question No-2\n| 1. Dropping out | 2. Work in laws house |\n\nAns to the Question No-10\nDear Sadia, Take care of your familyes.",
            word_count=20
        ),
        stage2_verification=Stage2VerificationResult(
            verified_transcript="Ans to the Question No-2\n| 1. Dropping out | 2. Work in laws house |\n\nAns to the Question No-10\nDear Sadia, Take care of your familyes.",
            total_corrections_count=0
        ),
        stage3_errors=Stage3ErrorResult(errors=[letter_err])
    )

    extraction = ExtractionResult(
        script_id="TEST_ALIGN",
        image_path="test.pdf",
        timestamp="2026-09-12T00:00:00",
        pages=[p1],
        stage1_transcription=Stage1TranscriptionResult(raw_transcript=p1.stage1_transcription.raw_transcript, word_count=20),
        stage2_verification=Stage2VerificationResult(verified_transcript=p1.stage2_verification.verified_transcript, total_corrections_count=0),
        stage3_errors=Stage3ErrorResult(errors=[letter_err], total_error_count=1, spelling_error_count=1),
        metadata={
            "aligned_answers": [
                {
                    "q_no": "2",
                    "q_name": "Flow Chart",
                    "answer_text": "| 1. Dropping out | 2. Work in laws house |",
                    "page_numbers": [1],
                    "errors": [],
                    "word_count": 8,
                    "character_count": 40
                },
                {
                    "q_no": "10",
                    "q_name": "Informal Letter",
                    "answer_text": "Dear Sadia, Take care of your familyes.",
                    "page_numbers": [1],
                    "errors": [letter_err.model_dump()],
                    "word_count": 8,
                    "character_count": 38
                }
            ]
        }
    )

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        out_md = export_extraction_summary_markdown(extraction, tmp.name)
        with open(out_md, "r", encoding="utf-8") as f:
            md_text = f.read()

    # Verify Question 2 is marked Objective with zero essay deductions
    assert "### Question 2: Flow Chart `[Objective]`" in md_text
    assert "Objective Question" in md_text
    # Verify Question 10 shows the true spelling error
    assert "### Question 10: Informal Letter `[Subjective Writing]`" in md_text
    assert "familyes" in md_text
    assert "families" in md_text
