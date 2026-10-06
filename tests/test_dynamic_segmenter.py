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



# Question numbers come from the question paper, never from a built-in class-11 map: fingerprint
# tests therefore run against the SE_11_Q1 paper's sub-question names.
SE_11_Q1_SCHEMA_SUBQUESTIONS = [
    {"q_no": "1(A)", "name": "Q1(A): Multiple Choice Questions (Gaza Passage)"},
    {"q_no": "1(B)", "name": "Q1(B): Short Answer Questions (Gaza Passage)"},
    {"q_no": "2", "name": "Q2: Flow Chart (Adolescent Bride Miseries)"},
    {"q_no": "3", "name": "Q3: Poem Summary ('Hope is the thing with feathers')"},
    {"q_no": "4", "name": "Q4: Cloze Test with Clues (Success and Risk)"},
    {"q_no": "5", "name": "Q5: Cloze Test without Clues (Education)"},
    {"q_no": "6", "name": "Q6: Rearranging Sentences (Louis Pasteur)"},
    {"q_no": "7", "name": "Q7: Paragraph on Artificial Intelligence"},
    {"q_no": "8", "name": "Q8: Chart Analysis (USA Electricity Sources 1980)"},
    {"q_no": "9", "name": "Q9: Story Completion (Lion and Mouse)"},
    {"q_no": "10", "name": "Q10: Informal Letter (Future Plan after HSC)"},
    {"q_no": "11", "name": "Q11: Theme of Poem ('All people dream...')"},
]


def se_11_q1_schema():
    from src.core.schemas import ExtractedQuestion
    return ExtractedQuestion(question_id="SE_11_Q1", question_text="English 1st Paper", sub_questions=SE_11_Q1_SCHEMA_SUBQUESTIONS)

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


def test_bangladeshi_exam_header_variations():
    """Verify robust extraction across diverse Bangladeshi exam headings and OCR artifacts."""
    valid_order = ["1(A)", "1(B)", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    
    assert extract_header_qno("Ans of Question Number - 7\nIn our country...", valid_q_order=valid_order) == "7"
    assert extract_header_qno("Answer of the Question Number-2\n(i) -> Taking care", valid_q_order=valid_order) == "2"
    assert extract_header_qno("Ans of Question Number- 3\nSummary of the text", valid_q_order=valid_order) == "3"
    assert extract_header_qno("Ass of the Question Number- 1 (A)\na) -> ii", valid_q_order=valid_order) == "1(A)"
    assert extract_header_qno("1 (B)\nAns: The writer describes...", current_parent="1(A)", valid_q_order=valid_order) == "1(B)"
    assert extract_header_qno("Ansl. Answer of the Question Number-4\n(a) through (b) by", valid_q_order=valid_order) == "4"
    assert extract_header_qno("Ans to the Question No- 05\n(a) education", valid_q_order=valid_order) == "5"
    assert extract_header_qno("Question Number - 7\nParagraph about AI", valid_q_order=valid_order) == "7"
    assert extract_header_qno("No. 8\nThe pie chart indicates", valid_q_order=valid_order) == "8"
    assert extract_header_qno("Ans to the Question No. Z\nAI is good", valid_q_order=valid_order) == "7"
    assert extract_header_qno("Ans: 1\n(A) MCQ", valid_q_order=valid_order) == "1(A)"
    assert extract_header_qno("Ans: 10\nDear friend,", valid_q_order=valid_order) == "10"
    assert extract_header_qno("Ans to the Qhe o. No. 1(B)\n(a) Answer", valid_q_order=valid_order) == "1(B)"


def test_anti_overmatching_on_body_text():
    """Ensure body text containing numbers, dates, and percentages is never falsely parsed as headers."""
    valid_order = ["1(A)", "1(B)", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

    body1 = "In 1980 the main source was Coal which was 46%."
    body2 = "From 1980 to 2000, 15 different power plants operated across the region."
    body3 = "The rate increased by 2% in the second quarter."
    body4 = "There are 3 major factors influencing this situation."
    body5 = "The answer lies in the fact that 5 people were involved."

    assert extract_header_qno(body1, valid_q_order=valid_order) is None
    assert extract_header_qno(body2, valid_q_order=valid_order) is None
    assert extract_header_qno(body3, valid_q_order=valid_order) is None
    assert extract_header_qno(body4, valid_q_order=valid_order) is None
    assert extract_header_qno(body5, valid_q_order=valid_order) is None


def test_se_11_q1_0001_full_segmentation():
    """Test full multi-page segmentation with realistic SE_11_Q1_0001 headers."""
    q_schema = ExtractedQuestion(
        question_id="SE_11_Q1",
        language="english",
        question_text="HSC English 1st Paper",
        sub_questions=[
            {"q_no": "1(A)", "name": "Multiple Choice Questions", "marks": 5.0},
            {"q_no": "1(B)", "name": "Short Answer Questions", "marks": 10.0},
            {"q_no": "2", "name": "Flow Chart", "marks": 10.0},
            {"q_no": "3", "name": "Summary Writing", "marks": 10.0},
            {"q_no": "4", "name": "Cloze Test with Clues", "marks": 5.0},
            {"q_no": "5", "name": "Cloze Test without Clues", "marks": 10.0},
            {"q_no": "6", "name": "Rearranging Sentences", "marks": 10.0},
            {"q_no": "7", "name": "Paragraph Writing", "marks": 15.0},
            {"q_no": "8", "name": "Interpreting Graph / Chart", "marks": 15.0},
            {"q_no": "9", "name": "Completing a Story", "marks": 15.0},
            {"q_no": "10", "name": "Informal Letter", "marks": 10.0},
            {"q_no": "11", "name": "Theme of Poem", "marks": 8.0}
        ]
    )

    pages_data = [
        (1, "Ans of the Question Number- 1 (A)\n(a) -> i\n(b) -> ii\n\n1 (B)\n(a) Answer B"),
        (2, "Answer of the Question Number-2\n(i) -> Education -> (ii) -> Knowledge"),
        (3, "Ans of Question Number- 3\nIn this passage the author summarizes the impact."),
        (4, "Ansl. Answer of the Question Number-4\n(a) development (b) society"),
        (5, "Ans to Question Number - 5\n(a) resource (b) vital"),
        (6, "Ans of Question Number - 7\nArtificial Intelligence is a branch of computer science."),
        (7, "Ans to Question Number - 8\nThe pie chart represents electricity production in 1980."),
        (8, "Ans of Question Number - 9\nOnce upon a time an old farmer lived in a village."),
        (9, "Ans of Question Number - 10\nDear Rahim,\nI hope you are doing well."),
        (10, "Ans of Question Number - 11\nTheme: The poem reflects on the beauty of nature.")
    ]

    page_objs = []
    for p_no, text in pages_data:
        page_objs.append(PageExtractionResult(
            page_no=p_no,
            image_path=f"page_{p_no}.png",
            has_red_ink=False,
            red_pixel_count=0,
            stage1_transcription=Stage1TranscriptionResult(raw_transcript=text, word_count=len(text.split())),
            stage2_verification=Stage2VerificationResult(verified_transcript=text, total_corrections_count=0),
            stage3_errors=Stage3ErrorResult(errors=[])
        ))

    extraction = ExtractionResult(
        script_id="SE_11_Q1_0001",
        image_path="SE_11_Q1_0001.pdf",
        timestamp="2026-09-20T00:00:00",
        pages=page_objs,
        stage1_transcription=Stage1TranscriptionResult(raw_transcript="", word_count=0),
        stage2_verification=Stage2VerificationResult(verified_transcript="", total_corrections_count=0),
        stage3_errors=Stage3ErrorResult(errors=[])
    )

    aligned = segment_script_into_questions(extraction, question_obj=q_schema)
    segmented_q_nos = [item.q_no for item in aligned]

    # Exactly the 11 attempted questions must be segmented cleanly
    expected_qs = ["1(A)", "1(B)", "2", "3", "4", "5", "7", "8", "9", "10", "11"]
    for eq in expected_qs:
        assert eq in segmented_q_nos, f"Expected {eq} to be segmented, but got {segmented_q_nos}"


def test_problem6_dans_normalization():
    """Verify Dans does not force 1(B) and extracts correct target question number."""
    assert extract_header_qno("Dans to Q 10\nDear friend, how are you?") == "10"
    assert extract_header_qno("Dans to Q 7\nArtificial Intelligence is a blessing.") == "7"
    assert extract_header_qno("Dans to the Question No 1(B)\n(a) Because...") == "1(B)"


def test_problem6_padded_digits():
    """Verify 3-digit padded numbers like 011 or 007 resolve to normalized integers."""
    assert extract_header_qno("Answer to the Que: 011\nTheme: The poem is about life.") == "11"
    assert extract_header_qno("Ans to the Q No-007\nParagraph on trees.") == "7"
    assert extract_header_qno("Answer to the Question No: 010\nDear Father,") == "10"


def test_problem6_salutation_fingerprint_and_multipage_state_machine():
    """Verify letter starting with Dear <Name> without immediate signoff is recognized as letter/Q10,
    and continuation on page 2 remains attached without dumping into 1(A)."""
    p1 = PageExtractionResult(
        page_no=1,
        image_path="page_1.png",
        has_red_ink=False,
        red_pixel_count=0,
        stage1_transcription=Stage1TranscriptionResult(raw_transcript="Dear Karim,\nTake my cordial love. I hope you are well. In your last letter you wanted to know about our picnic.", word_count=22),
        stage2_verification=Stage2VerificationResult(verified_transcript="Dear Karim,\nTake my cordial love. I hope you are well. In your last letter you wanted to know about our picnic.", total_corrections_count=0),
        stage3_errors=Stage3ErrorResult(errors=[])
    )
    p2 = PageExtractionResult(
        page_no=2,
        image_path="page_2.png",
        has_red_ink=False,
        red_pixel_count=0,
        stage1_transcription=Stage1TranscriptionResult(raw_transcript="We had a wonderful time together.\nNo more today.\nYour loving friend,\nRahim", word_count=13),
        stage2_verification=Stage2VerificationResult(verified_transcript="We had a wonderful time together.\nNo more today.\nYour loving friend,\nRahim", total_corrections_count=0),
        stage3_errors=Stage3ErrorResult(errors=[])
    )

    extraction = ExtractionResult(
        script_id="TEST_LETTER_SCRIPT",
        image_path="test_letter.pdf",
        timestamp="2026-09-29T00:00:00",
        pages=[p1, p2],
        stage1_transcription=Stage1TranscriptionResult(raw_transcript="", word_count=0),
        stage2_verification=Stage2VerificationResult(verified_transcript="", total_corrections_count=0),
        stage3_errors=Stage3ErrorResult(errors=[])
    )

    aligned = segment_script_into_questions(extraction, question_obj=se_11_q1_schema())
    q_nos = [item.q_no for item in aligned]

    # Must be segmented as Q10, never dumped into default 1(A)
    assert "10" in q_nos
    assert "1(A)" not in q_nos

    # Both pages should be united under Q10
    q10_item = next(item for item in aligned if item.q_no == "10")
    assert "Dear Karim" in q10_item.answer_text
    assert "Your loving friend" in q10_item.answer_text


def test_generalized_graph_and_story_fingerprints():
    from src.pipeline.answer_segmenter import detect_structural_fingerprint
    from src.core.schemas import ExtractedQuestion

    # 1. Graph with non-year statistical description (proportions/percentages across categories)
    graph_text = (
        "The pie chart shows the percentage allocation of national budget.\n"
        "Education received 25%, while defense was allocated 18%. The rate of healthcare expenditure was 12%."
    )
    assert detect_structural_fingerprint(graph_text, question_obj=se_11_q1_schema()) == "8"

    # 2. Completing Story with unseen traditional opening
    story_unseen = (
        "Devotion to Mother\n"
        "Many days ago, a young boy lived with his ailing mother in a remote village."
    )
    assert detect_structural_fingerprint(story_unseen, question_obj=se_11_q1_schema()) == "9"

    # 3. Dynamic story matching against schema
    schema = ExtractedQuestion(
        question_id="ENG_101",
        question_text="English Paper",
        sub_questions=[
            {"q_no": "9", "name": "Story: A King and His Astrologer", "marks": 15.0}
        ]
    )
    schema_story = (
        "A King and His Astrologer\n"
        "There was a king who was fond of knowing his future."
    )
    assert detect_structural_fingerprint(schema_story, question_obj=schema) == "9"

    # 4. Poem Theme with generalized terminology (no hardcoded poem titles)
    theme_text = (
        "Theme:\n"
        "The central message of the poem emphasizes perseverance in the face of hardship."
    )
    assert detect_structural_fingerprint(theme_text, question_obj=se_11_q1_schema()) == "11"



