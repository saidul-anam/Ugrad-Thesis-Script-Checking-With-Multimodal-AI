import pytest
from src.core.schemas import (
    ExtractedQuestion,
    ExtractionResult,
    PageExtractionResult,
    Stage1TranscriptionResult,
    Stage2VerificationResult,
    Stage3ErrorResult,
    LinguisticErrorItem
)
from src.pipeline.answer_segmenter import (
    normalize_header_text,
    extract_header_qno,
    segment_script_into_questions,
    build_question_entity_index,
    compute_entity_overlap
)


def se_10_q1_schema():
    """Build schema for SSC English 1st Paper (SE_10_Q1)."""
    sub_questions = [
        {"q_no": "1", "name": "MCQ (Zahir Raihan Passage)", "marks": 7.0},
        {"q_no": "2", "name": "Comprehension Questions (Zahir Raihan)", "marks": 10.0},
        {"q_no": "3", "name": "Cloze without Clues", "marks": 5.0},
        {"q_no": "4", "name": "Information Transfer Table (Arsenic)", "marks": 5.0},
        {"q_no": "5", "name": "Summary Writing (Arsenic)", "marks": 10.0},
        {"q_no": "6", "name": "Matching Columns Table (Email)", "marks": 5.0},
        {"q_no": "7", "name": "Rearranging Sentences (King Solomon & Sheba)", "marks": 8.0},
        {"q_no": "8", "name": "Poetry Comprehension Questions", "marks": 10.0},
        {
            "q_no": "9",
            "name": "Story Questions: Merchant of Venice, Rosamond, Purple Jar",
            "marks": 10.0,
            "prompt": "Answer any 5 of the following questions: Shylock condition loan Antonio, Rosamond purple jar shoes, Portia casket trick"
        },
        {"q_no": "10", "name": "Completing Story (The Foolish King and The Ghosts)", "marks": 15.0},
        {"q_no": "11", "name": "Dialogue Writing (Benefits of Early Rising)", "marks": 15.0},
    ]
    question_text = """
    8. Answer any 5 of the following questions from poems:
    a. What does the poet say in the poem "Books"?
    b. How are the woods described in "Stopping by Woods on a Snowy Evening"?
    c. What is unique about the mothers in "Two Mothers Remembered"?
    e. Why does the poet call time a "gipsy" in "Time, You Old Gipsy Man"?
    f. What does the "cruel crawling foam" signify in "The Sands of Dee"?

    9. Answer any 5 of the following questions from stories:
    a. What condition did Shylock place on the loan he gave to Antonio?
    b. Why did Rosamond's mother refuse to buy jewelry?
    c. What does "all that glitters is not gold" signify in The Merchant of Venice?
    d. Why did Rosamond choose the purple jar over the shoes?
    f. How did Rosamond's decision affect her daily life?
    h. What lesson did Rosamond learn from her experience?

    10. Read the beginning of the story and complete it:
    Once there lived a king in an island...

    11. Write a dialogue between you and your friend on the benefits of early rising.
    """
    return ExtractedQuestion(
        question_id="SE_10_Q1",
        language="english",
        question_text=question_text,
        sub_questions=sub_questions
    )


def test_header_normalization_am_arn_anis():
    """Verify that OCR ligature misreads 'Am.', 'Arn', 'Anis' normalize to 'Ans'."""
    assert "Ans to" in normalize_header_text("Am. to question no. 2")
    assert "Ans to" in normalize_header_text("Am. to the question no. 10")
    assert "Ans to" in normalize_header_text("Arn to the question no. 8")
    assert "Ans" in normalize_header_text("Anis to question no. 3")


def test_entity_index_mining():
    """Verify that dynamic entity indexer extracts key topic terms for questions."""
    paper = se_10_q1_schema()
    index = build_question_entity_index(paper, [str(i) for i in range(1, 12)])

    assert "9" in index
    q9_ents = index["9"]
    # Check that key prompt terms were extracted
    assert any(term in q9_ents for term in ["shylock", "antonio", "rosamond", "purple", "shoes"])

    assert "8" in index
    q8_ents = index["8"]
    assert any(term in q8_ents for term in ["woods", "evening", "mothers", "gipsy", "books"])


def test_sequence_collision_remapped_to_q9():
    """
    Verify that an OCR slip producing 'Am. to question no. 2' after Q8 has been completed
    is detected as an impossible backwards jump and remapped to Q9 via entity overlap.
    """
    paper = se_10_q1_schema()
    valid_order = [str(i) for i in range(1, 12)]
    e_index = build_question_entity_index(paper, valid_order)

    student_text = (
        "Am. to question no. 2\n\n"
        "a) Antonio have to pay a penalty if he cannot prepay the loan. "
        "The penalty is Shylock will take a pound of flesh from Antonio's body.\n"
        "b) Rosamond is a seven year old girl. She does not know which is necessary. "
        "Her mother refused to buy jewelry.\n"
        "d) Rosamond chose the purple jar over the shoes."
    )

    detected_q = extract_header_qno(
        student_text,
        current_parent="8",
        question_obj=paper,
        valid_q_order=valid_order,
        entity_index=e_index
    )

    assert detected_q == "9", f"Expected Q9 to be identified, but got: {detected_q}"


def test_headless_story_questions_matched_via_entity():
    """Verify that story questions written with NO header match Q9 via semantic fingerprint."""
    paper = se_10_q1_schema()
    valid_order = [str(i) for i in range(1, 12)]
    e_index = build_question_entity_index(paper, valid_order)

    student_text = (
        "a) Antonio have to pay a penalty. Shylock will take a pound of flesh.\n"
        "b) Rosamond mother refused to buy jewelry because it is unnecessary.\n"
        "d) Rosamond chose the purple jar over the shoes."
    )

    detected_q = extract_header_qno(
        student_text,
        current_parent="8",
        question_obj=paper,
        valid_q_order=valid_order,
        entity_index=e_index
    )

    assert detected_q == "9", f"Expected headless section to match Q9, got: {detected_q}"


def test_full_segmentation_se_10_q1_0009_scenario():
    """
    End-to-end multi-page simulation of SE_10_Q1_0009 failure mode:
    - Page 7-8: Q8 Poetry questions
    - Page 9-10: Antonio & Rosamond text with 'Am. to question no. 2'
    - Page 11: Part B Q11 Dialogue
    - Page 14: Q10 Story
    Must segment Q9 into distinct AlignedAnswerItem with correct pages and error attribution.
    """
    paper = se_10_q1_schema()

    p7_text = (
        "Ans. to the question no.8\n\n"
        "b) In the poem 'Stopping by Woods on a Snowy Evening' the woods are described as lovely, dark and deep.\n"
        "c) In the poem 'Two Mothers Remembered' the uniqueness of the mothers is their responsibility."
    )
    p8_text = (
        "e) In the poem 'Time, You Old Gipsy Man' the poet calls time a gipsy because it wanders.\n"
        "h) In the poem 'Books' the poet describes books as magical boxes."
    )
    p9_text = (
        "Am. to question no. 2\n\n"
        "a) Antonio have to pay a penalty if he cannot prepay the loan. Shylock will take a pound of flesh.\n"
        "b) Rosamond is a seven year old girl. Her mother refused to buy jewelry."
    )
    p10_text = (
        "d) Rosamond was very excited for buying the purple jar over the shoes.\n"
        "f) Rosamond's purple jar was a white glass jar. It affected her daily life.\n"
        "h) Rosamond learnt lessons from her experience."
    )
    p11_text = (
        "Part-B : Writing Part\n"
        "Ans. to the question no. 11\n"
        "Sumon : Hi, Afif! How are you?\n"
        "Afif : Fine. I can't sleep well."
    )
    p14_text = (
        "Am. to the question no. 10\n"
        "The Foolish King and The Ghosts\n"
        "Once there lived a king in an island. There were green trees everywhere."
    )

    pages = [
        PageExtractionResult(
            page_no=7,
            image_path="page_7.png",
            stage1_transcription=Stage1TranscriptionResult(raw_transcript=p7_text, word_count=len(p7_text.split())),
            stage2_verification=Stage2VerificationResult(verified_transcript=p7_text, total_corrections_count=0),
            stage3_errors=Stage3ErrorResult(errors=[])
        ),
        PageExtractionResult(
            page_no=8,
            image_path="page_8.png",
            stage1_transcription=Stage1TranscriptionResult(raw_transcript=p8_text, word_count=len(p8_text.split())),
            stage2_verification=Stage2VerificationResult(verified_transcript=p8_text, total_corrections_count=0),
            stage3_errors=Stage3ErrorResult(errors=[])
        ),
        PageExtractionResult(
            page_no=9,
            image_path="page_9.png",
            stage1_transcription=Stage1TranscriptionResult(raw_transcript=p9_text, word_count=len(p9_text.split())),
            stage2_verification=Stage2VerificationResult(verified_transcript=p9_text, total_corrections_count=0),
            stage3_errors=Stage3ErrorResult(errors=[
                LinguisticErrorItem(
                    error_type="grammar",
                    erroneous_text="Antonio have to pay",
                    suggested_correction="Antonio has to pay",
                    context_sentence="Antonio have to pay a penalty if he cannot prepay the loan",
                    explanation="Subject-verb agreement error",
                    question_no="2"  # Note: originally extracted with question_no 2
                )
            ])
        ),
        PageExtractionResult(
            page_no=10,
            image_path="page_10.png",
            stage1_transcription=Stage1TranscriptionResult(raw_transcript=p10_text, word_count=len(p10_text.split())),
            stage2_verification=Stage2VerificationResult(verified_transcript=p10_text, total_corrections_count=0),
            stage3_errors=Stage3ErrorResult(errors=[])
        ),
        PageExtractionResult(
            page_no=11,
            image_path="page_11.png",
            stage1_transcription=Stage1TranscriptionResult(raw_transcript=p11_text, word_count=len(p11_text.split())),
            stage2_verification=Stage2VerificationResult(verified_transcript=p11_text, total_corrections_count=0),
            stage3_errors=Stage3ErrorResult(errors=[])
        ),
        PageExtractionResult(
            page_no=14,
            image_path="page_14.png",
            stage1_transcription=Stage1TranscriptionResult(raw_transcript=p14_text, word_count=len(p14_text.split())),
            stage2_verification=Stage2VerificationResult(verified_transcript=p14_text, total_corrections_count=0),
            stage3_errors=Stage3ErrorResult(errors=[])
        ),
    ]

    extraction = ExtractionResult(
        script_id="SE_10_Q1_0009",
        image_path="SE_10_Q1_0009.pdf",
        timestamp="2026-10-09T00:00:00",
        pages=pages,
        stage1_transcription=Stage1TranscriptionResult(raw_transcript="", word_count=0),
        stage2_verification=Stage2VerificationResult(verified_transcript="", total_corrections_count=0),
        stage3_errors=Stage3ErrorResult(errors=[])
    )

    aligned = segment_script_into_questions(extraction, question_obj=paper)
    by_q = {item.q_no: item for item in aligned}

    # 1. Q9 must be present!
    assert "9" in by_q, f"Question 9 missing from segmented questions: {list(by_q.keys())}"

    # 2. Q8 must only contain pages [7, 8], NOT [7, 8, 9, 10]
    assert by_q["8"].page_numbers == [7, 8], f"Q8 pages incorrect: {by_q['8'].page_numbers}"
    assert "Stopping by Woods" in by_q["8"].answer_text
    assert "Antonio" not in by_q["8"].answer_text

    # 3. Q9 must contain pages [9, 10]
    assert by_q["9"].page_numbers == [9, 10], f"Q9 pages incorrect: {by_q['9'].page_numbers}"
    assert "Antonio" in by_q["9"].answer_text
    assert "Rosamond" in by_q["9"].answer_text

    # 4. Q10 and Q11 must be preserved cleanly
    assert "10" in by_q
    assert "11" in by_q

    # 5. Linguistic error on Antonio must be attributed to Q9, NOT Q8 or Q2!
    q9_errors = by_q["9"].errors
    assert len(q9_errors) == 1, f"Expected 1 error on Q9, found {len(q9_errors)}"
    assert q9_errors[0]["erroneous_text"] == "Antonio have to pay"
    assert q9_errors[0]["question_no"] == "9"
