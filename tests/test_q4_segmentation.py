import pytest
from src.utils.export_utils import load_extraction_artifacts
from src.utils.question_utils import load_question_for_script
from src.pipeline.answer_segmenter import segment_script_into_questions


def test_se_08_q1_0002_q4_segmented_properly():
    """Ensure Question 4 with plural 'Answers to the Question no.4' and table format is segmented correctly."""
    ext = load_extraction_artifacts("data_without_pipeline/english/se_08_q1/SE_08_Q1_0002")
    q_obj = load_question_for_script("SE_08_Q1_0002", lang="english")

    answers = segment_script_into_questions(ext, q_obj)
    q_dict = {a.q_no: a for a in answers}

    assert "4" in q_dict, "Question 4 must be segmented from script"
    q4 = q_dict["4"]
    assert 3 in q4.page_numbers, f"Q4 should be on page 3, found: {q4.page_numbers}"
    assert "Traditional" in q4.answer_text
    assert "well-known" in q4.answer_text
    assert "cheap" in q4.answer_text
    assert "bright" in q4.answer_text
    assert "unknown" in q4.answer_text


def test_se_08_q1_0002_all_questions_segmented():
    """Ensure all attempted questions Q1-Q13 are segmented on their respective pages."""
    ext = load_extraction_artifacts("data_without_pipeline/english/se_08_q1/SE_08_Q1_0002")
    q_obj = load_question_for_script("SE_08_Q1_0002", lang="english")

    answers = segment_script_into_questions(ext, q_obj)
    q_dict = {a.q_no: a for a in answers}

    for expected_q in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]:
        assert expected_q in q_dict, f"Expected Q{expected_q} to be segmented, missing from {list(q_dict.keys())}"

    # Verify Q2 has both page 1 and page 2 (continuation items c, d, e)
    assert 2 in q_dict["2"].page_numbers, f"Q2 should include continuation page 2, got {q_dict['2'].page_numbers}"

    # Verify Q11 story has pages 9, 10, 11
    assert set(q_dict["11"].page_numbers) == {9, 10, 11}

    # Verify Q12 paragraph has pages 12, 13, 14, 15
    assert set(q_dict["12"].page_numbers) == {12, 13, 14, 15}


def test_se_08_q1_0011_all_attempted_questions_segmented():
    """Ensure all 11 attempted questions in SE_08_Q1_0011 are segmented despite prefix numbering and out-of-order sequence."""
    ext = load_extraction_artifacts("data_without_pipeline/english/se_08_q1/SE_08_Q1_0011")
    q_obj = load_question_for_script("SE_08_Q1_0011", lang="english")

    answers = segment_script_into_questions(ext, q_obj)
    q_dict = {a.q_no: a for a in answers}

    expected_attempted = ["1", "2", "3", "4", "5", "6", "7", "8", "11", "12", "13"]
    for expected_q in expected_attempted:
        assert expected_q in q_dict, f"Expected Q{expected_q} to be segmented, missing from {list(q_dict.keys())}"

    # Verify Q7 summary continuation across pages 8 and 9
    assert set(q_dict["7"].page_numbers) == {8, 9}
    assert "Lord Byron was an English poet" in q_dict["7"].answer_text
    assert "Honours of Idleness" in q_dict["7"].answer_text

    # Verify Q8 matching table across pages 9 and 10 with numberless header
    assert set(q_dict["8"].page_numbers) == {9, 10}
    assert "(a+v) = Deforestation" in q_dict["8"].answer_text
    assert "carbon-di-oxide is increasing" in q_dict["8"].answer_text

    # Verify Q11 answered out-of-order before Q6/Q7 is retained on pages 6 and 7
    assert set(q_dict["11"].page_numbers) == {6, 7}
    assert "The Honest woodcutter" in q_dict["11"].answer_text

    # Verify Q6 answered on page 7 is not swallowed by Q11
    assert set(q_dict["6"].page_numbers) == {7}
