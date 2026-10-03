import pytest
from src.core.schemas import ExtractedQuestion
from src.utils.question_utils import extract_question_reference_numerals
from src.prompts.stage1_verbatim import build_stage1_prompt
from src.prompts.stage2_verification import build_stage2_prompt


def test_extract_question_reference_numerals():
    q = ExtractedQuestion(
        question_id="SE_11_Q1",
        language="english",
        question_text="English 1st Paper Examination 2026",
        sub_questions=[
            {
                "q_no": "8",
                "name": "Q8: Chart Analysis (USA Electricity Sources 1980)",
                "text": "Data commentary describing electricity generation shares (Coal 46%, Natural gas 24%, Hydro 15%, Oil 12%, Nuclear 2%)."
            }
        ]
    )

    nums = extract_question_reference_numerals(q)
    assert "46%" in nums
    assert "24%" in nums
    assert "15%" in nums
    assert "12%" in nums
    assert "2%" in nums
    assert "1980" in nums


def test_prompts_inject_reference_numerals():
    numerals = ["46%", "24%", "15%", "12%", "2%"]
    
    p1 = build_stage1_prompt(
        question_reference_numerals=numerals
    )
    assert "EXAM QUESTION REFERENCE NUMERALS" in p1
    assert "'46%'" in p1
    assert "distinguish open-top '6' from double-loop '8'" in p1

    p2 = build_stage2_prompt(
        stage1_transcript="The chart shows electricity in 1980.",
        question_reference_numerals=numerals
    )
    assert "TARGET EXAM QUESTION REFERENCE NUMERALS" in p2
    assert "'46%'" in p2
    assert "distinguish open-top '6' from double-loop '8'" in p2
