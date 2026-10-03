import pytest
from PIL import Image
from unittest.mock import MagicMock
from src.engine.base_engine import BaseVLMEngine
from src.pipeline.stage0b_teacher_marks import Stage0bTeacherMarkExtractor


class DummyEngine(BaseVLMEngine):
    def __init__(self, response_text: str):
        self.response_text = response_text

    def generate_multimodal(self, *args, **kwargs) -> str:
        return self.response_text

    def generate_text(self, *args, **kwargs) -> str:
        return self.response_text

    def get_engine_info(self):
        return {"name": "dummy"}


def test_stage0b_clamps_score_exceeding_max_marks():
    """Verify that a score exceeding syllabus max marks (e.g. 8 on a 5-mark question) is clamped."""
    # Model hallucinated 8 on Question 4(A) which has max 5 marks
    dummy_json = """[
        {"question_no": "4(A)", "mark_value": "8", "location": "left margin"}
    ]"""
    engine = DummyEngine(dummy_json)
    extractor = Stage0bTeacherMarkExtractor(engine)
    img = Image.new("RGB", (200, 200), color="white")

    q_max = {"4(A)": 5.0, "4": 5.0}
    res = extractor.run(
        image=img,
        candidate_questions=["4(A)"],
        question_max_marks=q_max
    )
    assert len(res.teacher_marks) == 1
    # Should be clamped to 5
    assert res.teacher_marks[0].mark_value == "5"


def test_stage0b_disambiguates_circled_one():
    """Verify that a circled '01' misread as '6' on a 5-mark question is corrected to '1'."""
    dummy_json = """[
        {"question_no": "1(A)", "mark_value": "6", "location": "left margin"}
    ]"""
    engine = DummyEngine(dummy_json)
    extractor = Stage0bTeacherMarkExtractor(engine)
    img = Image.new("RGB", (200, 200), color="white")

    q_max = {"1(A)": 5.0}
    res = extractor.run(
        image=img,
        candidate_questions=["1(A)"],
        question_max_marks=q_max
    )
    assert len(res.teacher_marks) == 1
    # Circled 01 -> 1
    assert res.teacher_marks[0].mark_value == "1"


def test_stage0b_resolves_base_question_number():
    """Verify that question_max_marks lookup matches base question number when subpart is missing."""
    dummy_json = """[
        {"question_no": "4", "mark_value": "7", "location": "left margin"}
    ]"""
    engine = DummyEngine(dummy_json)
    extractor = Stage0bTeacherMarkExtractor(engine)
    img = Image.new("RGB", (200, 200), color="white")

    # Only "4(A)" in max_marks
    q_max = {"4(A)": 5.0}
    res = extractor.run(
        image=img,
        candidate_questions=["4"],
        question_max_marks=q_max
    )
    assert len(res.teacher_marks) == 1
    assert res.teacher_marks[0].mark_value == "5"


def test_stage0b_rolls_up_loose_subitem_letters_dynamically():
    """Verify loose sub-item letters (e.g. 'c', 'g') roll up to candidate questions on the page."""
    dummy_json = """[
        {"question_no": "C", "mark_value": "0.5", "location": "left margin"},
        {"question_no": "D", "mark_value": "0.5", "location": "left margin"}
    ]"""
    engine = DummyEngine(dummy_json)
    extractor = Stage0bTeacherMarkExtractor(engine)
    img = Image.new("RGB", (200, 200), color="white")

    # Candidate is Question 2 (English 2nd paper - Prepositions)
    res = extractor.run(
        image=img,
        candidate_questions=["2"],
        question_max_marks={"2": 5.0}
    )
    assert len(res.teacher_marks) == 2
    for m in res.teacher_marks:
        assert m.question_no == "2"
        assert m.mark_value == "0.5"

