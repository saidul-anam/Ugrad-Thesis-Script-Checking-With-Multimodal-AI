import pytest
from src.core.schemas import TeacherMarkItem
from src.pipeline.stage0b_teacher_marks import align_orphan_marks


def test_align_single_orphan_mark():
    """When a page contains a single candidate question, any orphan mark must be aligned to it."""
    marks = [
        TeacherMarkItem(question_no=None, mark_value="10", location="left margin")
    ]
    candidates = ["8"]
    aligned = align_orphan_marks(marks, candidate_questions=candidates)

    assert len(aligned) == 1
    assert aligned[0].question_no == "8"
    assert aligned[0].mark_value == "10"
    assert "aligned to 8" in aligned[0].location


def test_align_unrecognized_question_labels():
    """Labels like 'unknown', 'not specified', 't specified' should be recognized as orphans."""
    marks = [
        TeacherMarkItem(question_no="not specified", mark_value="7", location="left margin"),
        TeacherMarkItem(question_no="t specified", mark_value="5", location="left margin (bottom)")
    ]
    candidates = ["8", "9"]
    aligned = align_orphan_marks(marks, candidate_questions=candidates)

    assert len(aligned) == 2
    q_nos = [m.question_no for m in aligned]
    assert "8" in q_nos
    assert "9" in q_nos


def test_spatial_vertical_ordering_alignment():
    """Top mark aligns to earlier question, bottom mark aligns to subsequent question."""
    marks = [
        TeacherMarkItem(question_no=None, mark_value="6", location="left margin (bottom)", y_position="bottom"),
        TeacherMarkItem(question_no=None, mark_value="5", location="left margin (top)", y_position="top")
    ]
    candidates = ["1(A)", "1(B)"]
    aligned = align_orphan_marks(marks, candidate_questions=candidates)

    assert len(aligned) == 2
    # y_position='top' should align to 1(A), 'bottom' should align to 1(B)
    top_item = next(m for m in aligned if m.question_no == "1(A)")
    bottom_item = next(m for m in aligned if m.question_no == "1(B)")

    assert top_item.mark_value == "5"
    assert bottom_item.mark_value == "6"


def test_preserve_already_assigned_marks():
    """Marks that are already correctly attributed should not be overwritten."""
    marks = [
        TeacherMarkItem(question_no="1(A)", mark_value="5", location="left margin next to Q1(A)"),
        TeacherMarkItem(question_no=None, mark_value="6", location="left margin", y_position="bottom")
    ]
    candidates = ["1(A)", "1(B)"]
    aligned = align_orphan_marks(marks, candidate_questions=candidates)

    assert len(aligned) == 2
    q_nos = {m.question_no: m.mark_value for m in aligned}
    assert q_nos["1(A)"] == "5"
    assert q_nos["1(B)"] == "6"
