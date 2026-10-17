"""Rubric linter: the question paper, rubric and answer key must agree (src/utils/rubric_lint.py)."""

import copy
from pathlib import Path

import pytest

from src.core.schemas import ExtractedQuestion
from src.utils.rubric_lint import ERROR, WARNING, check_bands, lint_paper, lint_repo_paper, printed_marks

RUBRIC = {
    "total_marks": 6,
    "question_map": [
        {"question_no": "1", "task_type": "MCQ", "mode": "A", "standard_marks": 2, "breakdown": "1 x 2 = 2"},
        {"question_no": "2", "task_type": "Story", "mode": "C", "standard_marks": 4},
    ],
    "scoring_modes": {"A_ITEM": {"questions": ["1"]}, "C_BAND": {"questions": ["2"]}},
    "mode_specifications": {
        "mode_a_item_scored": {"tasks": {"Q1_MCQ": {"allocation": "1 x 2 = 2"}}},
        "mode_c_band_scored": {
            "hard_caps": {"short_answer": {"applies_to": ["Story"], "threshold_words": 20, "max_awarded": 1.0}},
            "performance_bands_by_max_mark": {"4": {"band_1": "2.5 - 4.0", "band_2": "0.0 - 2.0"}},
            "tasks": {"Q2_Story": {"max_mark": 4, "criteria_ceilings": {"content": 2, "language": 2}}},
        },
    },
}
PAPER = ExtractedQuestion(question_id="SE_09_Q1", question_text="1. Choose the answer: 1 x 2 = 2\n2. Write a story. 4",
                          sub_questions=[{"q_no": "1", "max_marks": 2, "task_type": "MCQ"},
                                         {"q_no": "2", "max_marks": 4, "task_type": "Story"}])
KEY = {"question_id": "SE_09_Q1", "verified": True,
       "mode_a": {"1": {"task_type": "MCQ", "mark_per_item": 1.0,
                        "items": [{"label": "a", "accepted": ["i"]}, {"label": "b", "accepted": ["ii"]}]}}}


def _errors(issues):
    return [i.message for i in issues if i.level == ERROR]


def test_consistent_files_have_no_issues():
    assert lint_paper("SE_09_Q1", RUBRIC, PAPER, KEY) == []


def test_band_gap_and_ceiling_mismatch_are_errors():
    r = copy.deepcopy(RUBRIC)
    mc = r["mode_specifications"]["mode_c_band_scored"]
    mc["performance_bands_by_max_mark"]["4"]["band_1"] = "3.0 - 4.0"        # 2.5 uncovered
    mc["tasks"]["Q2_Story"]["criteria_ceilings"]["language"] = 1
    errs = _errors(lint_paper("SE_09_Q1", r, PAPER, KEY))
    assert any("uncovered: 2.5" in e for e in errs)
    assert any("ceilings add up to 3" in e for e in errs)


def test_key_items_and_marks_must_match_the_rubric():
    k = copy.deepcopy(KEY)
    k["mode_a"]["1"]["items"].pop()
    k["mode_a"]["1"]["mark_per_item"] = 2.0
    errs = _errors(lint_paper("SE_09_Q1", RUBRIC, PAPER, k))
    assert any("answer key has 1 items, the rubric scores 2" in e for e in errs)
    assert any("mark_per_item 2" in e for e in errs)


def test_paper_and_rubric_must_have_the_same_questions_and_marks():
    paper = PAPER.model_copy(update={"sub_questions": [{"q_no": "1", "max_marks": 2}, {"q_no": "2", "max_marks": 5},
                                                       {"q_no": "3", "max_marks": 1}]})
    errs = _errors(lint_paper("SE_09_Q1", RUBRIC, paper, KEY))
    assert any("Q3" not in e and "add up to 6" in e for e in errs)
    assert any("rubric gives 4 marks, question paper 5" in e for e in errs)
    assert any("not in the rubric" in e for e in [str(i) for i in lint_paper("SE_09_Q1", RUBRIC, paper, KEY)])


def test_printed_marks_and_unverified_key_are_warnings():
    paper = PAPER.model_copy(update={"question_text": "1. Choose: 1 x 3 = 3\n2. Write a story."})
    k = dict(KEY, verified=False)
    issues = lint_paper("SE_09_Q1", RUBRIC, paper, k)
    assert not _errors(issues)
    warnings = [i.message for i in issues if i.level == WARNING]
    assert any("printed paper gives 3 marks" in w for w in warnings)
    assert any("not marked verified" in w for w in warnings)
    assert printed_marks("10. Answer any five: $2 \\times 5 = 10$") == {"10": 10.0}


def test_check_bands_reads_both_range_styles():
    assert check_bands({"a": "4-5", "b": "0.0 - 3.5"}, 5) == []
    assert check_bands({"a": "4-5", "b": "0-3"}, 5) == ["bands leave these marks uncovered: 3.5"]


@pytest.mark.parametrize("paper", ["SE_06_Q1", "SE_07_Q1", "SE_08_Q1", "SE_10_Q1", "SE_11_Q1"])
def test_repo_papers_have_no_lint_errors(paper):
    if not Path(f"outputs/questions/english/{paper}.json").exists():
        pytest.skip("question paper JSON not on this machine (outputs/ is not in git)")
    assert _errors(lint_repo_paper(paper)) == []


def test_report_names_rubric_and_key_status():
    from src.utils.export_utils import _key_status_lines
    prov = {"rubric": {"path": "configs/rubrics/english_class10_ssc.yaml"},
            "answer_key": {"path": "configs/answer_keys/SE_10_Q1.yaml", "verified": False}}
    assert _key_status_lines(prov) == ["- **Rubric / Answer Key**: `english_class10_ssc.yaml` / `SE_10_Q1.yaml` "
                                       "(**not yet verified by a teacher**)"]
    assert _key_status_lines({}) == []
