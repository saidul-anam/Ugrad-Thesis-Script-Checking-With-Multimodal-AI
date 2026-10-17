import yaml

from src.core.schemas import ExtractedQuestion
from src.pipeline.stage4_modes import rubric_is_mode_based
from src.utils.rubric_resolver import resolve_rubric_path, rubric_mismatches


def _paper_from(rubric_path, question_id):
    rubric = yaml.safe_load(open(rubric_path, encoding="utf-8"))
    subs = [{"q_no": str(q["question_no"]), "marks": float(q["standard_marks"])} for q in rubric["question_map"]]
    return rubric, ExtractedQuestion(question_id=question_id, question_text="", sub_questions=subs)


def test_english_scripts_route_to_their_own_class_rubric():
    """Each English paper gets its class rubric; there is no shared English fallback rubric any more."""
    assert resolve_rubric_path("SE_11_Q1_0002").endswith("english_class11_hsc.yaml")
    assert resolve_rubric_path("SE_10_Q1_0002").endswith("english_class10_ssc.yaml")
    assert resolve_rubric_path("SE_08_Q1_0002").endswith("english_class8.yaml")
    assert resolve_rubric_path("SE_06_Q1_0002").endswith("english_class6.yaml")
    rubric = yaml.safe_load(open(resolve_rubric_path("SE_11_Q1_0002"), encoding="utf-8"))
    assert rubric_is_mode_based(rubric)
    q_nos = [str(q.get("question_no")) for q in rubric.get("question_map", [])]
    assert "1(A)" in q_nos and "6" in q_nos


def test_rubric_for_another_paper_is_refused():
    rubric11, paper11 = _paper_from("configs/rubrics/english_class11_hsc.yaml", "SE_11_Q1")
    rubric10, paper10 = _paper_from("configs/rubrics/english_class10_ssc.yaml", "SE_10_Q1")
    assert rubric_mismatches(paper11, rubric11) == []
    problems = rubric_mismatches(paper10, rubric11)
    assert any("missing from the rubric" in p for p in problems)
    assert any("Q10 is worth 15" in p for p in problems)
    # legacy rubrics without a question map are not checked
    assert rubric_mismatches(paper10, {"criteria": []}) == []
