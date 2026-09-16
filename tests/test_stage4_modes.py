import yaml

from src.core.schemas import AlignedAnswerItem, ExtractedQuestion
from src.engine.mock_engine import MockGemmaEngine
from src.pipeline.stage4_evaluator import Stage4Evaluator
from src.pipeline.stage4_modes import (
    load_rubric_specs, rubric_is_mode_based, load_answer_key, key_for_question,
    score_mode_a, score_mode_b, score_mode_c, snap_half, band_for, verbatim_overlap,
    build_mode_a_prompt, build_mode_b_prompt, build_mode_c_prompt, source_text_for_question,
    _matches_accepted,
)

RUBRIC = yaml.safe_load(open("configs/rubrics/english_writing.yaml", "r", encoding="utf-8"))
KEY = load_answer_key("SE_11_Q1")


def test_matches_accepted_spelling_tolerance_and_lexicon_gate():
    # Minor typographical slips on words >= 5 letters (not legitimate other words)
    assert _matches_accepted("atain", ["attain"]) is True
    assert _matches_accepted("helth", ["health"]) is True
    
    # Real words that are different parts of speech or different words must NOT match
    assert _matches_accepted("healthy", ["health"]) is False
    assert _matches_accepted("rights", ["right"]) is False
    
    # Exact match
    assert _matches_accepted("health", ["health"]) is True
    assert _matches_accepted("right", ["right"]) is True


def test_rubric_specs_loaded_per_mode():
    assert rubric_is_mode_based(RUBRIC)
    specs = load_rubric_specs(RUBRIC)
    assert specs["1(A)"].mode == "A" and specs["1(A)"].max_mark == 5
    assert specs["1(B)"].mode == "B" and 2.0 in specs["1(B)"].scale
    assert specs["7"].mode == "C" and specs["7"].criteria_ceilings["context_content_data"] == 3
    assert specs["10"].layout_components["letter"]
    assert specs["9"].bands["band_4"] == "6-7"
    assert "paragraph_split" in specs["7"].hard_caps
    assert load_rubric_specs({"subject": "Bangla", "penalties": {}}) == {}


def test_answer_key_lookup():
    assert KEY["question_id"] == "SE_11_Q1"
    sec, entry = key_for_question(KEY, "1(A)")
    assert sec == "mode_a" and len(entry["items"]) == 5
    sec, entry = key_for_question(KEY, "6")
    assert entry["correct_sequence"][0] == "c"
    sec, entry = key_for_question(KEY, "2")
    assert sec == "mode_b"
    assert key_for_question(KEY, "7") == ("", {})


def test_score_mode_a_mcq_and_cloze():
    specs = load_rubric_specs(RUBRIC)
    _, key = key_for_question(KEY, "1(A)")
    parsed = {"items": [
        {"item_label": "a", "candidate_answer": "(iii) being displaced lives outside the country"},
        {"item_label": "b", "candidate_answer": "ii"},
        {"item_label": "c", "candidate_answer": "iv"},          # wrong
        {"item_label": "d", "candidate_answer": "i, ii"},       # multiple selection -> 0
        {"item_label": "e", "candidate_answer": ""},            # blank
    ]}
    sr = score_mode_a(parsed, specs["1(A)"], key)
    assert sr.awarded == 2.0
    statuses = {it["item_label"]: it["status"] for it in sr.items}
    assert statuses == {"a": "correct", "b": "correct", "c": "incorrect", "d": "incorrect", "e": "not_attempted"}
    _, key4 = key_for_question(KEY, "4")
    parsed4 = {"items": [{"item_label": "a", "candidate_answer": "attain"}, {"item_label": "e", "candidate_answer": "courage"},
                         {"item_label": "g", "candidate_answer": "discouraged"}, {"item_label": "j", "candidate_answer": "disaster"}]}
    sr4 = score_mode_a(parsed4, specs["4"], key4)
    assert sr4.awarded == 1.5          # 'courage' (wrong form) earns 0


def test_score_mode_a_rearrangement_positional():
    specs = load_rubric_specs(RUBRIC)
    _, key = key_for_question(KEY, "6")
    sr = score_mode_a({"student_sequence": ["c", "h", "j", "a", "e", "d", "g", "i", "b", "f"]}, specs["6"], key)
    assert sr.awarded == 8.0
    sr2 = score_mode_a({"student_sequence": []}, specs["6"], key)
    assert sr2.awarded == 0.0 and all(it["status"] == "not_attempted" for it in sr2.items)


def test_score_mode_b_clamps_to_scale():
    specs = load_rubric_specs(RUBRIC)
    _, key = key_for_question(KEY, "1(B)")
    parsed = {"items": [{"item_label": "a", "marks_awarded": 1.7}, {"item_label": "b", "marks_awarded": 2.9},
                        {"item_label": "c", "marks_awarded": 0.2}, {"item_label": "d", "marks_awarded": 1.0}, {"item_label": "e", "marks_awarded": 0}]}
    sr = score_mode_b(parsed, specs["1(B)"], key)
    assert [it["marks_awarded"] for it in sr.items] == [1.5, 2.0, 0.0, 1.0, 0.0]
    assert sr.awarded == 4.5


def test_score_mode_c_ceilings_caps_and_band():
    specs = load_rubric_specs(RUBRIC)
    parsed = {"raw_subscores": {"context_content_data": 3.4, "structure_format_brevity": 5.0, "language_mechanics": 1.5, "originality_comparisons_paraphrase": 2.0},
              "structural_audit": {"paragraph_subdivisions": True}, "feedback_summary": "ok", "positive_aspects": ["x"], "frequent_errors": ["y"]}
    sr = score_mode_c(parsed, specs["7"], "some paragraph text")
    assert sr.subscores["structure_format_brevity"] == 3.0     # clamped to ceiling
    assert sr.raw_total == 9.5
    assert sr.cap_applied and sr.cap_reason == "Paragraph_Subdivisions" and sr.awarded == 5.0
    assert sr.band == "Band 2"
    # graph: personal opinion cap at 6
    parsed_g = {"raw_subscores": {"context_content_data": 3, "structure_format_brevity": 3, "language_mechanics": 2, "originality_comparisons_paraphrase": 2},
                "structural_audit": {"external_facts_or_personal_opinions": True}}
    sr_g = score_mode_c(parsed_g, specs["8"], "chart text")
    assert sr_g.awarded == 6.0 and sr_g.cap_reason == "Graph_External_Facts"
    # theme: verbatim copying computed in code
    poem = "All people dream but not equally those who dream by night in the dusty recesses of their mind wake in the morning to find that it was vanity"
    answer = "Those who dream by night in the dusty recesses of their mind wake in the morning to find that it was vanity."
    assert verbatim_overlap(answer, poem) > 0.5
    parsed_t = {"raw_subscores": {"context_content_data": 3, "structure_format_brevity": 2, "language_mechanics": 1, "originality_comparisons_paraphrase": 2}, "structural_audit": {}}
    sr_t = score_mode_c(parsed_t, specs["11"], answer, poem)
    assert sr_t.cap_applied and sr_t.awarded == 4.0
    # no cap when not binding
    parsed_ok = {"raw_subscores": {"context_content_data": 2, "structure_format_brevity": 2, "language_mechanics": 1, "originality_comparisons_paraphrase": 1}, "structural_audit": {}}
    sr_ok = score_mode_c(parsed_ok, specs["7"], "text")
    assert not sr_ok.cap_applied and sr_ok.awarded == 6.0 and sr_ok.band == "Band 3"


def test_helpers():
    assert snap_half(7.3) == 7.5 and snap_half(7.24) == 7.0 and snap_half(0.26) == 0.5
    assert band_for(8.0, {"band_4": "8-10", "band_3": "6-7", "band_0": "0"}) == "Band 4"
    assert band_for(0.0, {"band_4": "8-10", "band_0": "0"}) == "Band 0"
    q = ExtractedQuestion(question_id="X", question_text="3. Summarize the following text. 10\nHope is the thing with feathers\nThat perches in the soul\n\n4. Read the following", sub_questions=[])
    src = source_text_for_question("3", q.question_text)
    assert "feathers" in src and "Summarize" not in src


def test_prompts_carry_mode_tags_and_no_marks():
    specs = load_rubric_specs(RUBRIC)
    ans = AlignedAnswerItem(q_no="1(A)", answer_text="a) iii b) ii", page_numbers=[1])
    _, key = key_for_question(KEY, "1(A)")
    p = build_mode_a_prompt(ans, specs["1(A)"], key, "Choose the correct answer")
    assert "MODE A (ITEM-SCORED)" in p and "candidate_answer" in p
    ans_b = AlignedAnswerItem(q_no="1(B)", answer_text="a) because ...", page_numbers=[1])
    _, key_b = key_for_question(KEY, "1(B)")
    pb = build_mode_b_prompt(ans_b, specs["1(B)"], key_b, "Answer the questions")
    assert "MODE B (POINT-SCORED)" in pb and "2: Accurate" in pb
    ans_c = AlignedAnswerItem(q_no="7", answer_text="AI is ...", page_numbers=[1])
    pc = build_mode_c_prompt(ans_c, specs["7"], "Write a paragraph", "No errors")
    assert "MODE C (BAND-SCORED)" in pc and "context_content_data" in pc and "Paragraph" in pc


def test_evaluator_rubric_driven_with_mock_engine_missing_and_totals():
    evaluator = Stage4Evaluator(MockGemmaEngine())
    q = ExtractedQuestion(question_id="SE_11_Q1", question_text="1. Read the passage\nA. Choose\nB. Answer\n\n7. Write a paragraph 10", total_marks=100.0,
                          sub_questions=[{"q_no": "1(A)", "max_marks": 5.0}, {"q_no": "1(B)", "max_marks": 10.0}, {"q_no": "6", "max_marks": 10.0}, {"q_no": "7", "max_marks": 10.0}])
    answers = [
        AlignedAnswerItem(q_no="1(A)", answer_text="a) iii b) ii c) iii d) i e) i", page_numbers=[1], word_count=10),
        AlignedAnswerItem(q_no="1(B)", answer_text="a) ... b) ... c) ... d) ... e) ...", page_numbers=[1], word_count=30),
        AlignedAnswerItem(q_no="6", answer_text="c h j a e d g i f b", page_numbers=[2], word_count=10),
        AlignedAnswerItem(q_no="7", answer_text="AI is a branch of computer science ...", page_numbers=[3], word_count=120),
    ]
    res = evaluator.evaluate_modular(answers, q, RUBRIC, ground_truth_marks={"1(A)": 5, "1(B)": 6, "6": 10, "7": 6, "8": 5})
    assert res.rubric_driven
    by_q = {qe.q_no: qe for qe in res.question_evaluations}
    assert by_q["1(A)"].task_mode == "A_ITEM" and by_q["1(A)"].awarded_marks == 2.0     # mock answers "iii" everywhere -> a and c correct
    assert by_q["6"].awarded_marks == 10.0
    assert by_q["1(B)"].task_mode == "B_POINT" and by_q["1(B)"].awarded_marks == 7.5
    assert by_q["7"].task_mode == "C_BAND" and by_q["7"].awarded_marks == 7.5 and by_q["7"].performance_band == "Band 3"
    assert "8" in res.missing_questions and by_q["8"].scoring_status == "missing"
    assert res.mae_vs_human is not None and res.mae_including_missing is not None
    assert res.mae_including_missing >= res.mae_vs_human - 1e-9 or by_q["8"].human_ground_truth is None
    assert res.scored_with_gt == 4
