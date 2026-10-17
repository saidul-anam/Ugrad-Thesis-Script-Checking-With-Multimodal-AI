import yaml

from src.core.schemas import AlignedAnswerItem, ExtractedQuestion
from src.engine.mock_engine import MockGemmaEngine
from src.pipeline.stage4_evaluator import Stage4Evaluator
from src.pipeline.stage4_modes import (
    load_rubric_specs, rubric_is_mode_based, load_answer_key, key_for_question,
    score_mode_a, score_mode_b, score_mode_c, snap_half, band_for, verbatim_overlap,
    build_mode_a_prompt, build_mode_b_prompt, build_mode_c_prompt, source_text_for_question,
    _matches_accepted, selected_option_labels,
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
    assert specs["7"].criteria_ceilings["organization_cohesion"] == 3
    assert specs["10"].layout_components["letter"]
    assert specs["9"].bands["band_4"] == "6-7"
    assert "summary_violations" in specs["7"].hard_caps
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


def test_selected_option_labels_ignores_option_text():
    # one choice: articles and labels inside the option text are not extra selections
    assert selected_option_labels("(ii) He was a patriotic and courageous person") == {"ii"}
    assert selected_option_labels("iv. all of i, ii & iii.", ["iv", "all of i, ii & iii"]) == {"iv"}
    assert selected_option_labels("ii. Household manager and a caregiver.") == {"ii"}
    assert selected_option_labels("e.g. a test") == set()
    # two choices
    assert selected_option_labels("i, ii") == {"i", "ii"}
    assert selected_option_labels("ii and iii") == {"ii", "iii"}
    assert selected_option_labels("(iii) (iv) Adjective") == {"iii", "iv"}
    assert selected_option_labels("(ii) municipal area (iii) rural area") == {"ii", "iii"}


def test_score_mode_a_rearrangement_positional():
    specs = load_rubric_specs(RUBRIC)
    _, key = key_for_question(KEY, "6")
    sr = score_mode_a({"student_sequence": ["c", "h", "j", "a", "e", "d", "g", "i", "b", "f"]}, specs["6"], key)
    # Under dual-engine (positional=8, LCS=9), awarded is max(8, 9) = 9.0
    assert sr.awarded == 9.0
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
    parsed = {"raw_subscores": {"context_content_data": 3.4, "organization_cohesion": 5.0, "language_mechanics": 1.5, "originality_comparisons_paraphrase": 2.0},
              "structural_audit": {}, "feedback_summary": "ok", "positive_aspects": ["x"], "frequent_errors": ["y"]}
    sr = score_mode_c(parsed, specs["7"], "some paragraph text")
    assert sr.subscores["organization_cohesion"] == 3.0     # clamped to ceiling
    assert sr.raw_total == 9.5
    # Paragraph structure constraints removed to prevent penalizing OCR layout artifacts
    assert not sr.cap_applied and sr.awarded == 9.5
    assert sr.band == "Band 4"
    # graph: personal opinion cap at 6 (backward compatibility alias structure_format_brevity -> organization_cohesion)
    parsed_g = {"raw_subscores": {"context_content_data": 3, "structure_format_brevity": 3, "language_mechanics": 2, "originality_comparisons_paraphrase": 2},
                "structural_audit": {"external_facts_or_personal_opinions": True}}
    sr_g = score_mode_c(parsed_g, specs["8"], "chart text")
    assert sr_g.awarded == 6.0 and sr_g.cap_reason == "graph_external_facts"
    assert sr_g.subscores["organization_cohesion"] == 3.0
    # theme: verbatim copying computed in code
    poem = "All people dream but not equally those who dream by night in the dusty recesses of their mind wake in the morning to find that it was vanity"
    answer = "Those who dream by night in the dusty recesses of their mind wake in the morning to find that it was vanity."
    assert verbatim_overlap(answer, poem) > 0.5
    parsed_t = {"raw_subscores": {"context_content_data": 3, "organization_cohesion": 2, "language_mechanics": 1, "originality_comparisons_paraphrase": 2}, "structural_audit": {}}
    sr_t = score_mode_c(parsed_t, specs["11"], answer, poem)
    assert sr_t.cap_applied and sr_t.awarded == 4.0
    # no cap when not binding
    parsed_ok = {"raw_subscores": {"context_content_data": 2, "organization_cohesion": 2, "language_mechanics": 1, "originality_comparisons_paraphrase": 1}, "structural_audit": {}}
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
    assert "8" in res.missing_questions and by_q["8"].scoring_status == "missing_no_segment"
    assert res.mae_vs_human is not None and res.mae_including_missing is not None
    assert res.mae_including_missing >= res.mae_vs_human - 1e-9 or by_q["8"].human_ground_truth is None
    assert res.scored_with_gt == 4


def test_mode_c_multi_tier_self_healing_parsing():
    specs = load_rubric_specs(RUBRIC)
    spec_7 = specs["7"]

    # Tier 2: raw_subscores is missing, but marks are inside criteria_reasoning
    parsed_tier2 = {
        "criteria_reasoning": {
            "context_content_data": {"marks": 3.0, "justification": "good data"},
            "organization_cohesion": {"marks": 2.0, "justification": "logical structure"},
            "language_mechanics": {"marks": 1.5, "justification": "minor slips"},
            "originality_comparisons_paraphrase": {"marks": 1.0, "justification": "paraphrased"}
        },
        "structural_audit": {"attempted": True}
    }
    sr2 = score_mode_c(parsed_tier2, spec_7, "A valid paragraph of more than twenty words written by the student.")
    assert sr2.subscores["context_content_data"] == 3.0
    assert sr2.subscores["organization_cohesion"] == 2.0
    assert sr2.raw_total == 7.5
    assert sr2.awarded == 7.5

    # No numbers at all, only narrative mentioning Band 2 (Good): never turned into a mark
    parsed_tier3 = {
        "criteria_reasoning": {
            "context_content_data": {"justification": "good content"},
            "organization_cohesion": {"justification": "coherent flow"},
            "language_mechanics": {"justification": "clear expression"},
            "originality_comparisons_paraphrase": {"justification": "good attempt"}
        },
        "examiner_reasoning": "The student earns marks for content and organization, placing the total in Band 2 Good.",
        "structural_audit": {"attempted": True}
    }
    # Write a paragraph of > 15 words
    student_writing = "Tree plantation is very important for our environment. Trees give us oxygen and shelter and protect against global warming."
    sr3 = score_mode_c(parsed_tier3, spec_7, student_writing)
    assert sr3.unscored and sr3.awarded == 0.0


def test_source_text_for_question_group_header_and_fail_open():
    exam_paper = """
Read the text and answer the questions 5, 6 & 7 :
Lord Byron was an English poet who lived from 1788 to 1824. His father, Captain John Byron, died only three years after his birth. At the age of ten, after the death of his grand uncle, he inherited the family title and estate. Byron attended a number of schools for his education and received his M. A. Degree from Cambridge in 1808.

5. Complete the table.
6. True or False.
7. Write a summary of the passage in your own words. 10
8. Match the sentences.
"""
    # Should resolve the ~60-word passage from the preceding header
    src = source_text_for_question("7", exam_paper)
    assert "Lord Byron" in src and "Captain John Byron" in src

    # Fail-open test: when source passage is short/unresolved (<25 words), length cap is bypassed
    specs = load_rubric_specs(RUBRIC)
    spec_sum = specs["7"]
    spec_sum.task_type = "Summary"
    parsed = {"raw_subscores": {"context_content_data": 3.0, "organization_cohesion": 2.0, "language_mechanics": 1.0, "originality_comparisons_paraphrase": 1.0}}
    short_src = "Instruction only"
    sr = score_mode_c(parsed, spec_sum, "A summary text consisting of fifty words here to test the brevity bypass...", short_src)
    assert not sr.cap_applied
    assert any("cap not checked" in n for n in sr.notes)


def test_matches_accepted_enhanced():
    from src.pipeline.stage4_modes import _matches_accepted
    # 1. OCR space healing
    assert _matches_accepted("com pletely", ["completely"])
    assert _matches_accepted("was published com pletely", ["published completely"])

    # 2. Boilerplate stripping
    assert _matches_accepted("Ans: conventional", ["conventional"])
    assert _matches_accepted("Correct Ans: Lord Byron was an English poet", ["lord byron was an english poet", "english poet"], "True_False")

    # 3. Multi-word phrase containment
    assert _matches_accepted("the age of ten, after his grand uncle's death", ["at the age of ten", "age of 10"], "Table_Completion")
    assert _matches_accepted("was Published completely in six separate Volumes.", ["published completely in six separate volumes"], "Table_Completion")



def test_last_question_prompt_keeps_its_text():
    from src.pipeline.stage4_evaluator import get_sub_question_prompt_and_marks
    q = ExtractedQuestion(
        question_id="X_Q1", language="english", total_marks=20.0,
        question_text="1. Write a paragraph on trees. 10\n2. Write a dialogue about early rising. 10\nClass Test : 10",
        sub_questions=[{"q_no": "1", "title": "Paragraph", "max_marks": 10},
                       {"q_no": "2", "title": "Dialogue", "max_marks": 10},
                       {"q_no": "3", "title": "Story", "max_marks": 5, "instructions": "Complete the story."}],
    )
    assert get_sub_question_prompt_and_marks("2", q)[0] == "Write a dialogue about early rising. 10"
    assert get_sub_question_prompt_and_marks("1", q)[0] == "Write a paragraph on trees. 10"
    # not in the paper text: the sub-question's own instructions are used
    assert "Complete the story." in get_sub_question_prompt_and_marks("3", q)[0]


def test_structured_hard_caps_from_rubric():
    from src.pipeline.stage4_modes import QuestionSpec, apply_hard_caps, describe_hard_caps
    caps = {
        "heavy_verbatim_copying": {"threshold_ratio": 0.5, "max_awarded": 3.0, "applies_to": ["Summary"]},
        "short_length_below_50_words": {"threshold_words": 50, "max_awarded": 4.0, "applies_to": ["Story"]},
        "off_topic": {"max_awarded": 2.0, "applies_to": ["Summary", "Story"]},
    }
    src = " ".join(f"word{i} alpha beta gamma" for i in range(30))
    summ = QuestionSpec(q_no="5", mode="C", task_type="summary", max_mark=10, hard_caps=caps)   # case-insensitive
    assert apply_hard_caps(summ, src[:200], src, {})[:2] == (3.0, "heavy_verbatim_copying")
    assert apply_hard_caps(summ, "my own short words about the passage here", src, {})[0] is None
    assert apply_hard_caps(summ, "my own words", src, {"off_topic": True})[:2] == (2.0, "off_topic")
    story = QuestionSpec(q_no="10", mode="C", task_type="Story", max_mark=15, hard_caps=caps)
    assert apply_hard_caps(story, "a short story of nine words written here", "", {})[:2] == (4.0, "short_length_below_50_words")
    # free-text caps and other task types are ignored
    para = QuestionSpec(q_no="12", mode="C", task_type="Paragraph", max_mark=10, hard_caps={**caps, "x": "text cap"})
    assert apply_hard_caps(para, "few words", src, {"off_topic": True})[0] is None
    assert any("off_topic" in line for line in describe_hard_caps(summ))


def test_mode_a_and_c_carry_no_additive_penalty():
    engine = MockGemmaEngine()
    ev = Stage4Evaluator(engine)
    specs = load_rubric_specs(RUBRIC)
    errs = [{"error_type": "spelling", "erroneous_text": f"wrd{i}", "suggested_correction": "word"} for i in range(6)]
    ans_c = AlignedAnswerItem(q_no="7", answer_text="Trees are important for us. " * 20, page_numbers=[1], word_count=120, errors=errs)
    qe = ev._evaluate_rubric_driven(ans_c, specs["7"], "", {}, "Write a paragraph.", 10.0, "", 0.0, 0.1, 512, False, None)
    assert qe.linguistic_penalty == 0.0


def test_snap_half_rounds_halves_up():
    assert snap_half(7.25) == 7.5 and snap_half(7.75) == 8.0 and snap_half(7.2) == 7.0 and snap_half(0.25) == 0.5


def test_mode_b_answer_any_scores_first_n_attempted():
    from src.pipeline.stage4_modes import QuestionSpec
    spec = QuestionSpec(q_no="10", mode="B", task_type="Short_Answer", max_mark=10, mark_per_item=2, n_items=5,
                        scale={0.0: "", 1.0: "", 2.0: ""})
    key = {"items": [{"label": c} for c in "abcdefgh"], "marks_available": 2}
    parsed = {"items": [{"item_label": c, "candidate_answer": ("Not attempted" if c == "b" else f"answer {c}"),
                         "marks_awarded": (0 if c in "ac" else 2)} for c in "abcdefgh"]}
    sr = score_mode_b(parsed, spec, key)
    st = {it["item_label"]: it["status"] for it in sr.items}
    # attempted in order: a c d e f | g h not counted; b blank
    assert st["b"] == "not_attempted" and st["g"] == "not_counted" and st["h"] == "not_counted"
    assert sr.awarded == 6.0
