import pytest
from src.pipeline.stage4_modes import QuestionSpec, score_mode_a, compute_lcs_alignment
from src.pipeline.token_guard import sanitize_rearrangement_sequence


EXPECTED_SEQ = ["c", "h", "j", "a", "e", "d", "g", "i", "f", "b"]
SPEC = QuestionSpec(
    q_no="6",
    mode="A",
    task_type="Rearrangement",
    max_mark=10.0,
    mark_per_item=1.0,
    n_items=10
)
KEY_ENTRY = {
    "task_type": "Rearrangement",
    "mark_per_item": 1.0,
    "correct_sequence": EXPECTED_SEQ
}


def test_q6_perfect_match():
    """Verify that a perfect sequence earns 10.0 marks."""
    parsed = {"student_sequence": EXPECTED_SEQ}
    res = score_mode_a(parsed, SPEC, KEY_ENTRY)
    assert res.awarded == 10.0
    assert len(res.items) == 10
    assert all(it["status"] == "correct" for it in res.items)


def test_q6_homoglyph_repair_script_0002():
    """Verify that Script 0002 ('o' instead of 'j') is repaired and earns 10.0 marks."""
    raw = ["c", "h", "o", "a", "e", "d", "g", "i", "f", "b"]
    parsed = {"student_sequence": raw}
    res = score_mode_a(parsed, SPEC, KEY_ENTRY)
    assert res.awarded == 10.0
    assert any("repaired_homoglyph:o->j" in n or "repaired_substitution:o->j" in n for n in res.notes)


def test_q6_multi_homoglyph_repair():
    """Verify that multiple OCR glitches in one answer ('o'->'j' AND 'l'->'i') are all repaired."""
    raw = ["c", "h", "o", "a", "e", "d", "g", "l", "f", "b"]
    cleaned, anomalies = sanitize_rearrangement_sequence(raw)
    assert cleaned == EXPECTED_SEQ
    parsed = {"student_sequence": raw}
    res = score_mode_a(parsed, SPEC, KEY_ENTRY)
    assert res.awarded == 10.0


def test_q6_single_omission_lcs_cascade_prevention():
    """
    CRITICAL TEST: Verify that omitting sentence 'j' (producing 9 letters)
    earns 9.0 marks via LCS rather than collapsing to 2.0 marks under naive slot checking.
    """
    # Student wrote 9 sentences, missing 'j'
    student = ["c", "h", "a", "e", "d", "g", "i", "f", "b"]
    parsed = {"student_sequence": student}
    res = score_mode_a(parsed, SPEC, KEY_ENTRY)

    # Naive slot comparison would give only 2 marks (c and h). LCS must give 9 marks.
    assert res.awarded == 9.0
    # Positional matches should be 2, but LCS matches should be 9
    assert any("slot matches: 2, LCS relative order matches: 9" in n for n in res.notes)


def test_q6_prefix_noise_insertion_lcs_protection():
    """Verify that a stray token at the beginning shifts slots but LCS protects remaining 9 items."""
    # Stray noise token 'x' at start, pushing everything right
    student = ["x", "c", "h", "j", "a", "e", "d", "g", "i", "f"]
    parsed = {"student_sequence": student}
    res = score_mode_a(parsed, SPEC, KEY_ENTRY)
    # Naive slot comparison gives 0 marks. LCS must rescue the 9 correct items.
    assert res.awarded == 9.0


def test_q6_adjacent_swap_preserves_partial_marks():
    """Verify that swapping an adjacent pair (j and a) earns 9.0 marks via LCS."""
    student = ["c", "h", "a", "j", "e", "d", "g", "i", "f", "b"]
    parsed = {"student_sequence": student}
    res = score_mode_a(parsed, SPEC, KEY_ENTRY)
    assert res.awarded == 9.0


def test_q6_gibberish_sequence_earns_minimal_marks():
    """Verify that completely scrambled or backwards sequence receives minimal marks."""
    student = ["b", "f", "i", "g", "d", "e", "a", "j", "h", "c"]  # exact reverse
    parsed = {"student_sequence": student}
    res = score_mode_a(parsed, SPEC, KEY_ENTRY)
    assert res.awarded <= 2.0


def test_q6_script_0010_empirical():
    """Verify Script 0010 sequence scoring."""
    raw = ["c", "a", "e", "d", "g", "i", "h", "f", "b"]
    cleaned, anomalies = sanitize_rearrangement_sequence(raw)
    parsed = {"student_sequence": cleaned}
    res = score_mode_a(parsed, SPEC, KEY_ENTRY)
    # Slot match is 1 (c). LCS relative order matches 8 (c, a, e, d, g, i, f, b).
    assert res.awarded >= 1.0
    assert any("slot matches: 1" in n for n in res.notes)

