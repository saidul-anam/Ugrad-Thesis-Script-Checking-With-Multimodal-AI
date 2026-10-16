from dataclasses import dataclass
from typing import List, Optional

from src.pipeline.line_reconciler import (
    _tokens_match,
    align_lines,
    disagreement_spans,
    has_physical_midline_stroke,
    parse_choice_digit,
    replace_span,
)


def test_align_lines_skips_junk_and_keeps_order():
    t = ["Sumon : You have to rise early.", "Afif : Why?", "Sumon : Because it helps."]
    r = ["", "Sumon : You have to rise early.", "CS CamScanner noise", "Afif ; Why ?", "Sumon : Because it help."]
    assert align_lines(t, r) == [(0, 1), (1, 3), (2, 4)]


def test_disagreement_on_strike_flag_only():
    spans = disagreement_spans("the king did learning studying", "the king did [struck: learning] studying")
    assert len(spans) == 1
    start, end, new = spans[0]
    assert (start, end) == (3, 4) and new[0].struck


def test_punctuation_and_spacing_are_not_disagreements():
    assert disagreement_spans("Afif : Why?", "Afif ; Why ?") == []


def test_replace_span_keeps_rest_of_line():
    line = "Oh, I can now not realize the fact."
    start, end, new = disagreement_spans(line, "Oh, I can now [struck: not] realize the fact.")[0]
    assert replace_span(line, start, end, new) == "Oh, I can now [struck: not] realize the fact."


def test_replace_span_word_substitution():
    line = "you more refreshmant."
    assert disagreement_spans(line, "you more refreshment.") == []          # default scope: strikes only
    start, end, new = disagreement_spans(line, "you more refreshment.", scope="all")[0]
    assert replace_span(line, start, end, new) == "you more refreshment."


def test_fuzzy_token_matching_guard():
    # Short words (<= 3 chars) MUST match exactly to prevent preposition/article drift
    assert _tokens_match("in", "on") is False
    assert _tokens_match("cat", "car") is False
    assert _tokens_match("to", "of") is False
    assert _tokens_match("the", "she") is False
    assert _tokens_match("in", "in") is True

    # Longer words (>= 4 chars) allow 1-character OCR typos
    assert _tokens_match("contaminate", "contiminate") is True
    assert _tokens_match("birth", "binth") is True
    assert _tokens_match("desertation", "desertatior") is True
    assert _tokens_match("mother", "mothen") is True

    # Completely different long words must not match
    assert _tokens_match("contaminate", "communicate") is False
    assert _tokens_match("mother", "father") is False


def test_fuzzy_token_alignment_rescuing_ocr_typos():
    line = "iv) Ans: drinking arsenic contaminate water"
    # The crop OCR reads 'contiminate' (with an 'i') instead of 'contaminate'
    read = "iv) Ans: [struck: drinking arsenic contiminate water]"

    spans = disagreement_spans(line, read, scope="strikes")
    assert len(spans) == 1
    start, end, new = spans[0]
    assert (start, end) == (2, 6)

    # Verifies spelling immutability: applies struck tag while keeping authentic 'contaminate'
    result = replace_span(line, start, end, new)
    assert result == "iv) Ans: [struck: drinking arsenic contaminate water]"
    assert "contiminate" not in result


def test_surface_spelling_immutability():
    # Student authentic misspelling 'Remmen'
    line = "c) Ans. The mothers in poem Two Mothers Remmen Remembered"
    read = "c) Ans. The mothers in poem Two Mothers [struck: Remmin] Remembered"

    spans = disagreement_spans(line, read, scope="strikes")
    assert len(spans) == 1
    start, end, new = spans[0]
    assert replace_span(line, start, end, new) == "c) Ans. The mothers in poem Two Mothers [struck: Remmen] Remembered"


@dataclass
class MockRegion:
    x: int
    y: int
    w: int
    h: int
    is_underline: bool = False


def test_midline_stroke_filtering():
    # Line box: [x1=100, y1=100, x2=800, y2=150] (height=50, mid=125)
    line_box = (100, 100, 800, 150)

    # 1. Underline at baseline (y=145, h=4) -> relative y = (147 - 100) / 50 = 0.94 -> rejected
    underline_r = [MockRegion(x=200, y=145, w=200, h=4, is_underline=True)]
    assert has_physical_midline_stroke(line_box, strikethrough_regions=underline_r) is False

    # 2. Midline strike (y=122, h=4) -> relative y = (124 - 100) / 50 = 0.48 -> accepted
    midline_r = [MockRegion(x=200, y=122, w=200, h=4, is_underline=False)]
    assert has_physical_midline_stroke(line_box, strikethrough_regions=midline_r) is True

    # 3. Midline strike horizontally outside the line box (x=900) -> rejected
    outside_r = [MockRegion(x=900, y=122, w=200, h=4, is_underline=False)]
    assert has_physical_midline_stroke(line_box, strikethrough_regions=outside_r) is False


def test_choice_digit_parsing():
    assert parse_choice_digit("Choice: 2", alt_slot="2") == 1
    assert parse_choice_digit("Choice: 1", alt_slot="2") == 0
    assert parse_choice_digit(
        "Observation: A clear horizontal pen stroke cuts through the letters.\nChoice: 2", alt_slot="2"
    ) == 1
    assert parse_choice_digit(
        "Observation: The letters are clean and untouched.\nChoice: 1", alt_slot="1"
    ) == 1
    assert parse_choice_digit("Candidate 2 matches the ink.", alt_slot="2") == 1
    assert parse_choice_digit("Candidate 1 matches the ink.", alt_slot="2") == 0
