from src.pipeline.line_reconciler import align_lines, disagreement_spans, replace_span


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
    start, end, new = disagreement_spans(line, "you more refreshment.")[0]
    assert replace_span(line, start, end, new) == "you more refreshment."
