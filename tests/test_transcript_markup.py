from src.utils.transcript_markup import normalize_markup, struck_word_spans


def test_well_formed_text_is_unchanged():
    t = "Her father made [struck: po] Portia a\nS[struck: o]umon said [unclear: ane | are] [illegible] incom[truncated]"
    assert normalize_markup(t) == t


def test_unclosed_opener_closes_at_end_of_its_line():
    t = "[struck: Once there live a king. There\n[struck: were green trees]\nTrees that saved"
    assert normalize_markup(t) == "[struck: Once there live a king. There]\n[struck: were green trees]\nTrees that saved"


def test_nested_struck_is_flattened():
    assert normalize_markup("[struck: sunset and the [struck: with] fresh]") == "[struck: sunset and the with fresh]"
    assert normalize_markup("[struck: [struck: affected] affected]") == "[struck: affected affected]"


def test_unclosed_outer_with_inner_tag():
    t = "problems. [struck: The king don't know]\n[struck: sunset and also feel the [struck: with]\nnext line"
    assert normalize_markup(t) == "problems. [struck: The king don't know]\n[struck: sunset and also feel the with]\nnext line"


def test_tag_spanning_lines_strikes_each_line():
    assert normalize_markup("a [struck: b c\nd e] f") == "a [struck: b c]\n[struck: d e] f"


def test_stray_closer_is_dropped():
    assert normalize_markup("palace in the island]\nnext") == "palace in the island\nnext"


def test_literal_brackets_are_text():
    assert normalize_markup("see [a] here") == "see [a] here"


def test_empty_struck_removed():
    assert normalize_markup("word [struck: ] next") == "word  next"


def test_struck_word_spans():
    assert struck_word_spans("a [struck: b c] d") == [("a", False), ("b", True), ("c", True), ("d", False)]
