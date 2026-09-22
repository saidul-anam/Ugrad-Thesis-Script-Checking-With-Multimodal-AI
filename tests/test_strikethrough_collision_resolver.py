import pytest
from src.utils.strikethrough_collision_resolver import resolve_strikethrough_collisions


def test_duplicate_word_stutter():
    text = "In our world many countrie's childs are are poor economy."
    resolved, diffs = resolve_strikethrough_collisions(text)
    assert "[struck: are] are" in resolved
    assert len(diffs) == 1
    assert diffs[0]["type"] == "duplicate_word_stutter"


def test_preposition_collision():
    text = "Teachers also use AI by for giving better education to the student."
    resolved, diffs = resolve_strikethrough_collisions(text)
    assert "[struck: by] for" in resolved
    assert len(diffs) == 1
    assert diffs[0]["type"] == "preposition_collision"


def test_fragment_stutter():
    text = "They can increse incre their general knowledge."
    resolved, diffs = resolve_strikethrough_collisions(text)
    assert "[struck: increse] incre" in resolved
    assert len(diffs) == 1
    assert diffs[0]["type"] == "fragment_stutter"


def test_legitimate_phrases_not_touched():
    # "he had had enough" should NOT be modified
    text1 = "He had had enough time to finish the exam."
    resolved1, diffs1 = resolve_strikethrough_collisions(text1)
    assert "[struck" not in resolved1
    assert len(diffs1) == 0

    # "out of" should NOT be modified
    text2 = "They ran out of resources quickly."
    resolved2, diffs2 = resolve_strikethrough_collisions(text2)
    assert "[struck" not in resolved2
    assert len(diffs2) == 0

    # "according to" should NOT be modified
    text3 = "According to the passage, the climate is changing."
    resolved3, diffs3 = resolve_strikethrough_collisions(text3)
    assert "[struck" not in resolved3
    assert len(diffs3) == 0

    # "about to" should NOT be modified
    text4 = "He was about to finish the task."
    resolved4, diffs4 = resolve_strikethrough_collisions(text4)
    assert "[struck" not in resolved4
    assert len(diffs4) == 0

    # "thin thing" should NOT be modified
    text5 = "They rely on this thin thing."
    resolved5, diffs5 = resolve_strikethrough_collisions(text5)
    assert "[struck" not in resolved5
    assert len(diffs5) == 0


def test_means_determiner_stutter():
    text = "Artificial Intelligane means a it is a software programme."
    resolved, diffs = resolve_strikethrough_collisions(text)
    assert "[struck: a] it is" in resolved
    assert len(diffs) == 1
