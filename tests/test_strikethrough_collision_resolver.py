from src.utils.strikethrough_collision_resolver import (
    resolve_strikethrough_collisions,
    ground_and_reconcile_strikethroughs,
)
from src.pipeline.stage0_strikethrough_detector import StrikethroughRegion, StrikethroughBlock


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


def test_itemized_draft_stutter():
    text = "d) overcone inspiring overcone"
    resolved, diffs = resolve_strikethrough_collisions(text)
    assert "d) [struck: overcone inspiring] overcone" in resolved
    assert len(diffs) == 1
    assert diffs[0]["type"] == "itemized_draft_stutter"


def test_short_prefix_stutter():
    text = "make sure po Portia had married a wise man."
    resolved, diffs = resolve_strikethrough_collisions(text)
    assert "[struck: po] Portia" in resolved
    assert len(diffs) == 1
    assert diffs[0]["type"] == "short_prefix_stutter"


def test_draft_restart_multiline_block():
    text = (
        "Answer to Question No-10\n"
        "'Once there live a king in an island. There\n"
        "were green trees everywhere in the island.\n"
        "The king decided to build a magnificent\n"
        "palace in the island\n"
        "\n"
        "Trees that saved a kingdom.\n"
        "\n"
        "Once there lived a king in an island. There\n"
        "were green trees everywhere in the island.\n"
        "The king decided to build a magnificent\n"
        "palace in the island. so he ordered his"
    )
    resolved, diffs = resolve_strikethrough_collisions(text)
    assert "[struck: 'Once there live a king in an island. There]" in resolved
    assert "[struck: were green trees everywhere in the island.]" in resolved
    assert "[struck: The king decided to build a magnificent]" in resolved
    assert "[struck: palace in the island]" in resolved
    assert "Trees that saved a kingdom." in resolved
    assert "Once there lived a king in an island. There" in resolved
    assert any(d["type"] == "draft_restart_multiline_block" for d in diffs)


def test_ground_and_reconcile_unwraps_false_t_bar_strike():
    """When the VLM hallucinates a [struck: cut] due to a 't' crossbar, but no Stage 0 stroke exists, unwrap it."""
    text = (
        "People were starving as the fruit and vegetable\n"
        "plants were [struck: cut] down too. All the\n"
        "animals started to leave."
    )
    # Stage 0 detected zero strikethroughs on this page
    resolved, diffs = ground_and_reconcile_strikethroughs(
        text,
        strikethrough_regions=[],
        strikethrough_blocks=[]
    )
    assert "plants were cut down too." in resolved
    assert "[struck: cut]" not in resolved
    assert any(d["type"] == "false_t_bar_strike_unwrapped" for d in diffs)


def test_ground_and_reconcile_preserves_genuine_strike_with_optical_region():
    """When a word like 'cut' or 'had' is legitimately struck and Stage 0 detected a stroke, preserve [struck: ...]."""
    text = (
        "Her father made this strange plan to\n"
        "make sure [struck: po] Portia [struck: had] married a\n"
        "wise man."
    )
    # Line 2 (out of 3 lines) is at ~50% vertical position
    region = StrikethroughRegion(
        x=200, y=250, w=40, h=3,
        y_pct=50.0, y2_pct=51.0,
        x_pct=20.0, x2_pct=24.0,
        angle=0.0, confidence=0.85,
        is_multi_word=False, is_underline=False
    )
    resolved, diffs = ground_and_reconcile_strikethroughs(
        text,
        strikethrough_regions=[region],
        strikethrough_blocks=[]
    )
    assert "[struck: had]" in resolved
    assert "[struck: po]" in resolved


def test_duplicate_phrase_stutter():
    """Two identical multi-word phrases in succession must be resolved to [struck: ...]."""
    text = "which is also what gipsy men gipsy men tend to do."
    resolved, diffs = resolve_strikethrough_collisions(text)
    assert "[struck: gipsy men] gipsy men" in resolved
    assert len(diffs) == 1
    assert diffs[0]["type"] == "duplicate_phrase_stutter"


def test_ground_and_reconcile_unwraps_rearrangement_indices():
    """Formula strings and question indices like '(a + iv + ii)' must NEVER be struck."""
    text = (
        "Answer to Question No-6\n"
        "[struck: a t i v t i i] Email has brought about a revolution\n"
        "[struck: b t v t i v } ] Massages can be transmitted\n"
        "[struck: c t i t v] It is far cheaper than telephone calls."
    )
    resolved, diffs = ground_and_reconcile_strikethroughs(
        text,
        strikethrough_regions=[],
        strikethrough_blocks=[]
    )
    assert "a t i v t i i Email has brought about a revolution" in resolved
    assert "b t v t i v } Massages can be transmitted" in resolved
    assert "c t i t v It is far cheaper than telephone calls." in resolved
    assert "[struck:" not in resolved


def test_ground_and_reconcile_never_snaps_mcq_options():
    """Draft block snapping must NEVER trigger on MCQ options or Answer headers."""
    text = (
        "Answer to Question no-1\n"
        "c) Ans: ii) students faced severe restrictions\n"
        "d) Ans: ii) He wished films to serve as a voice for\n"
        "C) Ans: v) struggle to survive]\n"
        "d) Ans: ii) He wished films to serve as a voice for truth and justice"
    )
    block = StrikethroughBlock(
        y_pct=10.0, y2_pct=50.0,
        x_pct=10.0, x2_pct=85.0,
        line_count=4,
        stroke_type="parallel_horizontal",
        confidence=0.95
    )
    resolved, diffs = ground_and_reconcile_strikethroughs(
        text,
        strikethrough_regions=[],
        strikethrough_blocks=[block]
    )
    # None of the MCQ lines should be struck
    assert "[struck: d) Ans:" not in resolved
    assert "[struck: C) Ans:" not in resolved
    assert len(diffs) == 0


def test_ground_and_reconcile_snaps_quoted_draft_block():
    """When Stage 0 detected a multi-line block and the VLM output quoted draft lines without [struck: ...], snap them."""
    text = (
        "Answer to Question No-10\n"
        "'Once there live a king in an island. There\n"
        "were green trees everywhere in the island.\n"
        "The king decided to build a magnificent\n"
        "palace in the island]\n"
        "Trees that saved a kingdom.\n"
        "Once there lived a king in an island."
    )
    # Block covers lines 2-5 (~25% to ~65% of page)
    block = StrikethroughBlock(
        y_pct=25.0, y2_pct=65.0,
        x_pct=10.0, x2_pct=85.0,
        line_count=4,
        stroke_type="parallel_horizontal",
        confidence=0.95
    )
    resolved, diffs = ground_and_reconcile_strikethroughs(
        text,
        strikethrough_regions=[],
        strikethrough_blocks=[block]
    )
    assert "[struck: Once there live a king in an island. There]" in resolved
    assert "[struck: were green trees everywhere in the island.]" in resolved
    assert "[struck: The king decided to build a magnificent]" in resolved
    assert "[struck: palace in the island]" in resolved
    assert "Answer to Question No-10" in resolved
    assert "Trees that saved a kingdom." in resolved
    assert any(d["type"] == "draft_block_snapped_to_struck" for d in diffs)


def test_ground_and_reconcile_unwraps_underlined_header():
    """When a heading like 'Answer to Question No-10' was falsely tagged as struck, unwrap it."""
    text = (
        "[struck: Answer to Question No-10]\n"
        "Trees that saved a kingdom."
    )
    resolved, diffs = ground_and_reconcile_strikethroughs(
        text,
        strikethrough_regions=[],
        strikethrough_blocks=[]
    )
    assert "Answer to Question No-10" in resolved
    assert "[struck:" not in resolved
    assert any(d["type"] == "underlined_header_unwrapped" for d in diffs)


def test_ground_and_reconcile_preserves_prefix_strikes_without_optical_region():
    """Prefix stutters like [struck: po] or [struck: di] must NEVER be unwrapped even if Stage 0 missed the stroke."""
    text = "make sure [struck: po] Portia had not been told."
    resolved, diffs = ground_and_reconcile_strikethroughs(
        text,
        strikethrough_regions=[],
        strikethrough_blocks=[]
    )
    assert "[struck: po]" in resolved
    assert len(diffs) == 0



