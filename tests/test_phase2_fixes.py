import pytest
from types import SimpleNamespace
from src.pipeline.allograph_calibrator import check_cursive_topology, PLAUSIBLE_ALLOGRAPH_PAIRS
from src.utils.strikethrough_collision_resolver import ground_and_reconcile_strikethroughs
from src.pipeline.stage2_verifier import run_stage2_pre_analysis


def test_cursive_n_r_allograph_topology():
    """Verify that cursive n <-> r is in PLAUSIBLE_ALLOGRAPH_PAIRS and check_cursive_topology works."""
    assert frozenset({'n', 'r'}) in PLAUSIBLE_ALLOGRAPH_PAIRS

    vocab = {"ordered", "hotter", "worker", "weather", "eary", "threes"}
    assert check_cursive_topology("ondered", vocab) == ("allograph_n_r", "ordered")
    assert check_cursive_topology("weathen", vocab) == ("allograph_n_r", "weather")
    assert check_cursive_topology("hotten", vocab) == ("allograph_n_r", "hotter")
    assert check_cursive_topology("thnees", vocab) == ("allograph_n_r", "threes")


def test_strikethrough_tag_inversion_spatial_resolution():
    """Verify that tag inversion 'w1 [struck: w2]' is inverted when stroke is physically over w1."""
    text = "The temperature [struck: heat level] increased rapidly."
    # Simulate Stage 0 optical strike over 'temperature' (x: 5% - 25%, y: 50%)
    stroke = SimpleNamespace(
        y_pct=50.0,
        y2_pct=52.0,
        x_pct=5.0,
        x2_pct=25.0,
        is_underline=False,
        confidence=0.95
    )
    resolved, diffs = ground_and_reconcile_strikethroughs(
        text=text,
        strikethrough_regions=[stroke],
        strikethrough_blocks=None
    )
    assert "[struck: temperature] heat level" in resolved
    assert len(diffs) >= 1
    assert diffs[0]["type"] == "strikethrough_tag_inversion_resolved"


def test_cascaded_page_router_pristine_vs_anomaly():
    """Verify 5-point CPU anomaly detection for Stage 2 Fast Mode bypass."""
    # 1. Clean transcript with 0 strikes, 0 glitches, 0 splits -> should NOT trigger Stage 2
    clean_text = "Once upon a time there was a beautiful forest with tall trees and green plants."
    rep_clean = run_stage2_pre_analysis(
        stage1_transcript=clean_text,
        strikethrough_regions=None,
        strikethrough_blocks=None,
        question_vocab={"forest", "trees", "plants"}
    )
    assert rep_clean.should_trigger_stage2 is False

    # 2. Page with detected strike regions -> MUST trigger Stage 2
    stroke = SimpleNamespace(y_pct=30.0, y2_pct=31.0, confidence=0.9, is_underline=False)
    rep_strike = run_stage2_pre_analysis(
        stage1_transcript=clean_text,
        strikethrough_regions=[stroke],
        strikethrough_blocks=None
    )
    assert rep_strike.should_trigger_stage2 is True

    # 3. Page with cursive n<->r glitch candidate ('ondered') -> MUST trigger Stage 2
    glitch_text = "The king ondered his men to cut the trees."
    rep_glitch = run_stage2_pre_analysis(
        stage1_transcript=glitch_text,
        strikethrough_regions=None,
        strikethrough_blocks=None,
        question_vocab={"ordered", "king", "trees"}
    )
    assert rep_glitch.should_trigger_stage2 is True
    assert any(g["word"] == "ondered" for g in rep_glitch.likely_glitches)
