"""
Unit tests for Edge Truncation Detector & Cross-Line Stitcher.
"""

import pytest
from src.pipeline.edge_truncation_detector import (
    extract_line_end_tokens,
    stitch_cross_line_truncations,
    is_right_edge_truncation
)


def test_extract_line_end_tokens():
    sample_transcript = """
we should increase the use of renewabl
energy and natural resources.
The lion saw the mouse wor[truncated]
and was about to catch it.
--- Page Break ---
In 1980 electricity was pro
duced by natural coal.
"""
    info = extract_line_end_tokens(sample_transcript)
    assert "renewabl" in info["line_end_tokens"]
    assert "wor" in info["line_end_tokens"]
    assert "pro" in info["line_end_tokens"]
    assert "wor" in info["explicit_truncated"]


def test_stitch_cross_line_truncations():
    lexicon = {"university", "renewable", "intend", "passing", "produced"}

    # Hyphenated
    text1 = "Admitted into Dhaka Universi-\nty next year."
    stitched1, diffs1 = stitch_cross_line_truncations(text1, lexicon)
    assert "University" in stitched1
    assert "Universi-\nty" not in stitched1
    assert len(diffs1) == 1

    # Truncated tag with newline
    text2 = "Admitted into Dhaka Universi[truncated]\nty next year."
    stitched2, diffs2 = stitch_cross_line_truncations(text2, lexicon)
    assert "University" in stitched2
    assert len(diffs2) == 1

    # Unhyphenated split word where concatenation is in lexicon
    text3 = "we should increase renewabl\ne energy."
    stitched3, diffs3 = stitch_cross_line_truncations(text3, lexicon)
    assert "renewable" in stitched3
    assert len(diffs3) == 1

    text4 = "He had the inten\nd to study medicine."
    stitched4, diffs4 = stitch_cross_line_truncations(text4, lexicon)
    assert "intend" in stitched4
    assert len(diffs4) == 1


def test_is_right_edge_truncation_detection():
    transcript = """
we want to save these natural resources
we should increase the use of renewabl
energy.
The villagers lived in a small villa
and worked hard.
Coal was the highest pro
duced energy source.
At the right edge he had becon
a great doctor.
"""
    lexicon = {"renewable", "village", "produced", "become", "words"}

    # 1. 'renewabl' at line end -> 'renewable' (prefix match)
    assert is_right_edge_truncation(
        erroneous_text="renewabl",
        suggested_correction="renewable",
        context_sentence="we should increase the use of renewabl",
        transcript=transcript,
        lexicon=lexicon
    ) is True

    # 2. 'villa' at line end -> 'village'
    assert is_right_edge_truncation(
        erroneous_text="villa",
        suggested_correction="village",
        context_sentence="The villagers lived in a small villa",
        transcript=transcript,
        lexicon=lexicon
    ) is True

    # 3. 'pro' at line end -> 'produced'
    assert is_right_edge_truncation(
        erroneous_text="pro",
        suggested_correction="produced",
        context_sentence="Coal was the highest pro",
        transcript=transcript,
        lexicon=lexicon
    ) is True

    # 4. 'becon' at line end -> 'become' (near prefix with final character distortion)
    assert is_right_edge_truncation(
        erroneous_text="becon",
        suggested_correction="become",
        context_sentence="At the right edge he had becon",
        transcript=transcript,
        lexicon=lexicon
    ) is True

    # 5. Explicit [truncated] tag anywhere
    assert is_right_edge_truncation(
        erroneous_text="wor[truncated]",
        suggested_correction="words",
        context_sentence="in other wor[truncated] they were happy"
    ) is True


def test_protection_against_false_suppression():
    """Ensure genuine student errors are strictly preserved and NOT suppressed."""
    transcript = """
we want to save these natural resources
we should renewabl energy in all sectors
and visit their familyes every year.
They went to scholl yesterday.
"""
    lexicon = {"renewable", "families", "school"}

    # 1. Genuine misspelling in the MIDDLE of a line
    # 'renewabl' is in the middle of line 2, not at the end of the line!
    assert is_right_edge_truncation(
        erroneous_text="renewabl",
        suggested_correction="renewable",
        context_sentence="we should renewabl energy in all sectors",
        transcript=transcript,
        lexicon=lexicon
    ) is False

    # 2. Genuine misspelling at the end of a line, but NOT a truncation prefix
    # 'familyes' ends line 3, but 'families' does not start with 'familyes' (wrong suffix)
    assert is_right_edge_truncation(
        erroneous_text="familyes",
        suggested_correction="families",
        context_sentence="and visit their familyes every year.",
        transcript=transcript,
        lexicon=lexicon
    ) is False

    # 3. 'scholl' at the end of line 4 -> 'school' (not a prefix)
    assert is_right_edge_truncation(
        erroneous_text="scholl",
        suggested_correction="school",
        context_sentence="They went to scholl yesterday.",
        transcript=transcript,
        lexicon=lexicon
    ) is False
