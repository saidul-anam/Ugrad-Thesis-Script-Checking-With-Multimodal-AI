"""
Unit tests for pipeline logic hardening fixes:
1. Punctuation run suppression.
2. Nested struck tag flattening.
3. Sub-question header normalization (Bans: -> (B) Ans:, Dans: -> (D) Ans:).
4. Stage 3b strikeout suspect phonetic signal neutralization.
"""

import pytest
from src.pipeline.stage1_transcriber import sanitize_and_normalize_stage1_output
from src.pipeline.answer_segmenter import normalize_header_text
from src.pipeline.arbitration.gate import EvidenceArbitrationGate
from src.core.schemas import ArbitrationCandidate
from src.core.config import ArbitrationConfig


def test_punctuation_run_collapse():
    raw = "The student answered and’s’’’’’’’’ well."
    cleaned = sanitize_and_normalize_stage1_output(raw)
    assert "’s’’’’’’’’" not in cleaned
    assert "’s" in cleaned

    dots = "Wait..... what happened,,,,, here????"
    cleaned_dots = sanitize_and_normalize_stage1_output(dots)
    assert "....." not in cleaned_dots
    assert ",,,,," not in cleaned_dots


def test_nested_struck_tag_flattening():
    raw = "The [struck: [struck: wrong word]] was written."
    cleaned = sanitize_and_normalize_stage1_output(raw)
    assert "[struck: [struck:" not in cleaned
    assert "[struck: wrong word]" in cleaned


def test_fused_subpart_header_normalization():
    text1 = "Dans: The author illustrates hope."
    norm1 = normalize_header_text(text1)
    assert "(D) Ans: The author illustrates hope." == norm1

    text2 = "Bans: The second answer."
    norm2 = normalize_header_text(text2)
    assert "(B) Ans: The second answer." == norm2


def test_strikeout_phonetic_neutralization():
    """Verify that strikethrough suspects have phonetic_signal = None (not 1.00)."""
    cand = ArbitrationCandidate(
        candidate_id="1:2:strike",
        error_index=2,
        error_type="strikethrough_suspect",
        erroneous_text="many",
        suggested_correction="many of",
        candidate_token="many",
        intended_token="[struck]",
        context_sentence="helps many us",
    )

    class MockEngine:
        pass

    cfg = ArbitrationConfig(save_crops=False)
    gate = EvidenceArbitrationGate(
        engine=MockEngine(),
        cfg=cfg,
        script_id="test_script",
        output_dir="/tmp",
        page_images=[],
        page_transcripts={},
        clean_image_fn=None,
        full_transcript="",
        lexicon=set(),
    )
    record = gate._score_candidate(cand)
    assert record.evidence.phonetic_signal is None
    assert record.evidence.phonetic_plausibility is None
