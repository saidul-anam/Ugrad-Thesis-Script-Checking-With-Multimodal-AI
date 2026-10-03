"""
Tests for Pipeline-Wide Dynamic Transcript Healing & Strikethrough Reconciliation.

Verifies:
1. OCR Allograph Healing: Misread cursive/allograph tokens are healed in context.
2. Strikethrough Suspect Healing: Confirmed struck tokens are wrapped in [struck: ...].
3. Stutter Collision Healing: Repeated tokens or preposition collisions are resolved to [struck: ...].
4. Genuine Error Protection: Confirmed student errors are NEVER normalized.
5. Idempotent Tagging: Struck tokens are never double-wrapped.
6. Rubric Text Cleansing: clean_rubric_answer removes [struck: ...] drafts for Stage 4.
"""

import os
import tempfile
from unittest.mock import MagicMock
from PIL import Image

import pytest
from src.pipeline.token_guard import clean_rubric_answer
from src.pipeline.orchestrator import (
    _normalize_token_in_context,
    _replace_token_preserving_case,
    _apply_normalizations,
)
from src.core.schemas import (
    AlignedAnswerItem,
    LinguisticErrorItem,
)
from src.engine.mock_engine import MockGemmaEngine
from src.pipeline.arbitration import EvidenceArbitrationGate
from src.core.config import ArbitrationConfig


def test_clean_rubric_answer():
    """Verify that [struck: ...] tags are removed while regular text and paragraphs are preserved."""
    raw = (
        "In this time every person should have a dream, and [struck: allay] always focus on your dream.\n\n"
        "A bird sings the tune without words and [struck: her] And that kept so many warm."
    )
    cleaned = clean_rubric_answer(raw)
    assert "[struck:" not in cleaned
    assert "her" not in cleaned
    assert "allay" not in cleaned
    assert "In this time every person should have a dream, and always focus on your dream." in cleaned
    assert "A bird sings the tune without words and And that kept so many warm." in cleaned


def test_ocr_allograph_healing():
    """Verify that an OCR misread token is healed within its context sentence."""
    text = "In this time every person should have a dream, and allay focus on your dream."
    context = "In this time every person should have a dream, and allay focus on your dream."
    c_word = "allay"
    i_word = "always"

    healed = _normalize_token_in_context(
        text=text,
        c_word=c_word,
        i_word=i_word,
        erroneous_text="allay",
        context=context,
        lexicon={"always", "dream", "person", "focus"},
    )
    assert "allay" not in healed
    assert "always focus on your dream" in healed


def test_strikethrough_suspect_healing():
    """Verify that a confirmed struck token is enclosed in [struck: <tok>] without double wrapping."""
    text = "A bird sings the tune without words and her And that kept so many warm."
    context = "A bird sings the tune without words and her And that kept so many warm."
    c_word = "her"
    i_word = "[struck: her]"

    healed = _normalize_token_in_context(
        text=text,
        c_word=c_word,
        i_word=i_word,
        erroneous_text="her And that kept so many warm",
        context=context,
        lexicon={"and", "her", "that", "kept", "so", "many", "warm"},
    )
    assert "[struck: her]" in healed
    assert "and [struck: her] And" in healed

    # Idempotence test: Running normalization again on already-healed text must NOT produce nested tags
    healed_again = _normalize_token_in_context(
        text=healed,
        c_word=c_word,
        i_word=i_word,
        erroneous_text="her And that kept so many warm",
        context=context,
        lexicon={"and", "her", "that", "kept", "so", "many", "warm"},
    )
    assert "[struck: [struck:" not in healed_again
    assert healed_again == healed


def test_collision_stutter_healing():
    """Verify that repeated word stutters are healed to [struck: word] word."""
    text = "Hydro-electric power was was 16% in total."
    context = "Hydro-electric power was was 16% in total."
    c_word = "was was"
    i_word = "[struck: was] was"

    healed = _normalize_token_in_context(
        text=text,
        c_word=c_word,
        i_word=i_word,
        erroneous_text="was was",
        context=context,
        lexicon={"was", "power", "total"},
    )
    assert "[struck: was] was" in healed
    assert "power [struck: was] was 16%" in healed


def test_genuine_error_not_normalized():
    """Verify that genuine student errors are untouched by normalization."""
    text = "The goverment should take steps."
    cleared = [
        {
            "candidate": "goverment",
            "intended_word": "government",
            "verdict": "GENUINE_ERROR",
            "normalize": False,  # Gate sets normalize=False for genuine errors
        }
    ]
    ans = AlignedAnswerItem(q_no="1", answer_text=text, page_numbers=[1], word_count=6)
    page_res = MagicMock()
    page_res.page_no = 1
    page_res.stage2_verification.verified_transcript = text

    count = _apply_normalizations(
        cleared=cleared,
        ans=ans,
        page_results=[page_res],
        errors=[],
        lexicon={"government", "should", "take", "steps"},
    )
    assert count == 0
    assert ans.answer_text == text
    assert page_res.stage2_verification.verified_transcript == text


def test_gate_cleared_strikethrough_normalization():
    """Verify that EvidenceArbitrationGate outputs normalize=True and [struck: ...] for strikethrough collisions."""
    img = Image.new("RGB", (600, 300), (255, 255, 255))
    err = LinguisticErrorItem(
        error_type="grammar",
        erroneous_text="was was",
        suggested_correction="was",
        context_sentence="Hydro-electric power was was 16% in total.",
        explanation="Stutter duplicate copula.",
    )
    with tempfile.TemporaryDirectory() as d:
        cfg = ArbitrationConfig(consensus_samples=2, use_bbox_localization=False)
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(),
            cfg=cfg,
            script_id="mock",
            output_dir=d,
            page_images=[(1, img, "p1.png")],
            page_transcripts={1: err.context_sentence},
            clean_image_fn=None,
            full_transcript=err.context_sentence,
            lexicon={"hydro-electric", "power", "was", "total"},
        )
        confirmed, cleared, records = gate.arbitrate([err], q_no="3", answer_text=err.context_sentence)
        
        # Must be cleared with zero deduction and normalized
        assert len(confirmed) == 0
        assert len(cleared) == 1
        amb = cleared[0]
        assert amb["verdict"] == "HANDWRITING_AMBIGUITY"
        assert amb["normalize"] is True
        assert amb["intended_word"] == "[struck: was] was"

