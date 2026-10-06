import pytest
from unittest.mock import MagicMock
from PIL import Image

from src.core.schemas import Stage2VerificationResult, AutocorrectionDiffItem
from src.pipeline.stage2_verifier import (
    is_spurious_reversion,
    apply_anchored_diff,
    extract_ghost_corrections_from_notes,
    extract_unlisted_strikethroughs,
    purge_empty_struck_tags,
    Stage2Verifier,
    PROTECTED_FUNCTION_WORDS,
)


# ---------------------------------------------------------------------------
# Problem 4: Protected Function Words & Priming Resistance
# ---------------------------------------------------------------------------

def test_protected_function_words_blocklist():
    """Verify high-frequency English function words cannot be mutated to rare nouns or homoglyphs."""
    # Classic Problem 4 bug: 'have' -> 'hare'
    assert is_spurious_reversion("have", "hare") is True
    assert is_spurious_reversion("Have", "Hare") is True
    
    # Preposition & pronoun mutations
    assert is_spurious_reversion("with", "whith") is True
    assert is_spurious_reversion("that", "thad") is True
    assert is_spurious_reversion("from", "froom") is True
    assert is_spurious_reversion("their", "thier") is True

    # Strikethrough wrapping of function words MUST still be permitted
    assert is_spurious_reversion("have", "[struck: have]") is False
    assert is_spurious_reversion("with", "[struck: with]") is False
    assert is_spurious_reversion("was", "[struck: was]") is False

    # Genuine student spelling mistakes on open-class words must NOT be blocked
    assert is_spurious_reversion("bornito", "bortito") is False
    assert is_spurious_reversion("village", "villagc") is False


# ---------------------------------------------------------------------------
# Problem 14: Context-Anchored Window Substitution
# ---------------------------------------------------------------------------

def test_apply_anchored_diff_exact_context():
    """Verify replacement using exact context snippet."""
    text = "The quick brown fox jumps over the lazy dog."
    updated, success = apply_anchored_diff(
        text=text,
        stage1_target="fox",
        actual="cat",
        context_snippet="brown fox jumps"
    )
    assert success is True
    assert updated == "The quick brown cat jumps over the lazy dog."


def test_apply_anchored_diff_multi_occurrence_disambiguation():
    """Verify that multiple occurrences of a target are correctly disambiguated by context."""
    text = (
        "Paragraph 1: It was a cold night.\n"
        "Paragraph 2: The water was warm.\n"
        "Paragraph 3: The pie chart was clear.\n"
        "Paragraph 4: Hydro power was 16%."
    )
    # Target: replace 'was' with '[struck: was]' ONLY in Paragraph 3
    updated, success = apply_anchored_diff(
        text=text,
        stage1_target="was",
        actual="[struck: was]",
        context_snippet="The pie chart was clear"
    )
    assert success is True
    assert "Paragraph 1: It was a cold night." in updated
    assert "Paragraph 2: The water was warm." in updated
    assert "Paragraph 3: The pie chart [struck: was] clear." in updated
    assert "Paragraph 4: Hydro power was 16%." in updated


def test_apply_anchored_diff_fuzzy_whitespace_context():
    """Verify replacement when context snippet has different whitespace or newline formatting."""
    text = "In 1980 the percentage\nwas Hydro-electrice power was 16%."
    updated, success = apply_anchored_diff(
        text=text,
        stage1_target="was",
        actual="[struck: was]",
        context_snippet="percentage was Hydro-electrice"
    )
    assert success is True
    assert "[struck: was] Hydro-electrice" in updated


def test_apply_anchored_diff_ambiguous_without_context_aborts():
    """Verify that ambiguous targets with multiple occurrences and no context abort rather than corrupting top of page."""
    text = "He was here. She was there. Everyone was happy."
    updated, success = apply_anchored_diff(
        text=text,
        stage1_target="was",
        actual="[struck: was]",
        context_snippet=""
    )
    # Must abort to prevent corrupting first occurrence
    assert success is False
    assert updated == text


def test_apply_anchored_diff_single_occurrence_unambiguous():
    """Verify single unambiguous occurrence is safely replaced even without context."""
    text = "The unique elephant marched forward."
    updated, success = apply_anchored_diff(
        text=text,
        stage1_target="elephant",
        actual="mammoth",
        context_snippet=""
    )
    assert success is True
    assert updated == "The unique mammoth marched forward."


# ---------------------------------------------------------------------------
# Problem 17: Programmatic Ghost Correction Extraction
# ---------------------------------------------------------------------------

def test_extract_ghost_corrections_0006_p9():
    """Test extracting ghost correction on 'low' from actual 0006/p9 verification_notes."""
    notes = (
        "The transcription was largely accurate. A strike-through was identified on the word 'low' "
        "in the phrase 'this is the low percentage lowest percentage', which was missing from the "
        "Stage 1 output. The student's spelling errors ('electrice', 'eel', 'wed') were correctly "
        "preserved by Stage 1 and maintained here."
    )
    transcript = (
        "nuclear was 2%. and this is the low percentage\n"
        "lowest percentage. After that, oil was 12%."
    )
    ghost_diffs = extract_ghost_corrections_from_notes(notes, transcript)
    assert len(ghost_diffs) == 1
    assert ghost_diffs[0].stage1_output == "low"
    assert ghost_diffs[0].actual_handwritten == "[struck: low]"
    assert "lowest percentage" in ghost_diffs[0].context_snippet


def test_extract_ghost_corrections_0022_p16():
    """Test extracting ghost corrections on 'indicates to' and 'dream into reality' from 0022/p16 notes."""
    notes = (
        "Stage 1 failed to identify specific strike-throughs. I have applied [struck: ...] to 'indicates to' "
        "(Region 7) and the first instance of 'dream into reality' (Region 8) as per the provided coordinates "
        "and visual evidence. The student wrote the phrase 'dream into reality' twice, striking through the first attempt."
    )
    transcript = (
        "The following poem indicates to seek\n"
        "our dreams and hopes seriously. All of\n"
        "the people can dream but can't make the\n"
        "dream into reality.\n"
    )
    ghost_diffs = extract_ghost_corrections_from_notes(notes, transcript)
    assert len(ghost_diffs) == 2
    targets = {d.stage1_output for d in ghost_diffs}
    assert "indicates to" in targets
    assert "dream into reality" in targets


def test_purge_empty_struck_tags():
    """Verify empty or whitespace-only strikethrough tags from margin noise are wiped cleanly."""
    raw = "Ans to the Question no:11\n~~~~~~~~~~~~~~~~~~~~~~\n[struck:  ] [struck:  ] [struck:  ] [struck:  ]\nThe poem."
    cleaned = purge_empty_struck_tags(raw)
    assert "[struck:" not in cleaned
    assert "The poem." in cleaned


def test_extract_unlisted_strikethroughs():
    """Verify extracting strikethrough tags present in raw verified transcript but omitted from diffs."""
    s1 = "According to the graph, In 1980 the percentage was Hydro-electrice power was 16%."
    raw_v = "According to the graph, In 1980 the percentage [struck: was] Hydro-electrice power was 16%."
    diffs = extract_unlisted_strikethroughs(s1, raw_v)
    assert len(diffs) == 1
    assert diffs[0].stage1_output == "was"
    assert diffs[0].actual_handwritten == "[struck: was]"


# ---------------------------------------------------------------------------
# Problem 1 & Full Stage2Verifier Integration Mock Test
# ---------------------------------------------------------------------------

def test_stage2_verifier_full_flow_recovers_ghost_correction():
    """Verify full Stage2Verifier.run pipeline using mock engine on 0006/p9 scenario."""
    s1_transcript = (
        "Answer to the question No. 8\n\n"
        "The pie chart describes about the sources\n"
        "of the USA electricity in 1980.\n\n"
        "According to the graph, In 1980 the percentage\n"
        "was Hydro-electrice power was 16%. Then we see\n"
        "nuclear was 2%. and this is the low percentage\n"
        "lowest percentage. After that, oil was 12%."
    )
    # VLM returned empty silent_corrections_fixed, but mentioned strike-through in notes
    mock_response = """```json
    {
      "verified_transcript": "Answer to the question No. 8\\n\\nThe pie chart describes about the sources of the USA electricity in 1980.",
      "silent_corrections_fixed": [],
      "total_corrections_count": 0,
      "verification_notes": "A strike-through was identified on the word 'low' in the phrase 'this is the low percentage lowest percentage', which was missing from the Stage 1 output."
    }
    ```"""
    mock_engine = MagicMock()
    mock_engine.generate_multimodal.return_value = mock_response

    verifier = Stage2Verifier(mock_engine, legacy_filters=True)
    result = verifier.run(
        image=Image.new("RGB", (100, 100)),
        stage1_transcript=s1_transcript
    )

    # 1. Strikethrough on 'low' must be recovered
    assert "[struck: low]" in result.verified_transcript
    # 2. Stage 1 immutable base: student spelling 'electrice' must NOT be autocorrected
    assert "Hydro-electrice" in result.verified_transcript
    # 3. Corrections count reflects recovered ghost correction
    assert result.total_corrections_count >= 1
    assert any(d.stage1_output == "low" for d in result.silent_corrections_fixed)


def test_strikethrough_anti_nesting_and_duplicate_prevention():
    """Verify that strikethrough tags never nest and unlisted synthesis skips declared targets."""
    # 1. Direct apply_anchored_diff test on already struck token
    text = "The [struck: both] options were considered."
    updated, success = apply_anchored_diff(
        text=text,
        stage1_target="both",
        actual="[struck: both]",
        context_snippet="The both options"
    )
    # Must NOT produce [struck: [struck: both]]
    assert "[struck: [struck:" not in updated
    assert updated == text

    # 2. Purge un-nesting test
    nested_text = "Answer (b) [struck: [struck: both]] and (c) [struck: [struck: [struck: healthy]]]"
    cleaned = purge_empty_struck_tags(nested_text)
    assert "[struck: [struck:" not in cleaned
    assert "[struck: both]" in cleaned
    assert "[struck: healthy]" in cleaned

    # 3. Duplicate unlisted extraction skips declared targets
    s1 = "I decided me and my family enjoy"
    raw_v = "I decided [struck: me] and my family enjoy"
    diffs = extract_unlisted_strikethroughs(
        stage1_text=s1,
        raw_verified_text=raw_v,
        already_declared_targets={"me"}
    )
    assert len(diffs) == 0


# ---------------------------------------------------------------------------
# Stage 2 Redesign: Pre-Analysis, Delta Patches, Drift & Hallucination Defense
# ---------------------------------------------------------------------------

def test_pre_analysis_flags_curriter_and_preserves_poverly():
    """Verify Phase 1 pre-analysis classifies 'curriter' as OCR glitch and 'desenibe' as student error."""
    from src.pipeline.stage2_verifier import run_stage2_pre_analysis
    transcript = (
        "Ans to the Question No - 01\n"
        "The curriter cannot write beautiful words about Gaza because desenibe famine siege covered the whole area."
    )
    report = run_stage2_pre_analysis(stage1_transcript=transcript)

    # curriter must be flagged as likely OCR glitch via ligature_curr_wr topology with candidate 'writer'
    assert any(g["word"] == "curriter" and g["candidate"] == "writer" for g in report.likely_glitches)
    assert "curriter" in report.formatted_prompt_block
    assert "writer" in report.formatted_prompt_block

    # desenibe must be classified as likely student error (not an OCR glitch)
    assert "desenibe" in report.likely_student_errors
    assert not any(g["word"] == "desenibe" for g in report.likely_glitches)

    # Must trigger Stage 2 because of curriter glitch
    assert report.should_trigger_stage2 is True


def test_pre_analysis_strikethrough_gap_and_split_tokens():
    """Verify pre-analysis detects strikethrough discrepancy and pen-lift split tokens."""
    from src.pipeline.stage2_verifier import run_stage2_pre_analysis
    from src.pipeline.stage0_strikethrough_detector import StrikethroughRegion

    transcript = "The USA elec tricity was high. Pen haps they will succeed."
    regions = [
        StrikethroughRegion(x=10, y=50, w=100, h=5, y_pct=10.0, y2_pct=11.0, x_pct=5.0, x2_pct=25.0, is_multi_word=False),
        StrikethroughRegion(x=10, y=100, w=120, h=5, y_pct=20.0, y2_pct=21.0, x_pct=5.0, x2_pct=30.0, is_multi_word=False),
    ]
    report = run_stage2_pre_analysis(
        stage1_transcript=transcript,
        strikethrough_regions=regions
    )
    # Stage 1 has 0 [struck:] tags, but 2 optical strokes -> strike gap of 2
    assert report.strike_gap_count == 2
    assert "Strikethrough gap: Stage 0.5 detected 2 cross-out strokes" in report.formatted_prompt_block
    assert report.should_trigger_stage2 is True


def test_delta_only_patch_application_curriter_to_writer():
    """Verify that Stage 2 successfully applies 'curriter' -> 'writer' patch on immutable base."""
    s1_transcript = (
        "The curriter cannot write beautiful words about Gaza because poverly famine siege "
        "covered the whole area. Moreover, deprivation in every house fear and sickness."
    )
    mock_json = """```json
    {
      "proposed_patches": [
        {
          "patch_type": "ligature_fix",
          "stage1_target": "curriter",
          "replacement": "writer",
          "confidence": "high",
          "reason": "Verified against image: cursive 'w' with two bowls segmented as 'cu' + 'rr'.",
          "context_anchor": "The curriter cannot write beautiful words"
        },
        {
          "patch_type": "no_change",
          "stage1_target": "poverly",
          "replacement": "poverly",
          "confidence": "high",
          "reason": "Genuine student spelling error for poverty.",
          "context_anchor": "because poverly famine siege"
        }
      ],
      "verification_notes": "Restored writer and preserved poverly."
    }
    ```"""
    mock_engine = MagicMock()
    mock_engine.generate_multimodal.return_value = mock_json

    verifier = Stage2Verifier(mock_engine, legacy_filters=True)
    result = verifier.run(
        image=Image.new("RGB", (100, 100)),
        stage1_transcript=s1_transcript
    )

    # 1. 'curriter' must be surgically replaced with 'writer'
    assert "The writer cannot write" in result.verified_transcript
    assert "curriter" not in result.verified_transcript

    # 2. Student spelling 'poverly' must be strictly preserved
    assert "poverly" in result.verified_transcript

    # 3. Untouched rest of the page remains 100% frozen
    assert "deprivation in every house fear and sickness." in result.verified_transcript

    # 4. Patch records populated
    assert result.total_corrections_count == 1
    assert result.silent_corrections_fixed[0].stage1_output == "curriter"
    assert result.silent_corrections_fixed[0].actual_handwritten == "writer"


def test_reject_hallucinated_target_mpona():
    """Verify that hallucinated targets like '[struck: mpona]' are blocked by target existence check."""
    s1_transcript = "Answer to Question 10\nSome student answer without any mpona text."
    mock_json = """```json
    {
      "proposed_patches": [
        {
          "patch_type": "strikethrough",
          "stage1_target": "mpona",
          "replacement": "[struck: mpona]",
          "confidence": "high",
          "reason": "Hallucinated strike tag",
          "context_anchor": "non-existent context"
        }
      ],
      "verification_notes": "Attempted hallucinated strike tag"
    }
    ```"""
    mock_engine = MagicMock()
    mock_engine.generate_multimodal.return_value = mock_json

    verifier = Stage2Verifier(mock_engine, legacy_filters=True)
    # Even if target is mentioned, if it doesn't match context or doesn't exist, it is rejected
    result = verifier.run(
        image=Image.new("RGB", (100, 100)),
        stage1_transcript="Answer to Question 10\nSome student answer without that word."
    )
    assert "[struck: mpona]" not in result.verified_transcript
    assert result.total_corrections_count == 0


def test_reject_function_word_drift_see_to_the():
    """Verify that autoregressive drift mutating 'see' -> 'the' is blocked by function word shield."""
    s1_transcript = "The writer can't write beautiful words about Gaza because he see the poverty."
    mock_json = """```json
    {
      "proposed_patches": [
        {
          "patch_type": "ligature_fix",
          "stage1_target": "see",
          "replacement": "the",
          "confidence": "high",
          "reason": "Spurious VLM slip mutating see to the",
          "context_anchor": "because he see the poverty"
        }
      ],
      "verification_notes": "Tried to corrupt see into the"
    }
    ```"""
    mock_engine = MagicMock()
    mock_engine.generate_multimodal.return_value = mock_json

    verifier = Stage2Verifier(mock_engine, legacy_filters=True)
    result = verifier.run(
        image=Image.new("RGB", (100, 100)),
        stage1_transcript=s1_transcript
    )
    # Must preserve 'he see the poverty' and reject mutation to 'the the'
    assert "he see the poverty" in result.verified_transcript
    assert "the the" not in result.verified_transcript
    assert result.total_corrections_count == 0


def test_smart_trigger_gate_clean_bypass():
    """Verify smart trigger gate returns should_trigger_stage2 == False for clean transcripts."""
    from src.pipeline.stage2_verifier import run_stage2_pre_analysis
    clean_transcript = "Answer to Question 1\nThe quick brown fox jumps over the lazy dog."
    report = run_stage2_pre_analysis(
        stage1_transcript=clean_transcript,
        strikethrough_regions=[]
    )
    assert report.should_trigger_stage2 is False
    assert len(report.likely_glitches) == 0
    assert report.strike_gap_count == 0


def test_header_strikethrough_shield():
    """Verify that question headers are protected from being marked as [struck: ...]."""
    s1_transcript = "Ans: the Q. No. 3\nHope is the thing with feathers."
    mock_json = """```json
    {
      "proposed_patches": [
        {
          "patch_type": "strikethrough",
          "stage1_target": "Ans: the Q. No. 3",
          "replacement": "[struck: Ans: the Q. No. 3]",
          "confidence": "high",
          "reason": "Spurious ruled line detected near header",
          "context_anchor": "Ans: the Q. No. 3 Hope"
        }
      ],
      "verification_notes": "Attempted to strike out question header"
    }
    ```"""
    mock_engine = MagicMock()
    mock_engine.generate_multimodal.return_value = mock_json

    verifier = Stage2Verifier(mock_engine, legacy_filters=True)
    result = verifier.run(
        image=Image.new("RGB", (100, 100)),
        stage1_transcript=s1_transcript
    )
    assert "[struck: Ans: the Q. No. 3]" not in result.verified_transcript
    assert "Ans: the Q. No. 3" in result.verified_transcript
    assert result.total_corrections_count == 0


def test_student_error_shield_phonetic_vs_cursive():
    """
    Verify that genuine student phonetic misspellings (desenibe, renny, Pull-time)
    are preserved, while genuine cursive stroke topology (curriter -> writer, remore -> remove)
    is accepted.
    """
    s1_transcript = "Now I am going to desenibe the future plan. I will be renny happy. I want to be a curriter and remore poverty."
    mock_json = """```json
    {
      "proposed_patches": [
        {
          "patch_type": "ligature_fix",
          "stage1_target": "desenibe",
          "replacement": "describe",
          "confidence": "high",
          "reason": "Autocorrecting student phonetic error",
          "context_anchor": "going to desenibe the future"
        },
        {
          "patch_type": "ligature_fix",
          "stage1_target": "renny",
          "replacement": "really",
          "confidence": "high",
          "reason": "Autocorrecting student error",
          "context_anchor": "will be renny happy"
        },
        {
          "patch_type": "ligature_fix",
          "stage1_target": "curriter",
          "replacement": "writer",
          "confidence": "high",
          "reason": "Restoring cursive w-split",
          "context_anchor": "be a curriter and"
        },
        {
          "patch_type": "ligature_fix",
          "stage1_target": "remore",
          "replacement": "remove",
          "confidence": "high",
          "reason": "Restoring cursive v-r ligature",
          "context_anchor": "and remore poverty"
        }
      ],
      "verification_notes": "Tested student error shield"
    }
    ```"""
    mock_engine = MagicMock()
    mock_engine.generate_multimodal.return_value = mock_json

    verifier = Stage2Verifier(mock_engine, legacy_filters=True)
    result = verifier.run(
        image=Image.new("RGB", (100, 100)),
        stage1_transcript=s1_transcript
    )

    # 1. Phonetic student errors must be PRESERVED (autocorrection blocked)
    assert "desenibe" in result.verified_transcript
    assert "describe" not in result.verified_transcript
    assert "renny" in result.verified_transcript
    assert "really" not in result.verified_transcript

    # 2. Cursive topology glitches must be ACCEPTED (restored)
    assert "writer" in result.verified_transcript
    assert "curriter" not in result.verified_transcript
    assert "remove" in result.verified_transcript
    assert "remore" not in result.verified_transcript

    # Exactly 2 patches applied (curriter->writer, remore->remove)
    assert result.total_corrections_count == 2


def test_strikethrough_quota_caps_flooding():
    """Verify that a page is protected from strikethrough flooding (quota <= 3 new strikes)."""
    s1_transcript = "word1 word2 word3 word4 word5 word6 word7"
    mock_json = """```json
    {
      "proposed_patches": [
        {"patch_type": "strikethrough", "stage1_target": "word1", "replacement": "[struck: word1]", "confidence": "high", "reason": "strike", "context_anchor": "word1 word2"},
        {"patch_type": "strikethrough", "stage1_target": "word2", "replacement": "[struck: word2]", "confidence": "high", "reason": "strike", "context_anchor": "word2 word3"},
        {"patch_type": "strikethrough", "stage1_target": "word3", "replacement": "[struck: word3]", "confidence": "high", "reason": "strike", "context_anchor": "word3 word4"},
        {"patch_type": "strikethrough", "stage1_target": "word4", "replacement": "[struck: word4]", "confidence": "high", "reason": "strike", "context_anchor": "word4 word5"},
        {"patch_type": "strikethrough", "stage1_target": "word5", "replacement": "[struck: word5]", "confidence": "high", "reason": "strike", "context_anchor": "word5 word6"}
      ],
      "verification_notes": "Attempted to flood strikethroughs"
    }
    ```"""
    mock_engine = MagicMock()
    mock_engine.generate_multimodal.return_value = mock_json

    verifier = Stage2Verifier(mock_engine, legacy_filters=True)
    result = verifier.run(
        image=Image.new("RGB", (100, 100)),
        stage1_transcript=s1_transcript
    )
    # Only at most 3 strikethroughs applied
    import re
    struck_count = len(re.findall(r'\[struck:', result.verified_transcript))
    assert struck_count <= 3
    assert result.total_corrections_count <= 3


def test_orchestrator_normalization_guards():
    """Verify that _normalize_token_in_context blocks 'see' -> 'the' and consecutive duplicates."""
    from src.pipeline.orchestrator import _normalize_token_in_context
    from src.utils.linguistic_sanitizer import get_english_lexicon

    lex = get_english_lexicon()
    text = "he see the poverty, siege and famine"

    # 1. 'see' -> 'the' must be blocked (creates 'he the the poverty' and violates function word shield)
    res1 = _normalize_token_in_context(text, "see", "the", "he see", text, lex)
    assert res1 == text
    assert "the the" not in res1

    # 2. Block creating duplicate words
    text2 = "they were in danger"
    res2 = _normalize_token_in_context(text2, "were", "in", "they were", text2, lex)
    assert res2 == text2
    assert "in in" not in res2

    # 3. Valid allograph normalization (curriter -> writer) is permitted
    text3 = "the curriter cannot write"
    res3 = _normalize_token_in_context(text3, "curriter", "writer", "the curriter", text3, lex)
    assert "writer" in res3
    assert "curriter" not in res3




def test_default_stage2_applies_structurally_valid_patches_without_word_filters():
    """Default path: no word lists or quotas; a patch whose target exists is applied, tags stay balanced."""
    import json as _json
    from unittest.mock import MagicMock
    engine = MagicMock()
    engine.generate_multimodal.return_value = _json.dumps({"proposed_patches": [
        {"stage1_target": "see the", "replacement": "see to", "context_anchor": "we see the river"},
        {"stage1_target": "draft one\ndraft two", "replacement": "[struck: draft one\ndraft two]", "context_anchor": ""},
    ], "verification_notes": ""})
    out = Stage2Verifier(engine).run(image=None, stage1_transcript="we see the river\ndraft one\ndraft two\nend",
                                     pre_analysis=None).verified_transcript
    assert out == "we see to river\n[struck: draft one]\n[struck: draft two]\nend"
