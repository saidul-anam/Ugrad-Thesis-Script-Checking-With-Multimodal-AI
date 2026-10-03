import pytest
import os
import tempfile
from typing import List

from src.pipeline.stage1_transcriber import sanitize_and_normalize_stage1_output
from src.pipeline.split_token_stitcher import stitch_pen_lift_splits
from src.pipeline.stage0b_teacher_marks import reconcile_document_teacher_marks
from src.pipeline.stage2_verifier import is_spurious_reversion, is_header_strikethrough, extract_ghost_corrections_from_notes
from src.core.schemas import TeacherMarkItem, RawTierRecord
from src.utils.export_utils import export_raw_tier_csv


# ---------------------------------------------------------------------------
# Test Fix 1: Stage 1 Normalization & Degeneracy Suppression (P1, P2, P13, P14)
# ---------------------------------------------------------------------------

def test_stage1_html_stripping():
    """Verify HTML presentation tags like <u>A</u> are stripped cleanly."""
    raw = "Ans to the Question No 1(A):\n<u>A</u> (i) being displaced\n<u>(d)</u> another option"
    cleaned = sanitize_and_normalize_stage1_output(raw)
    assert "<u>" not in cleaned
    assert "</u>" not in cleaned
    assert "A (i) being displaced" in cleaned
    assert "(d) another option" in cleaned


def test_stage1_latex_arrows():
    """Verify LaTeX math symbols like $\\rightarrow$ are converted to plain text arrows."""
    raw = "Sequence: a $\\rightarrow$ b \\rightarrow c $\\to$ d"
    cleaned = sanitize_and_normalize_stage1_output(raw)
    assert "$\\rightarrow$" not in cleaned
    assert "\\rightarrow" not in cleaned
    assert "$\\to$" not in cleaned
    assert "a -> b -> c -> d" in cleaned


def test_stage1_illegible_flood_cap():
    """Verify consecutive [illegible] tokens are collapsed and capped at 15 per page."""
    # 50 consecutive [illegible] markers from bleed-through noise
    flood = "start " + " [illegible]" * 50 + " end"
    cleaned = sanitize_and_normalize_stage1_output(flood)
    count = cleaned.count("[illegible]")
    assert count <= 15
    assert "start" in cleaned
    assert "end" in cleaned


def test_stage1_repetition_loop_squash():
    """Verify repetitive lines from VLM looping on ruled lines are squashed."""
    raw = (
        "[struck: I want to be a nurse] [struck: I] want to be a nurse\n"
        "[struck: I want to be a nurse] [struck: I] want to be a nurse\n"
        "[struck: I want to be a nurse] [struck: I] want to be a nurse\n"
        "[struck: I want to be a nurse] [struck: I] want to be a nurse\n"
        "[struck: I want to be a nurse] [struck: I] want to be a nurse\n"
        "And next line follows."
    )
    cleaned = sanitize_and_normalize_stage1_output(raw)
    lines = [l for l in cleaned.splitlines() if "[struck: I want to be a nurse]" in l]
    assert len(lines) <= 2
    assert "And next line follows." in cleaned


# ---------------------------------------------------------------------------
# Test Fix 2: Morphological Token Stitching (P10)
# ---------------------------------------------------------------------------

def test_split_token_stitcher_phrasal_verbs():
    """Verify valid two-word phrases (come back, far away) are NOT falsely stitched."""
    lexicon = {"come", "back", "comeback", "far", "away", "faraway", "day", "dreaming", "daydreaming", "cannot", "electricity"}
    
    # "come back" should remain two words
    text1 = "They will come back tomorrow."
    stitched1, diffs1 = stitch_pen_lift_splits(text1, lexicon)
    assert "come back" in stitched1
    assert "comeback" not in stitched1

    # "far away" should remain two words
    text2 = "He is far away from home."
    stitched2, diffs2 = stitch_pen_lift_splits(text2, lexicon)
    assert "far away" in stitched2
    assert "faraway" not in stitched2

    # "can not" should be stitched to "cannot"
    text3 = "Students can not leave early."
    stitched3, diffs3 = stitch_pen_lift_splits(text3, lexicon)
    assert "cannot" in stitched3

    # Genuine syllable split "elec tricity" should be stitched
    text4 = "Using elec tricity wisely."
    stitched4, diffs4 = stitch_pen_lift_splits(text4, lexicon)
    assert "electricity" in stitched4


# ---------------------------------------------------------------------------
# Test Fix 3: Document-Level Teacher Mark Reconciler (P7)
# ---------------------------------------------------------------------------

def test_reconcile_document_teacher_marks_dedup():
    """Verify duplicate marks for the same question are merged."""
    marks = [
        TeacherMarkItem(question_no="11", mark_value="6", location="left margin", y_position="top"),
        TeacherMarkItem(question_no="11", mark_value="6", location="left margin", y_position="top"),
        TeacherMarkItem(question_no="8", mark_value="5", location="full page (mid)", y_position="mid"),
        TeacherMarkItem(question_no="8", mark_value="6", location="left margin (zone zoom)", y_position="mid"),
    ]
    reconciled = reconcile_document_teacher_marks(marks)
    q_map = {m.question_no: m for m in reconciled}
    
    # Q11 deduplicated to single entry
    assert "11" in q_map
    assert q_map["11"].mark_value == "6"
    assert sum(1 for m in reconciled if m.question_no == "11") == 1

    # Q8 conflict resolved in favor of targeted zoom reading (6)
    assert "8" in q_map
    assert q_map["8"].mark_value == "6"
    assert "reconciled conflict" in q_map["8"].location


# ---------------------------------------------------------------------------
# Test Fix 4: Stage 2 Structural Header Immunity (P5 & P9)
# ---------------------------------------------------------------------------

def test_stage2_exam_header_immunity():
    """Verify structural exam headers (Ans:, Ans to the, Q.) cannot be mutated by Stage 2."""
    assert is_spurious_reversion("Ans:", "Ann:") is True
    assert is_spurious_reversion("Ans", "Ann") is True
    assert is_spurious_reversion("Ans to the", "Ann to the") is True
    assert is_spurious_reversion("Q.1", "O.1") is True
    assert is_header_strikethrough("Ans to Question 1", "[struck: Ans to Question 1]") is True


def test_stage2_ghost_notes_protection():
    """Verify free text in notes mentioning ordinary words (salam, privacy) is not marked struck."""
    notes = "Student opened with greetings: wrote 'salam' and discusses privacy. No strikethroughs."
    transcript = "salam to you. privacy is important."
    diffs = extract_ghost_corrections_from_notes(notes, transcript)
    targets = {d.stage1_output for d in diffs}
    assert "salam" not in targets
    assert "privacy" not in targets


# ---------------------------------------------------------------------------
# Test Fix 5: Raw Tier CSV Export Upsert (P16)
# ---------------------------------------------------------------------------

def test_export_raw_tier_csv_upsert():
    """Verify CSV export does not accumulate duplicate rows on re-runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "dataset.csv")
        rec1 = RawTierRecord(script_id="SE_001", page_no=1, question_no="1", transcript_text="First version")
        rec2 = RawTierRecord(script_id="SE_001", page_no=2, question_no="2", transcript_text="Page two")
        
        # Initial run
        export_raw_tier_csv([rec1, rec2], csv_file)
        
        # Re-run updating page 1
        rec1_updated = RawTierRecord(script_id="SE_001", page_no=1, question_no="1", transcript_text="Updated version")
        export_raw_tier_csv([rec1_updated], csv_file)
        
        # Read back
        import csv
        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        
        # Must have exactly 2 rows, not 3 (upserted, not blindly appended)
        assert len(rows) == 2
        p1_row = next(r for r in rows if r["page_no"] == "1")
        assert p1_row["transcript_text"] == "Updated version"


# ---------------------------------------------------------------------------
# Test Fix 6: Stage 0.5 Underline vs Strikethrough Baseline Discriminator (Phase 2)
# ---------------------------------------------------------------------------

def test_stage0_strikethrough_discriminates_underlines():
    """Verify underlines along the baseline with descender tails are classified as underlines, not strikethroughs."""
    import numpy as np
    import cv2
    from PIL import Image
    from src.pipeline.stage0_strikethrough_detector import StrikethroughDetector

    detector = StrikethroughDetector(min_line_width=25, max_line_height=8)
    arr = np.full((300, 500), 255, dtype=np.uint8)

    # Heading text
    cv2.putText(arr, "Ans to Question No-01", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    # Underline stroke
    cv2.line(arr, (35, 108), (420, 108), 0, 2)
    # Descender dots below the underline
    cv2.circle(arr, (180, 114), 2, 0, -1)
    cv2.circle(arr, (240, 114), 2, 0, -1)

    res = detector.detect(Image.fromarray(arr))
    assert res.has_strikethrough is False
    assert res.region_count == 0
    assert len(res.underlines) >= 1
    assert res.underlines[0].is_underline is True


# ---------------------------------------------------------------------------
# Test Fix 7: Stage 3 Sensitivity Cascade & Proper Noun Shielding (Phase 3)
# ---------------------------------------------------------------------------

def test_stage3_sensitivity_cascade_two_pass():
    """Verify Stage 3 triggers Pass 2 when an essay > 80 words yields 0 errors on Pass 1."""
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.stage3_error_analyzer import Stage3ErrorAnalyzer
    from src.core.schemas import LinguisticErrorItem

    # Create mock engine where pass 1 returns 0 errors, pass 2 returns recovered errors
    class CascadeMockEngine(MockGemmaEngine):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        def generate_text(self, prompt, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                # Pass 1: returns zero errors
                return '{"errors": [], "linguistic_summary": "Initial pass clean"}'
            else:
                # Pass 2: returns recovered agreement error
                return '''{
                    "errors": [
                        {
                            "error_type": "grammar",
                            "erroneous_text": "he do",
                            "suggested_correction": "he does",
                            "context_sentence": "Every day he do his homework.",
                            "explanation": "Subject-verb agreement error"
                        }
                    ],
                    "linguistic_summary": "Recovered subtler grammar error"
                }'''

    engine = CascadeMockEngine()
    analyzer = Stage3ErrorAnalyzer(engine)
    essay = " ".join(["Education is the most powerful weapon which you can use to change the world."] * 8)
    assert len(essay.split()) > 80

    res = analyzer.run(verified_transcript=essay)
    assert engine.call_count == 2
    assert res.total_error_count == 1
    assert res.errors[0].erroneous_text == "he do"
    assert res.errors[0].suggested_correction == "he does"


def test_stage3_proper_noun_and_cultural_shield():
    """Verify Bengali/NCTB cultural terms and mid-sentence capitalized proper nouns are shielded."""
    from src.utils.linguistic_sanitizer import verify_and_filter_stage3_errors
    from src.core.schemas import LinguisticErrorItem

    errors = [
        LinguisticErrorItem(error_type="spelling", erroneous_text="salam", suggested_correction="peace",
                            context_sentence="Convey my salam to all.", explanation="Spelling error"),
        LinguisticErrorItem(error_type="spelling", erroneous_text="Tarun", suggested_correction="Turn",
                            context_sentence="Then the boy Tarun went home.", explanation="Spelling error"),
        LinguisticErrorItem(error_type="spelling", erroneous_text="definitly", suggested_correction="definitely",
                            context_sentence="This is definitly true.", explanation="Spelling error"),
    ]
    filtered = verify_and_filter_stage3_errors(errors)
    err_texts = [e.erroneous_text for e in filtered]
    assert "salam" not in err_texts
    assert "Tarun" not in err_texts
    assert "definitly" in err_texts


# ---------------------------------------------------------------------------
# Test Fix 8: Stage 0 Red Ink Telea Inpainting Default Disabled (Phase 4)
# ---------------------------------------------------------------------------

def test_stage0_red_ink_telea_inpainting_default_disabled():
    """Verify RedInkDetector defaults enable_inpainting to False and uses Telea when enabled."""
    import numpy as np
    import cv2
    from PIL import Image
    from src.pipeline.stage0_red_ink_detector import RedInkDetector

    detector_default = RedInkDetector()
    assert detector_default.enable_inpainting is False

    # When enabled, inpainting cleanly produces image
    detector_inpaint = RedInkDetector(enable_inpainting=True)
    arr = np.full((200, 300, 3), 255, dtype=np.uint8)
    # Draw red mark in body zone
    cv2.line(arr, (100, 50), (150, 100), (0, 0, 255), 3)
    res = detector_inpaint.detect(Image.fromarray(arr))
    assert res.clean_image is not None


# ---------------------------------------------------------------------------
# Test Fix 9: Stage 2 Length Disparity Guard & Multi-line Squasher (Phase 5)
# ---------------------------------------------------------------------------

def test_stage2_length_disparity_fallback():
    """Verify Stage 2 falls back to Stage 1 base if verified transcript drifts >15% in length."""
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.stage2_verifier import Stage2Verifier

    class TruncatingEngine(MockGemmaEngine):
        def generate_multimodal(self, prompt, **kwargs):
            return '''{
                "verified_transcript": "Truncated sentence.",
                "proposed_patches": [],
                "verification_notes": "Truncated heavily"
            }'''

    verifier = Stage2Verifier(TruncatingEngine())
    s1 = "This is a very long and detailed handwritten exam script that has many paragraphs and lots of sentences written by the student."
    from PIL import Image
    res = verifier.run(image=Image.new("RGB", (100, 100)), stage1_transcript=s1)
    # Severe length disparity (>15%) must fallback to Stage 1 transcript
    assert res.verified_transcript == s1


def test_stage1_multiline_cycle_squasher():
    """Verify multi-line repeating cycles (e.g. 2-line or 3-line loop attractors) are squashed."""
    from src.pipeline.stage1_transcriber import sanitize_and_normalize_stage1_output

    text = (
        "Line 1: Education is very important\n"
        "Line 2: It enlightens the mind\n"
        "Line 1: Education is very important\n"
        "Line 2: It enlightens the mind\n"
        "Line 1: Education is very important\n"
        "Line 2: It enlightens the mind\n"
        "Line 3: Conclusion follows"
    )
    squashed = sanitize_and_normalize_stage1_output(text)
    # Multi-line loop squashed to single cycle + conclusion
    assert squashed.count("Education is very important") == 1
    assert "Conclusion follows" in squashed

