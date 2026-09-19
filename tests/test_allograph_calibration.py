"""
Tests for Generalized Allograph Calibration and Pen-Lift Split Token Stitcher.
Verifies self-supervised discovery across SE_11_Q1_0023, SE_11_Q1_0021, and SE_11_Q1_0010.
"""

import pytest
from typing import Set

from src.pipeline.allograph_calibrator import AllographCalibrator
from src.pipeline.split_token_stitcher import stitch_pen_lift_splits
from src.pipeline.arbitration.writer_profile import WriterProfile
from src.utils.linguistic_sanitizer import get_english_lexicon


@pytest.fixture
def english_lexicon() -> Set[str]:
    lex = get_english_lexicon()
    # Ensure domain words are present
    lex.update({
        "source", "sources", "dreamer", "dreamers", "algorithm", "algorithms",
        "feature", "features", "second", "seconds", "work", "works",
        "application", "applications", "business", "businesses", "sector", "sectors",
        "teach", "teaches", "easy", "generator", "generators", "assistant", "assistants",
        "issue", "issues", "demerit", "demerits", "dangerous", "electricity", "problem",
        "problems", "daytime", "sickness", "remove", "village", "his", "with",
        "hydro", "hydro-electric", "hydroelectric"
    })
    return lex


def test_se_11_q1_0023_terminal_y_discovery(english_lexicon):
    """Verify that SE_11_Q1_0023's terminal cursive 's' (read as 'y') is automatically discovered."""
    sample_transcript = """
    In this poem the poet wanted to describe the difference between daytime dream and
    nighttime dream. The nighttime dreams are useless, the nighttime dreamery dreams while
    sleeping but in real life they don't work for this. On the other hand day time dreamery
    are dangerouy, they work hard for their dream and make them come to reality.
    
    Basically AI runne with some wonderful algorithmy and programme.
    AI hay many featurey. It can solve complex mathematical problem.
    AI can give answer within a secondy. It worky very fast.
    AI has many applicationy in real life.
    get his life back. AI is also used in Education, Bank, Businesy and every sectory of life.
    Teachers can teachey student with more information and eaye way.
    It has some threatening issuey too. Which has some demerity.
    
    The pie chart illustrates the sourcey of electricity.
    """
    calibrator = AllographCalibrator(min_support=3)
    profile = calibrator.calibrate(
        script_id="SE_11_Q1_0023",
        transcript=sample_transcript,
        lexicon=english_lexicon
    )

    # Must discover terminal_y -> s
    assert "terminal_y" in profile.discovered_allographs
    assert profile.discovered_allographs["terminal_y"] == "s"

    # Must have adapted all the non-words
    adapted = profile.adapted_words
    assert adapted.get("sourcey") == "sources"
    assert adapted.get("dreamery") == "dreamers"
    assert adapted.get("dangerouy") == "dangerous"
    assert adapted.get("algorithmy") == "algorithms"
    assert adapted.get("featurey") == "features"
    assert adapted.get("secondy") == "seconds"
    assert adapted.get("worky") == "works"
    assert adapted.get("applicationy") == "applications"
    assert adapted.get("businesy") == "business"
    assert adapted.get("sectory") == "sectors"

    # Test transcript normalization
    calibrated_text, diffs = calibrator.apply_adaptations(sample_transcript, profile)
    assert "sources" in calibrated_text
    assert "dreamers" in calibrated_text
    assert "algorithms" in calibrated_text
    assert "features" in calibrated_text
    assert "seconds" in calibrated_text
    assert len(diffs) >= 8


def test_dictionary_immutability(english_lexicon):
    """Verify that legitimate English words ending in 'y' are NEVER corrupted."""
    text_with_legit_y = "The poverty of people living in reality is a hungry journey only for every day."
    calibrator = AllographCalibrator(min_support=2)
    profile = calibrator.calibrate("test_script", text_with_legit_y, english_lexicon)

    calibrated_text, diffs = calibrator.apply_adaptations(text_with_legit_y, profile)
    # None of these legitimate dictionary words should be changed!
    assert "poverty" in calibrated_text
    assert "reality" in calibrated_text
    assert "hungry" in calibrated_text
    assert "only" in calibrated_text
    assert "every" in calibrated_text
    assert len(diffs) == 0


def test_pen_lift_split_stitching(english_lexicon):
    """Verify pen-lift intra-word split stitching."""
    sample_text = """
    The USA elec tricuity in 1980 was large.
    It helps them solve complex pro blems.
    Day time dreamers are inspiring.
    It operates in non- formal educational sectors.
    Hy dro - electric power is renewable.
    """
    stitched_text, diffs = stitch_pen_lift_splits(
        sample_text,
        lexicon=english_lexicon,
        allograph_map={"terminal_y": "s"}
    )

    assert "electricity" in stitched_text
    assert "problems" in stitched_text
    assert "daytime" in stitched_text.lower()
    assert "non-formal" in stitched_text
    assert "hydro-electric" in stitched_text.lower()
    assert len(diffs) >= 4


def test_writer_profile_allograph_matching():
    """Verify WriterProfile.is_writer_allograph correctly identifies candidate pairs."""
    prof = WriterProfile(
        script_id="SE_11_Q1_0023",
        discovered_allographs={"terminal_y": "s", "curvy_s": "s", "cursive_vr": "v"},
        adapted_words={"sourcey": "sources", "dreamery": "dreamers", "algorithmy": "algorithms"}
    )

    assert prof.is_writer_allograph("sourcey", "sources") is True
    assert prof.is_writer_allograph("dreamery", "dreamers") is True
    assert prof.is_writer_allograph("dangerouy", "dangerous") is True
    assert prof.is_writer_allograph("hin", "his") is True
    assert prof.is_writer_allograph("rumore", "remove") is True
    # Unrelated error should return False
    assert prof.is_writer_allograph("deficite", "deficit") is False


def test_arbitration_gate_with_allograph_profile(english_lexicon):
    """Verify that EvidenceArbitrationGate scores profile allographs as HANDWRITING_AMBIGUITY."""
    from src.core.schemas import ArbitrationCandidate
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.arbitration.gate import EvidenceArbitrationGate
    from src.core.config import ArbitrationConfig
    import tempfile
    from PIL import Image

    prof = WriterProfile(
        script_id="SE_11_Q1_0023",
        discovered_allographs={"terminal_y": "s"},
        adapted_words={"sourcey": "sources"}
    )

    cand = ArbitrationCandidate(
        candidate_id="8:10:0",
        error_index=10,
        error_type="spelling",
        erroneous_text="sourcey",
        suggested_correction="sources",
        candidate_token="sourcey",
        intended_token="sources",
        context_sentence="That It indicate that the sourcey of electricity is mainly natural element.",
        page_no=8,
    )
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    with tempfile.TemporaryDirectory() as d:
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(),
            cfg=ArbitrationConfig(),
            script_id="SE_11_Q1_0023",
            output_dir=d,
            page_images=[(8, img, "p8.png")],
            page_transcripts={8: "That It indicate that the sourcey of electricity is mainly natural element."},
            clean_image_fn=None,
            full_transcript="That It indicate that the sourcey of electricity is mainly natural element.",
            lexicon=english_lexicon,
            profile=prof
        )
        rec = gate._score_candidate(cand)
        # Should be classified as HANDWRITING_AMBIGUITY (0 deduction)
        assert rec.verdict == "HANDWRITING_AMBIGUITY"
        assert any("Writer allograph" in note for note in rec.evidence.notes)


def test_dynamic_26_letter_allograph_discovery(english_lexicon):
    """Verify that arbitrary repeated character substitutions (e.g. 'b' <-> 'f') are dynamically discovered."""
    english_lexicon.update({"feature", "features", "famous", "fortune", "farmer", "forest", "hospital", "house", "horse"})
    transcript = """
    The beature was presented in the bamous city with great bortune.
    This beature has many advantages in the bamous world.
    A single random typo like hoss does not repeat.
    """
    calibrator = AllographCalibrator(min_support=2)
    profile = calibrator.calibrate(
        script_id="test_dynamic",
        transcript=transcript,
        lexicon=english_lexicon
    )
    # Should discover allograph_b_f
    assert "allograph_b_f" in profile.discovered_allographs
    assert profile.discovered_allographs["allograph_b_f"] == "f"
    assert profile.adapted_words.get("beature") == "feature"
    assert profile.adapted_words.get("bamous") == "famous"
    assert profile.adapted_words.get("bortune") == "fortune"
    # Single isolated error should NOT be promoted as a rule
    assert "allograph_s_r" not in profile.discovered_allographs


def test_sentence_chunking_integrity():
    """Verify sentence-bounded long answer chunking does not drop sentences or words."""
    from src.pipeline.orchestrator import _chunk_text_by_sentences

    sentences = [
        "Artificial intelligence is rapidly transforming modern education across the world.",
        "Students can now access personalized learning tutors whenever they require assistance with complex topics.",
        "Teachers are able to automate grading and spend significantly more time mentoring students individually.",
        "However there are ethical considerations including student privacy and algorithm bias that require strict guidelines.",
        "In developing countries infrastructure and electrical access remain formidable bottlenecks to broad adoption.",
        "Future policy must balance rapid technological integration with equal educational accessibility."
    ]
    full_text = " ".join(sentences)

    chunks = _chunk_text_by_sentences(full_text, target_words=40)
    assert len(chunks) >= 2
    # Reconstructed words should match the original words completely
    reconstructed = " ".join(chunks)
    assert reconstructed.split() == full_text.split()

