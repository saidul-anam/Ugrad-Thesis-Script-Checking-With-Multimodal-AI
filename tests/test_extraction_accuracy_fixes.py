"""
Tests for extraction accuracy fixes:
- LaTeX arrow sanitization and repair
- Declared-only Stage 2 updates
- Consensus override in arbitration (powerdul, reallly)
- Header regex extraction (01 (A), 01 (B), que no: 3, due no: 7)
- Token guard and protected tag shield
"""

import re
import pytest
from src.pipeline.stage2_verifier import sanitize_latex_json
from scripts.repair_stage2_checkpoints import repair_text
from src.pipeline.arbitration.consensus import consensus_signal
from src.pipeline.arbitration.fusion import fuse, quantize, signals_from_evidence
from src.core.schemas import ArbitrationEvidence, ArbitrationCandidate, ArbitrationRecord
from src.core.config import ArbitrationWeights, ArbitrationConfig
from src.pipeline.answer_segmenter import extract_header_qno, normalize_header_text
from src.pipeline.token_guard import sanitize_rearrangement_sequence, normalize_roman_numeral, is_token_in_protected_tag


def test_sanitize_latex_json():
    raw_json = '{"verified_transcript": "1. Step $\\rightarrow$ 2. Step $\\times$ 3"}'
    sanitized = sanitize_latex_json(raw_json)
    assert "\\\\rightarrow" in sanitized
    assert "\\\\times" in sanitized


def test_repair_text_arrows():
    # Simulate carriage return / corruption in Stage 2 text
    corrupted = "1. Step $\r\tightarrow$ 2. Step $\right\rarrow$ 3. Step $\rightarrow$"
    raw = "1. Step $\\rightarrow$ 2. Step $\\rightarrow$ 3. Step $\\rightarrow$"
    repaired, arrows, tags = repair_text(corrupted, raw)
    assert arrows >= 2
    assert r"$\rightarrow$" in repaired
    assert "\r" not in repaired
    assert "$ightarrow" not in repaired
    assert "\tightarrow" not in repaired


def test_repair_text_restore_struck_tags():
    raw = "The student wrote [struck: wrong_word] correct_word here."
    verified_omitted = "The student wrote wrong_word correct_word here."
    repaired, arrows, tags = repair_text(verified_omitted, raw)
    assert tags == 1
    assert "[struck: wrong_word]" in repaired


def test_consensus_signal_high_agreement():
    # When agreement on intended is 100% and word_conf is 1.0 (e.g. powerful vs powerdul)
    sig = consensus_signal(agr_candidate=0.0, agr_intended=1.0, word_conf=1.0)
    assert sig >= 0.95, f"Expected consensus signal >= 0.95 for unanimous consensus, got {sig}"


def test_consensus_override_arbitration():
    # Simulate the exact case of 'powerdul' where consensus is 100% on 'powerful'
    # but forced choice hallucinates 'candidate'
    ev = ArbitrationEvidence(
        phonetic_plausibility=0.8,
        phonetic_signal=0.2,
        consensus_signal=1.0,
        consensus_agreement_candidate=0.0,
        consensus_agreement_intended=1.0,
        forced_choice="candidate",
        forced_choice_confidence=1.0,
        forced_choice_signal=0.5, # neutralized by consensus override
    )
    w = ArbitrationWeights(bias=-0.3, phonetic=0.6, writer=1.2, consensus=2.6, forced_choice=2.0)
    score = fuse(ev, w)
    # Check that score reaches ambiguity threshold (>= 0.65)
    assert score >= 0.65, f"Expected ambiguity score >= 0.65, got {score}"
    verdict = quantize(score, threshold_ambiguity=0.65, threshold_genuine=0.35)
    assert verdict == "HANDWRITING_AMBIGUITY"


def test_header_extraction_subparts():
    # Script 0010 case: Ans to the Question No - 01 (A) vs 01 (B)
    h_1a = extract_header_qno("Ans to the Question No - 01 (A)\nSome content")
    assert h_1a == "1(A)"

    h_1b = extract_header_qno("Ans to the Question No - 01 (B)\nSome content")
    assert h_1b == "1(B)"

    # Script 0015 cases: que no: 3, due no: 7
    h_3 = extract_header_qno("Ans to the que no:3\nContent")
    assert h_3 == "3"

    h_7 = extract_header_qno("Ans to the due no: 7\nContent")
    assert h_7 == "7"

    h_5 = extract_header_qno("Ans to the que: no: 5\nContent")
    assert h_5 == "5"


def test_token_guard_rearrangement():
    raw = "c -> j -> a -> e -> h -> d -> g -> i -> f -> b"
    seq, anomalies = sanitize_rearrangement_sequence(raw)
    assert len(seq) == 10
    assert "j" in seq
    assert len(anomalies) == 0

    # Test substitution repair: 'o' instead of 'j'
    raw_corrupt = "c -> o -> a -> e -> h -> d -> g -> i -> f -> b"
    seq_repaired, anomalies = sanitize_rearrangement_sequence(raw_corrupt)
    assert "j" in seq_repaired
    assert "o" not in seq_repaired
    assert len(anomalies) == 1
    assert "repaired_substitution:o->j" in anomalies[0]


def test_token_guard_protected_tag():
    full_text = "The [struck: that] coal is the most powerful sector"
    assert is_token_in_protected_tag("that", full_text) is True
    assert is_token_in_protected_tag("coal", full_text) is False

    unclear_text = "We saw [unclear: reality | reallly] of Gaza"
    assert is_token_in_protected_tag("reality", unclear_text) is True
    assert is_token_in_protected_tag("Gaza", unclear_text) is False


def test_candidate_selector_table_and_cultural_shielding():
    from src.pipeline.arbitration.candidate_selector import lexicon_candidates
    text = (
        "| From | To | Stamp |\n"
        "| Azimpur, BDR | Manitgong, | |\n"
        "| enogate | shimiliya-7 | |\n"
        "Convey my salam to your parents."
    )
    lexicon = {"from", "to", "stamp", "convey", "my", "your", "parents", "elongate", "salaam"}
    cands = lexicon_candidates(text, lexicon)
    cand_tokens = [c.erroneous_text.lower() for c in cands]
    # enogate is inside a markdown table -> must NOT be flagged as a spelling error
    assert "enogate" not in cand_tokens
    # salam is an NCTB cultural term -> must NOT be flagged as a spelling error
    assert "salam" not in cand_tokens


def test_arbitration_gate_neither_yields_uncertain():
    from src.core.schemas import ArbitrationCandidate
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.arbitration.gate import EvidenceArbitrationGate
    from src.core.config import ArbitrationConfig
    import tempfile
    from PIL import Image

    cand = ArbitrationCandidate(
        candidate_id="10:13:0",
        error_index=13,
        error_type="spelling",
        erroneous_text="ned",
        suggested_correction="need",
        candidate_token="ned",
        intended_token="need",
        context_sentence="So I ned to work hard on my dreem",
        page_no=1,
    )
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    with tempfile.TemporaryDirectory() as d:
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(),
            cfg=ArbitrationConfig(),
            script_id="test_script",
            output_dir=d,
            page_images=[(1, img, "p1.png")],
            page_transcripts={1: "So I ned to work hard on my dreem"},
            clean_image_fn=None,
            full_transcript="So I ned to work hard on my dreem",
            lexicon={"need", "to", "work", "hard"},
        )
        rec = gate._score_candidate(cand)
        # Without crop or with forced_choice neither, verdict must be UNCERTAIN (benefit of doubt)
        assert rec.verdict == "UNCERTAIN"


def test_answer_segmenter_question_anchored_error_attribution():
    from src.pipeline.answer_segmenter import segment_script_into_questions
    from src.core.schemas import ExtractionResult, PageExtractionResult, Stage1TranscriptionResult, Stage2VerificationResult, Stage3ErrorResult, LinguisticErrorItem

    dummy_errors = Stage3ErrorResult(
        errors=[], total_error_count=0, spelling_error_count=0,
        grammar_error_count=0, syntax_error_count=0, punctuation_error_count=0,
        linguistic_summary=""
    )
    p1 = PageExtractionResult(
        page_no=1,
        stage1_transcription=Stage1TranscriptionResult(raw_transcript="Ans to the que no:3\nHope is a bird that sings."),
        stage2_verification=Stage2VerificationResult(verified_transcript="Ans to the que no:3\nHope is a bird that sings.\n\nAns to the que no:4\na) attain\nb) why"),
        stage3_errors=dummy_errors,
    )
    p2 = PageExtractionResult(
        page_no=2,
        stage1_transcription=Stage1TranscriptionResult(raw_transcript="Ans to the que: no: 10\nDate: 27.6.2026\nMy dear Arif\nAfter passin HSC exam I prepared me to get admitted."),
        stage2_verification=Stage2VerificationResult(verified_transcript="Ans to the que: no: 10\nDate: 27.6.2026\nMy dear Arif\nAfter passin HSC exam I prepared me to get admitted."),
        stage3_errors=dummy_errors,
    )

    errors = [
        LinguisticErrorItem(error_type="grammar", erroneous_text="sings", suggested_correction="sing", context_sentence="Hope is a bird that sings.", question_no="3", explanation="Subject-verb agreement"),
        LinguisticErrorItem(error_type="spelling", erroneous_text="passin", suggested_correction="passing", context_sentence="After passin HSC exam", question_no="10", explanation="Spelling"),
    ]

    extraction = ExtractionResult(
        script_id="test_script",
        image_path="test.pdf",
        timestamp="2026-09-17T00:00:00",
        pages=[p1, p2],
        stage1_transcription=Stage1TranscriptionResult(raw_transcript=""),
        stage2_verification=Stage2VerificationResult(verified_transcript=""),
        stage3_errors=Stage3ErrorResult(errors=errors, total_error_count=2, spelling_error_count=1, grammar_error_count=1, syntax_error_count=0, punctuation_error_count=0, linguistic_summary=""),
    )

    answers = segment_script_into_questions(extraction)
    ans_dict = {a.q_no: a for a in answers}

    assert "3" in ans_dict
    assert "4" in ans_dict
    assert "10" in ans_dict

    # Q3 must have its 1 error
    assert len(ans_dict["3"].errors) == 1
    assert ans_dict["3"].errors[0]["erroneous_text"] == "sings"

    # Q4 was on the same page as Q3, but must have ZERO leaked errors!
    assert len(ans_dict["4"].errors) == 0

    # Q10 must have its own 1 error
    assert len(ans_dict["10"].errors) == 1
    assert ans_dict["10"].errors[0]["erroneous_text"] == "passin"


def test_consensus_context_anchored_target_alignment():
    from src.pipeline.arbitration.consensus import agreement, tokenize
    samples = [
        "the people of my. village. I will be verry happy if you",
        "the people of my. village. I will be verry happy if you",
        "the people of my. village. I will be verry happy if you",
        "the people of my. village. I will be verry happy if you",
        "the people of my village. I will be verry happy if you",
    ]
    ref_line = "the people of my village. I will be verry happy if you"
    ctx = "of my village. I will be renny happy if you join with me."

    # Even though candidate='renny' and intended='runny', context anchoring locates 'verry'
    agr_c, agr_i, word_conf, aligned = agreement(samples, ref_line, "renny", "runny", context_sentence=ctx)
    assert len(aligned) == 5
    assert all(t == "verry" for t in aligned)


def test_writer_anchor_benefit_of_the_doubt_rillage():
    from src.core.schemas import ArbitrationCandidate
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.arbitration.gate import EvidenceArbitrationGate
    from src.core.config import ArbitrationConfig
    import tempfile
    from PIL import Image

    cand = ArbitrationCandidate(
        candidate_id="10:7:0",
        error_index=7,
        error_type="spelling",
        erroneous_text="rillage",
        suggested_correction="village",
        candidate_token="rillage",
        intended_token="village",
        context_sentence="Many farmers of my rillage are not educated.",
        page_no=1,
    )
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    with tempfile.TemporaryDirectory() as d:
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(),
            cfg=ArbitrationConfig(),
            script_id="SE_11_Q1_0011",
            output_dir=d,
            page_images=[(1, img, "p1.png")],
            page_transcripts={1: "Many farmers of my rillage are not educated."},
            clean_image_fn=None,
            full_transcript="Many farmers of my rillage are not educated.",
            lexicon={"village", "farmers", "educated"},
        )
        # Establish 'village' as a writer anchor word (used correctly elsewhere)
        gate.profile.anchors.append("village")
        rec = gate._score_candidate(cand)
        # Benefit of Doubt ensures verdict is UNCERTAIN (never GENUINE_ERROR)
        assert rec.verdict == "UNCERTAIN"
        assert any("Benefit of Doubt applied" in note for note in rec.evidence.notes)


def test_lexicon_candidates_initial_confusion_expansion():
    from src.pipeline.arbitration.candidate_selector import lexicon_candidates
    text = "Many farmers of my rillage are not educated."
    lexicon = {"many", "farmers", "of", "my", "village", "are", "not", "educated"}
    cands = lexicon_candidates(text, lexicon)
    assert len(cands) >= 1
    # Check that 'village' is suggested as the correction for 'rillage' via initial letter expansion
    cand = cands[0]
    assert cand.erroneous_text == "rillage"
    assert cand.suggested_correction == "village"


def test_parallel_consensus_exact_output_match():
    from src.pipeline.arbitration.consensus import ConsensusTranscriber
    from src.engine.mock_engine import MockGemmaEngine
    from PIL import Image

    engine = MockGemmaEngine()
    img = Image.new("RGB", (200, 60), (255, 255, 255))
    ref = "the people of my village. I will be very happy if you"

    t_serial = ConsensusTranscriber(engine, n_samples=5, parallel_workers=1)
    res_serial = t_serial.run(img, ref, "renny", "very")

    t_parallel = ConsensusTranscriber(engine, n_samples=5, parallel_workers=4)
    res_parallel = t_parallel.run(img, ref, "renny", "very")

    # Outputs must be 100% bit-for-bit identical
    assert res_serial["samples"] == res_parallel["samples"]
    assert res_serial["agreement_candidate"] == res_parallel["agreement_candidate"]
    assert res_serial["agreement_intended"] == res_parallel["agreement_intended"]
    assert res_serial["consensus_token"] == res_parallel["consensus_token"]
    assert res_serial["consensus_token_agreement"] == res_parallel["consensus_token_agreement"]


def test_orchestrator_parallel_workers_parameter():
    from src.core.config import PipelineConfig
    from src.pipeline.orchestrator import ScriptCheckingPipeline
    from src.engine.mock_engine import MockGemmaEngine

    cfg = PipelineConfig()
    cfg.pipeline.parallel_workers = 4
    cfg.arbitration.consensus_parallel_workers = 4

    engine = MockGemmaEngine()
    pipeline = ScriptCheckingPipeline(engine=engine, config=cfg)
    assert pipeline.config.pipeline.parallel_workers == 4
    assert pipeline.config.arbitration.consensus_parallel_workers == 4



