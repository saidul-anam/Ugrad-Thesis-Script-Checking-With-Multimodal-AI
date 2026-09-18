"""
Tests for extraction accuracy fixes:
- LaTeX arrow sanitization and repair
- Declared-only Stage 2 updates
- Consensus override in arbitration (powerdul, reallly)
- Header regex extraction (01 (A), 01 (B), que no: 3, due no: 7)
- Token guard and protected tag shield
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
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
        # Benefit of Doubt ensures verdict is never GENUINE_ERROR
        assert rec.verdict in ("UNCERTAIN", "HANDWRITING_AMBIGUITY")
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


def test_cursive_w_to_cu_candidate_selection():
    from src.pipeline.arbitration.candidate_selector import differing_token_pairs, lexicon_candidates

    # 1. Test differing token pairs accepts 2-char ligature split
    pairs1 = differing_token_pairs("cuith", "with")
    assert pairs1 == [("cuith", "with")]

    pairs2 = differing_token_pairs("pocuer", "power")
    assert pairs2 == [("pocuer", "power")]

    # 2. Test lexicon candidate generation locates 'with' from 'cuith' via expanded initial confusion
    lexicon = {"he", "came", "with", "us", "together"}
    cands = lexicon_candidates("he came cuith us", lexicon)
    assert len(cands) == 1
    assert cands[0].erroneous_text == "cuith"
    assert cands[0].suggested_correction == "with"


def test_palmer_r_to_re_arbitration():
    from src.core.schemas import ArbitrationCandidate
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.arbitration.gate import EvidenceArbitrationGate
    from src.core.config import ArbitrationConfig
    import tempfile
    from PIL import Image

    cand = ArbitrationCandidate(
        candidate_id="3:1:0",
        error_index=1,
        error_type="spelling",
        erroneous_text="troee",
        suggested_correction="tree",
        candidate_token="troee",
        intended_token="tree",
        context_sentence="playing on a troee quite happily",
        page_no=1,
    )
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    with tempfile.TemporaryDirectory() as d:
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(),
            cfg=ArbitrationConfig(),
            script_id="SE_11_Q1_0007",
            output_dir=d,
            page_images=[(1, img, "p1.png")],
            page_transcripts={1: "playing on a troee quite happily"},
            clean_image_fn=None,
            full_transcript="playing on a troee quite happily",
            lexicon={"playing", "on", "a", "tree", "quite", "happily"},
        )
        rec = gate._score_candidate(cand)
        # Never penalize as genuine error
        assert rec.verdict != "GENUINE_ERROR"
        assert any("Palmer cursive 'r'<->'re' shelf" in note for note in rec.evidence.notes)


def test_looped_s_terminal_arbitration():
    from src.core.schemas import ArbitrationCandidate
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.arbitration.gate import EvidenceArbitrationGate
    from src.core.config import ArbitrationConfig
    import tempfile
    from PIL import Image

    cand = ArbitrationCandidate(
        candidate_id="10:2:0",
        error_index=2,
        error_type="spelling",
        erroneous_text="examis",
        suggested_correction="exams",
        candidate_token="examis",
        intended_token="exams",
        context_sentence="after passing HSC examis",
        page_no=1,
    )
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    with tempfile.TemporaryDirectory() as d:
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(),
            cfg=ArbitrationConfig(),
            script_id="SE_11_Q1_0010",
            output_dir=d,
            page_images=[(1, img, "p1.png")],
            page_transcripts={1: "after passing HSC examis"},
            clean_image_fn=None,
            full_transcript="after passing HSC examis",
            lexicon={"after", "passing", "hsc", "exams"},
        )
        rec = gate._score_candidate(cand)
        assert rec.verdict != "GENUINE_ERROR"
        assert any("looped base 's' variant" in note for note in rec.evidence.notes)


def test_visual_evidence_supremacy_grammar_not_excused():
    from src.core.schemas import ArbitrationCandidate
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.arbitration.gate import EvidenceArbitrationGate
    from src.core.config import ArbitrationConfig
    import tempfile
    from PIL import Image

    cand = ArbitrationCandidate(
        candidate_id="1(B):1:0",
        error_index=1,
        error_type="grammar",
        erroneous_text="The author go",
        suggested_correction="The author goes",
        candidate_token="go",
        intended_token="goes",
        context_sentence="The author go back to the reality of gaza",
        page_no=1,
    )
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    with tempfile.TemporaryDirectory() as d:
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(),
            cfg=ArbitrationConfig(),
            script_id="SE_11_Q1_0010",
            output_dir=d,
            page_images=[(1, img, "p1.png")],
            page_transcripts={1: "The author go back to the reality of gaza"},
            clean_image_fn=None,
            full_transcript="The author go back to the reality of gaza",
            lexicon={"the", "author", "go", "goes", "back", "to", "reality"},
        )
        class MockCrop:
            image = img
            transcript = "The author go back"
            method = "bbox"
            match_ratio = 1.0
            path = "crop.png"

        gate.localizer.locate = lambda c: MockCrop()
        gate.consensus.run = lambda *args, **kwargs: {
            "samples": ["The author go back"] * 5,
            "agreement_candidate": 1.0,
            "agreement_intended": 0.0,
            "word_confidence": 1.0,
            "signal": 0.0,
            "consensus_token": "go",
            "consensus_token_agreement": 1.0,
            "consensus": None,
        }
        gate.judge.judge = lambda *args, **kwargs: {
            "choice": "candidate",
            "confidence": 1.0,
            "reason": "Clear letters g and o",
            "order": "candidate_first",
            "signal": 0.0,
        }
        rec = gate._score_candidate(cand)
        # MUST be GENUINE_ERROR because physical ink is verified to say 'go'
        assert rec.verdict == "GENUINE_ERROR"
        assert not any("looped base 's' variant" in note for note in rec.evidence.notes)


def test_blind_loop_e_c_arbitration():
    from src.core.schemas import ArbitrationCandidate
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.arbitration.gate import EvidenceArbitrationGate
    from src.core.config import ArbitrationConfig
    import tempfile
    from PIL import Image

    cand = ArbitrationCandidate(
        candidate_id="10:3:0",
        error_index=3,
        error_type="spelling",
        erroneous_text="villagc",
        suggested_correction="village",
        candidate_token="villagc",
        intended_token="village",
        context_sentence="the people of my villagc",
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
            page_transcripts={1: "the people of my villagc"},
            clean_image_fn=None,
            full_transcript="the people of my villagc",
            lexicon={"the", "people", "of", "my", "village"},
        )
        rec = gate._score_candidate(cand)
        assert rec.verdict != "GENUINE_ERROR"
        assert any("blind loop 'e'<->'c' ligature" in note for note in rec.evidence.notes)


def test_permissive_headers_grammar():
    from src.pipeline.answer_segmenter import extract_header_qno

    # Test short-hand forms
    assert extract_header_qno("Ans: 2\nContent") == "2"
    assert extract_header_qno("Answer: 2\nContent") == "2"
    assert extract_header_qno("Ans- 2\nContent") == "2"
    assert extract_header_qno("Ans 2\nContent") == "2"
    assert extract_header_qno("Answer 2\nContent") == "2"
    assert extract_header_qno("Ans No: 2\nContent") == "2"
    assert extract_header_qno("Answer No. 2\nContent") == "2"
    assert extract_header_qno("Q-2\nContent") == "2"
    assert extract_header_qno("Ques: 2\nContent") == "2"
    assert extract_header_qno("No. 2\nContent") == "2"
    assert extract_header_qno("1(A)\nContent") == "1(A)"
    assert extract_header_qno("1(B)\nContent") == "1(B)"
    assert extract_header_qno("1(b)\nContent") == "1(B)"

    # Test Bengali short-hand and standard forms
    assert extract_header_qno("২ নং উত্তর\nবিষয়বস্তু") == "2"
    assert extract_header_qno("২নং প্রশ্নের উত্তর\nবিষয়বস্তু") == "2"
    assert extract_header_qno("উত্তর: ২\nবিষয়বস্তু") == "2"
    assert extract_header_qno("উত্তর নং ২\nবিষয়বস্তু") == "2"
    assert extract_header_qno("১(ক) নং প্রশ্নের উত্তর\nবিষয়বস্তু") == "1(A)"


def test_structural_fingerprinting():
    from src.pipeline.answer_segmenter import detect_structural_fingerprint

    # 1. Flowchart
    flowchart_text = """
    (i) Curtailing their economic opportunities ->
    (ii) Increasing their vulnerability ->
    (iii) Creating social inequality
    """
    assert detect_structural_fingerprint(flowchart_text) == "2"

    # 2. Rearranging Table
    rearrange_text = """
    | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
    | c | h | j | a | e | d | g | i | f | b  |
    """
    assert detect_structural_fingerprint(rearrange_text) == "6"

    # 3. Informal Letter / Email with Stamp
    letter_text = """
    Dear Rahim,
    Take my love. I received your letter yesterday.
    Yours ever,
    Karim

    +-------------------+
    | From: Karim       | [STAMP]
    | To: Rahim         |
    +-------------------+
    """
    assert detect_structural_fingerprint(letter_text) == "10"

    # 4. Data Graph / Chart Interpretation
    graph_text = """
    The graph shows the percentage of internet users in Bangladesh from 2010 to 2020.
    In 2010, the rate was only 5% whereas in 2020 it increased to 40%.
    """
    assert detect_structural_fingerprint(graph_text) == "8"

    # 5. Completing Story
    story_text = """
    The Lion and the Mouse
    Once upon a time, a lion was sleeping in a forest. A little mouse began running up and down upon him.
    """
    assert detect_structural_fingerprint(story_text) == "9"

    # 6. Poem Theme
    theme_text = """
    Theme:
    The poem deals with the importance of dreams in human life.
    """
    assert detect_structural_fingerprint(theme_text) == "11"


def test_dynamic_topic_matching_without_prefix():
    from src.pipeline.answer_segmenter import extract_header_qno
    from src.core.schemas import ExtractedQuestion

    q_obj = ExtractedQuestion(
        question_id="SE_11_Q1",
        question_text="English 1st Paper",
        sub_questions=[
            {"q_no": "7", "name": "Artificial Intelligence", "marks": 10},
            {"q_no": "9", "name": "Story Writing: The Lion and the Mouse", "marks": 15},
        ]
    )

    # Title written directly without "title:" or "ans:"
    assert extract_header_qno("Artificial Intelligence\nArtificial intelligence is transforming the world.", question_obj=q_obj) == "7"
    assert extract_header_qno("The Lion and the Mouse\nOnce there lived a lion in a forest.", question_obj=q_obj) == "9"


def test_stage4_unattempted_question_handling():
    from src.pipeline.stage4_evaluator import Stage4Evaluator
    from src.engine.mock_engine import MockGemmaEngine
    from src.core.schemas import AlignedAnswerItem, ExtractedQuestion

    evaluator = Stage4Evaluator(engine=MockGemmaEngine())
    q_obj = ExtractedQuestion(
        question_id="SE_11_Q1",
        question_text="English 1st Paper",
        sub_questions=[
            {"q_no": "1(A)", "name": "MCQ", "marks": 5},
            {"q_no": "4", "name": "Cloze with clues", "marks": 5},
        ]
    )

    # Student only answered 1(A); question 4 was completely unattempted
    answers = [
        AlignedAnswerItem(q_no="1(A)", answer_text="(a) i (b) ii (c) iii", page_numbers=[1], word_count=6)
    ]
    rubric = {
        "subject": "English",
        "questions": {
            "1(A)": {"mode": "A", "max_mark": 5.0, "task_type": "mcq"},
            "4": {"mode": "A", "max_mark": 5.0, "task_type": "cloze_with_clues"},
        }
    }
    # Human examiner also gave 0 for unattempted Q4
    gt_marks = {"1(A)": 5.0, "4": 0.0}

    res = evaluator.evaluate_modular(answers, q_obj, rubric, ground_truth_marks=gt_marks)
    by_q = {qe.q_no: qe for qe in res.question_evaluations}

    # Q4 must be in missing_questions, awarded 0.0, marked missing, and delta vs human 0.0 is 0.0
    assert "4" in res.missing_questions
    assert by_q["4"].awarded_marks == 0.0
    assert by_q["4"].scoring_status == "missing"
    assert "unattempted" in by_q["4"].examiner_feedback.lower()
    assert by_q["4"].delta == 0.0


def test_se_11_q1_0007_flowchart_isolation():
    """Verify that Question 2 with shorthand heading and flowchart arrows is properly isolated from 1(B)."""
    from src.pipeline.answer_segmenter import segment_script_into_questions
    from src.core.schemas import (
        ExtractionResult, PageExtractionResult, Stage1TranscriptionResult,
        Stage2VerificationResult, Stage3ErrorResult, ExtractedQuestion, LinguisticErrorItem
    )

    q_obj = ExtractedQuestion(
        question_id="SE_11_Q1",
        question_text="English 1st Paper",
        sub_questions=[
            {"q_no": "1(A)", "name": "MCQ", "marks": 5},
            {"q_no": "1(B)", "name": "Short Questions", "marks": 10},
            {"q_no": "2", "name": "Flow Chart", "marks": 5},
        ]
    )

    # Page 1: 1(A) and 1(B)
    p1_text = "Ans to the Question No: 01 (A)\n(a) iii (b) ii\n\n(B)\nAns to the Q. No. 1(B)\n(a) Girls are often marginalized."
    p1 = PageExtractionResult(
        page_no=1,
        image_path="/tmp/p1.png",
        stage1_transcription=Stage1TranscriptionResult(raw_transcript=p1_text, word_count=20),
        stage2_verification=Stage2VerificationResult(verified_transcript=p1_text, total_corrections_count=0, silent_corrections_fixed=[]),
        stage3_errors=Stage3ErrorResult(total_error_count=0, errors=[], linguistic_summary="")
    )

    # Page 2: Question 2 with shorthand header "Ans: to the Ques No. 2" and flowchart arrows
    p2_text = """Ans: to the Ques No. 2
    (i) Curtailing their economie non-formal educational opportunities ->
    (ii) Increasing their vulnerability to violence ->
    (iii) Excluding them from decision making"""
    
    # Error on Page 2: "economie" inside flowchart
    p2_err = LinguisticErrorItem(
        error_index=1,
        error_type="spelling",
        erroneous_text="economie",
        suggested_correction="economic",
        explanation="Spelling error",
        context_sentence="Curtailing their economie non-formal educational opportunities",
        page_no=2,
        question_no="1(B)"  # Stale tag from before segmentation
    )

    p2 = PageExtractionResult(
        page_no=2,
        image_path="/tmp/p2.png",
        stage1_transcription=Stage1TranscriptionResult(raw_transcript=p2_text, word_count=25),
        stage2_verification=Stage2VerificationResult(verified_transcript=p2_text, total_corrections_count=0, silent_corrections_fixed=[]),
        stage3_errors=Stage3ErrorResult(total_error_count=1, errors=[p2_err], linguistic_summary="1 spelling error")
    )

    ext = ExtractionResult(
        script_id="SE_11_Q1_0007",
        image_path="/tmp/test.pdf",
        timestamp="2026-09-17T00:00:00",
        pages=[p1, p2],
        stage1_transcription=Stage1TranscriptionResult(raw_transcript=f"{p1_text}\n{p2_text}", word_count=45),
        stage2_verification=Stage2VerificationResult(verified_transcript=f"{p1_text}\n{p2_text}", total_corrections_count=0, silent_corrections_fixed=[]),
        stage3_errors=Stage3ErrorResult(total_error_count=1, errors=[p2_err], linguistic_summary="1 spelling error")
    )

    answers = segment_script_into_questions(ext, q_obj)
    by_q = {a.q_no: a for a in answers}

    # Verify all 3 questions exist distinctly
    assert "1(A)" in by_q
    assert "1(B)" in by_q
    assert "2" in by_q

    # Crucial assertion: 1(B) must NOT contain the flowchart text or its error
    assert "economie" not in by_q["1(B)"].answer_text
    assert len(by_q["1(B)"].errors) == 0

    # Crucial assertion: Question 2 must contain the flowchart text and the error is correctly attributed to 2
    assert "economie" in by_q["2"].answer_text
    assert len(by_q["2"].errors) == 1
    assert by_q["2"].errors[0]["erroneous_text"] == "economie"
    assert by_q["2"].errors[0]["question_no"] == "2"


def test_headless_q3_summary_semantic_detection():
    """Verify that a student answer for Question 3 with ZERO header and no 'Summary' keyword is detected semantically."""
    from src.pipeline.answer_segmenter import extract_header_qno
    from src.core.schemas import ExtractedQuestion

    q_paper_text = """
    3. Summarize the following text. 10
    "Hope" is the thing with feathers —
    That perches in the soul —
    And sings the tune without the words—
    And never stops — at all—
    And sweetest — in the Gale — is heard —
    And sore must be the storm —
    That could abash the little Bird
    That kept so many warm —
    
    4. Fill in the blanks with suitable words...
    """

    q_obj = ExtractedQuestion(
        question_id="SE_11_Q1",
        question_text=q_paper_text,
        sub_questions=[
            {"q_no": "2", "name": "Flow Chart", "marks": 10},
            {"q_no": "3", "name": "Summary", "marks": 10},
            {"q_no": "4", "name": "Cloze with clues", "marks": 5},
        ]
    )

    # Student writes with ZERO header: no "Ans to the Q No 3", no "3.", no "Summary:"
    headless_summary = """Hope is what keeps humans alive. If our hope dies or our soul dies too. Hope is the sweetest thing a man could taste on his soul. Sometimes it might get a little rough but we shouldn't give up no matter what happens."""

    detected_q = extract_header_qno(
        headless_summary,
        current_parent="2",
        question_obj=q_obj,
        valid_q_order=["1(A)", "1(B)", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    )

    assert detected_q == "3"


def test_cursive_v_r_ligature_safeguard():
    from src.core.schemas import ArbitrationCandidate
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.arbitration.gate import EvidenceArbitrationGate
    from src.core.config import ArbitrationConfig
    import tempfile
    from PIL import Image

    # Test 1: The exact SE_11_Q1_0010 case: 'rumore' vs 'remove'
    cand1 = ArbitrationCandidate(
        candidate_id="3:3:0",
        error_index=3,
        error_type="spelling",
        erroneous_text="rumore",
        suggested_correction="remove",
        candidate_token="rumore",
        intended_token="remove",
        context_sentence="never rumore hope from all the good things",
        page_no=2,
    )
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    with tempfile.TemporaryDirectory() as d:
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(),
            cfg=ArbitrationConfig(),
            script_id="SE_11_Q1_0010",
            output_dir=d,
            page_images=[(2, img, "p2.png")],
            page_transcripts={2: "never rumore hope from all the good things"},
            clean_image_fn=None,
            full_transcript="never rumore hope from all the good things",
            lexicon={"never", "remove", "hope", "from", "all", "the", "good", "things"},
        )
        rec1 = gate._score_candidate(cand1)
        assert rec1.verdict == "HANDWRITING_AMBIGUITY"
        assert any("cursive 'v'<->'r' ligature" in note for note in rec1.evidence.notes)

    # Test 2: 'remore' vs 'remove' (Stage 1 initial output)
    cand2 = ArbitrationCandidate(
        candidate_id="3:3:1",
        error_index=3,
        error_type="spelling",
        erroneous_text="remore",
        suggested_correction="remove",
        candidate_token="remore",
        intended_token="remove",
        context_sentence="never remore hope from all the good things",
        page_no=2,
    )
    with tempfile.TemporaryDirectory() as d:
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(),
            cfg=ArbitrationConfig(),
            script_id="SE_11_Q1_0010",
            output_dir=d,
            page_images=[(2, img, "p2.png")],
            page_transcripts={2: "never remore hope from all the good things"},
            clean_image_fn=None,
            full_transcript="never remore hope from all the good things",
            lexicon={"never", "remove", "hope", "from", "all", "the", "good", "things"},
        )
        rec2 = gate._score_candidate(cand2)
        assert rec2.verdict == "HANDWRITING_AMBIGUITY"
        assert any("cursive 'v'<->'r' ligature" in note for note in rec2.evidence.notes)


def test_curvy_s_n_safeguard():
    from src.core.schemas import ArbitrationCandidate
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.arbitration.gate import EvidenceArbitrationGate
    from src.core.config import ArbitrationConfig
    import tempfile
    from PIL import Image

    # SE_11_Q1_0021 case: 'hin' vs 'his' ('cut the net with hin sharp teeth')
    cand = ArbitrationCandidate(
        candidate_id="9:9:0",
        error_index=9,
        error_type="spelling",
        erroneous_text="hin",
        suggested_correction="his",
        candidate_token="hin",
        intended_token="his",
        context_sentence="cut the net with hin sharp teeth.",
        page_no=10,
    )
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    with tempfile.TemporaryDirectory() as d:
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(),
            cfg=ArbitrationConfig(),
            script_id="SE_11_Q1_0021",
            output_dir=d,
            page_images=[(10, img, "p10.png")],
            page_transcripts={10: "cut the net with hin sharp teeth."},
            clean_image_fn=None,
            full_transcript="cut the net with hin sharp teeth.",
            lexicon={"cut", "the", "net", "with", "his", "sharp", "teeth"},
        )
        rec = gate._score_candidate(cand)
        assert rec.verdict == "HANDWRITING_AMBIGUITY"
        assert any("curvy 's'<->'n'" in note for note in rec.evidence.notes)


def test_aborted_draft_struck_safeguard():
    from src.core.schemas import ArbitrationCandidate
    from src.engine.mock_engine import MockGemmaEngine
    from src.pipeline.arbitration.gate import EvidenceArbitrationGate
    from src.core.config import ArbitrationConfig
    import tempfile
    from PIL import Image

    # SE_11_Q1_0021 case: 'possi' before 'positively'
    cand1 = ArbitrationCandidate(
        candidate_id="10:5:0",
        error_index=5,
        error_type="spelling",
        erroneous_text="possi",
        suggested_correction="posse",
        candidate_token="possi",
        intended_token="posse",
        context_sentence="out of our village see this possi positively and they promise that they",
        page_no=13,
    )
    # SE_11_Q1_0021 case: 'grap' before 'pie-chart'
    cand2 = ArbitrationCandidate(
        candidate_id="8:14:0",
        error_index=14,
        error_type="spelling",
        erroneous_text="grap",
        suggested_correction="graph",
        candidate_token="grap",
        intended_token="graph",
        context_sentence="However analyzing the grap pie-chart it can be said that the chart is very up-to-date",
        page_no=8,
    )
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    with tempfile.TemporaryDirectory() as d:
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(),
            cfg=ArbitrationConfig(),
            script_id="SE_11_Q1_0021",
            output_dir=d,
            page_images=[(13, img, "p13.png"), (8, img, "p8.png")],
            page_transcripts={
                13: "out of our village see this possi positively and they promise that they",
                8: "However analyzing the grap pie-chart it can be said that the chart is very up-to-date",
            },
            clean_image_fn=None,
            full_transcript="out of our village see this possi positively and they promise that they However analyzing the grap pie-chart",
            lexicon={"out", "of", "our", "village", "see", "this", "positively", "and", "they", "promise", "that", "analyzing", "the", "pie-chart"},
        )
        rec1 = gate._score_candidate(cand1)
        assert rec1.verdict == "HANDWRITING_AMBIGUITY"
        assert any("Aborted / struck-through draft" in note for note in rec1.evidence.notes)

        rec2 = gate._score_candidate(cand2)
        assert rec2.verdict == "HANDWRITING_AMBIGUITY"
        assert any("Aborted / struck-through draft" in note for note in rec2.evidence.notes)


if __name__ == "__main__":
    import sys
    import inspect

    current_module = sys.modules[__name__]
    test_funcs = [
        obj for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("test_")
    ]
    passed = 0
    failed = 0
    for fn in test_funcs:
        try:
            fn()
            passed += 1
            print(f"PASS: {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL: {fn.__name__}: {e}")

    print(f"\nResult: {passed} passed, {failed} failed out of {len(test_funcs)} tests.")
    if failed > 0:
        sys.exit(1)








