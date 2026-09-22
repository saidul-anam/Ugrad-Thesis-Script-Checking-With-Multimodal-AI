import os
import json
import tempfile

from PIL import Image, ImageDraw

from src.core.config import ArbitrationConfig
from src.core.schemas import LinguisticErrorItem, ArbitrationCandidate
from src.engine.mock_engine import MockGemmaEngine
from src.pipeline.arbitration import EvidenceArbitrationGate, attribute_error_to_page
from src.pipeline.arbitration.localizer import LineLocalizer, match_ratio
from src.pipeline.stage3_error_analyzer import Stage3ErrorAnalyzer


def _three_line_page():
    """Three text lines rendered with the default bitmap font, then upscaled 3x so each line is ~30px tall."""
    img = Image.new("RGB", (400, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, txt in enumerate(["the pie chart illustrodes the sources", "coal is the most powerdul sector", "hope you are well"]):
        d.text((30, 30 + i * 55), txt, fill=(0, 0, 0))
    return img.resize((1200, 600), Image.NEAREST)


def test_attribute_error_to_page():
    pages = {1: "Ans to Q1\nthe author want do imagine", 2: "coal is the most powerdul sector"}
    assert attribute_error_to_page("powerdul", "coal is the most powerdul sector", pages) == 2
    assert attribute_error_to_page("want do", "the author want do imagine", pages) == 1
    assert attribute_error_to_page("zzz", "nothing here", pages) is None


def test_match_ratio_windows():
    assert match_ratio("coal is the most powerdul sector", "From the chart coal is the most powerdul sector in USA", "powerdul") > 0.8
    assert match_ratio("completely different words here", "coal is the most powerdul sector", "powerdul") < 0.5


def test_projection_lines_and_locate_graceful():
    img = _three_line_page()
    loc = LineLocalizer(MockGemmaEngine(), [(1, img, "p1.png")], {1: "line one\nline two\nline three"},
                        clean_image_fn=None, use_bbox=False, min_ratio=0.6, search_window=3)
    lines = loc.projection_lines(1)
    assert len(lines) >= 3
    cand = ArbitrationCandidate(candidate_id="1:0:0", error_index=0, candidate_token="powerdul",
                                intended_token="powerful", context_sentence="coal is the most powerdul sector", page_no=1)
    # mock re-reads never match the context -> no crop, no exception
    assert loc.locate(cand) is None
    assert loc.model_calls >= 1


def test_gate_end_to_end_with_mock_engine():
    img = _three_line_page()
    errors = [
        LinguisticErrorItem(error_type="spelling", erroneous_text="powerdul", suggested_correction="powerful",
                            context_sentence="coal is the most powerdul sector", explanation=""),
        LinguisticErrorItem(error_type="grammar", erroneous_text="want do", suggested_correction="want to",
                            context_sentence="the author want do imagine", explanation=""),
        LinguisticErrorItem(error_type="syntax", erroneous_text="for want of new dream", suggested_correction="He wanted a new dream",
                            context_sentence="for want of new dream", explanation=""),
    ]
    with tempfile.TemporaryDirectory() as d:
        cfg = ArbitrationConfig(consensus_samples=2, use_bbox_localization=True)
        gate = EvidenceArbitrationGate(
            engine=MockGemmaEngine(), cfg=cfg, script_id="mock", output_dir=d,
            page_images=[(1, img, "p1.png")],
            page_transcripts={1: "the author want do imagine\ncoal is the most powerdul sector"},
            clean_image_fn=None, full_transcript="the the the to to want do coal is the most powerdul sector",
            lexicon={"the", "to", "want", "coal", "is", "most", "sector", "author", "imagine"},
        )
        confirmed, cleared, records = gate.arbitrate(errors, q_no="8")
        assert len(records) == 2                      # syntax error not gated
        assert {r.candidate.candidate_token for r in records} == {"powerdul", "do"}
        assert all(r.verdict in ("GENUINE_ERROR", "HANDWRITING_AMBIGUITY", "UNCERTAIN") for r in records)
        assert all(0.0 <= r.ambiguity_score <= 1.0 for r in records)
        # the un-gated syntax error is always kept
        assert any(e.error_type == "syntax" for e in confirmed)
        res = gate.finalize()
        assert res.total_candidates == 2
        assert os.path.exists(os.path.join(d, "stage3b_arbitration.json"))
        assert os.path.exists(os.path.join(d, "writer_profile.json"))
        with open(os.path.join(d, "stage3b_arbitration.json")) as f:
            data = json.load(f)
        assert data["mode"] == "evidence" and len(data["records"]) == 2
        # writer profile learned t>d from the transcript's 'do' near 'to'? ('do' is a lexicon word -> not a near-miss)
        # but candidate evidence is added after scoring
        assert data["records"][0]["evidence"]["edit_ops"]


def test_legacy_path_unchanged_with_mock():
    analyzer = Stage3ErrorAnalyzer(MockGemmaEngine())
    errors = [LinguisticErrorItem(error_type="spelling", erroneous_text="powerdul", suggested_correction="powerful",
                                  context_sentence="x", explanation="")]
    confirmed, cleared = analyzer.arbitrate_visual_errors(image=Image.new("RGB", (50, 50)), errors=errors, gate=None)
    assert confirmed == errors and cleared == []


def test_gate_adopts_consensus_token():
    gate = EvidenceArbitrationGate(
        engine=MockGemmaEngine(), cfg=ArbitrationConfig(), script_id="test", output_dir="/tmp",
        page_images=[(1, Image.new("RGB", (100, 100)), "p1.png")],
        page_transcripts={1: "test"}, clean_image_fn=None, full_transcript="test", lexicon={"test"}
    )
    gate.consensus.run = lambda *args, **kwargs: {
        "samples": ["verry", "verry"], "agreement_candidate": 0.0, "agreement_intended": 0.0,
        "word_confidence": 1.0, "signal": 0.5, "consensus": None,
        "consensus_token": "verry", "consensus_token_agreement": 1.0
    }
    cand = ArbitrationCandidate(candidate_id="1:0:0", candidate_token="renny", intended_token="runny",
                                erroneous_text="renny", suggested_correction="runny", error_type="spelling",
                                error_index=0, page_no=1)
    class Crop:
        image = Image.new("RGB", (50, 20))
        transcript = "renny"
        method = "bbox"
        match_ratio = 1.0
        path = "/tmp/crop.png"
    gate.localizer.locate = lambda c: Crop()

    rec = gate._score_candidate(cand)
    assert cand.intended_token == "verry"
    assert rec.evidence.consensus_agreement_intended == 1.0
    assert rec.evidence.consensus_signal == 1.0


def test_gate_clears_strikethrough_suspect_on_crop():
    import numpy as np
    import cv2
    
    # Create image crop with a horizontal strike line through letters
    arr = np.full((100, 200), 255, dtype=np.uint8)
    cv2.putText(arr, "many", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.line(arr, (15, 52), (120, 52), 0, 2)
    crop_img = Image.fromarray(arr)

    gate = EvidenceArbitrationGate(
        engine=MockGemmaEngine(), cfg=ArbitrationConfig(), script_id="test", output_dir="/tmp",
        page_images=[(1, Image.new("RGB", (100, 100)), "p1.png")],
        page_transcripts={1: "it helps many us"}, clean_image_fn=None, full_transcript="it helps many us",
        lexicon={"it", "helps", "us", "many"}
    )
    class StrikeCrop:
        image = crop_img
        transcript = "it helps many us"
        method = "bbox"
        match_ratio = 1.0
        path = "/tmp/crop.png"
    gate.localizer.locate = lambda c: StrikeCrop()

    cand = ArbitrationCandidate(
        candidate_id="7:0:strike",
        error_index=0,
        error_type="strikethrough_suspect",
        erroneous_text="helps many us",
        suggested_correction="helps many of us",
        candidate_token="many",
        intended_token="[struck]",
        context_sentence="it is a software programme and it helps many us by solve many problems",
        page_no=1,
    )
    rec = gate._score_candidate(cand)
    assert rec.verdict == "HANDWRITING_AMBIGUITY"
    assert rec.ambiguity_score >= 0.85
    assert any("Optical strikethrough" in note for note in rec.evidence.notes)

