from PIL import Image

from src.pipeline.arbitration.consensus import (
    needleman_wunsch,
    progressive_consensus,
    word_confidences,
    aligned_token,
    agreement,
    consensus_signal,
    default_augmentations,
    apply_augmentation,
    ConsensusTranscriber,
)
from src.engine.mock_engine import MockGemmaEngine


def test_needleman_wunsch_alignment_lengths():
    a, b = needleman_wunsch("kitten", "sitting")
    assert len(a) == len(b)
    assert a.replace("-", "") == "kitten" and b.replace("-", "") == "sitting"


def test_progressive_consensus_majority_and_votes():
    samples = ["the sector", "dhe sector", "the secfor", "the sector", "the sector"]
    res = progressive_consensus(samples)
    assert res.consensus == "the sector"
    assert res.n == 5
    wc = dict(word_confidences(res))
    assert abs(wc["the"] - 0.8) < 1e-9
    assert abs(wc["sector"] - 0.8) < 1e-9
    assert all(0 < v <= 1 for v in res.column_votes)


def test_progressive_consensus_edge_cases():
    assert progressive_consensus([]).consensus == ""
    assert progressive_consensus(["only"]).consensus == "only"
    res = progressive_consensus(["ab", "ab", "abc"])
    assert res.consensus == "ab"


def test_aligned_token_positions():
    ref = ["the", "most", "powerdul", "sector"]
    assert aligned_token(["the", "most", "powerful", "sector"], ref, 2) == "powerful"
    assert aligned_token(["the", "most", "sector"], ref, 2) is None
    assert aligned_token(["the", "most", "power", "ful", "sector"], ref, 2) in ("power", "ful")


def test_agreement_and_signal():
    samples = ["the sector", "dhe sector", "the secfor", "the sector", "the sector"]
    agr_c, agr_i, word_conf, aligned = agreement(samples, "dhe sector", "dhe", "the")
    assert abs(agr_c - 0.2) < 1e-9 and abs(agr_i - 0.8) < 1e-9
    assert word_conf is not None and 0 < word_conf <= 1
    sig = consensus_signal(agr_c, agr_i, word_conf)
    assert sig > 0.5
    assert consensus_signal(1.0, 0.0, 1.0) == 0.0
    assert consensus_signal(0.0, 1.0, None) == 1.0


def test_augmentations_and_mock_transcriber():
    augs = default_augmentations(5, True, 0.7)
    assert len(augs) == 5 and augs[0].temperature == 0.7
    img = Image.new("RGB", (400, 60), (255, 255, 255))
    for a in augs:
        out = apply_augmentation(img, a)
        assert out.size[0] > 0 and out.size[1] > 0
    tr = ConsensusTranscriber(MockGemmaEngine(), n_samples=3)
    res = tr.run(img, "mock line of handwriting text", "handwritng", "handwriting")
    assert tr.model_calls == 3
    assert len(res["samples"]) == 3
    assert res["signal"] is not None
