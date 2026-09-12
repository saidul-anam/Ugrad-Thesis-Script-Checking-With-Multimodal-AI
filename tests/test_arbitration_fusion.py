from src.core.config import ArbitrationWeights
from src.core.schemas import ArbitrationEvidence
from src.pipeline.arbitration.fusion import (
    fuse,
    fuse_signals,
    quantize,
    fit_weights,
    choose_thresholds,
    GENUINE,
    AMBIGUITY,
    UNCERTAIN,
)


def test_fusion_monotone_and_none_skipping():
    w = ArbitrationWeights()
    low = fuse_signals({"phonetic": 0.1, "writer": 0.0, "consensus": 0.1, "forced_choice": 0.1}, w)
    high = fuse_signals({"phonetic": 0.9, "writer": 0.8, "consensus": 0.9, "forced_choice": 0.9}, w)
    assert low < 0.35 and high > 0.65
    partial = fuse_signals({"phonetic": None, "writer": None, "consensus": 0.9, "forced_choice": None}, w)
    assert 0.5 < partial < high
    none_all = fuse_signals({}, w)
    assert abs(none_all - 1 / (1 + 2.718281828 ** 0.3)) < 1e-3


def test_fuse_from_evidence_and_quantize():
    ev = ArbitrationEvidence(phonetic_signal=0.9, writer_prior=0.8, consensus_signal=0.9, forced_choice_signal=0.95)
    s = fuse(ev, ArbitrationWeights())
    assert quantize(s, 0.65, 0.35) == AMBIGUITY
    assert quantize(0.1, 0.65, 0.35) == GENUINE
    assert quantize(0.5, 0.65, 0.35) == UNCERTAIN


def test_fit_weights_recovers_separable_set():
    rows = ([{"phonetic": 0.9, "writer": 0.7, "consensus": 0.9, "forced_choice": 0.9}] * 8 +
            [{"phonetic": 0.2, "writer": 0.0, "consensus": 0.1, "forced_choice": 0.2}] * 8)
    labels = [1] * 8 + [0] * 8
    w = fit_weights(rows, labels, epochs=500)
    assert all(getattr(w, k) >= 0 for k in ("phonetic", "writer", "consensus", "forced_choice"))
    hi = fuse_signals(rows[0], w)
    lo = fuse_signals(rows[-1], w)
    assert hi > 0.8 and lo < 0.2
    thr_amb, thr_gen = choose_thresholds([fuse_signals(r, w) for r in rows], labels)
    assert thr_gen < thr_amb


def test_choose_thresholds_fallback_small_data():
    assert choose_thresholds([0.1, 0.9], [0, 1]) == (0.65, 0.35)
