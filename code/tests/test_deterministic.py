"""Tests for the offline (no-API) logic: full marks, cross-check, routing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import fullmarks, ground_truth, normalize_bn as nb, analyze


def test_metrics_empty_batch_no_crash():
    # an all-blank batch (no gold scores) must not raise, just return empty
    df = pd.DataFrame([{"id": "x", "subject": "Bangla", "status": "NO_REDMARK",
                        "gold_score": None, "full_marks": 2.5, "gold_usable": False,
                        "score_rubric": 0.0}])
    out = analyze.metrics_by_subject(df, "rubric")
    assert out.empty
    assert list(out.columns) == ["subject", "variant", "n", "exact_agreement",
                                 "within_0.5", "normalized_mae"]


# ---- full marks from rubric (sum of per-clause 'X নম্বর' allocations) ----
def test_fullmarks_rubric_sum():
    rb = "নম্বর বণ্টন: ৪টি স্থানের নাম লিখলে (০.৫ \\(\\times \\) ৪) = ০২ নম্বর। গ্লাইকোলাইসিস লিখলে ০.৫ নম্বর।"
    assert fullmarks.full_marks_from_rubric(rb) == 2.5


def test_fullmarks_rubric_single_clause():
    rb = "নম্বর বণ্টন: পা্চটি এনজাইমের ভূমিকা লেখার জন্য (০.৫ \\(\\times \\) ৫) = ২.৫ নম্বর।"
    assert fullmarks.full_marks_from_rubric(rb) == 2.5


def test_fullmarks_rubric_ocr_spaced_word():
    rb = "নম্ব র বণ্টনঃ ... নির্ণয় করার জন্য ০১ নম্ব র । ... জন্য ০২ নম্ব র ।"
    assert fullmarks.full_marks_from_rubric(rb) == 3.0


def test_fullmarks_marks_col_semantics():
    # score/full tokens -> denominator is the total
    assert fullmarks.full_marks_from_marks_col("7/10") == 10.0
    assert fullmarks.full_marks_from_marks_col("10/10") == 10.0
    assert fullmarks.full_marks_from_marks_col("3/5") == 5.0
    # half-mark token -> value a/b is the total (5/2 = 2.5)
    assert fullmarks.full_marks_from_marks_col("5/2") == 2.5
    assert fullmarks.full_marks_from_marks_col("1/2") == 0.5


# ---- marks-token semantics + cross-check ----
def test_parse_marks_token():
    assert ground_truth.parse_marks_token("7/10") == {"full": 10.0, "score": 7.0, "kind": "score_over_full"}
    assert ground_truth.parse_marks_token("5/2") == {"full": 2.5, "score": None, "kind": "half_mark_total"}
    assert ground_truth.parse_marks_token("") is None


def test_crosscheck_halfmark_has_no_score_to_verify():
    # 5/2 carries only the total; red mark is the sole score -> NO_MARKSCOL
    cc = ground_truth.cross_check(2.5, "5/2")
    assert cc["status"] == "NO_MARKSCOL" and cc["gold_score"] == 2.5


def test_crosscheck_gold_outof_reading():
    cc = ground_truth.cross_check(7.0, "7/10")
    assert cc["status"] == "GOLD" and cc["matched"] == 7.0


def test_crosscheck_flag():
    # red 4 disagrees with the token's score of 7 -> FLAG
    cc = ground_truth.cross_check(4.0, "7/10")
    assert cc["status"] == "FLAG" and cc["gold_score"] == 4.0


def test_crosscheck_no_redmark():
    assert ground_truth.cross_check(None, "7/10")["status"] == "NO_REDMARK"


def test_crosscheck_no_markscol():
    cc = ground_truth.cross_check(3.0, "")
    assert cc["status"] == "NO_MARKSCOL" and cc["gold_score"] == 3.0


# ---- question routing ----
def test_route_prefers_english():
    r = {"questionEN": "Define bivalent and synapsis.", "questionBN": "গার্বলড"}
    d = nb.route_question(r)
    assert d["source"] == "EN"


def test_route_genuine_bengali_unicode():
    r = {"questionEN": "", "questionBN": "নিচের শব্দগুলোর বিপরীতার্থক শব্দ লেখ"}
    d = nb.route_question(r)
    assert d["source"] == "BN_UNICODE"


def test_route_bijoy_garbled_flagged():
    r = {"questionEN": "", "questionBN": "উ্ঃঃধঢ়যৎুহ্ং সবষধহড়ংঃরপঃ্ং রং ধহ ধহরসধষ ড়ভ যিরপয পষধংং"}
    d = nb.route_question(r)
    assert d["source"] == "BN_GARBLED"


def test_bn_digits():
    assert nb.bn_digits_to_ascii("২.৫") == "2.5"
    assert nb.bn_digits_to_ascii("১০") == "10"
