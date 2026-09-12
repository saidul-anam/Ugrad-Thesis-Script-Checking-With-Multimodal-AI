import os
import tempfile

from src.pipeline.arbitration.symbolic_evidence import align_chars
from src.pipeline.arbitration.writer_profile import (
    WriterProfile,
    build_writer_profile,
    add_consensus_disagreements,
    add_candidate_evidence,
    writer_prior,
    save_writer_profile,
    load_writer_profile,
)

LEX = {"the", "chart", "shows", "that", "and", "of", "to", "sector", "with", "this", "from", "coal", "is", "most"}


def test_anchor_near_misses_learn_writer_confusions():
    transcript = (
        "From dhe chart we explore dhad coal is dhe most sector\n"
        "the the the that that to to to with this from"
    )
    prof = build_writer_profile("s1", transcript, LEX)
    assert "the" in prof.anchors and "that" in prof.anchors
    # 'dhe' x2 -> t>d twice ; 'dhad' -> t>d (two substitutions, len 4 anchor allowed)
    assert prof.pair_counts.get("t>d", 0) >= 3
    assert all(e["source"] == "transcript" for e in prof.evidence)


def test_writer_prior_uses_learned_counts_only():
    prof = WriterProfile(script_id="x")
    ops = align_chars("illustrodes", "illustrates")   # a>o, t>d
    prior, cnt = writer_prior(prof, ops, min_count=2)
    assert prior == 0.0 and cnt == 0
    prof.add_pair("t>d", "transcript")
    prof.add_pair("t>d", "transcript")
    prof.add_pair("t>d", "transcript")
    prior, cnt = writer_prior(prof, ops, min_count=2)
    assert 0.0 < prior < 0.5          # only one of two ops explained
    prof.add_pair("a>o", "consensus")
    prof.add_pair("a>o", "consensus")
    prior2, _ = writer_prior(prof, ops, min_count=2)
    assert prior2 > prior
    # below min_count contributes nothing
    prof2 = WriterProfile(script_id="y")
    prof2.add_pair("f>d", "stage3")
    assert writer_prior(prof2, align_chars("powerdul", "powerful"), min_count=2)[0] == 0.0


def test_consensus_disagreements_feed_profile():
    prof = WriterProfile(script_id="z")
    columns = [["t", "d", "t", "t"], ["h", "h", "h", "h"], ["e", "e", "e", "-"]]
    majority = ["t", "h", "e"]
    added = add_consensus_disagreements(prof, columns, majority)
    assert added == 2
    assert prof.pair_counts["t>d"] == 1
    assert prof.pair_counts["e>∅"] == 1


def test_candidate_evidence_and_persistence():
    prof = WriterProfile(script_id="p")
    add_candidate_evidence(prof, "powerdul", "powerful")
    assert prof.pair_counts["f>d"] == 1
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "writer_profile.json")
        save_writer_profile(prof, path)
        loaded = load_writer_profile(path)
        assert loaded.pair_counts == prof.pair_counts
