from src.pipeline.arbitration.symbolic_evidence import (
    align_chars,
    edit_op_keys,
    metaphone,
    phonetic_plausibility,
    symbolic_evidence,
)


def test_align_chars_describes_edit_path():
    ops = [str(o) for o in align_chars("powerdul", "powerful") if o.op != "match"]
    assert ops == ["sub:f>d"]
    keys = edit_op_keys(align_chars("electricidty", "electricity"))
    assert keys == ["∅>d"]
    keys = edit_op_keys(align_chars("abot", "about"))
    assert keys == ["u>∅"]
    assert edit_op_keys(align_chars("same", "same")) == []


def test_metaphone_basic_codes():
    assert metaphone("thoughts").startswith("0")      # TH -> 0
    assert metaphone("knight") == metaphone("night")
    assert metaphone("phone")[0] == "F"
    assert metaphone("") == ""


def test_phonetic_plausibility_bounds_and_neutrality():
    assert phonetic_plausibility("familyes", "families") >= 0.75
    assert phonetic_plausibility("fullfill", "fulfill") == 1.0
    assert phonetic_plausibility("accroding", "according") == 1.0
    assert phonetic_plausibility("বর্নিত", "বর্ণিত") == 0.5    # non-Latin -> neutral
    assert phonetic_plausibility("", "x") == 0.5
    v = phonetic_plausibility("thouths", "thoughts")
    assert 0.0 <= v <= 1.0


def test_symbolic_evidence_tuple():
    pp, ops = symbolic_evidence("illustrodes", "illustrates")
    assert 0.0 <= pp <= 1.0
    assert "sub:t>d" in ops and "sub:a>o" in ops
