from src.utils.text_metrics import (
    levenshtein,
    cer,
    wer,
    edit_ops,
    normalize_transcript,
    score_transcription,
)


def test_levenshtein_basic():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein(["a", "b"], ["a", "c", "d"]) == 2


def test_cer_and_wer_known_values():
    ref = "the cat sat"
    hyp = "the cat sit"
    assert abs(cer(ref, hyp) - 1 / len(ref)) < 1e-9
    assert abs(wer(ref, hyp) - 1 / 3) < 1e-9
    assert cer("", "") == 0.0
    assert cer("", "x") == 1.0
    assert wer("a b", "a b") == 0.0


def test_edit_ops_counts():
    subs, ins, dels = edit_ops("abc", "abd")
    assert (subs, ins, dels) == (1, 0, 0)
    subs, ins, dels = edit_ops("abc", "abcd")
    assert (subs, ins, dels) == (0, 1, 0)
    subs, ins, dels = edit_ops("abcd", "abc")
    assert (subs, ins, dels) == (0, 0, 1)


def test_normalize_transcript_tags_and_layout():
    raw = "Ans: to the Q. No. 1\n\n[struck: overcoming] overcome [unclear: thouths] [illegible]\n\n--- Page Break ---\n\nnext"
    norm = normalize_transcript(raw)
    assert "[struck" not in norm and "overcoming" not in norm
    assert "thouths" in norm and "[unclear" not in norm
    assert "[illegible]" in norm
    assert "page break" not in norm
    assert norm.endswith("next")
    assert normalize_transcript("A  B\n\nC") == "a b c"


def test_score_transcription_silent_correction_probe():
    lexicon = {"the", "chart", "shows", "sources", "of", "electricity", "powerful", "families"}
    reference = "The chart illustrodes the sources of electricidty. familyes"
    # hypothesis autocorrected two student non-words, preserved one
    hypothesis = "The chart illustrates the sources of electricity. familyes"
    s = score_transcription(reference, hypothesis, lexicon=lexicon)
    assert s.student_nonwords == 3
    assert s.nonwords_preserved == 1
    assert sorted(s.nonwords_lost) == ["electricidty", "illustrodes"]
    assert abs(s.silent_correction_rate - 2 / 3) < 1e-9
    assert 0 < s.cer < 0.2
    assert 0 < s.wer < 0.5
    d = s.as_dict()
    assert d["silent_correction_rate"] is not None


def test_score_transcription_without_lexicon():
    s = score_transcription("hello world", "hello world")
    assert s.cer == 0.0 and s.wer == 0.0
    assert s.silent_correction_rate is None
