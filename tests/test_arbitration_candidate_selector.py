from src.core.schemas import LinguisticErrorItem
from src.pipeline.arbitration.candidate_selector import (
    tokenize,
    differing_token_pairs,
    select_candidates,
)


def _err(etype, text, corr, ctx="ctx"):
    return LinguisticErrorItem(error_type=etype, erroneous_text=text, suggested_correction=corr,
                               context_sentence=ctx, explanation="")


def test_tokenize_strips_punctuation_keeps_apostrophe():
    assert tokenize("Hello, world! it's \"Done\".") == ["hello", "world", "it's", "done"]


def test_single_token_pairs():
    assert differing_token_pairs("powerdul", "powerful") == [("powerdul", "powerful")]
    assert differing_token_pairs("stanled", "started") == [("stanled", "started")]
    assert differing_token_pairs("mangsimbee", "mistreated") == []


def test_multi_token_grammar_pairs_are_gated():
    assert differing_token_pairs("want do", "want to") == [("do", "to")]
    assert differing_token_pairs("Do the", "to the") == [("do", "to")]
    assert differing_token_pairs("he fed thad", "he felt that") == [("fed", "felt"), ("thad", "that")]


def test_structural_changes_are_not_gated():
    assert differing_token_pairs("he go", "he goes to") == []
    assert differing_token_pairs("a", "an apple") == []
    assert differing_token_pairs("Yours even", "Yours sincerely") == []


def test_fusion_tokens_are_gated():
    # Single token OCR bleed-through / ligature corrected to multi-word phrase
    assert differing_token_pairs("noth", "not a") == [("noth", "not")]


def test_lexicon_candidates_hygiene():
    from src.pipeline.arbitration.candidate_selector import lexicon_candidates
    from src.utils.linguistic_sanitizer import get_english_lexicon
    lex = get_english_lexicon()
    # Ensure proper names like 'ronny' are NOT suggested for 'renny'
    cands = lexicon_candidates("I will be renny happy if you join with me.", lex)
    for c in cands:
        if c.erroneous_text.lower() == "renny":
            assert c.suggested_correction.lower() != "ronny"


def test_select_candidates_any_error_type():
    errors = [
        _err("grammar", "want do imagine", "wants to imagine"),
        _err("spelling", "powerdul", "powerful"),
        _err("syntax", "for want of new dream", "He wanted a new dream"),
    ]
    cands = select_candidates(errors, q_no="7", max_edits=2)
    ids = [(c.error_index, c.candidate_token, c.intended_token) for c in cands]
    assert (0, "want", "wants") in ids
    assert (0, "do", "to") in ids
    assert (1, "powerdul", "powerful") in ids
    assert cands[0].candidate_id.startswith("7:0:")
    assert cands[0].question_no == "7"


def test_select_candidates_strikethrough_suspects():
    errors = [
        _err("grammar", "helps many us", "helps many of us", ctx="it is a software and it helps many us by solve many problems"),
        _err("syntax", "the percentage was Hydro-electrice power was 16%", "Hydro-electric power was 16%", ctx="According to the graph, In 1980 the percentage was Hydro-electrice power was 16%."),
    ]
    cands = select_candidates(errors, q_no="7")
    suspects = [(c.error_type, c.candidate_token, c.intended_token) for c in cands]
    assert ("strikethrough_suspect", "many", "[struck]") in suspects
    assert ("strikethrough_suspect", "was", "[struck]") in suspects

