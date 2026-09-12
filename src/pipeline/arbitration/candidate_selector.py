"""
Candidate selection for the Stage 3b ambiguity gate (pure python, no model calls).

Any Stage 3 error — spelling, grammar or syntax — whose erroneous span and suggested correction
differ in one or more token pairs that are each within `max_edits` character edits is a
candidate for visual arbitration. This catches perceptual misreads that Stage 3 labelled as
grammar ("want do" -> "want to") as well as classic spelling candidates.
"""

import re
import difflib
from typing import List, Tuple, Optional, Sequence, Set

from src.core.schemas import LinguisticErrorItem, ArbitrationCandidate


_STRIP_PUNCT = re.compile(r"^[^\wঀ-৿']+|[^\wঀ-৿']+$", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokens with surrounding punctuation stripped (apostrophes kept)."""
    out = []
    for raw in (text or "").split():
        t = _STRIP_PUNCT.sub("", raw).lower()
        if t:
            out.append(t)
    return out


def levenshtein(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def differing_token_pairs(erroneous: str, correction: str, max_edits: int = 2) -> List[Tuple[str, str]]:
    """
    Return the (read, intended) token pairs in which the two spans differ.

    - Equal-length token sequences: every differing position is a pair; if any pair exceeds
      `max_edits`, the error is not a stroke-level candidate and [] is returned.
    - Unequal lengths: aligned with difflib; handles single-to-multi or multi-to-single token
      mismatches (e.g. 'noth' -> 'not a', 'inall' -> 'in all', 'per fect' -> 'perfect')
      when a root token or joined string is within `max_edits`.
    """
    e_toks = tokenize(erroneous)
    c_toks = tokenize(correction)
    if not e_toks or not c_toks:
        return []

    pairs: List[Tuple[str, str]] = []
    if len(e_toks) == len(c_toks):
        for a, b in zip(e_toks, c_toks):
            if a == b:
                continue
            if levenshtein(a, b) > max_edits:
                return []
            pairs.append((a, b))
        return pairs

    # Case: Single-token error corrected to a multi-word phrase (e.g. fusion "noth" -> "not a", "alot" -> "a lot")
    if len(e_toks) == 1 and len(c_toks) > 1:
        e_word = e_toks[0]
        joined_c = "".join(c_toks)
        if levenshtein(e_word, joined_c) <= max_edits:
            best_c = min(c_toks, key=lambda c: (levenshtein(e_word, c), -len(c)))
            return [(e_word, best_c)]

    # Case: Multi-token error corrected to single token (e.g. split word "in stead" -> "instead")
    if len(e_toks) > 1 and len(c_toks) == 1:
        c_word = c_toks[0]
        joined_e = "".join(e_toks)
        if levenshtein(joined_e, c_word) <= max_edits:
            return [(joined_e, c_word)]

    sm = difflib.SequenceMatcher(a=e_toks, b=c_toks, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            return []
        for a, b in zip(e_toks[i1:i2], c_toks[j1:j2]):
            if levenshtein(a, b) > max_edits:
                return []
            pairs.append((a, b))
    return pairs


_ALPHA_WORD = re.compile(r"^[a-z]+$")


def lexicon_candidates(
    answer_text: str,
    lexicon: Set[str],
    question_vocab: Optional[Set[str]] = None,
    existing_errors: Optional[List[LinguisticErrorItem]] = None,
    min_len: int = 4,
    max_edits: int = 2,
) -> List[LinguisticErrorItem]:
    """
    Dictionary scan (no model, no letter rules): every alphabetic token of the answer that is not a
    dictionary word, not question vocabulary, not a likely proper noun, and not already covered by a
    Stage 3 error, becomes a spelling candidate whose intended word is the closest dictionary word
    (prefer words that occur in the question vocabulary or elsewhere in the answer).
    """
    if not answer_text or not lexicon:
        return []
    q_vocab = {w.lower() for w in (question_vocab or set())}
    covered = set()
    for e in existing_errors or []:
        covered.update(tokenize(e.erroneous_text or ""))
    raw_tokens = answer_text.split()
    answer_lex = {t for t in tokenize(answer_text) if t in lexicon}
    seen: Set[str] = set()
    out: List[LinguisticErrorItem] = []
    for i, raw in enumerate(raw_tokens):
        tok = _STRIP_PUNCT.sub("", raw)
        low = tok.lower()
        if len(low) < min_len or not _ALPHA_WORD.match(low) or low in seen:
            continue
        if low in lexicon or low in q_vocab or low in covered:
            continue
        prev = raw_tokens[i - 1] if i > 0 else ""
        sentence_start = i == 0 or prev.endswith((".", "!", "?", ":")) or prev.lower().startswith("ans")
        if tok[:1].isupper() and not sentence_start:
            continue  # likely a proper noun
        if "[" in raw or "]" in raw or "-" in raw.strip(".,;:!?\"'()"):
            continue  # transcription markers / hyphenated compounds (hydro-electric)

        # Check if merging with adjacent tokens forms a valid dictionary word
        # (handwriting spacing / syllable separation artifact, e.g. "per" + "fect" -> "perfect")
        if i > 0 and not raw_tokens[i - 1].endswith((".", "!", "?", ":", ";")):
            prev_clean = _STRIP_PUNCT.sub("", raw_tokens[i - 1]).lower()
            if prev_clean and (prev_clean + low) in lexicon:
                continue
        if i + 1 < len(raw_tokens) and not raw.endswith((".", "!", "?", ":", ";")):
            next_clean = _STRIP_PUNCT.sub("", raw_tokens[i + 1]).lower()
            if next_clean and (low + next_clean) in lexicon:
                continue
        # closest dictionary word within max_edits (prefer question vocab / words used in the answer)
        def _rank(w: str) -> tuple:
            # smaller edit distance first; on ties prefer same length (substitution-only, the perceptual case)
            return (levenshtein(low, w), abs(len(w) - len(low)), w)
        pool = {w for w in (q_vocab | answer_lex) if abs(len(w) - len(low)) <= max_edits and len(w) >= 3}
        pool.update(difflib.get_close_matches(low, [w for w in lexicon if w[:1] == low[:1] and abs(len(w) - len(low)) <= 1 and len(w) >= 3], n=5, cutoff=0.8))
        best, best_d = None, max_edits + 1
        if pool:
            cand_w = min(pool, key=_rank)
            best, best_d = cand_w, levenshtein(low, cand_w)
        if best is None or best_d > max_edits or best == low:
            continue
        seen.add(low)
        ctx_lo, ctx_hi = max(0, i - 6), min(len(raw_tokens), i + 7)
        out.append(LinguisticErrorItem(
            error_type="spelling",
            erroneous_text=tok,
            suggested_correction=best,
            context_sentence=" ".join(raw_tokens[ctx_lo:ctx_hi]),
            explanation="Non-dictionary token found by lexicon scan (not flagged by Stage 3); arbitrated against the ink.",
        ))
    return out


def select_candidates(
    errors: List[LinguisticErrorItem],
    q_no: Optional[str] = None,
    max_edits: int = 2,
) -> List[ArbitrationCandidate]:
    """Build ArbitrationCandidate objects for every gate-able token pair in the error list."""
    cands: List[ArbitrationCandidate] = []
    q_key = str(q_no) if q_no is not None else "ALL"
    for idx, err in enumerate(errors):
        pairs = differing_token_pairs(err.erroneous_text or "", err.suggested_correction or "", max_edits)
        for p_idx, (read_tok, intended_tok) in enumerate(pairs):
            cands.append(ArbitrationCandidate(
                candidate_id=f"{q_key}:{idx}:{p_idx}",
                error_index=idx,
                error_type=(err.error_type or "spelling").lower(),
                erroneous_text=err.erroneous_text or "",
                suggested_correction=err.suggested_correction or "",
                candidate_token=read_tok,
                intended_token=intended_tok,
                context_sentence=err.context_sentence or "",
                question_no=str(q_no) if q_no is not None else getattr(err, "question_no", None),
            ))
    return cands
