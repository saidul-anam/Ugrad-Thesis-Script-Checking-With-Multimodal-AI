"""
High-Stakes Token Guard: Verification and protection for mark-critical tokens.

Protects tokens where exact recognition directly determines awarded marks:
  1. Q6 Sentence Rearrangement: permutation of letters (a-j).
  2. Q1(A) Multiple Choice: valid Roman numerals ((i)-(v)).
  3. Rubric Tag Shield: excludes text within [struck: ...] and [unclear: ...]
     from linguistic error extraction and grading penalties.
"""

import re
from typing import List, Optional, Tuple, Set, Union, Dict

VALID_REARRANGEMENT_LETTERS: Set[str] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}

REARRANGEMENT_HOMOGLYPHS: dict = {
    "o": "j",
    "0": "j",
    "l": "i",
    "1": "i",
    "|": "i",
    "!": "i",
    "u": "v",
    "v": "u",
    "c": "e",
    "e": "c",
    "q": "g",
    "g": "q",
    "t": "f",
    "f": "t",
    "d": "o",
}


def sanitize_rearrangement_sequence(raw_seq: Union[str, List[str]]) -> Tuple[List[str], List[str]]:
    """
    Parse and validate sentence rearrangement sequence letters.
    Returns (cleaned_sequence, detected_anomalies).
    
    Handles arrow notation ($\rightarrow$, ->), commas, spaces.
    Applies multi-token OCR homoglyph resolution (e.g. 'o'/'0' -> 'j', 'l'/'1' -> 'i')
    and single-omission substitutions.
    """
    if isinstance(raw_seq, list):
        tokens = [str(x).strip().lower() for x in raw_seq if str(x).strip()]
    else:
        # Extract letter and digit tokens (handling OCR substitutions like 0, 1, |)
        tokens = re.findall(r'[a-zA-Z0-9|!]', str(raw_seq).lower())
    
    if not tokens:
        return [], ["no_letters_found"]

    anomalies: List[str] = []
    seen = {t for t in tokens if t in VALID_REARRANGEMENT_LETTERS}
    missing = [l for l in sorted(VALID_REARRANGEMENT_LETTERS) if l not in seen]

    # Step 1: Multi-token homoglyph mapping
    for i, tok in enumerate(tokens):
        if tok not in VALID_REARRANGEMENT_LETTERS:
            cand = REARRANGEMENT_HOMOGLYPHS.get(tok)
            if cand and cand in missing:
                tokens[i] = cand
                missing.remove(cand)
                anomalies.append(f"repaired_homoglyph:{tok}->{cand}")

    # Step 2: Single-substitution fallback if exactly 1 invalid token remains and 1 valid letter is missing
    seen_after = {t for t in tokens if t in VALID_REARRANGEMENT_LETTERS}
    invalid_after = [t for t in tokens if t not in VALID_REARRANGEMENT_LETTERS]
    missing_after = [l for l in sorted(VALID_REARRANGEMENT_LETTERS) if l not in seen_after]

    if len(tokens) == 10 and len(invalid_after) == 1 and len(missing_after) == 1:
        bad_tok = invalid_after[0]
        repair_tok = missing_after[0]
        tokens = [repair_tok if t == bad_tok else t for t in tokens]
        anomalies.append(f"repaired_substitution:{bad_tok}->{repair_tok}")
    elif invalid_after:
        anomalies.append(f"invalid_tokens:{','.join(invalid_after)}")

    return tokens, anomalies


def normalize_roman_numeral(token: str) -> Optional[str]:
    """Normalize OCR variations of Roman numerals (i)-(v)."""
    t = token.strip().lower().strip("()[].,")
    mapping = {
        "1": "i", "i": "i", "(i)": "i",
        "2": "ii", "ii": "ii", "11": "ii", "ll": "ii", "(ii)": "ii",
        "3": "iii", "iii": "iii", "111": "iii", "lll": "iii", "(iii)": "iii",
        "4": "iv", "iv": "iv", "lv": "iv", "1v": "iv", "(iv)": "iv",
        "5": "v", "v": "v", "(v)": "v",
    }
    return mapping.get(t)


def is_token_in_protected_tag(token: str, full_text: str) -> bool:
    """
    Check if a candidate token resides within [struck: ...], [unclear: ...], or [truncated].
    Protected tokens must never incur language mark deductions.
    """
    if not token or not full_text:
        return False

    # Find all protected spans
    protected_spans = []
    for m in re.finditer(r'\[(struck|unclear):([^\]]+)\]', full_text, re.IGNORECASE):
        protected_spans.append((m.start(), m.end(), m.group(2)))

    # Also find word[truncated] or [truncated: word]
    for m in re.finditer(r'(\b\w+)?\s*\[truncated(?::\s*([^\]]+))?\]', full_text, re.IGNORECASE):
        if m.group(1):
            protected_spans.append((m.start(), m.end(), m.group(1)))
        if m.group(2):
            protected_spans.append((m.start(), m.end(), m.group(2)))

    token_lower = token.strip().lower()
    for start, end, inner in protected_spans:
        if token_lower in inner.lower():
            return True

    return False


def filter_protected_errors(errors: List[dict], full_text: str) -> Tuple[List[dict], List[dict]]:
    """
    Filter out any linguistic error whose erroneous_text is inside a [struck:] or [unclear:] tag.
    Returns (retained_errors, protected_cleared_errors).
    """
    retained = []
    cleared = []

    for err in errors:
        err_txt = str(err.get("erroneous_text") or err.get("candidate_token") or "").strip()
        ctx = str(err.get("context_sentence") or "").strip()
        if is_token_in_protected_tag(err_txt, full_text) or is_token_in_protected_tag(err_txt, ctx):
            err["cleared_reason"] = "protected_tag"
            cleared.append(err)
        else:
            retained.append(err)

    return retained, cleared


def clean_rubric_answer(text: str) -> str:
    """
    Clean student answer text for rubric scoring by removing struck drafts
    while preserving paragraph structure and un-struck student writing.
    """
    if not text:
        return ""
    t = re.sub(r'\[struck:\s*[^\]]*\]', ' ', text, flags=re.IGNORECASE)
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in t.splitlines()]
    return "\n".join(lines).strip()

