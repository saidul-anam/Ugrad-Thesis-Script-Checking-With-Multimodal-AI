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


def sanitize_rearrangement_sequence(
    raw_seq: Union[str, List[str]],
    expected_letters: Optional[Set[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Parse and validate sentence rearrangement sequence letters.
    Returns (cleaned_sequence, detected_anomalies).
    
    Supports dynamic expected letter sets derived from question key / schema,
    handling arrow notation, commas, spaces, multi-token OCR homoglyphs, and single substitutions.
    """
    valid_set = {str(l).strip().lower() for l in expected_letters} if expected_letters else VALID_REARRANGEMENT_LETTERS

    if isinstance(raw_seq, list):
        tokens = [str(x).strip().lower() for x in raw_seq if str(x).strip()]
    else:
        tokens = re.findall(r'[a-zA-Z0-9|!]', str(raw_seq).lower())
    
    if not tokens:
        return [], ["no_letters_found"]

    anomalies: List[str] = []
    seen = {t for t in tokens if t in valid_set}
    missing = [l for l in sorted(valid_set) if l not in seen]

    # Step 1: Multi-token homoglyph mapping (only if token is not in valid_set and candidate is missing)
    for i, tok in enumerate(tokens):
        if tok not in valid_set:
            cand = REARRANGEMENT_HOMOGLYPHS.get(tok)
            # Guard: only substitute if candidate letter is legitimately missing from sequence
            if cand and cand in missing:
                tokens[i] = cand
                missing.remove(cand)
                anomalies.append(f"repaired_homoglyph:{tok}->{cand}")

    # Step 2: Single-substitution fallback if exactly 1 invalid token remains and 1 valid letter is missing
    seen_after = {t for t in tokens if t in valid_set}
    invalid_after = [t for t in tokens if t not in valid_set]
    missing_after = [l for l in sorted(valid_set) if l not in seen_after]

    if len(tokens) == len(valid_set) and len(invalid_after) == 1 and len(missing_after) == 1:
        bad_tok = invalid_after[0]
        repair_tok = missing_after[0]
        tokens = [repair_tok if t == bad_tok else t for t in tokens]
        anomalies.append(f"repaired_substitution:{bad_tok}->{repair_tok}")
    elif invalid_after:
        anomalies.append(f"invalid_tokens:{','.join(invalid_after)}")

    return tokens, anomalies


# Canonical Roman numerals from 1 to 20
_INT_TO_ROMAN = {
    1: "i", 2: "ii", 3: "iii", 4: "iv", 5: "v",
    6: "vi", 7: "vii", 8: "viii", 9: "ix", 10: "x",
    11: "xi", 12: "xii", 13: "xiii", 14: "xiv", 15: "xv",
    16: "xvi", 17: "xvii", 18: "xviii", 19: "xix", 20: "xx"
}
_ROMAN_TO_INT = {v: k for k, v in _INT_TO_ROMAN.items()}

# OCR letter-to-numeral confusions
_OCR_ROMAN_HOMOGLYPHS = {
    "1": "i", "l": "i", "|": "i", "!": "i",
    "11": "ii", "ll": "ii", "l1": "ii", "1l": "ii",
    "111": "iii", "lll": "iii",
    "1v": "iv", "lv": "iv", "|v": "iv",
    "v1": "vi", "vl": "vi", "v|": "vi",
    "v11": "vii", "vll": "vii",
    "v111": "viii", "vlll": "viii",
    "1x": "ix", "lx": "ix", "|x": "ix",
    "x1": "xi", "xl": "xi",
    "x11": "xii", "xll": "xii",
}


def normalize_roman_numeral(token: str) -> Optional[str]:
    """
    Normalize OCR variations and digits to lowercase Roman numerals (i)-(xx).
    Handles parens, brackets, OCR letter slips (e.g. 1/l/| -> i, lv -> iv).
    """
    t = token.strip().lower().strip("()[].,")
    if not t:
        return None

    # Direct Roman match
    if t in _ROMAN_TO_INT:
        return t

    # OCR homoglyph match
    if t in _OCR_ROMAN_HOMOGLYPHS:
        return _OCR_ROMAN_HOMOGLYPHS[t]

    # Digit match e.g. "1" -> "i", "5" -> "v", "10" -> "x"
    if t.isdigit():
        val = int(t)
        if val in _INT_TO_ROMAN:
            return _INT_TO_ROMAN[val]

    return None


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

