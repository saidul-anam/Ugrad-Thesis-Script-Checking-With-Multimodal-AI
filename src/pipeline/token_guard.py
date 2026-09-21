"""
High-Stakes Token Guard: Verification and protection for mark-critical tokens.

Protects tokens where exact recognition directly determines awarded marks:
  1. Q6 Sentence Rearrangement: permutation of letters (a-j).
  2. Q1(A) Multiple Choice: valid Roman numerals ((i)-(v)).
  3. Rubric Tag Shield: excludes text within [struck: ...] and [unclear: ...]
     from linguistic error extraction and grading penalties.
"""

import re
from typing import List, Optional, Tuple, Set

VALID_REARRANGEMENT_LETTERS: Set[str] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
VALID_ROMAN_NUMERALS: Set[str] = {"i", "ii", "iii", "iv", "v"}


def sanitize_rearrangement_sequence(raw_seq: str) -> Tuple[List[str], List[str]]:
    """
    Parse and validate sentence rearrangement sequence letters.
    Returns (cleaned_sequence, detected_anomalies).
    
    Handles arrow notation ($\rightarrow$, ->), commas, spaces.
    If 9 of 10 letters are valid and an invalid token (e.g. 'o' instead of 'j') is present,
    repairs the missing letter.
    """
    # Extract letter tokens
    tokens = re.findall(r'\b[a-zA-Z]\b', raw_seq.lower())
    if not tokens:
        return [], ["no_letters_found"]

    anomalies = []
    # Check if this looks like a 10-item rearrangement attempt
    if len(tokens) == 10:
        seen = set(tokens)
        invalid = [t for t in tokens if t not in VALID_REARRANGEMENT_LETTERS]
        missing = [l for l in VALID_REARRANGEMENT_LETTERS if l not in seen]

        # Single substitution repair (e.g. 'o' for 'j' on script 0002)
        if len(invalid) == 1 and len(missing) == 1:
            bad_tok = invalid[0]
            repair_tok = missing[0]
            tokens = [repair_tok if t == bad_tok else t for t in tokens]
            anomalies.append(f"repaired_substitution:{bad_tok}->{repair_tok}")
        elif invalid:
            anomalies.append(f"invalid_tokens:{','.join(invalid)}")

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
