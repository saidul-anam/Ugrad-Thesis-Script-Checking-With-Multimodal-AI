"""
Strikethrough Collision & False-Start Stutter Resolver.

Detects untagged student cross-outs and cancelled draft collisions that the VLM
missed during optical OCR (e.g. 'are are', 'by for giving', 'increse incre').
Encloses the superseded draft in `[struck: ...]` so that downstream rubric evaluation
(Stage 4) and error analysis (Stage 3) do not unfairly penalize the student.
"""

import re
from typing import List, Dict, Any, Tuple, Set, Optional

# Prepositions that never legitimately appear consecutively without punctuation/conjunction
COMMON_PREPOSITIONS = {
    "by", "for", "to", "in", "on", "at", "with", "from", "of", "into", "about", "through"
}

# Legitimate adjacent preposition pairs in English that should NOT be treated as collisions
LEGITIMATE_PREP_PAIRS = {
    ("out", "of"),
    ("as", "to"),
    ("as", "for"),
    ("up", "to"),
    ("in", "to"),
    ("on", "to"),
    ("from", "to"),
    ("because", "of"),
    ("instead", "of"),
    ("according", "to"),
    ("due", "to"),
    ("prior", "to"),
    ("next", "to"),
    ("close", "to"),
    ("about", "to"),
    ("down", "to"),
}

# Legitimate duplicate word phrases or structural headers in English exams
LEGITIMATE_DUPLICATES = {
    "had",      # "he had had enough"
    "that",     # "know that that is true"
    "answer",   # "Answer to Q. No. 1 \n Answer:"
    "question", # "Question: Question 1"
}

# Legitimate consecutive words sharing a prefix
LEGITIMATE_PREFIX_PAIRS = {
    ("thin", "thing"),
    ("some", "someone"),
    ("some", "somebody"),
    ("every", "everyone"),
    ("every", "everybody"),
    ("with", "within"),
    ("with", "without"),
}


def resolve_strikethrough_collisions(
    text: str,
    custom_protected: Optional[Set[str]] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Scan transcript text for untagged strikethrough collisions and false-start stutters.
    Returns:
        (resolved_text, list_of_diffs)
    """
    if not text:
        return text, []

    diffs: List[Dict[str, Any]] = []
    resolved = text

    # Helper to check if a span is already inside [struck: ...]
    def is_inside_struck(pos: int, target_text: str) -> bool:
        last_open = target_text.rfind("[struck:", 0, pos)
        if last_open != -1:
            next_close = target_text.find("]", last_open)
            if next_close == -1 or next_close > pos:
                return True
        return False

    # -------------------------------------------------------------------------
    # 1. Duplicate Word Stutters (e.g. "childs are are poor" -> "childs [struck: are] are poor")
    # -------------------------------------------------------------------------
    def _replace_duplicate_stutter(match: re.Match) -> str:
        word = match.group(1)
        full_match = match.group(0)

        # Check if word is protected
        if word.lower() in LEGITIMATE_DUPLICATES:
            return full_match
        if custom_protected and word.lower() in custom_protected:
            return full_match
        
        # Determine the two actual words (preserving case)
        parts = full_match.split()
        w1 = parts[0]
        w2 = parts[1]
        
        diffs.append({
            "type": "duplicate_word_stutter",
            "original": full_match,
            "superseded": w1,
            "kept": w2,
            "resolved": f"[struck: {w1}] {w2}"
        })
        return f"[struck: {w1}] {w2}"

    # Use backreference \1 to match repeated word case-insensitively
    resolved = re.sub(
        r'\b([a-zA-Z]{2,})\s+\1\b',
        _replace_duplicate_stutter,
        resolved,
        flags=re.IGNORECASE
    )

    # -------------------------------------------------------------------------
    # 2. Contradictory Adjacent Preposition Collisions (e.g. "use AI by for giving")
    # -------------------------------------------------------------------------
    def _replace_preposition_collision(match: re.Match) -> str:
        prep1 = match.group(1)
        prep2 = match.group(2)
        full_match = match.group(0)

        p1_low = prep1.lower()
        p2_low = prep2.lower()

        if (p1_low, p2_low) in LEGITIMATE_PREP_PAIRS:
            return full_match

        if p1_low in COMMON_PREPOSITIONS and p2_low in COMMON_PREPOSITIONS and p1_low != p2_low:
            diffs.append({
                "type": "preposition_collision",
                "original": full_match,
                "superseded": prep1,
                "kept": prep2,
                "resolved": f"[struck: {prep1}] {prep2}"
            })
            return f"[struck: {prep1}] {prep2}"
        return full_match

    # Match adjacent single prepositions separated by normal spacing on the same line
    resolved = re.sub(
        r'\b(' + '|'.join(COMMON_PREPOSITIONS) + r')[ \t]{1,2}(' + '|'.join(COMMON_PREPOSITIONS) + r')\b',
        _replace_preposition_collision,
        resolved,
        flags=re.IGNORECASE
    )

    # -------------------------------------------------------------------------
    # 3. Known Exam Stutter Patterns
    # -------------------------------------------------------------------------
    # Pattern 3a: "means a it is a" -> "means [struck: a] it is a"
    resolved_3a = re.sub(
        r'\b(means|signifies|indicates)\s+([a-zA-Z])\s+(it\s+is)\b',
        r'\1 [struck: \2] \3',
        resolved,
        flags=re.IGNORECASE
    )
    if resolved_3a != resolved:
        diffs.append({
            "type": "verb_determiner_stutter",
            "original": "means a it is",
            "resolved": "means [struck: a] it is"
        })
        resolved = resolved_3a

    # Pattern 3b: Fragment-to-Word collisions (e.g. "increse incre" -> "[struck: increse] incre")
    def _replace_fragment_stutter(match: re.Match) -> str:
        frag = match.group(1)
        word = match.group(2)
        full_match = match.group(0)

        f_low = frag.lower()
        w_low = word.lower()

        if (f_low, w_low) in LEGITIMATE_PREFIX_PAIRS:
            return full_match

        # If f_low and w_low share prefix >= 4 chars, and are not legitimate separate words
        if len(f_low) >= 4 and len(w_low) >= 4 and w_low.startswith(f_low[:4]):
            diffs.append({
                "type": "fragment_stutter",
                "original": full_match,
                "superseded": frag,
                "kept": word,
                "resolved": f"[struck: {frag}] {word}"
            })
            return f"[struck: {frag}] {word}"
        return full_match

    resolved = re.sub(
        r'\b([a-zA-Z]{4,})\s+([a-zA-Z]{4,})\b',
        lambda m: _replace_fragment_stutter(m) if m.group(1).lower() != m.group(2).lower() and m.group(2).lower().startswith(m.group(1).lower()[:4]) else m.group(0),
        resolved
    )

    return resolved, diffs
