"""
Split Token Stitcher: Automated Pen-Lift & Syllable Separation Merger.

Students frequently pause their pens or leave unintended spaces inside long words:
  - "elec tricuity" / "elec tricity" -> "electricity"
  - "pro blems" -> "problems"
  - "sec tory" -> "sectors"
  - "day time" -> "daytime"
  - "non- formal" -> "non-formal"
  - "Hy dro - electric" -> "Hydro-electric"

This module detects and stitches these split tokens before Stage 3 linguistic evaluation,
eliminating false spelling and syntax errors without human intervention.
"""

import re
from typing import List, Tuple, Set, Optional, Dict, Any


_PUNCT_SPLIT = re.compile(r"^([^\wঀ-৿]*)([\wঀ-৿]+(?:-[\wঀ-৿]+)*)([^\wঀ-৿]*)$", re.UNICODE)

# Common short function words that should NEVER be stitched when both are valid words
_STOP_WORDS = {
    "to", "in", "on", "at", "is", "as", "it", "he", "we", "an", "by", "or", "so",
    "my", "me", "us", "no", "if", "do", "up", "be", "am", "the", "and", "for", "of"
}


def _clean_word(token: str) -> Tuple[str, str, str]:
    """Return (leading_punct, clean_word, trailing_punct)."""
    m = _PUNCT_SPLIT.match(token)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return "", token, ""


def stitch_pen_lift_splits(
    text: str,
    lexicon: Set[str],
    question_vocab: Optional[Set[str]] = None,
    allograph_map: Optional[Dict[str, str]] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Scan text for intra-word pen-lift splits and merge them into canonical words.
    
    Returns:
        (stitched_text, diff_records)
    """
    if not text:
        return text, []

    q_vocab = {w.lower() for w in (question_vocab or set())}
    combined_vocab = lexicon | q_vocab
    diffs: List[Dict[str, Any]] = []

    # 1. First Pass: Fix broken hyphen compounds like "non- formal" -> "non-formal", "Hy dro - electric"
    def _fix_hyphens(match: re.Match) -> str:
        w1, w2 = match.group(1), match.group(2)
        diffs.append({
            "original": match.group(0),
            "stitched": f"{w1}-{w2}",
            "reason": "broken hyphen compound"
        })
        return f"{w1}-{w2}"

    cleaned_text = re.sub(r'\b([A-Za-z]+)\s*-\s+([A-Za-z]+)\b', _fix_hyphens, text)
    cleaned_text = re.sub(r'\b([A-Za-z]+)\s+-\s*([A-Za-z]+)\b', _fix_hyphens, cleaned_text)

    # 2. Syllable and Pen-Lift Split Detection across lines and spaces
    lines = cleaned_text.split("\n")
    processed_lines = []

    for line in lines:
        stripped_line = line.strip()
        # Skip markdown table rows, page breaks, or question headers
        if not stripped_line or stripped_line.startswith("|") or stripped_line.startswith("---") or stripped_line.startswith("Ans to"):
            processed_lines.append(line)
            continue

        tokens = line.split(" ")
        new_tokens = []
        i = 0
        while i < len(tokens):
            cur_tok = tokens[i]
            if not cur_tok.strip():
                new_tokens.append(cur_tok)
                i += 1
                continue

            if i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                if next_tok.strip():
                    p1_lead, w1, p1_trail = _clean_word(cur_tok)
                    p2_lead, w2, p2_trail = _clean_word(next_tok)

                    # Only evaluate if neither token has hard sentence-terminating punctuation between them
                    if w1 and w2 and not p1_trail.endswith((".", "!", "?", ":", ";", ")", "]")):
                        w1_low = w1.lower()
                        w2_low = w2.lower()

                        # Apply allograph mapping if provided (e.g. 'y' -> 's' on terminal syllables like 'sectory' -> 'sectors')
                        w2_mapped = w2_low
                        if allograph_map and "terminal_y" in allograph_map and w2_low.endswith("y") and not w2_low.endswith("ly"):
                            w2_mapped = w2_low[:-1] + "s"

                        # Syllable fix for cursive 'ri' read as 'u' (e.g. tricuity -> tricity)
                        if "tricuity" in w2_low:
                            w2_mapped = w2_low.replace("tricuity", "tricity")

                        cand_merged = w1_low + w2_low
                        cand_mapped = w1_low + w2_mapped

                        is_split = False
                        merged_word = ""

                        # Condition A: cand_merged or cand_mapped is a valid dictionary/domain word
                        cand_no_hyphen = cand_merged.replace("-", "")
                        if (cand_merged in combined_vocab or cand_no_hyphen in combined_vocab) and len(cand_merged) >= 4:
                            # If both w1 and w2 are already valid words, prevent merging common short phrases (e.g. "he is", "to do")
                            if w1_low in combined_vocab and w2_low in combined_vocab and "-" not in cand_merged:
                                if w1_low not in _STOP_WORDS and w2_low not in _STOP_WORDS:
                                    is_split = True
                                    merged_word = cand_merged
                            else:
                                is_split = True
                                merged_word = cand_merged
                        elif cand_mapped in combined_vocab and len(cand_mapped) >= 4:
                            is_split = True
                            merged_word = cand_mapped

                        if is_split:
                            # Preserve casing: Capitalize if first word was capitalized
                            final_word = merged_word.capitalize() if w1[:1].isupper() else merged_word
                            stitched_token = f"{p1_lead}{final_word}{p2_trail}"
                            diffs.append({
                                "original": f"{cur_tok} {next_tok}",
                                "stitched": stitched_token,
                                "reason": f"pen-lift syllable split ('{w1}' + '{w2}' -> '{final_word}')"
                            })
                            new_tokens.append(stitched_token)
                            i += 2
                            continue

            new_tokens.append(cur_tok)
            i += 1

        processed_lines.append(" ".join(new_tokens))

    return "\n".join(processed_lines), diffs
