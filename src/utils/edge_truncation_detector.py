"""
Deterministic Edge Truncation Detector & Right-Margin Clipper.

Identifies words and fragments physically cut off at the right edge of pages,
gutters, or camera photo boundaries:
1. Detects explicit '[truncated]' tags produced by Stage 1/2 VLM.
2. Identifies line-end tokens preceding line breaks or page breaks.
3. Stitches cross-line broken words (e.g., 'inten' + 'd' -> 'intend', 'p' + 'assing' -> 'passing').
4. Validates whether an extracted error is an edge-truncation artifact to suppress false penalties.
"""

import re
from typing import List, Set, Tuple, Optional, Dict, Any

_PUNCT_STRIP = re.compile(r"^[^\wঀ-৿]+|[^\wঀ-৿]+$", re.UNICODE)
_TRUNCATED_TAG_PATTERN = re.compile(r"\[truncated(?::\s*([^\]]+))?\]", re.IGNORECASE)


def extract_line_end_tokens(transcript: str) -> Dict[str, Any]:
    """
    Parse transcript into line-end tokens and tagged truncated words.
    
    Returns:
        {
            "line_end_tokens": set of lowercase tokens at the end of physical lines,
            "explicit_truncated": set of lowercase tokens explicitly tagged with [truncated],
            "raw_line_ends": list of raw last-token strings per line
        }
    """
    if not transcript:
        return {
            "line_end_tokens": set(),
            "explicit_truncated": set(),
            "raw_line_ends": []
        }

    line_end_tokens: Set[str] = set()
    explicit_truncated: Set[str] = set()
    raw_line_ends: List[str] = []

    # 1. Capture explicitly tagged [truncated] tokens anywhere in text
    for m in re.finditer(r'(\b[\wঀ-৿]+)?\s*\[truncated(?::\s*([^\]]+))?\]', transcript, re.IGNORECASE):
        prefix_word = m.group(1)
        inner_word = m.group(2)
        if prefix_word:
            explicit_truncated.add(prefix_word.lower().strip())
        if inner_word:
            explicit_truncated.add(inner_word.lower().strip())

    # 2. Extract right-edge (last) token of each line
    lines = transcript.splitlines()
    for line in lines:
        stripped = line.strip()
        # Skip empty lines, page breaks, exam headers, or markdown tables
        if not stripped:
            continue
        if stripped.startswith("---") or stripped.startswith("|") or stripped.startswith("#"):
            continue
        if re.match(r"(?im)^d?ans\s*:\s*(?:to\s+)?q(?:uestion)?", stripped):
            continue

        words = stripped.split()
        if not words:
            continue

        last_word_raw = words[-1]
        raw_line_ends.append(last_word_raw)

        # Check if last word had [truncated] attached
        if "[truncated" in last_word_raw.lower():
            clean_tag = re.sub(r'\[truncated(?::\s*[^\]]+)?\]', '', last_word_raw, flags=re.IGNORECASE)
            clean_tok = _PUNCT_STRIP.sub('', clean_tag).lower()
            if clean_tok:
                explicit_truncated.add(clean_tok)
                line_end_tokens.add(clean_tok)
        else:
            clean_tok = _PUNCT_STRIP.sub('', last_word_raw).lower()
            if clean_tok:
                line_end_tokens.add(clean_tok)

    return {
        "line_end_tokens": line_end_tokens,
        "explicit_truncated": explicit_truncated,
        "raw_line_ends": raw_line_ends
    }


def stitch_cross_line_truncations(text: str, lexicon: Set[str]) -> Tuple[str, List[Dict[str, str]]]:
    """
    Stitch words split across line breaks where line k ends with a truncated fragment
    and line k+1 starts with the continuation, forming a valid dictionary word.
    
    Examples:
      - 'Universi-\\nty' -> 'University'
      - 'Universi[truncated]\\nty' -> 'University'
      - 'inten\\nd to' -> 'intend to'
      - 'renewabl\\ne energy' -> 'renewable energy'
      - 'p\\nassing' -> 'passing'
    """
    if not text:
        return text, []

    diffs: List[Dict[str, str]] = []
    lexicon_lower = {w.lower() for w in lexicon}

    # 1. Hyphenated and explicitly tagged line wraps
    # e.g., 'Universi-[truncated]\nty' or 'Universi-\nty' or 'Universi[truncated]\nty'
    def _stitch_tagged_or_hyphenated(match: re.Match) -> str:
        part1 = match.group(1)
        part2 = match.group(2)
        combined = (part1 + part2).lower()
        if combined in lexicon_lower or (len(part1) >= 3 and len(part2) >= 2):
            diffs.append({"original": match.group(0), "stitched": part1 + part2})
            return part1 + part2
        return match.group(0)

    # Hyphen + optional [truncated]
    stitched = re.sub(
        r'(\b[a-zA-Z]{2,})-\s*(?:\[truncated\])?\s*\n\s*([a-zA-Z]{1,}\b)',
        _stitch_tagged_or_hyphenated,
        text,
        flags=re.IGNORECASE
    )

    # [truncated] without hyphen across newline
    stitched = re.sub(
        r'(\b[a-zA-Z]{2,})\s*\[truncated\]\s*\n\s*([a-zA-Z]{1,}\b)',
        _stitch_tagged_or_hyphenated,
        stitched,
        flags=re.IGNORECASE
    )

    # 2. Unhyphenated split words across newline: part1 on line k, part2 on line k+1
    # Only if concatenation strictly forms a valid English word in the lexicon
    def _stitch_unhyphenated_lexicon(match: re.Match) -> str:
        part1 = match.group(1)
        part2 = match.group(2)
        combined = (part1 + part2).lower()
        if combined in lexicon_lower and combined != part1.lower() and combined != part2.lower():
            diffs.append({"original": match.group(0), "stitched": part1 + part2})
            return part1 + part2
        return match.group(0)

    stitched = re.sub(
        r'(\b[a-zA-Z]{2,})\s*\n\s*([a-zA-Z]{1,}\b)',
        _stitch_unhyphenated_lexicon,
        stitched
    )

    return stitched, diffs


def is_right_edge_truncation(
    erroneous_text: str,
    suggested_correction: str,
    context_sentence: str = "",
    transcript: Optional[str] = None,
    lexicon: Optional[Set[str]] = None
) -> bool:
    """
    Check if an extracted error is a right-edge / margin truncation artifact.
    
    True if:
    1. erroneous_text is tagged [truncated] or contains [truncated].
    2. erroneous_text is located at the right edge of a physical line in transcript or context,
       AND:
       - suggested_correction starts with erroneous_text (prefix match: e.g. 'renewabl' -> 'renewable',
         'wor' -> 'words', 'pro' -> 'produced', 'villa' -> 'village', 'organi' -> 'organization'), OR
       - erroneous_text is a near-prefix (within 1 char edit, e.g. 'becon' -> 'become'), OR
       - erroneous_text is a short line-end fragment (<= 3 chars, e.g. 'pro', 'co', 'ar', 'w', 'p', 't')
         that was corrected to a full word.
    """
    err_raw = (erroneous_text or "").strip()
    corr_raw = (suggested_correction or "").strip()
    if not err_raw or not corr_raw:
        return False

    # 1. Explicit [truncated] tag in erroneous text or context
    if "[truncated" in err_raw.lower() or "[truncated" in context_sentence.lower():
        return True

    err_clean = _PUNCT_STRIP.sub('', err_raw).lower()
    corr_clean = _PUNCT_STRIP.sub('', corr_raw).lower()
    if not err_clean or not corr_clean:
        return False

    # A full identical word is not a truncation
    if err_clean == corr_clean:
        return False

    # 2. Check position: Is err_clean at the right edge of a physical line?
    is_at_line_end = False

    # Regex for a word at the end of a line (followed only by optional punctuation/spaces and newline or end of string)
    line_end_regex = re.compile(rf'\b{re.escape(err_clean)}[^\w\s]*\s*(?:\n|\r|---|\Z)', re.IGNORECASE)

    if context_sentence and transcript:
        # Find context in transcript to see physical line breaks for THIS specific instance
        ctx_clean = " ".join(context_sentence.split()).lower()
        # Look for the sentence in transcript
        for line in transcript.splitlines():
            line_str = line.strip()
            # If this line ends with err_clean and shares words with context_sentence
            if re.search(rf'\b{re.escape(err_clean)}[^\w\s]*$', line_str, re.IGNORECASE):
                # Check if this line is part of the context sentence
                line_words = set(_PUNCT_STRIP.sub('', w).lower() for w in line_str.split())
                ctx_words = set(_PUNCT_STRIP.sub('', w).lower() for w in context_sentence.split())
                if len(line_words & ctx_words) >= min(2, len(line_words)):
                    is_at_line_end = True
                    break
        if not is_at_line_end:
            # Check if erroneous_text was in the middle of a line in transcript
            # (e.g. "... renewabl energy ...")
            if re.search(rf'\b{re.escape(err_clean)}\s+[a-zA-Z]', transcript, re.IGNORECASE):
                # If context sentence also shows it mid-line, it is definitely mid-line
                if re.search(rf'\b{re.escape(err_clean)}\s+[a-zA-Z]', context_sentence, re.IGNORECASE):
                    return False
    elif transcript:
        end_info = extract_line_end_tokens(transcript)
        if err_clean in end_info["explicit_truncated"]:
            return True
        if err_clean in end_info["line_end_tokens"]:
            is_at_line_end = True
        elif line_end_regex.search(transcript):
            is_at_line_end = True
        else:
            return False

    # Check context sentence line boundaries if newline is present
    if not is_at_line_end and context_sentence:
        if line_end_regex.search(context_sentence):
            is_at_line_end = True
        elif context_sentence.strip().lower().endswith(err_clean) or context_sentence.strip().lower().endswith(err_raw.lower()):
            is_at_line_end = True

    # Known classic margin break stems (even without transcript line coordinates)
    KNOWN_TRUNCATION_STEMS = {
        "universi", "organi", "technolo", "renewabl", "environ",
        "intellig", "diffic", "communi", "satisf"
    }
    if err_clean in KNOWN_TRUNCATION_STEMS and corr_clean.startswith(err_clean):
        return True

    # 3. Linguistic Prefix & Morphology Check
    if is_at_line_end:
        # Case A: Exact prefix (e.g. 'renewabl' -> 'renewable', 'wor' -> 'words', 'villa' -> 'village', 'pro' -> 'produced')
        if corr_clean.startswith(err_clean):
            return True

        # Case B: Near-prefix where final letter is slightly ambiguous/distorted before edge
        # e.g., 'becon' -> 'become' (stem 'beco' matches, len >= 4, last char 'n' vs 'm' ligature at boundary)
        if len(err_clean) >= 4 and len(corr_clean) >= 4:
            stem_len = min(len(err_clean) - 1, len(corr_clean) - 1)
            if stem_len >= 3 and err_clean[:stem_len] == corr_clean[:stem_len]:
                return True

        # Case C: Short fragments at the line end (<= 3 chars) that the LLM corrected to a full word
        # e.g., 'pro' -> 'produced', 'co' -> 'comes' or 'coal', 'ar' -> 'are', 'w' -> 'which', 'p' -> 'passing'
        if len(err_clean) <= 3 and len(corr_clean) >= len(err_clean):
            if corr_clean.startswith(err_clean):
                return True
            if len(err_clean) == 1 and corr_clean.startswith(err_clean):
                return True

    return False
