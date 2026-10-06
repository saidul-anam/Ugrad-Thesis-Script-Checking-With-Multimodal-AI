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

    # 1c. Multi-Word Phrase Duplicate Stutters (e.g. "gipsy men gipsy men" -> "[struck: gipsy men] gipsy men")
    def _replace_phrase_duplicate_stutter(match: re.Match) -> str:
        full_match = match.group(0)
        p1 = match.group(1)
        p2 = match.group(2)
        diffs.append({
            "type": "duplicate_phrase_stutter",
            "original": full_match,
            "superseded": p1,
            "kept": p2,
            "resolved": f"[struck: {p1}] {p2}"
        })
        return f"[struck: {p1}] {p2}"

    resolved = re.sub(
        r'\b([a-zA-Z]{3,}\s+[a-zA-Z]{3,})\s+(\1)\b',
        _replace_phrase_duplicate_stutter,
        resolved,
        flags=re.IGNORECASE
    )

    # 1b. Itemized Draft Stutters (e.g. "d) wordA ... wordA" -> "d) [struck: wordA ...] wordA")
    def _replace_itemized_stutter(match: re.Match) -> str:
        prefix = match.group(1)
        word1 = match.group(2)
        middle = match.group(3)
        word2 = match.group(4)
        if word1.lower() == word2.lower() and "[struck:" not in match.group(0):
            resolved_line = f"{prefix}[struck: {word1} {middle}] {word2}"
            diffs.append({
                "type": "itemized_draft_stutter",
                "original": match.group(0),
                "superseded": f"{word1} {middle}",
                "kept": word2,
                "resolved": resolved_line
            })
            return resolved_line
        return match.group(0)

    resolved = re.sub(
        r'(?m)^(\s*[a-jA-J]\)\s+)([a-zA-Z]{3,})\s+([a-zA-Z\s]+?)\s+([a-zA-Z]{3,})\s*$',
        _replace_itemized_stutter,
        resolved
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
        lambda m: _replace_fragment_stutter(m) if m.group(1).lower() != m.group(2).lower() and (m.group(2).lower().startswith(m.group(1).lower()[:4]) or m.group(1).lower().startswith(m.group(2).lower()[:4])) else m.group(0),
        resolved
    )

    # Pattern 3c: Orphaned determiner before subject pronoun clause (e.g. "means a it is" -> "means [struck: a] it is")
    def _replace_determiner_pronoun_stutter(match: re.Match) -> str:
        lead = match.group(1)
        det = match.group(2)
        clause = match.group(3)
        diffs.append({
            "type": "determiner_pronoun_stutter",
            "original": match.group(0),
            "superseded": det,
            "kept": clause,
            "resolved": f"{lead}[struck: {det}] {clause}"
        })
        return f"{lead}[struck: {det}] {clause}"

    resolved = re.sub(
        r'(\b\w+\s+)\b([aA]|an|the)\s+(it\s+is|he\s+is|she\s+is|they\s+are|we\s+are)\b',
        _replace_determiner_pronoun_stutter,
        resolved,
        flags=re.IGNORECASE
    )

    # Pattern 3d: Short non-word prefix stutter (e.g. "po Portia", "di decisions", "ha had")
    common_short_words = {
        "to", "in", "on", "at", "by", "of", "an", "as", "is", "it", "so", "no", "my", "we", "he", "do", "go", "if", "up", "me", "us", "am", "or"
    }

    def _replace_short_prefix_stutter(match: re.Match) -> str:
        frag = match.group(1)
        word = match.group(2)
        f_low = frag.lower()
        w_low = word.lower()
        if f_low in common_short_words or f_low in LEGITIMATE_DUPLICATES:
            return match.group(0)
        if len(w_low) >= 3 and w_low.startswith(f_low):
            diffs.append({
                "type": "short_prefix_stutter",
                "original": match.group(0),
                "superseded": frag,
                "kept": word,
                "resolved": f"[struck: {frag}] {word}"
            })
            return f"[struck: {frag}] {word}"
        return match.group(0)

    resolved = re.sub(
        r'\b([a-zA-Z]{2,3})\s+([a-zA-Z]{3,})\b',
        _replace_short_prefix_stutter,
        resolved
    )

    # -------------------------------------------------------------------------
    # 4. Multi-Line Draft Restart / Narrative Retake Detection
    # -------------------------------------------------------------------------
    # When a student writes an opening draft of a story/answer (2-6 lines), crosses it out,
    # and restarts the story/answer below with an identical or near-identical opening line:
    lines = resolved.split("\n")
    if len(lines) >= 6:
        for i in range(len(lines)):
            l1 = lines[i].strip()
            if not l1 or l1.startswith("[struck:") or l1.lower().startswith("answer") or l1.lower().startswith("ans"):
                continue
            words1 = re.findall(r'\b[a-zA-Z]+\b', l1)
            if len(words1) < 4:
                continue

            for j in range(i + 2, min(len(lines), i + 12)):
                l2 = lines[j].strip()
                if not l2 or l2.startswith("[struck:"):
                    continue
                words2 = re.findall(r'\b[a-zA-Z]+\b', l2)
                if len(words2) < 4:
                    continue

                # Check if first 4-5 words match (allowing minor inflection like live vs lived)
                match_cnt = 0
                for w_a, w_b in zip(words1[:5], words2[:5]):
                    if w_a.lower() == w_b.lower() or (len(w_a) >= 4 and len(w_b) >= 4 and w_a[:4].lower() == w_b[:4].lower()):
                        match_cnt += 1
                    else:
                        break

                if match_cnt >= 4:
                    # Found draft restart! Lines from i up to j (stopping at empty line or title before j) form initial draft.
                    block_end = i + 1
                    while block_end < j and lines[block_end].strip() and not lines[block_end].strip().lower().startswith("answer"):
                        block_end += 1

                    modified_any = False
                    for k in range(i, block_end):
                        curr_line = lines[k].strip()
                        if curr_line and not curr_line.startswith("[struck:") and not curr_line.lower().startswith("answer"):
                            diffs.append({
                                "type": "draft_restart_multiline_block",
                                "original": curr_line,
                                "superseded": curr_line,
                                "kept": f"restarted at line {j+1}",
                                "resolved": f"[struck: {curr_line}]"
                            })
                            lines[k] = f"[struck: {curr_line}]"
                            modified_any = True
                    if modified_any:
                        resolved = "\n".join(lines)
                    break

    return resolved, diffs


COMMON_ENGLISH_SHORT_FUNCTION_WORDS = {
    "to", "in", "on", "at", "by", "of", "an", "as", "is", "it", "so", "no", "my",
    "we", "he", "do", "go", "if", "up", "me", "us", "am", "or", "and", "the"
}


def is_student_prefix_fragment(token: str) -> bool:
    """
    Returns True if a token appears to be an aborted prefix stutter / false start (e.g. 'po', 'di', 'ha', 'fr', 'inste')
    rather than a complete grammatical English word.
    Authentic prefix false starts should always remain marked as [struck: ...].
    """
    clean = token.lower().strip()
    if len(clean) <= 1:
        return True
    if len(clean) == 2 and clean not in COMMON_ENGLISH_SHORT_FUNCTION_WORDS:
        # Non-standard 2-letter syllables like "po", "di", "ha", "fr", "ea"
        return True
    if clean.endswith(("'", "`")) or clean in {"inste", "autocran", "anim", "pres", "recoo"}:
        return True
    return False


def ground_and_reconcile_strikethroughs(
    text: str,
    strikethrough_regions: Optional[List[Any]] = None,
    strikethrough_blocks: Optional[List[Any]] = None,
    underlines: Optional[List[Any]] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Spatially ground and reconcile transcript strikethroughs using Stage 0 optical detection.
    
    1. Unwraps false-positive strikes on natural crossbar words (e.g. '[struck: cut]' -> 'cut')
       caused by cursive 't' or 'f' ascenders when no Stage 0 strikethrough exists on that line.
    2. Unwraps false strikes on rearrangement question indices (e.g. '(a + iv + ii)').
    3. Snaps untagged draft paragraphs to '[struck: ...]' when Stage 0 detected a multi-line block.
    4. Unwraps false strikes on question headers where an underline was present.
    """
    if not text:
        return text, []

    diffs: List[Dict[str, Any]] = []
    lines = text.split("\n")
    n_lines = max(1, len(lines))

    # Helper: Check if an optical strike exists in the vertical neighborhood of a line
    def has_optical_strike_near(y_pct: float, tol: float = 7.0) -> bool:
        if strikethrough_regions:
            for r in strikethrough_regions:
                if not getattr(r, "is_underline", False):
                    ry = getattr(r, "y_pct", 0.0)
                    ry2 = getattr(r, "y2_pct", ry)
                    if (ry - tol) <= y_pct <= (ry2 + tol):
                        return True
        if strikethrough_blocks:
            for b in strikethrough_blocks:
                by = getattr(b, "y_pct", 0.0)
                by2 = getattr(b, "y2_pct", by)
                if (by - tol) <= y_pct <= (by2 + tol):
                    return True
        return False

    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        line_y_pct = (float(idx) + 0.5) / float(n_lines) * 100.0

        # ---------------------------------------------------------------------
        # 1. False Strikethrough Elimination (Cursive 't'/'f' Crossbar Rule)
        # ---------------------------------------------------------------------
        def _unwrap_false_crossbar_strike(match: re.Match) -> str:
            full = match.group(0)
            inner = match.group(1).strip()
            # If the segment contains ascender crossbar letters ('t' or 'f') and is NOT a student's aborted prefix fragment
            has_crossbar_letter = bool(re.search(r'[tf]', inner, re.IGNORECASE))
            if has_crossbar_letter and not is_student_prefix_fragment(inner):
                # Verify whether an actual optical pen stroke exists in this line's vertical neighborhood
                if not has_optical_strike_near(line_y_pct, tol=6.0):
                    diffs.append({
                        "type": "false_t_bar_strike_unwrapped",
                        "original": full,
                        "resolved": inner,
                        "line_idx": idx,
                        "line_y_pct": round(line_y_pct, 1)
                    })
                    return inner
            return full

        line = re.sub(r'\[struck:\s*([a-zA-Z\s]+?)\]', _unwrap_false_crossbar_strike, line)

        # ---------------------------------------------------------------------
        # 2. Matching Question Index & Roman Numeral Protection
        # ---------------------------------------------------------------------
        # In Question 6/7 rearrangement/matching tables, students write tuples like:
        # '(a + iv + ii)', '1 | 2 | 3', 'i - b - iii', or column references.
        # If the VLM falsely wrapped these in [struck: ...], unwrap them.
        def _unwrap_rearrangement_index(match: re.Match) -> str:
            full = match.group(0)
            inner = match.group(1).strip()
            # Split by common formula/table punctuation and operators
            tokens = [t for t in re.split(r'[\s\+\-\=\/\|\(\)\[\]\{\}\:\.\,]+', inner) if t]
            # Matching tuples consist of 2+ short tokens (single letters, digits, roman numerals)
            is_matching_tuple = len(tokens) >= 2 and all(len(t) <= 3 for t in tokens)
            has_substantive_word = any(len(w) >= 4 and w.lower() not in {"ans", "answer"} for w in inner.split())
            if is_matching_tuple and not has_substantive_word:
                diffs.append({
                    "type": "question_index_unwrapped",
                    "original": full,
                    "resolved": inner,
                    "line_idx": idx
                })
                return inner
            return full

        line = re.sub(r'\[struck:\s*([^\]]+?)\]', _unwrap_rearrangement_index, line)

        # ---------------------------------------------------------------------
        # 3. Underline Header Protection
        # ---------------------------------------------------------------------
        if re.search(r'\[struck:\s*(?:ans|answer|question)\b', line, flags=re.IGNORECASE):
            unwrapped = re.sub(r'\[struck:\s*([^\]]+)\]', r'\1', line)
            diffs.append({
                "type": "underlined_header_unwrapped",
                "original": line,
                "resolved": unwrapped
            })
            line = unwrapped

        # ---------------------------------------------------------------------
        # 3b. Strikethrough Tag Inversion Disambiguation (Problem P4)
        # ---------------------------------------------------------------------
        # When a student strikes wordA and immediately writes replacement wordB,
        # the VLM may tag wordB instead: 'wordA [struck: wordB]'.
        # If Stage 0 optical detection shows the physical horizontal stroke covers
        # wordA and NOT wordB, invert the tags: '[struck: wordA] wordB'.
        if strikethrough_regions and "[struck:" in line:
            for m in list(re.finditer(r'(\b[\w\'-]+)\s+\[struck:\s*([^\]]+)\]', line)):
                w1 = m.group(1).strip()
                w2 = m.group(2).strip()
                if re.match(r'^(?:ans|answer|question|no|\d+[\)\.]|[a-zA-Z][\)\.])\b', w1, re.I):
                    continue
                if is_student_prefix_fragment(w2) or len(w2) <= 2:
                    continue
                line_len = max(1, len(line))
                w1_mid = ((m.start(1) + m.end(1)) / 2.0 / line_len) * 100.0
                w2_mid = ((m.start(2) + m.end(2)) / 2.0 / line_len) * 100.0
                for r in strikethrough_regions:
                    if getattr(r, "is_underline", False):
                        continue
                    ry = getattr(r, "y_pct", 0.0)
                    ry2 = getattr(r, "y2_pct", ry)
                    if (ry - 6.0) <= line_y_pct <= (ry2 + 6.0):
                        rx1 = getattr(r, "x_pct", 0.0)
                        rx2 = getattr(r, "x2_pct", 100.0)
                        covers_w1 = (rx1 - 8.0) <= w1_mid <= (rx2 + 8.0)
                        covers_w2 = (rx1 - 5.0) <= w2_mid <= (rx2 + 5.0)
                        if covers_w1 and not covers_w2:
                            inverted_span = f"[struck: {w1}] {w2}"
                            orig_span = m.group(0)
                            line = line.replace(orig_span, inverted_span, 1)
                            diffs.append({
                                "type": "strikethrough_tag_inversion_resolved",
                                "original": orig_span,
                                "resolved": inverted_span,
                                "line_idx": idx
                            })
                            break

        lines[idx] = line

    # -------------------------------------------------------------------------
    # 4. Multi-Line Draft Paragraph Snapping
    # -------------------------------------------------------------------------
    if strikethrough_blocks:
        for b in strikethrough_blocks:
            by1 = getattr(b, "y_pct", 0.0)
            by2 = getattr(b, "y2_pct", by1)
            b_type = getattr(b, "stroke_type", "")
            
            # Find lines whose estimated vertical center falls inside block range
            block_lines_idx = []
            for idx, line in enumerate(lines):
                if not line.strip():
                    continue
                ly = (float(idx) + 0.5) / float(n_lines) * 100.0
                if (by1 - 3.0) <= ly <= (by2 + 3.0):
                    block_lines_idx.append(idx)

            if len(block_lines_idx) >= 2:
                # CRITICAL SAFETY: Never snap blocks that contain question headers, sub-question letters, or MCQ answer labels!
                has_answer_labels = any(
                    bool(re.match(r'^(?:[a-zA-Z0-9][\)\.]\s*(?:Ans|ans)?|Ans|Answer to|Question)', lines[li].strip(), re.I))
                    for li in block_lines_idx
                )
                if has_answer_labels:
                    continue

                # Crucial Safety Condition:
                # ONLY snap parallel_horizontal blocks if:
                # 1. The lines are enclosed in draft quotation marks / brackets ('Once there live... palace in the island]'), OR
                # 2. b_type in {"x_cross", "steep_diagonal"}
                # Never snap ordinary text on parallel horizontal lines (to prevent ruled paper from snapping valid answers)!
                is_explicit_cross = b_type in {"x_cross", "steep_diagonal"}
                first_line_text = lines[block_lines_idx[0]].strip()
                last_line_text = lines[block_lines_idx[-1]].strip()
                draft_quotes_hint = (
                    first_line_text.startswith(("' ", "'", '"', "`")) or
                    last_line_text.endswith(("' ", "'", '"', "`", "]")) or
                    any(lines[li].strip().startswith(("' ", "'", '"', "`")) for li in block_lines_idx) or
                    any(lines[li].strip().endswith(("' ", "'", '"', "`", "]")) for li in block_lines_idx)
                )

                # Boundary expansion: if draft quotes are hinted, expand upward/downward to include full quoted block
                if draft_quotes_hint:
                    # Expand upward to opening quote if not already at start
                    curr_start = block_lines_idx[0]
                    for k in range(curr_start - 1, -1, -1):
                        k_text = lines[k].strip()
                        if not k_text or k_text.lower().startswith(("answer to", "ans ")) or re.match(r'^[a-zA-Z0-9][\)\.]', k_text):
                            break
                        block_lines_idx.insert(0, k)
                        if k_text.startswith(("' ", "'", '"', "`")):
                            break

                    # Expand downward to closing quote if not already at end
                    curr_end = block_lines_idx[-1]
                    for k in range(curr_end + 1, len(lines)):
                        k_text = lines[k].strip()
                        if not k_text or k_text.lower().startswith(("answer to", "ans ")) or re.match(r'^[a-zA-Z0-9][\)\.]', k_text):
                            break
                        block_lines_idx.append(k)
                        if k_text.endswith(("' ", "'", '"', "`", "]")):
                            break

                expanded_first = lines[block_lines_idx[0]].strip()
                expanded_last = lines[block_lines_idx[-1]].strip()
                has_draft_quotes = (
                    (expanded_first.startswith(("' ", "'", '"', "`")) or expanded_last.endswith(("' ", "'", '"', "`", "]"))) and
                    len(block_lines_idx) >= 3
                )

                # CRITICAL SAFETY: Never snap multi-line paragraphs based on bounding-box
                # line overlap alone. Snapping requires explicit draft quotes or brackets.
                if has_draft_quotes:
                    for li in block_lines_idx:
                        curr = lines[li].strip()
                        if curr and not curr.startswith("[struck:") and not curr.lower().startswith("answer to"):
                            cleaned_curr = re.sub(r"^['\"`\[]+|['\"`\]]+$", "", curr).strip()
                            resolved_line = f"[struck: {cleaned_curr}]"
                            diffs.append({
                                "type": "draft_block_snapped_to_struck",
                                "original": curr,
                                "resolved": resolved_line,
                                "block_type": b_type
                            })
                            lines[li] = resolved_line

    resolved = "\n".join(lines)
    return resolved, diffs
