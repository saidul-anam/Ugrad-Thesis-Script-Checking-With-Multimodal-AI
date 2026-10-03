"""
Deterministic Linguistic Sanitizer & Neuro-Symbolic Verification Gate.

Provides pre-cleaning for transcripts prior to LLM linguistic analysis and
post-validation of LLM-extracted error catalogs to eliminate false-positive
spelling errors caused by OCR artifacts, margin truncations, exam headers,
and misclassified words.
"""

import os
import re
import difflib
from typing import List, Set, Optional
from src.core.schemas import LinguisticErrorItem
from src.utils.edge_truncation_detector import (
    stitch_cross_line_truncations,
    is_right_edge_truncation
)


# Standard English dictionary loader with fallback
_SYSTEM_DICT_PATH = "/usr/share/dict/words"
_ENGLISH_LEXICON: Optional[Set[str]] = None

def get_english_lexicon() -> Set[str]:
    """Load standard English words dictionary."""
    global _ENGLISH_LEXICON
    if _ENGLISH_LEXICON is None:
        words = set()
        if os.path.exists(_SYSTEM_DICT_PATH):
            try:
                with open(_SYSTEM_DICT_PATH, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        raw = line.strip()
                        if raw and len(raw) > 1 and not raw.endswith("'s"):
                            words.add(raw.lower())
            except Exception:
                pass
        if not words:
            try:
                import nltk
                from nltk.corpus import words as nltk_words
                words = {w.lower() for w in nltk_words.words()}
            except Exception:
                pass
        _ENGLISH_LEXICON = words
    return _ENGLISH_LEXICON


# Regex patterns for exam headers, subparts, and metadata
_HEADER_PATTERNS = [
    # Lines starting with "Ans:", "Dans:", "Ans to Q.", "Question No.", etc.
    r"(?im)^\s*d?ans\s*:\s*(?:to\s+the\s+)?q(?:uestion)?\.?\s*(?:no\.?)?\s*[\w\(\)\.\s\-]+$",
    r"(?im)^\s*q(?:uestion)?\.?\s*(?:no\.?)?\s*\d+[\w\(\)\.\s\-]*$",
    # Sub-question bullet points like "a) -> (iii) ...", "b) -> (ii) ..."
    r"(?im)^\s*[a-j]\)\s*(?:->|\$?\s*\\rightarrow\s*\$?)\s*\([ivx]+\).*$",
    # Table dividers and standalone Roman/letter labels: "(A)", "(B)", "Figure: Flow chart"
    r"(?im)^\s*\([A-J]\)\s*$",
    r"(?im)^\s*figure\s*:\s*flow\s*chart\s*$",
    r"(?im)^\s*---\s*page\s*break\s*---\s*$",
]

_COMPILED_HEADER_REGEXES = [re.compile(p) for p in _HEADER_PATTERNS]


def sanitize_transcript_for_linguistic_analysis(text: str) -> str:
    """
    Pre-process raw transcript before sending to Stage 3 Linguistic Error Analyzer.
    
    1. Removes question headers, subpart labels, and OCR metadata so they cannot be
       falsely flagged as student errors (e.g., 'Dans' -> 'Ans').
    2. Stitches margin-wrapped and hyphenated word breaks (e.g., 'Universi-\\nty' -> 'University').
    """
    if not text:
        return ""

    # 1. Stitch cross-line word breaks (hyphenated, [truncated] tagged, or unhyphenated splits)
    lexicon = get_english_lexicon()
    stitched, _ = stitch_cross_line_truncations(text, lexicon)

    # 2. Clean residual [truncated] tags so brackets do not generate syntax/punctuation noise
    stitched = re.sub(r'\[truncated(?::\s*[^\]]+)?\]', '', stitched, flags=re.IGNORECASE)

    # 2b. Strip struck-through / cancelled text so deleted drafts do not trigger syntax/grammar penalties
    stitched = re.sub(r'\[struck:\s*[^\]]*\]', ' ', stitched, flags=re.IGNORECASE)
    stitched = re.sub(r'[ \t]+', ' ', stitched)

    # 3. Strip exam header prefixes at the beginning of lines (e.g. "Dans: The author..." -> "The author...")
    # Use [ \t] instead of \s to prevent consuming newlines
    stitched = re.sub(r"(?im)^[ \t]*d?ans\s*:\s*(?:to\s+(?:the\s+)?q(?:uestion)?\.?\s*(?:no\.?)?\s*[\w\(\)\.\- \t]*:?[ \t]*)?", "", stitched)

    # 4. Strip standalone exam headers and metadata line by line
    cleaned_lines = []
    for line in stitched.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        
        # Check against header patterns
        is_header = False
        for reg in _COMPILED_HEADER_REGEXES:
            if reg.match(line_str):
                is_header = True
                break
        
        if not is_header:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


NCTB_CULTURAL_TERMS: Set[str] = {
    "salam", "salaam", "assalamu", "alaikum", "nomoshkar", "adab",
    "eid", "puja", "boishakh", "pohela", "ekushey", "hartal", "bazaar", "bazar",
    "lungi", "saree", "kurta", "madrasah", "madrasa", "upazila", "thana", "union",
    "crore", "lakh", "taka", "paisa", "ghat", "char",
    "tarun", "kalam", "jamuna", "padma", "meghna", "surma", "karnafuli", "sundarbans",
    "rahim", "karim", "barkat", "rafiq", "jabbar", "shafiq", "bangabandhu", "mujib",
    "nazrul", "rabindranath", "tagore", "titumir", "rokeya", "bhashani", "hasina",
    "dhaka", "chittagong", "chattogram", "sylhet", "rajshahi", "khulna", "barisal",
    "rangpur", "mymensingh", "comilla", "cumilla", "bogura", "bogra"
}


def verify_and_filter_stage3_errors(
    errors: List[LinguisticErrorItem],
    question_vocab: Optional[Set[str]] = None,
    subject: str = "English",
    transcript: Optional[str] = None
) -> List[LinguisticErrorItem]:
    """
    Post-process and validate errors extracted by Stage 3 LLM.
    
    Rules enforced:
    1. Header Filter: Discards errors stemming from exam prefixes (e.g. 'Dans', 'Ans', 'Q. No').
    2. Edge Truncation Gate: Suppresses false spelling/grammar errors on words cut off at right margins/photo edges.
    3. Struck Text Filter: Discards errors on words the student crossed out with [struck: ...].
    4. OCR Artifact Detection: Cursive OCR slips (e.g. 'thad' -> 'that', 'beals' -> 'beats') are suppressed.
    5. Proper Noun / Question Whitelist: If the word or suggested correction is in the question paper
       vocabulary (e.g. 'Pasteur', 'Gaza', 'Joseph Meister'), immune from spelling penalties.
    6. Dictionary Gate: If erroneous_text is a valid dictionary word (e.g. 'really' in 'really of Gaza'),
       it CANNOT be a spelling error. Reclassifies to 'grammar' or 'syntax'.
    7. Single-Word Precision: For spelling errors, erroneous_text must be exactly 1 word. If multiple
       words are provided, reclassifies to 'syntax' or 'grammar'.
    8. Compound Words / Hyphenations: Common closed compounds in note-taking (e.g. 'healthrisk') are
       demoted or filtered.
    """
    if not errors:
        return []

    lexicon = get_english_lexicon() if "english" in subject.lower() else set()
    q_vocab = {w.lower() for w in question_vocab} if question_vocab else set()
    
    # Common headers to reject immediately
    header_tokens = {"ans", "dans", "q", "no", "q.", "no.", "qno", "section", "part", "question"}

    # Extract all struck tokens/phrases from original transcript so cancelled text is never penalized
    struck_texts = set()
    if transcript:
        for m in re.finditer(r'\[struck:\s*([^\]]*)\]', transcript, re.IGNORECASE):
            st = m.group(1).strip().lower()
            if st:
                struck_texts.add(st)
                for w in st.split():
                    clean_w = re.sub(r'[^\w]', '', w)
                    if clean_w:
                        struck_texts.add(clean_w)

    validated_errors: List[LinguisticErrorItem] = []

    for err in errors:
        etype = (err.error_type or "spelling").lower()
        err_text = (err.erroneous_text or "").strip()
        err_clean = re.sub(r"[^\w\s\-]", "", err_text).strip()
        words = err_clean.split()
        corr_text = (err.suggested_correction or "").strip()
        corr_clean = re.sub(r"[^\w\s\-]", "", corr_text).strip()
        # 0. Drop punctuation errors immediately (punctuation is excluded from error catalog)
        if "punct" in etype or "comma" in etype or "period" in etype or "hyphen" in etype:
            continue
        if not err_clean or not corr_clean:
            continue
        if err.explanation and any(pw in err.explanation.lower() for pw in ["missing comma", "missing period", "punctuation", "quotation mark", "missing dari", "apostrophe"]):
            continue

        # 0b. Discard pure capitalization / casing differences across all error types
        # In student handwriting, letter casing (e.g. C, P, S, W, O, K, V) is often ambiguous or stylistic.
        if "capital" in etype or "case" in etype or "casing" in etype:
            continue
        if err_clean.lower() == corr_clean.lower():
            # Error and correction are identical ignoring case: this is purely a capitalization difference
            continue
        err_strip = re.sub(r'^[^\w\u0980-\u09FF]+|[^\w\u0980-\u09FF]+$', '', err_clean).lower()
        corr_strip = re.sub(r'^[^\w\u0980-\u09FF]+|[^\w\u0980-\u09FF]+$', '', corr_clean).lower()
        if err_strip and corr_strip and err_strip == corr_strip:
            # Difference is purely capitalization and/or edge punctuation
            continue
        if err.explanation and any(kw in err.explanation.lower() for kw in [
            "capitalization", "capital letter", "capital letters", "uppercase", "lowercase",
            "capitalized", "capitalised", "capitalise", "capitalize", "capitalisation",
            "casing error", "case error", "upper case", "lower case", "small letter",
            "title case", "sentence case", "initial capital", "letter case"
        ]):
            continue

        # 0c. Discard errors on text that was crossed out / struck by the student
        if struck_texts:
            err_lower = err_clean.lower()
            if err_lower in struck_texts or (words and all(w.lower() in struck_texts for w in words)):
                continue

        # 1. Drop exam header artifacts (e.g. "Dans" -> "Ans")
        if err_clean.lower() in header_tokens or corr_clean.lower() in header_tokens:
            continue
        if any(h in err_clean.lower().split() for h in ["dans", "qno"]):
            continue

        # 2. Check right-edge / margin truncation across all error categories
        # (Catches spelling like 'renewabl' -> 'renewable', 'wor' -> 'words', 'villa' -> 'village',
        # and grammar like 'pro' -> 'produced', 'co' -> 'comes', 'w' -> 'which')
        if is_right_edge_truncation(err_clean, corr_clean, context_sentence=err.context_sentence, transcript=transcript, lexicon=lexicon):
            continue

        # 3. Check proper nouns, cultural terms, and question paper vocabulary
        err_low = err_clean.lower()
        corr_low = corr_clean.lower()
        if err_low in q_vocab or corr_low in q_vocab or err_low in NCTB_CULTURAL_TERMS:
            # If word is from question paper or cultural entity, it is not a student spelling error
            if "spell" in etype:
                continue

        # 3b. Mid-Sentence Capitalized Proper Noun Shield:
        # If the word appears capitalized in the context sentence and is NOT sentence-initial,
        # it functions grammatically as a proper noun / named entity (e.g. 'Tarun', 'Kalam', 'Jamuna').
        if "spell" in etype and err_text and err_text[0].isupper():
            ctx = err.context_sentence or ""
            pattern = r'(?:\b|[^\w])' + re.escape(err_text) + r'\b'
            m = re.search(pattern, ctx)
            if m:
                preceding = ctx[:m.start()].strip()
                if preceding and not preceding.endswith((".", "!", "?", ":", "\n")):
                    continue

        # 4. Spelling Error Gate
        if "spell" in etype:
            # Rule A: Multi-word phrase cannot be a spelling error
            if len(words) > 1:
                # Reclassify to grammar or syntax
                err.error_type = "syntax" if len(words) > 4 else "grammar"
                validated_errors.append(err)
                continue

            single_token = words[0].lower() if words else ""

            # Rule C: Compound word / note-taking spacing (e.g. "healthrisk" -> "health risk")
            if len(single_token) >= 6:
                parts = corr_clean.lower().split()
                if len(parts) == 2 and parts[0] in lexicon and parts[1] in lexicon:
                    # Valid compound spacing in notes, do not penalize as spelling
                    continue

            # Rule D: Line-break truncation fragments (e.g. "Universi" for "University", "renewabl" for "renewable")
            if is_right_edge_truncation(single_token, corr_clean, context_sentence=err.context_sentence, transcript=transcript, lexicon=lexicon):
                continue

            # Rule E: Dictionary Check - If erroneous word is actually a valid English word,
            # it is a grammatical / lexical choice error (e.g., "really" instead of "reality"),
            # NOT an orthographic spelling error.
            if single_token in lexicon and len(single_token) > 2:
                err.error_type = "grammar"
                validated_errors.append(err)
                continue

            # Rule F: Gibberish / OCR artifact non-words in notes
            if err.explanation and any(kw in err.explanation.lower() for kw in ["unclear", "nonsensical", "noise", "transliteration"]):
                # Mark as OCR noise rather than student spelling
                continue

            # Rule G: Orthographic & Phonetic Similarity Gate
            # A true student spelling error must share close orthographic similarity with its
            # suggested correction (e.g. 'Accroding' -> 'According' [0.89], 'systeme' -> 'system' [0.92]).
            # Disjoint word replacements (e.g. 'mangsimbee' -> 'mistreated' [0.30])
            # are OCR transcription garbles or illegible handwriting artifacts, NOT student spelling mistakes.
            corr_first = corr_clean.lower().split()[0] if corr_clean else ""
            if single_token and corr_first:
                sim = difflib.SequenceMatcher(None, single_token, corr_first).ratio()
                if sim < 0.60:
                    # Drop OCR artifact / disjoint hallucination
                    continue

            # Rule H: Table & Bullet note isolation
            # Shorthand notes in markdown tables or bulleted lists are bypassed.
            ctx = (err.context_sentence or "").strip()
            if ctx.startswith("|") and ctx.endswith("|"):
                continue
            if re.match(r"^[a-j]\)\s+[\w\s,]+$", ctx, re.IGNORECASE):
                continue

        validated_errors.append(err)

    return validated_errors
