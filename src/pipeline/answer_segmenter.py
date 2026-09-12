"""
Answer Segmenter & Question Aligner.

Segments multi-page handwritten exam transcripts into question-aligned answer blocks,
mapping continuation pages, headers, and linguistic errors to their corresponding exam questions.
"""

import re
from typing import List, Dict, Any, Optional
from src.core.schemas import (
    PageExtractionResult,
    ExtractionResult,
    ExtractedQuestion,
    AlignedAnswerItem
)


# Canonical header normalization mappings
HEADER_NORMALIZATION_RULES = [
    (re.compile(r'No[.,\s]*[Zz]\b', re.IGNORECASE), 'No. 7'),
    (re.compile(r'\bDans\b', re.IGNORECASE), 'Ans: to the Q. No. 1(B)'),
    (re.compile(r'\bdo the [Qq]\b', re.IGNORECASE), 'to the Q'),
    (re.compile(r'Qhe o\.', re.IGNORECASE), 'Q. No.'),
    (re.compile(r'Anseve?r', re.IGNORECASE), 'Answer'),
    (re.compile(r'\bAns(?:i?to|i\s*to)\b', re.IGNORECASE), 'Ans to'),
    (re.compile(r'\bAnsi\b', re.IGNORECASE), 'Ans'),
    (re.compile(r'\$?\s*\\?i?ghtarrow\s*\$?', re.IGNORECASE), ' -> '),
]

# Bengali to Arabic numeral and subpart mappings
BENGALI_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
BENGALI_SUBPART_MAP = {
    'ক': 'A', 'খ': 'B', 'গ': 'C', 'ঘ': 'D', 'ঙ': 'E', 'চ': 'F', 'ছ': 'G'
}


def to_arabic_digits(text: str) -> str:
    """Convert Bengali numerals in text to Arabic ASCII digits."""
    return text.translate(BENGALI_DIGIT_MAP) if text else text


HEADER_SPLIT_REGEX = re.compile(
    r'(?=(?:'
    # English question headers
    r'(?:\n|\A)\s*(?:Ans(?:i?to)?|Answer|Dans)[:\s]*(?:to\s+(?:the\s+)?)?Q(?:uestion)?[\s\.\,\-]*(?:No[\s\.\,\-]*)?\s*[0-9০-৯]{1,2}\s*(?:\([A-Za-z0-9\u0980-\u09FF]\))?|'
    r'(?:\n|\A)\s*Ans[:\s]+(?:to\s+)?Qhe\s+o\.\s*No\.\s*[0-9A-Za-z]+|'
    r'(?:\n|\A)\s*Q(?:uestion)?[\s\.\,\-]+(?:No[\s\.\,\-]*)?\s*[0-9০-৯]{1,2}\s*(?:\([A-Za-z0-9\u0980-\u09FF]\))?|'
    # Bengali question headers
    r'(?:\n|\A)\s*(?:[০-৯0-9]{1,2}\s*(?:[\(（][\u0980-\u09FFA-Za-z0-9]+[\)）]\s*)?(?:নং|নম্বর)?\s*প্রশ্নের?\s*উত্তর)|'
    r'(?:\n|\A)\s*(?:প্রশ্নের?\s*উত্তর\s*[:\-]?\s*[০-৯0-9]{1,2})|'
    # Subparts: (A), (B), (ক), (খ) on new line
    r'(?:\n|\A)\s*[\(（](?:[A-Da-d]|[ক-ঘ])[\)）]\s*(?:\n|Ans|Answer)|'
    # Theme/Poem headers
    r'(?:\n|\A)\s*Theme:\s+The\s+poem'
    r'))',
    re.IGNORECASE
)

HEADER_EXTRACT_REGEX = re.compile(
    r'(?:Ans(?:i?to)?|Answer|Dans)[:\s]*(?:to\s+(?:the\s+)?)?Q(?:uestion)?[\s\.\,\-]*(?:No[\s\.\,\-]*)?\s*([0-9]{1,2}(?:\([A-Za-z0-9]\))?|[A-B]\b)|'
    r'Q(?:uestion)?[\s\.\,\-]*(?:No[\s\.\,\-]*)?\s*([0-9]{1,2}(?:\([A-Za-z0-9]\))?|[A-B]\b)|'
    r'^\s*\(([A-B])\)',
    re.IGNORECASE | re.MULTILINE
)


def normalize_header_text(text: str) -> str:
    """Normalize OCR anomalies in question headings."""
    normalized = text
    for pat, repl in HEADER_NORMALIZATION_RULES:
        normalized = pat.sub(repl, normalized)
    return normalized


_KEYWORD_STOPWORDS = {
    "question", "questions", "write", "writing", "answer", "answers", "following", "paragraph",
    "with", "that", "this", "from", "into", "those", "their", "thing", "there", "these", "about",
    "which", "what", "when", "your", "have", "been", "were", "will", "than", "then", "them", "they",
}


def _extract_explicit_english_header(arabic_lines: str) -> Optional[str]:
    """Canonical q_no from an explicit 'Ans to the Q No ...' style header, else None."""
    match = HEADER_EXTRACT_REGEX.search(arabic_lines)
    if not match:
        return None
    for g in match.groups():
        if g:
            val = g.strip().upper()
            clean_val = re.sub(r'\b0+(\d+)', r'\1', val)
            if clean_val in ["1", "1A", "1(A)", "A"]:
                return "1(A)"
            elif clean_val in ["1B", "1(B)", "B"]:
                return "1(B)"
            elif clean_val in ["Z", "N"]:
                return "7"
            elif clean_val in [str(i) for i in range(1, 25)]:
                return clean_val
            sub_m = re.match(r'^([0-9]{1,2})\s*\(([A-Za-z0-9])\)$', clean_val)
            if sub_m:
                return f"{sub_m.group(1)}({sub_m.group(2).upper()})"
            return clean_val
    return None


def extract_header_qno(
    section_text: str,
    current_parent: Optional[str] = None,
    question_obj: Optional[ExtractedQuestion] = None,
    valid_q_order: Optional[List[str]] = None
) -> Optional[str]:
    """
    Extract canonical question number from the leading lines of a text section.
    Supports English & Bengali headers, dynamic schema keywords, and sub-parts.
    """
    first_lines = "\n".join(section_text.strip().split("\n")[:4])
    normalized = normalize_header_text(first_lines)
    arabic_lines = to_arabic_digits(normalized)
    first_lines_lower = first_lines.lower()

    # 1. Explicit numeric English header first (an explicit "Ans to the Q No 11" must never be
    #    overridden by topic keywords). Keyword matching (below) only runs when no header is present.
    explicit = _extract_explicit_english_header(arabic_lines)
    if explicit:
        return explicit

    # 1b. Dynamic Keyword/Topic Matching from question_obj: best-matching sub-question wins,
    #     stop words are ignored, and at least two distinctive words must match.
    if question_obj and question_obj.sub_questions:
        best_q, best_score = None, 0.0
        for sq in question_obj.sub_questions:
            sq_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
            sq_name = str(sq.get("name") or sq.get("title") or "").strip()
            if not sq_no or not sq_name:
                continue
            words = [
                w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', sq_name)
                if w.lower() not in _KEYWORD_STOPWORDS
            ]
            if not words:
                continue
            matches = sum(1 for w in set(words) if w in first_lines_lower)
            if matches >= 2 and any(h in first_lines_lower for h in ["ans", "q", "title", "theme", "paragraph", "topic", "no"]):
                score = matches / len(set(words))
                if score > best_score:
                    best_q, best_score = sq_no, score
        if best_q:
            return best_q

    # 2. Bengali Question Header Matching (supports both '১(ক) নং' and '১ নং প্রশ্নের উত্তর (ক)')
    beng_m = re.search(
        r'([0-9]{1,2})\s*(?:[\(（]([\u0980-\u09FFA-Za-z0-9]+)[\)）]\s*)?(?:নং|নম্বর)?\s*প্রশ্নের?\s*উত্তর(?:\s*[\(（]([\u0980-\u09FFA-Za-z0-9]+)[\)）])?',
        arabic_lines
    )
    if beng_m:
        q_num = beng_m.group(1)
        sub_part = beng_m.group(2) or beng_m.group(3)
        if sub_part:
            ascii_sub = BENGALI_SUBPART_MAP.get(sub_part, sub_part.upper())
            return f"{q_num}({ascii_sub})"
        return q_num

    # 3. (explicit English header already handled in step 1)

    # 4. Subpart (B) following (A) or subpart (খ) following (ক)
    if current_parent in ["1", "1(A)", "1A"]:
        if re.search(r'^\s*[\(（](?:B|b|খ)[\)）]', first_lines, re.MULTILINE):
            return "1(B)"

    # 5. Backward compatibility heuristics for English HSC
    if "theme:" in first_lines_lower and ("dream" in first_lines_lower or "poem" in first_lines_lower):
        return "11"
    if "artificial" in first_lines_lower or "indelligence" in first_lines_lower or re.search(r'\b(ai|a\.i\.)\b', first_lines_lower):
        if re.search(r'\b(ans|q|question|no)\b', first_lines_lower):
            return "7"

    return None


def segment_script_into_questions(
    extraction: ExtractionResult,
    question_obj: Optional[ExtractedQuestion] = None
) -> List[AlignedAnswerItem]:
    """
    Segment verified script text into aligned question answers.
    Combines page-by-page transcripts, tracks continuation pages,
    and maps linguistic errors to each answer segment.
    """
    # Build question catalog from question paper if available
    sub_q_names = {}
    valid_q_order = []
    if question_obj and question_obj.sub_questions:
        for sq in question_obj.sub_questions:
            q_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
            if q_no:
                valid_q_order.append(q_no)
                sub_q_names[q_no] = sq.get("name") or sq.get("title") or f"Question {q_no}"

    # Collect answers across all pages
    # Map: q_no -> {"text_parts": [], "pages": [], "errors": []}
    answer_buckets: Dict[str, Dict[str, Any]] = {}
    current_q_no: Optional[str] = None

    pages = extraction.pages or []

    # If pages are not available, segment the combined verified transcript directly
    if not pages:
        full_text = extraction.stage2_verification.verified_transcript or extraction.stage1_transcription.raw_transcript
        return _segment_raw_transcript(full_text, sub_q_names, valid_q_order, extraction.stage3_errors.errors)

    for p in pages:
        page_no = p.page_no
        page_text = (p.stage2_verification.verified_transcript or p.stage1_transcription.raw_transcript or "").strip()
        if not page_text:
            continue

        page_errors = [e.model_dump() for e in p.stage3_errors.errors] if p.stage3_errors else []

        # Split page text into sections if page contains multiple question headers
        normalized_page = normalize_header_text(page_text)
        sections = HEADER_SPLIT_REGEX.split(normalized_page)
        sections = [s for s in sections if s.strip()]

        if not sections:
            sections = [normalized_page]

        for s in sections:
            header_q = extract_header_qno(
                s,
                current_parent=current_q_no,
                question_obj=question_obj,
                valid_q_order=valid_q_order
            )
            if header_q:
                current_q_no = header_q

            # If still no active question, fallback to current_q_no, or first valid question
            target_q = current_q_no or (valid_q_order[0] if valid_q_order else "1(A)")
            if target_q not in answer_buckets:
                answer_buckets[target_q] = {
                    "text_parts": [],
                    "pages": set(),
                    "errors": []
                }

            answer_buckets[target_q]["text_parts"].append(s.strip())
            answer_buckets[target_q]["pages"].add(page_no)

        # Attribute page errors to active question(s)
        if current_q_no and current_q_no in answer_buckets:
            answer_buckets[current_q_no]["errors"].extend(page_errors)

    # Convert buckets to structured AlignedAnswerItem list
    aligned_items: List[AlignedAnswerItem] = []

    # Order by valid_q_order if known, else natural order
    all_keys = list(answer_buckets.keys())
    if valid_q_order:
        ordered_keys = [k for k in valid_q_order if k in answer_buckets] + [k for k in all_keys if k not in valid_q_order]
    else:
        ordered_keys = sorted(all_keys)

    for q in ordered_keys:
        b = answer_buckets[q]
        combined_ans = "\n\n".join(b["text_parts"]).strip()
        # Deduplicate error dicts by erroneous_text + sentence
        seen_errs = set()
        dedup_errors = []
        for err in b["errors"]:
            key = (err.get("erroneous_text", ""), err.get("context_sentence", ""))
            if key not in seen_errs:
                seen_errs.add(key)
                dedup_errors.append(err)

        words = len(combined_ans.split())
        aligned_items.append(AlignedAnswerItem(
            q_no=q,
            q_name=sub_q_names.get(q, f"Question {q}"),
            answer_text=combined_ans,
            page_numbers=sorted(list(b["pages"])),
            errors=dedup_errors,
            word_count=words,
            character_count=len(combined_ans)
        ))

    return aligned_items


def _segment_raw_transcript(
    full_text: str,
    sub_q_names: Dict[str, str],
    valid_q_order: List[str],
    all_errors: List[Any]
) -> List[AlignedAnswerItem]:
    """Fallback segmenter when page-level checkpoint structures are absent."""
    normalized = normalize_header_text(full_text)
    sections = HEADER_SPLIT_REGEX.split(normalized)
    sections = [s for s in sections if s.strip()]

    items = []
    cur_q = valid_q_order[0] if valid_q_order else "1(A)"
    for s in sections:
        hq = extract_header_qno(s, current_parent=cur_q, valid_q_order=valid_q_order)
        if hq:
            cur_q = hq
        items.append((cur_q, s.strip()))

    grouped: Dict[str, List[str]] = {}
    for q, txt in items:
        grouped.setdefault(q, []).append(txt)

    results = []
    for q, parts in grouped.items():
        ans_text = "\n\n".join(parts)
        results.append(AlignedAnswerItem(
            q_no=q,
            q_name=sub_q_names.get(q, f"Question {q}"),
            answer_text=ans_text,
            page_numbers=[1],
            errors=[e.model_dump() if hasattr(e, "model_dump") else e for e in all_errors],
            word_count=len(ans_text.split()),
            character_count=len(ans_text)
        ))
    return results
