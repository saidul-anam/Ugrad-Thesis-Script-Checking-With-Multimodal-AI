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
from src.utils.text_metrics import levenshtein


# Canonical header normalization mappings
HEADER_NORMALIZATION_RULES = [
    (re.compile(r'(?im)^\s*Dans\s+to\b'), 'Ans to'),
    (re.compile(r'(?im)^\s*([b-gB-G])\s*ans[:\s]+'), r'(\1) Ans: '),
    (re.compile(r'(?im)^\s*Dans[:\s]+'), '(d) Ans: '),
    (re.compile(r'\b(?:Ass|Ansl)\b', re.IGNORECASE), 'Ans'),
    (re.compile(r'No[.,\s]*[Zz]\b', re.IGNORECASE), 'No. 7'),
    (re.compile(r'\bDans\b', re.IGNORECASE), 'Ans'),
    (re.compile(r'\bdo the [Qq]\b', re.IGNORECASE), 'to the Q'),
    (re.compile(r'Qhe o\.', re.IGNORECASE), 'Q. No.'),
    (re.compile(r'Anseve?r', re.IGNORECASE), 'Answer'),
    (re.compile(r'\bAns(?:i?to|i\s*to)\b', re.IGNORECASE), 'Ans to'),
    (re.compile(r'\bAnsi\b', re.IGNORECASE), 'Ans'),
    (re.compile(r'\b(?:Ans\s*[:\s]*(?:(?:to|of)\s+(?:the\s+)?)?)(?:ques?|dues?)\b', re.IGNORECASE), 'Ans to the Q'),
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
    # English question headers (handles Ans/Answer/Dans/Ass/Ansl, to/of the, Question/Ques/Q/No/Number, optional leading [struck: ...])
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:Ansl?\.?\s*)?(?:Ans(?:i?to|i\s*to)?|Answer|Dans|Ass|Q(?:uestion|ues|ue|n)?|due|dues|No|Number|Num)\b[^\n\r0-9]{0,40}[0-9০-৯]{1,3}(?:\s*[\(（][A-Za-z0-9\u0980-\u09FF][\)）])?|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*Ans[:\s]+(?:(?:to|of)\s+)?Qhe\s+o\.\s*No\.\s*[0-9A-Za-z]+|'
    # Standalone question subpart headers: 1(A), 1(B), 1(a), 1(b)
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[0-9০-৯]{1,3}\s*[\(（][A-Za-z0-9\u0980-\u09FF][\)）]\s*(?:\n|\r\n|Ans|Answer|:|\.|\-|$)|'
    # Bengali question headers: supports '১ নং উত্তর', '১(ক) নং প্রশ্নের উত্তর', 'উত্তর: ১', 'উত্তর নং ১'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:[০-৯0-9]{1,3}\s*(?:[\(（][\u0980-\u09FFA-Za-z0-9]+[\)）]\s*)?(?:নং|নম্বর)?\s*(?:প্রশ্নের?\s*)?উত্তর)|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:(?:প্রশ্নের?\s*)?উত্তর\s*(?:নং|নম্বর)?\s*[:\-]?\s*[০-৯0-9]{1,3})|'
    # Subparts: (A), (B), (ক), (খ), circled letters Ⓐ-Ⓩ, or standalone letter line before sub-items
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[\(（](?:[A-Za-z]|[ক-ঘ])[\)）]\s*(?:\n|\r\n|Ans|Answer)|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[Ⓐ-Ⓩⓐ-ⓩ]\s*(?:\n|\r\n|Ans|Answer)|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[A-Za-z]\s*(?:\n|\r\n)\s*(?=[a-z0-9][\)\.]|\([a-z0-9]\))|'
    # Structural headers (Flowchart, Theme, Rearrange, Summary, Story, Chart)
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*Flow[\s\-]*chart\s*[:\-]|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*Theme\s*:\s+|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*Summary\s*:\s+|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*Rearrange\s*[:\-]|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:The\s+graph|The\s+chart)\s+shows\b|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*Once\s+upon\s+a\s+time\b'
    r'))',
    re.IGNORECASE
)

HEADER_EXTRACT_REGEX = re.compile(
    r'^\s*(?:\[struck:[^\]]*\]\s*)*(?:Ansl?\.?\s*)?(?:Ans(?:i?to|i\s*to)?|Answer|Dans|Ass|Q(?:uestion|ues|ue|n)?|due|dues|No|Number|Num)\b[^\n\r0-9]{0,40}([0-9]{1,3}[ \t]*(?:\([A-Za-z0-9]\))?|[A-Za-z]\b)|'
    r'^\s*(?:\[struck:[^\]]*\]\s*)*([0-9]{1,3})[ \t]*[\(（]([A-Za-z0-9])[\)）]\s*[\:\.\-]?\s*(?:\n|\r\n|Ans|Answer|$)|'
    r'^\s*(?:\[struck:[^\]]*\]\s*)*[\(（]([A-Za-z])[\)）]\s*(?:\n|\r\n|Ans|Answer|$)|'
    r'^\s*(?:\[struck:[^\]]*\]\s*)*([Ⓐ-Ⓩⓐ-ⓩ])\s*(?:\n|\r\n|Ans|Answer|$)|'
    r'^\s*(?:\[struck:[^\]]*\]\s*)*([A-Za-z])\s*$',
    re.IGNORECASE | re.MULTILINE
)

VALID_HEADER_INTERMEDIATE_WORDS = {
    "to", "of", "the", "question", "questions", "ques", "que", "qn", "no", "number", "num", "n", "q", "qhe", "o", "ans", "answer", "for", "in"
}



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


def _extract_explicit_english_header(
    arabic_lines: str,
    valid_q_order: Optional[List[str]] = None,
    current_parent: Optional[str] = None
) -> Optional[str]:
    """
    Schema-Constrained Canonical Anchor-and-Validate Header Parser.
    
    1. Checks for standalone subparts e.g. (A), (B), (a), (b), Ⓐ, Ⓑ.
    2. Detects header intent anchors (ans, answer, dans, ass, ansl, q, ques, question, no, number, num).
    3. Extracts candidate question number and subpart.
    4. Validates candidate against valid_q_order from question_obj to eliminate false positives.
    """
    first_lines = "\n".join(arabic_lines.strip().split("\n")[:4])

    # 1. Standalone subparts following any active parent question
    if current_parent:
        base_parent = re.sub(r'\(.*?\)|[A-Za-z\u0980-\u09FF]+$', '', str(current_parent)).strip()
        if base_parent.isdigit():
            sub_m = re.search(r'^\s*[\(（]([A-Za-z])[\)）]\s*(?:\n|\r\n|Ans|Answer|:|\.|\-|$)', first_lines, re.MULTILINE)
            if sub_m:
                cand = f"{base_parent}({sub_m.group(1).upper()})"
                if not valid_q_order or cand in valid_q_order:
                    return cand
            sub_circle = re.search(r'^\s*([Ⓐ-Ⓩⓐ-ⓩ])\s*(?:\n|\r\n|Ans|Answer|:|\.|\-|$)', first_lines, re.MULTILINE)
            if sub_circle:
                ch = sub_circle.group(1)
                let = chr(ord('A') + ord(ch) - ord('Ⓐ')) if 'Ⓐ' <= ch <= 'Ⓩ' else (chr(ord('A') + ord(ch) - ord('ⓐ')) if 'ⓐ' <= ch <= 'ⓩ' else 'A')
                cand = f"{base_parent}({let})"
                if not valid_q_order or cand in valid_q_order:
                    return cand

    # 2. Extract non-empty, non-structural leading lines
    clean_lines = []
    for l in first_lines.split("\n"):
        l_str = re.sub(r'^\s*\[struck:[^\]]*\]\s*', '', l).strip()
        if not l_str or l_str.startswith("|") or l_str.startswith("---"):
            continue
        clean_lines.append(l_str)

    if not clean_lines:
        return None

    # Check the first 2 clean lines for a header (handles page title on line 0)
    for line_idx in range(min(2, len(clean_lines))):
        top_line = clean_lines[line_idx]

        # OCR artifact 'No. Z' -> 7 or 2
        if re.search(r'\bNo[.,\s]*[Zz]\b', top_line, re.IGNORECASE):
            if valid_q_order:
                if "7" in valid_q_order:
                    return "7"
                if "2" in valid_q_order:
                    return "2"
            return "7"

        # Standalone question subpart format e.g. "1(A)" or "2(B)" or "3(b)"
        standalone_m = re.match(r'^\s*([0-9]{1,3})\s*[\(（]([A-Za-z])[\)）]\s*[\:\.\-]?', top_line)
        if standalone_m:
            candidate = f"{int(standalone_m.group(1))}({standalone_m.group(2).upper()})"
            if not valid_q_order or candidate in valid_q_order:
                return candidate

        # Connected header anchor pattern: requires anchor (Ans/Answer/Dans/Ass/Q/No) connected to a number
        # through valid header connectors (to, of, the, question, no, punctuation) or 1-edit OCR typos.
        header_m = re.match(
            r'^\s*(?:Ansl?\.?\s*)?(?:Ans(?:i?to|i\s*to)?|Answer|Dans|Ass|Q(?:uestion|ues|ue|n)?|due|dues|No|Number|Num)\b([^\n\r0-9]{0,40})([0-9]{1,3})\s*(?:[\(（\.\s]*([A-Za-z0-9])[\)）]?)?',
            top_line,
            re.IGNORECASE
        )

        if header_m:
            intermediate = header_m.group(1)
            inter_words = [w.lower() for w in re.findall(r'[A-Za-z]+', intermediate)]
            all_valid = True
            for w in inter_words:
                if w in VALID_HEADER_INTERMEDIATE_WORDS or len(w) <= 2:
                    continue
                # 1-char edit distance tolerance for common OCR misreads on connector words
                if any(levenshtein(w, canon) <= 1 for canon in ("question", "number", "answer")):
                    continue
                all_valid = False
                break
            if not all_valid:
                header_m = None

        if not header_m:
            # Check if line has subpart letter like "Ans (A)" or "Ans: B"
            sub_letter_m = re.search(r'^\s*(?:ans(?:wer)?)[:\s\.\-]+(?:[\(（]([A-Za-z])[\)）]|([A-Za-z])\b)', top_line, re.IGNORECASE)
            if sub_letter_m:
                let = (sub_letter_m.group(1) or sub_letter_m.group(2)).upper()
                base = re.sub(r'\(.*?\)|[A-Za-z\u0980-\u09FF]+$', '', str(current_parent or "")).strip()
                base = base if base.isdigit() else (valid_q_order[0] if valid_q_order else "1")
                cand = f"{base}({let})"
                if not valid_q_order or cand in valid_q_order:
                    return cand
            continue

        raw_num = str(int(header_m.group(2)))
        sub_letter = header_m.group(3).upper() if header_m.group(3) else None

        # If subpart wasn't on the same line, check if the immediately following line specifies the subpart
        if not sub_letter and len(clean_lines) > line_idx + 1:
            next_line = clean_lines[line_idx + 1].strip()
            next_sub = re.match(r'^\s*(?:[\(（]([A-Za-z])[\)）]|([Ⓐ-Ⓩⓐ-ⓩ]))\s*(?:\n|\r\n|Ans|Answer|:|\.|\-|$)', next_line)
            if next_sub:
                if next_sub.group(1):
                    sub_letter = next_sub.group(1).upper()
                elif next_sub.group(2):
                    ch = next_sub.group(2)
                    sub_letter = chr(ord('A') + ord(ch) - ord('Ⓐ')) if 'Ⓐ' <= ch <= 'Ⓩ' else (chr(ord('A') + ord(ch) - ord('ⓐ')) if 'ⓐ' <= ch <= 'ⓩ' else 'A')

        candidate = f"{raw_num}({sub_letter})" if sub_letter and sub_letter.isalpha() else raw_num

        # Schema-Constrained Validation
        if valid_q_order:
            if candidate in valid_q_order:
                return candidate

            # If candidate is a base number whose first subpart is in valid_q_order (e.g. cand='1', valid has '1(A)'):
            first_sub = f"{candidate}(A)"
            if first_sub in valid_q_order and candidate not in valid_q_order:
                return first_sub
            if candidate in valid_q_order:
                return candidate

            if sub_letter:
                normalized_cand = f"{raw_num}({sub_letter})"
                if normalized_cand in valid_q_order:
                    return normalized_cand

            continue

        if raw_num.isdigit() and 1 <= int(raw_num) <= 50:
            return candidate

    return None


def detect_structural_fingerprint(
    section_text: str,
    question_obj: Optional[ExtractedQuestion] = None,
    valid_q_order: Optional[List[str]] = None,
    current_parent: Optional[str] = None
) -> Optional[str]:
    """
    Detect question attribution from structural layout, notation, or compositional fingerprints.
    Covers: Flowcharts (->, boxes), Rearrange tables (| 1 | 2 | 3 |), Letter/Email envelopes,
    Statistical data charts (%, years), Narrative stories, and Poem Themes.
    """
    if not section_text or not section_text.strip():
        return None

    # Dynamically map question types to q_no using question_obj catalog
    q_map: Dict[str, str] = {}
    if question_obj and question_obj.sub_questions:
        for sq in question_obj.sub_questions:
            sq_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
            name_lower = str(sq.get("name") or sq.get("title") or "").lower()
            task_lower = str(sq.get("task_type") or "").lower()
            combined = f"{name_lower} {task_lower}".strip()
            if not sq_no:
                continue
            if "flow" in combined or ("chart" in combined and "pie" not in combined and "bar" not in combined):
                q_map["flowchart"] = sq_no
            elif "rearrange" in combined or "jumbled" in combined or "order" in combined:
                q_map["rearrange"] = sq_no
            elif "summary" in combined or "summariz" in combined:
                q_map["summary"] = sq_no
            elif "graph" in combined or "chart" in combined or "diagram" in combined:
                q_map["graph"] = sq_no
            elif "story" in combined or "completing" in combined:
                q_map["story"] = sq_no
            elif "dialogue" in combined or "conversation" in combined:
                q_map["dialogue"] = sq_no
            elif "letter" in combined or "email" in combined:
                q_map["letter"] = sq_no
            elif "theme" in combined or "poem" in combined or "poetry" in combined:
                q_map["theme"] = sq_no
            elif "paragraph" in combined or "composition" in combined or "essay" in combined:
                q_map["paragraph"] = sq_no
            elif "cloze" in combined:
                if "without" in combined:
                    q_map["cloze_no_clues"] = sq_no
                else:
                    q_map["cloze_clues"] = sq_no
            elif "table" in combined:
                q_map["table"] = sq_no
            elif "matching" in combined:
                q_map["matching"] = sq_no

    def _get_q(task_type: str) -> Optional[str]:
        val = q_map.get(task_type)
        if val and (not valid_q_order or val in valid_q_order):
            return val
        return None

    sec_lower = section_text.lower()
    first_lines = "\n".join(section_text.strip().split("\n")[:5]).lower()

    # 1. Flowchart: Directional arrows (->, →, ↓, \rightarrow), multiple boxes with numbered/roman steps
    arrows_count = section_text.count("->") + section_text.count("→") + section_text.count("↓") + len(re.findall(r'\\?i?ghtarrow', section_text, re.I))
    has_flow_kw = bool(re.search(r'\b(?:flow[\s\-]*chart)\b', first_lines, re.I))
    has_flow_boxes = (arrows_count >= 2) and bool(re.search(r'[\(\[]?(?:i|1|ii|2)[\)\]]', section_text, re.I))
    if has_flow_kw or has_flow_boxes:
        return _get_q("flowchart")

    # 2. Rearranging: Sequence table (| 1 | 2 | 3 |), sequence arrows, or rearrange keywords
    has_rearrange_kw = bool(re.search(r'\b(?:re[\s\-]*arrange|rearranging|order\s+of\s+events)\b', first_lines, re.I))
    has_rearrange_table = bool(re.search(r'\|\s*1\s*\|\s*2\s*\|\s*3\s*\|', section_text)) or bool(re.search(r'\b(?:1\s*\+\s*[a-j]|i\s*->\s*[ivx]+)', section_text, re.I))
    if has_rearrange_kw or has_rearrange_table:
        return _get_q("rearrange")

    # 3. Informal Letter / Email: Envelope box, [STAMP], salutation + sign-off
    has_envelope = bool(re.search(r'\b(?:\[?STAMP\]?|envelope)\b', section_text, re.I)) and bool(re.search(r'\b(?:from|to)\b', section_text, re.I))
    has_salutation = bool(re.search(r'^\s*(?:dear\s+[a-z]+|my\s+dear)\b', first_lines, re.I | re.MULTILINE))
    has_signoff = bool(re.search(r'\b(?:yours?\s+(?:ever|loving|faithfully|sincerely|truly|affectionately|friend)|lovingly\s+yours|your\s+(?:loving\s+)?friend)\b', sec_lower, re.I))
    if has_envelope or (has_salutation and (has_signoff or len(section_text.split()) >= 8)) or (has_signoff and current_parent in (q_map.get("letter"), None)):
        return _get_q("letter")

    # 4. Data Chart / Graph: Statistical description with percentages or trend metrics
    has_chart_kw = bool(re.search(r'\b(?:graph|chart|pie\s*chart|bar\s*chart|diagram)\b', first_lines, re.I))
    has_stats = bool(re.search(r'\b(?:percent|percentage|\%)\b', sec_lower, re.I)) and bool(re.search(r'\b\d+(?:\.\d+)?\b', sec_lower))
    has_chart_desc = bool(re.search(r'\b(?:the\s+(?:given\s+)?(?:graph|chart|diagram|table)\s+shows|shows\s+that|increased|decreased|highest|lowest|fluctuat|rate\s+of)\b', sec_lower, re.I))
    if (has_chart_kw and (has_stats or has_chart_desc)) or (has_stats and has_chart_desc):
        return _get_q("graph")

    # 5. Completing Story: Narrative openings or common story motifs
    has_story_opening = bool(re.search(r'\b(?:once\s+upon\s+a\s+time|once\s+(?:there\s+)?lived|once\s+there\s+was|there\s+(?:lived|was)\s+(?:a|an)|one\s+day\s+(?:a|an)?|long\s+(?:ago|time\s+ago)|many\s+days\s+ago)\b', first_lines, re.I))
    has_story_kw = bool(re.search(r'\b(?:completing\s+story|story\s*[:\-]|\btitle\s*[:\-])\b', first_lines, re.I))
    has_schema_story_title = False
    if question_obj and question_obj.sub_questions:
        for sq in question_obj.sub_questions:
            sq_name = str(sq.get("name") or sq.get("title") or "").lower()
            if "story" in sq_name:
                title_words = [w for w in re.findall(r'[a-z]{4,}', sq_name) if w not in _KEYWORD_STOPWORDS and w != "story"]
                if title_words and sum(1 for w in set(title_words) if w in first_lines) >= 2:
                    has_schema_story_title = True
                    break
    if has_story_opening or has_story_kw or has_schema_story_title:
        return _get_q("story")

    # 6. Poem Theme: Theme keyword, poem analysis phrasing
    has_theme_kw = bool(re.search(r'\b(?:theme\s*[:\-]|\bthe\s+(?:poem|stanza|verse|passage)\s+(?:deals\s+with|is\s+about|focuses\s+on)|central\s+theme\b)', first_lines, re.I))
    if has_theme_kw or ("theme:" in first_lines and any(w in first_lines for w in ["poem", "poet", "stanza", "verse", "central", "main", "message", "author"])):
        return _get_q("theme")

    # 7. Summary: Explicit summary keyword
    has_summary_kw = bool(re.search(r'\b(?:summary\s*[:\-]|\bthe\s+passage\s+(?:deals\s+with|is\s+about|summarizes))\b', first_lines, re.I))
    if has_summary_kw:
        return _get_q("summary")

    # 8. Dialogue / Conversation: Multiple speaker turns (e.g. "Afifa:", "Doctor:", "Myself:")
    speaker_turns = len(re.findall(r'(?im)^\s*[A-Z][a-z]{1,15}\s*:\s+', section_text))
    has_dialogue_kw = bool(re.search(r'\b(?:dialogue|conversation)\b', first_lines, re.I))
    if has_dialogue_kw or (speaker_turns >= 4 and len(section_text.split()) >= 20):
        return _get_q("dialogue")

    # 9. Semantic Source-Text Fingerprint for Headless Answers (Summary & Theme)
    # When a student writes NO header (no "Ans 3", no "3.", no "Summary:"), we match the text
    # against the source poem/passage printed on the question paper.
    if question_obj and getattr(question_obj, "question_text", None):
        from src.pipeline.stage4_modes import source_text_for_question
        sec_words = section_text.split()
        sec_tokens = {
            w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', section_text)
            if w.lower() not in _KEYWORD_STOPWORDS
        }

        # Check Summary source passage overlap
        summary_q = _get_q("summary")
        if summary_q:
            q_source = source_text_for_question(summary_q, question_obj.question_text)
            if q_source and sec_tokens and (20 <= len(sec_words) <= 200):
                q_tokens = {
                    w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', q_source)
                    if w.lower() not in _KEYWORD_STOPWORDS
                }
                overlap = q_tokens & sec_tokens
                if len(overlap) >= 3:
                    return summary_q

        # Check Theme source poem overlap
        theme_q = _get_q("theme")
        if theme_q:
            q_source = source_text_for_question(theme_q, question_obj.question_text)
            if q_source and sec_tokens and (15 <= len(sec_words) <= 150):
                q_tokens = {
                    w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', q_source)
                    if w.lower() not in _KEYWORD_STOPWORDS
                }
                overlap = q_tokens & sec_tokens
                if len(overlap) >= 3:
                    return theme_q

    return None


def extract_header_qno(
    section_text: str,
    current_parent: Optional[str] = None,
    question_obj: Optional[ExtractedQuestion] = None,
    valid_q_order: Optional[List[str]] = None
) -> Optional[str]:
    """
    Extract canonical question number from the leading lines of a text section.
    Supports English & Bengali headers, dynamic schema keywords, sub-parts,
    and structural & semantic content fingerprinting.
    """
    first_lines = "\n".join(section_text.strip().split("\n")[:4])
    normalized = normalize_header_text(first_lines)
    arabic_lines = to_arabic_digits(normalized)
    first_lines_lower = first_lines.lower()

    # 1. Explicit numeric English header first (an explicit "Ans to the Q No 11" must never be
    #    overridden by topic keywords). Keyword matching (below) only runs when no header is present.
    explicit = _extract_explicit_english_header(arabic_lines, valid_q_order=valid_q_order, current_parent=current_parent)
    if explicit:
        return explicit

    # 2. Bengali Question Header Matching (supports '১ নং উত্তর', '১(ক) নং', '১ নং প্রশ্নের উত্তর (ক)', 'উত্তর: ১')
    beng_m = re.search(
        r'([0-9]{1,2})\s*(?:[\(（]([\u0980-\u09FFA-Za-z0-9]+)[\)）]\s*)?(?:নং|নম্বর)?\s*(?:প্রশ্নের?\s*)?উত্তর(?:\s*[\(（]([\u0980-\u09FFA-Za-z0-9]+)[\)）])?',
        arabic_lines
    )
    if not beng_m:
        beng_m = re.search(
            r'(?:(?:প্রশ্নের?\s*)?উত্তর)\s*(?:নং|নম্বর)?\s*[:\-]?\s*([0-9]{1,2})\s*(?:[\(（]([\u0980-\u09FFA-Za-z0-9]+)[\)）])?',
            arabic_lines
        )
    if beng_m:
        q_num = beng_m.group(1)
        sub_part = beng_m.group(2) or (beng_m.group(3) if len(beng_m.groups()) >= 3 else None)
        if sub_part:
            ascii_sub = BENGALI_SUBPART_MAP.get(sub_part, sub_part.upper())
            return f"{q_num}({ascii_sub})"
        return q_num

    # 3. Dynamic Subpart (A/B/C/D) or (ক/খ/গ/ঘ) following active parent question
    if current_parent:
        base_parent = re.sub(r'\(.*?\)|[A-Za-z\u0980-\u09FF]+$', '', str(current_parent)).strip()
        if base_parent.isdigit():
            sub_patterns = [
                ("B", r'^\s*(?:[\(（](?:B|খ)[\)）]|[Ⓑ]|B(?:\s*[\:\.\-]|\s*$))(?:\s*[\:\.\-]?\s*(?:\n|\r\n|Ans|Answer|$))'),
                ("A", r'^\s*(?:[\(（](?:A|ক)[\)）]|[Ⓐ]|A(?:\s*[\:\.\-]|\s*$))(?:\s*[\:\.\-]?\s*(?:\n|\r\n|Ans|Answer|$))'),
                ("C", r'^\s*(?:[\(（](?:C|গ)[\)）]|[Ⓒ]|C(?:\s*[\:\.\-]|\s*$))(?:\s*[\:\.\-]?\s*(?:\n|\r\n|Ans|Answer|$))'),
                ("D", r'^\s*(?:[\(（](?:D|ঘ)[\)）]|[Ⓓ]|D(?:\s*[\:\.\-]|\s*$))(?:\s*[\:\.\-]?\s*(?:\n|\r\n|Ans|Answer|$))'),
            ]
            for sub_char, pat in sub_patterns:
                if re.search(pat, first_lines, re.MULTILINE | re.IGNORECASE):
                    cand = f"{base_parent}({sub_char})"
                    if not valid_q_order or cand in valid_q_order:
                        return cand

    # 4. Dynamic Keyword/Topic Matching from question_obj: best-matching sub-question wins
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
            is_strong_match = False
            if matches >= 2:
                is_strong_match = True
            elif matches == 1:
                matched_w = [w for w in set(words) if w in first_lines_lower]
                if matched_w and len(matched_w[0]) >= 6 and matched_w[0] not in {"english", "second", "paper", "write", "passage"}:
                    top_line = first_lines.split("\n")[0].lower()
                    if any(h in first_lines_lower for h in ["ans", "q", "title", "theme", "paragraph", "topic", "no"]) or matched_w[0] in top_line:
                        is_strong_match = True
            if is_strong_match:
                score = matches / len(set(words))
                if score > best_score:
                    best_q, best_score = sq_no, score
        if best_q:
            return best_q

    # 5. Structural & Semantic Content Fingerprinting (Flowchart, Rearrange table, Letter, Graph, Story, Theme)
    fp = detect_structural_fingerprint(section_text, question_obj=question_obj, valid_q_order=valid_q_order, current_parent=current_parent)
    if fp:
        return fp

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

    # Page-aware continuation state machine: track the last active question per page
    page_last_q: Dict[int, str] = {}

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

        for s_idx, s in enumerate(sections):
            header_q = extract_header_qno(
                s,
                current_parent=current_q_no,
                question_obj=question_obj,
                valid_q_order=valid_q_order
            )
            if header_q:
                current_q_no = header_q
            elif s_idx == 0 and not current_q_no:
                # Continuation from previous page if available
                prev_q = page_last_q.get(page_no - 1)
                if prev_q:
                    current_q_no = prev_q

            # If still no active question, fallback to current_q_no or valid_q_order[0]
            target_q = current_q_no or (valid_q_order[0] if valid_q_order else "1")
            current_q_no = target_q  # anchor active question

            if target_q not in answer_buckets:
                answer_buckets[target_q] = {
                    "text_parts": [],
                    "pages": set(),
                    "errors": []
                }

            answer_buckets[target_q]["text_parts"].append(s.strip())
            answer_buckets[target_q]["pages"].add(page_no)

        if current_q_no:
            page_last_q[page_no] = current_q_no

    # Gather all script-level errors from extraction
    all_extracted_errors = []
    if pages:
        for p in pages:
            if p.stage3_errors and p.stage3_errors.errors:
                for e in p.stage3_errors.errors:
                    d = e.model_dump() if hasattr(e, "model_dump") else dict(e)
                    d.setdefault("page_no", p.page_no)
                    all_extracted_errors.append(d)
    if not all_extracted_errors and extraction.stage3_errors and extraction.stage3_errors.errors:
        all_extracted_errors = [e.model_dump() if hasattr(e, "model_dump") else dict(e) for e in extraction.stage3_errors.errors]

    # Convert buckets to structured AlignedAnswerItem list
    aligned_items: List[AlignedAnswerItem] = []

    # Order by valid_q_order if known, else natural order
    all_keys = list(answer_buckets.keys())
    if valid_q_order:
        ordered_keys = [k for k in valid_q_order if k in answer_buckets] + [k for k in all_keys if k not in valid_q_order]
    else:
        ordered_keys = sorted(all_keys)

    # Pre-build combined answer texts for matching
    combined_texts = {}
    for q in ordered_keys:
        combined_texts[q] = "\n\n".join(answer_buckets[q]["text_parts"]).strip()

    # Attribute errors to questions accurately using context, erroneous text, and question_no
    question_errors: Dict[str, List[Dict[str, Any]]] = {q: [] for q in ordered_keys}
    for err in all_extracted_errors:
        qn = str(err.get("question_no") or "").strip().upper()
        needle = str(err.get("erroneous_text") or "").strip().lower()
        ctx = str(err.get("context_sentence") or "").strip().lower()

        matched_q = None
        # 1. Match by context sentence presence in answer text (highest ground-truth precision)
        if ctx:
            for q in ordered_keys:
                q_txt = combined_texts[q].lower()
                if ctx[:30] in q_txt or (len(ctx) > 15 and ctx[-20:] in q_txt):
                    matched_q = q
                    break

        # 2. Match by erroneous_text within word boundaries in answer text
        if not matched_q and needle:
            for q in ordered_keys:
                q_txt = combined_texts[q].lower()
                if re.search(r'\b' + re.escape(needle) + r'\b', q_txt):
                    matched_q = q
                    break

        # 3. Match by explicit question_no
        if not matched_q and qn:
            for q in ordered_keys:
                clean_q = q.upper().replace("(", "").replace(")", "").strip()
                clean_qn = qn.replace("(", "").replace(")", "").strip()
                if q.upper() == qn or clean_q == clean_qn:
                    matched_q = q
                    break

        # 4. Fallback: match by page presence if error has page_no or page_number
        err_page = err.get("page_no") if err.get("page_no") is not None else err.get("page_number")
        if not matched_q and err_page is not None:
            for q in ordered_keys:
                if err_page in answer_buckets[q]["pages"]:
                    matched_q = q
                    break

        if matched_q:
            err["question_no"] = matched_q
            question_errors[matched_q].append(err)

    for q in ordered_keys:
        b = answer_buckets[q]
        combined_ans = combined_texts[q]
        # Deduplicate error dicts by erroneous_text + sentence
        seen_errs = set()
        dedup_errors = []
        for err in question_errors[q]:
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
    cur_q = valid_q_order[0] if valid_q_order else "1"
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
