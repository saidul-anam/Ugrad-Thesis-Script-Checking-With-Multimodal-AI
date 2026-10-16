"""
Answer Segmenter & Question Aligner.

Segments multi-page handwritten exam transcripts into question-aligned answer blocks,
mapping continuation pages, headers, and linguistic errors to their corresponding exam questions.
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from src.core.schemas import (
    PageExtractionResult,
    ExtractionResult,
    ExtractedQuestion,
    AlignedAnswerItem
)
from src.utils.text_metrics import levenshtein


HEADER_NORMALIZATION_RULES = [
    # Leading number question headers: '2 NO.Q.Ans', '3 NO. Q. Ans', '4. NO. Q. Ans:-', '5 NO: Q. Ans:', '11 No. Q. Ans.', '13.NO.Q.Ans'
    (re.compile(r'(?im)^\s*(?:\[struck:[^\]]*\]\s*)*([0-9]{1,3})\s*[\.\:\-]?\s*(?:(?:NO|No|no)\b[\.\:\s]*(?:Q|Ques|Question|qn)\b|(?:Q|Ques|Question|qn)\b[\.\:\s]*(?:(?:NO|No|no)\b[\.\:\s]*)?|(?:NO|No|no)\b[\.\:\s]*)\s*(?:Ans|Answers?|Answen|Answering)?\s*[\:\.\-]?', re.IGNORECASE), r'Ans to the Q No. \1\n'),
    # Leading number with Ans headers: '1 Ans:', '2 Ans:', '2. Ans:', '3. Ans:', '(5 Ans:)', '11 Ans', '12 Ans', '13 Ans:'
    (re.compile(r'(?im)^\s*(?:\[struck:[^\]]*\]\s*)*[\(（]?([0-9]{1,3})\s*[\.\:\-]?[ \t]*(?:Ans|Answers?|Answen|Answering)\b\s*[\:\.\-\)]?', re.IGNORECASE), r'Ans to the Q No. \1\n'),
    # Standalone parenthesized question numbers on their own line: e.g. '(8)', '(5)'
    (re.compile(r'(?im)^\s*(?:\[struck:[^\]]*\]\s*)*[\(（]([0-9]{1,2})[\)）]\s*[\:\.\-]?\s*$', re.IGNORECASE), r'Ans to the Q No. \1\n'),
    # Standalone bare question number on its own line: e.g. '5.', '11.'
    (re.compile(r'(?im)^\s*(?:\[struck:[^\]]*\]\s*)*([0-9]{1,2})\s*[\.\:\-]\s*$', re.IGNORECASE), r'Ans to the Q No. \1\n'),
    # Bare number headers e.g. '11 no', '12 no', '13no', '1no', '2no', '4 নং', '9 নং'
    (re.compile(r'(?im)^\s*(?:\[struck:[^\]]*\]\s*)*([0-9০-৯]{1,2})\s*(?:no|num|নং)\b[\.\:\-]?', re.IGNORECASE), r'Ans to the Q No. \1\n'),
    # Numberless question headers: 'NO.Q-Ans', 'NO. Q. Ans', '_NO Q: Ai'
    (re.compile(r'(?im)^\s*(?:\[struck:[^\]]*\]\s*)*(?:_?\s*(?:NO|No|no)\b[\.\:\s]*(?:Q|Ques|Question|qn)\b[\.\:\s\-]*(?:Ans|Answers?|Answen|Answering|Ai)?)\s*[\:\.\-]?', re.IGNORECASE), r'Ans to the Q\n'),
    # Standalone Ans No X: e.g. 'Ans No 13.', 'Ans No. 12', 'Ans Num 5'
    (re.compile(r'(?im)^\s*(?:\[struck:[^\]]*\]\s*)*Ans\s*(?:No|Num|Number)[\.\:\s]+([0-9]{1,3})\b[\.\:\-]?', re.IGNORECASE), r'Ans to the Q No. \1\n'),
    # In-line sentence header: '...sentence. Answering to the Question No 8'
    (re.compile(r'([.?!;]\s+)(?:Ans(?:i?to|i\s*to)?|Answers?|Answen|Answering|Dans|Ass)\s+(?:to|of)\s+(?:the\s+)?(?:Questions?|Ques|Qus|Quse|due|dues|Qno|qn|Q\s*N)\s*(?:\b(?:number|num|no)\b\.?|[nN]o[\.\s]*ber\b)?\s*([0-9]{1,3})\b', re.IGNORECASE), r'\1\nAns to the Q No. \2\n'),
    (re.compile(r'(?im)^\s*(?:(?:\([^\)]*\)|Part[\s\-]*[A-Za-z0-9]+)\s*[:\-]\s*)?(?:Ans(?:i?to|i\s*to)?|Answers?|Answen|Answering|Dans|Ass|Am|Arn|Anis)\s+(?:to|of)\s+(?:the\s+)?(?:Questions?|Ques|Qus|Quse|due|dues|Qno|qn|Q\s*N)\s*(?:\b(?:number|num|no)\b\.?|[nN]o[\.\s]*ber\b)?\s*[\-:]?', re.IGNORECASE), 'Ans to the Q No. '),
    (re.compile(r'\bQus\b', re.IGNORECASE), 'Ques'),
    (re.compile(r'\bQno\b', re.IGNORECASE), 'Q No.'),
    (re.compile(r'\bQtn\b', re.IGNORECASE), 'Ques'),
    (re.compile(r'\bAnswen\b', re.IGNORECASE), 'Answer'),
    (re.compile(r'\bNo[\.\s]*ber\b', re.IGNORECASE), 'No.'),
    (re.compile(r'\bQ(?:uestion)?[\s\.\-]+ber\b', re.IGNORECASE), 'Q No.'),
    (re.compile(r'(?im)^\s*Dans\s+to\b'), 'Ans to'),
    (re.compile(r'(?im)^\s*([b-gB-G])\s*ans[:\s]+'), r'(\1) Ans: '),
    (re.compile(r'(?im)^\s*Dans[:\s]+'), '(d) Ans: '),
    (re.compile(r'(?im)^\s*(?:Am|Arn|Anis)[\.,\s]+(?:to|of|the|question|ques|no\b)', re.IGNORECASE), 'Ans to '),
    (re.compile(r'(?im)^\s*(?:Am|Arn|Anis)\.?\s+(?=[0-9])', re.IGNORECASE), 'Ans '),
    (re.compile(r'\b(?:Ass|Ansl)\b', re.IGNORECASE), 'Ans'),
    (re.compile(r'No[.,\s]*[Zz]\b', re.IGNORECASE), 'No. 7'),
    (re.compile(r'\bDans\b', re.IGNORECASE), 'Ans'),
    (re.compile(r'\bdo the [Qq]\b', re.IGNORECASE), 'to the Q'),
    (re.compile(r'Qhe o\.', re.IGNORECASE), 'Q. No.'),
    (re.compile(r'Anseve?r', re.IGNORECASE), 'Answer'),
    (re.compile(r'\bAns(?:i?to|i\s*to)\b', re.IGNORECASE), 'Ans to'),
    (re.compile(r'\bAnsi\b', re.IGNORECASE), 'Ans'),
    (re.compile(r'\b(?:Ans\s*[:\s]*(?:(?:to|of)\s+(?:the\s+)?)?)(?:ques?|qus?|dues?)\b', re.IGNORECASE), 'Ans to the Q'),
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
    # English question headers (handles Ans/Answer/Answers/Answering/Answen/Dans/Ass/Ansl/Am/Arn/Anis, to/of the, Question/Questions/Ques/Q/No/Number, optional leading [struck: ...], optional section prefixes)
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:(?:\([^\)]*\)|Part[\s\-]*[A-Za-z0-9]+)\s*[:\-]\s*)?(?:Ansl?\.?\s*)?(?:Ans(?:i?to|i\s*to)?|Answers?|Answen|Answering|Dans|Ass|Am|Arn|Anis|Q(?:uestions?|ues|ue|us|use|u|n|no|tn)?|due|dues|No|Number|Num)\b[^\n\r0-9]{0,40}[0-9০-৯]{1,3}(?:\s*[\(（][A-Za-z0-9\u0980-\u09FF][\)）])?|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:Ans|Answers?|Answen|Answering|Am|Arn)[:\s]+(?:(?:to|of)\s+)?Qhe\s+o\.\s*No\.\s*[0-9A-Za-z]+|'
    # Leading number question headers: 1. No. Q. Ans, 2 NO.Q.Ans, 3 NO. Q. Ans, 4. NO. Q. Ans:-, 13.NO.Q.Ans, 12 NO. Q. Ans, 11 no, 13no, 4 নং, 9 নং, 1 Ans:, 2 Ans:, (5 Ans:), 11 Ans
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[\(（]?[0-9০-৯]{1,3}\s*[\.\:\-]?[ \t]*(?:(?:NO|No|no)\b[\.\:\s]*(?:Q|Ques|Question|qn)\b|(?:Q|Ques|Question|qn)\b[\.\:\s]*(?:(?:NO|No|no)\b[\.\:\s]*)?|(?:NO|No|no|নং)\b[\.\:\s]*|(?:Ans|Answers?|Answen|Answering)\b)\s*(?:Ans|Answers?|Answen|Answering)?\s*[\:\.\-\)]?|'
    # Standalone parenthesized question number: '(8)', '(5)'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[\(（][0-9০-৯]{1,2}[\)）]\s*[\:\.\-]?(?:\r?\n|$)|'
    # Standalone bare question number on its own line: '5.', '11.'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[0-9০-৯]{1,2}\s*[\.\:\-][ \t]*(?:\r?\n|$)|'
    # Numberless question headers: NO.Q-Ans, NO. Q. Ans, Ans to the Q, _NO Q: Ai
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:_?\s*(?:NO|No|no)\b[\.\:\s]*(?:Q|Ques|Question|qn)\b[\.\:\s\-]*(?:Ans|Answers?|Answen|Answering|Ai)|(?:Ans|Answers?|Answen|Answering|Dans|Ass|Am|Arn|Anis)\s+(?:to|of)\s+(?:the\s+)?(?:Questions?|Ques|Q|qn))\s*[\:\.\-]?|'
    # Standalone question subpart headers: 1(A), 1(B), 1(a), 1(b), 3 a), 3. a), 3. (a)
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[0-9০-৯]{1,3}\s*[\(（][A-Za-z0-9\u0980-\u09FF][\)）]\s*(?:\n|\r\n|Ans|Answers?|:|\.|\-|$)|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[0-9০-৯]{1,3}\s*[\.\:\-]?[ \t]+[\(（]?[a-jA-J][\)）\.]\s+|'
    # Bengali question headers: supports '১ নং উত্তর', '১(ক) নং প্রশ্নের উত্তর', 'উত্তর: ১', 'উত্তর নং ১', '৪ নং', '9 নং'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:[০-৯0-9]{1,3}\s*(?:[\(（][\u0980-\u09FFA-Za-z0-9]+[\)）]\s*)?(?:নং|নম্বর)\s*(?:(?:প্রশ্নের?\s*)?উত্তর)?)|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:[০-৯0-9]{1,3}\s*(?:[\(（][\u0980-\u09FFA-Za-z0-9]+[\)）]\s*)?(?:প্রশ্নের?\s*)?উত্তর)|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:(?:প্রশ্নের?\s*)?উত্তর\s*(?:নং|নম্বর)?\s*[:\-]?\s*[০-৯0-9]{1,3})|'
    # Subparts: (A), (B), (ক), (খ), circled letters Ⓐ-Ⓩ, or standalone letter line before sub-items
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[\(（](?:[A-Za-z]|[ক-ঘ])[\)）]\s*(?:\n|\r\n|Ans|Answers?)|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[Ⓐ-Ⓩⓐ-ⓩ]\s*(?:\n|\r\n|Ans|Answers?)|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[A-Za-z]\s*(?:\n|\r\n)\s*(?=[a-z0-9][\)\.]|\([a-z0-9]\))|'
    # Structural headers (Flowchart, Theme, Rearrange, Summary, Story, Chart)
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*Flow[\s\-]*chart\s*[:\-]|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*Theme\s*:\s+|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*Summary\s*:\s+|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*Rearrange\s*[:\-]|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*The\s+(?:graph|chart)\s+shows\b|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*Once\s+upon\s+a\s+time\b'
    r'))',
    re.IGNORECASE
)

HEADER_EXTRACT_REGEX = re.compile(
    r'^\s*(?:\[struck:[^\]]*\]\s*)*(?:(?:\([^\)]*\)|Part[\s\-]*[A-Za-z0-9]+)\s*[:\-]\s*)?(?:Ansl?\.?\s*)?(?:Ans(?:i?to|i\s*to)?|Answers?|Answen|Answering|Dans|Ass|Am|Arn|Anis|Q(?:uestions?|ues|ue|us|use|u|n|no|tn)?|due|dues|No|Number|Num)\b[^\n\r0-9]{0,40}([0-9]{1,3}[ \t]*(?:\([A-Za-z0-9]\))?|[A-Za-z]\b)|'
    r'^\s*(?:\[struck:[^\]]*\]\s*)*(?:Ans(?:i?to|i\s*to)?|Answers?|Answen|Answering)?\s*No[\.\s]+([0-9]{1,3})\.?|'
    r'^\s*(?:\[struck:[^\]]*\]\s*)*([0-9]{1,3})\s*[\.\:\-]?\s*(?:(?:NO|No|no)\b[\.\:\s]*(?:Q|Ques|Question|qn)\b|(?:Q|Ques|Question|qn)\b[\.\:\s]*(?:(?:NO|No|no)\b[\.\:\s]*)?|(?:NO|No|no)\b[\.\:\s]*)\s*(?:Ans|Answers?|Answen|Answering)?\s*[\:\.\-]?|'
    r'^\s*(?:\[struck:[^\]]*\]\s*)*([0-9]{1,3})[ \t]*[\(（]([A-Za-z0-9])[\)）]\s*[\:\.\-]?\s*(?:\n|\r\n|Ans|Answers?|$)|'
    r'^\s*(?:\[struck:[^\]]*\]\s*)*[\(（]([A-Za-z])[\)）]\s*(?:\n|\r\n|Ans|Answers?|$)|'
    r'^\s*(?:\[struck:[^\]]*\]\s*)*([Ⓐ-Ⓩⓐ-ⓩ])\s*(?:\n|\r\n|Ans|Answers?|$)|'
    r'^\s*(?:\[struck:[^\]]*\]\s*)*([A-Za-z])\s*$',
    re.IGNORECASE | re.MULTILINE
)

VALID_HEADER_INTERMEDIATE_WORDS = {
    "to", "of", "the", "question", "questions", "quest", "qust", "ques", "que", "qus", "quse", "quss", "qusst", "qussgion", "qn", "qno", "qtn", "qns", "quesn", "questing", "qusstion", "no", "number", "num", "n", "q", "qhe", "o", "ans", "answer", "answering", "for", "in", "ber"
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


def build_question_entity_index(
    question_obj: Optional[ExtractedQuestion] = None,
    valid_q_order: Optional[List[str]] = None
) -> Dict[str, Set[str]]:
    """
    Build a dynamic index of distinctive keywords/entities for each question on the paper.
    Mines terms dynamically from question_obj sub_questions and question_text.
    Zero hardcoding of specific subjects or questions.
    """
    index: Dict[str, Set[str]] = {}
    if not question_obj:
        return index

    q_order = list(valid_q_order or [])
    if not q_order and getattr(question_obj, "sub_questions", None):
        for sq in question_obj.sub_questions:
            q_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
            if q_no and q_no not in q_order:
                q_order.append(q_no)

    for q in q_order:
        clean_q = re.sub(r'\(.*?\)', '', str(q)).strip()
        entities: Set[str] = set()

        # 1. Source text from question_obj.question_text
        if getattr(question_obj, "question_text", None):
            try:
                from src.pipeline.stage4_modes import source_text_for_question
                src = source_text_for_question(clean_q, question_obj.question_text)
                if src:
                    for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', src):
                        wl = w.lower()
                        if wl not in _KEYWORD_STOPWORDS:
                            entities.add(wl)
            except Exception:
                pass

        # 2. sub_questions catalog
        if getattr(question_obj, "sub_questions", None):
            for sq in question_obj.sub_questions:
                sq_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
                if sq_no in (q, clean_q):
                    for field in ("name", "title", "prompt", "question_prompt", "description", "printed_task"):
                        val = str(sq.get(field) or "")
                        for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', val):
                            wl = w.lower()
                            if wl not in _KEYWORD_STOPWORDS:
                                entities.add(wl)

        if entities:
            index[q] = entities

    # 3. Mine reading passages that introduce questions (e.g. 'Read the text and answer the questions 3 & 4 :')
    if getattr(question_obj, "question_text", None):
        passage_matches = re.findall(
            r'(?i)(?:Read\s+the\s+(?:text|passage)[^\n]*?(?:questions?|qn|no\.?)\s*([0-9\s,&and\-to]+)[\s:]*\n)(.*?)(?=\n\s*(?:Read\s+the\s+(?:text|passage)|[0-9]{1,2}\.\s|Part[\s\-]*[A-Z]|\Z))',
            question_obj.question_text,
            re.DOTALL
        )
        for q_targets, p_body in passage_matches:
            target_nums = re.findall(r'\b[0-9]{1,2}\b', q_targets)
            p_words = {
                w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', p_body)
                if w.lower() not in _KEYWORD_STOPWORDS
            }
            if p_words:
                for t_num in target_nums:
                    for q in q_order:
                        clean_q = re.sub(r'\(.*?\)', '', str(q)).strip()
                        if clean_q == t_num:
                            index.setdefault(q, set()).update(p_words)

    return index


def compute_entity_overlap(text: str, entities: Set[str]) -> int:
    """Count distinctive entity tokens from the set present in text."""
    if not text or not entities:
        return 0
    words = {w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', text)}
    return len(words & entities)


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

    for line_idx in range(min(2, len(clean_lines))):
        top_line = re.sub(r'\[struck:[^\]]*\]', '', clean_lines[line_idx]).strip()
        if not top_line:
            continue

        # OCR artifact 'No. Z' -> 7 or 2
        if re.search(r'\bNo[.,\s]*[Zz]\b', top_line, re.IGNORECASE):
            if valid_q_order:
                if "7" in valid_q_order:
                    return "7"
                if "2" in valid_q_order:
                    return "2"
            return "7"

        # Standalone question subpart format e.g. "1(A)" or "2(B)" or "3(b)" or "3 a)" or "3. a)"
        standalone_m = re.match(r'^\s*([0-9]{1,3})\s*[\(（]([A-Za-z])[\)）]\s*[\:\.\-]?', top_line)
        if not standalone_m:
            standalone_sub_m = re.match(r'^\s*([0-9]{1,3})\s*[\.\:\-]?[ \t]+[\(（]?([a-jA-J])[\)）\.]\s+', top_line)
            if standalone_sub_m:
                cand_base = str(int(standalone_sub_m.group(1)))
                if valid_q_order and cand_base in valid_q_order:
                    # Parent question is directly in valid_q_order (e.g. Q3 Cloze where items are a, b, c...)
                    return cand_base
                cand_sub = f"{cand_base}({standalone_sub_m.group(2).upper()})"
                if not valid_q_order or cand_sub in valid_q_order:
                    return cand_sub
        if standalone_m:
            candidate = f"{int(standalone_m.group(1))}({standalone_m.group(2).upper()})"
            if not valid_q_order or candidate in valid_q_order:
                return candidate

        # Leading number question headers e.g. "2 NO.Q.Ans", "3 NO. Q. Ans", "1 Ans:", "2. Ans:", "(5 Ans:)", "11 Ans", "13.NO.Q.Ans", "12 NO. Q. Ans", "11 no", "13no", "1no", "2no", "4 নং", "9 নং"
        leading_num_m = re.match(
            r'^\s*(?:\[struck:[^\]]*\]\s*)*[\(（]?([0-9]{1,3})\s*[\.\:\-]?[ \t]*(?:(?:NO|No|no)\b[\.\:\s]*(?:Q|Ques|Question|qn)\b|(?:Q|Ques|Question|qn)\b[\.\:\s]*(?:(?:NO|No|no)\b[\.\:\s]*)?|(?:NO|No|no|নং)\b[\.\:\s]*|(?:Ans|Answers?|Answen|Answering)\b)\s*(?:Ans|Answers?|Answen|Answering)?\s*[\:\.\-\)]?',
            top_line,
            re.IGNORECASE
        )
        if not leading_num_m:
            # Bare number headers like '11 no', '13no', '1no', '2 no'
            leading_num_m = re.match(
                r'^\s*(?:\[struck:[^\]]*\]\s*)*([0-9]{1,2})\s*(?:no|num)\b[\.\:\-]?',
                top_line,
                re.IGNORECASE
            )
        if not leading_num_m:
            # Standalone parenthesized question numbers e.g. "(8)" or bare question line "5."
            leading_num_m = re.match(
                r'^\s*(?:\[struck:[^\]]*\]\s*)*(?:[\(（]([0-9]{1,2})[\)）]\s*[\:\.\-]?|([0-9]{1,2})\s*[\.\:\-])\s*$',
                top_line,
                re.IGNORECASE
            )
        if leading_num_m:
            cand_raw = leading_num_m.group(1) or (leading_num_m.group(2) if len(leading_num_m.groups()) > 1 else None)
            if cand_raw:
                cand_num = str(int(cand_raw))
                if not valid_q_order or cand_num in valid_q_order:
                    return cand_num

        # Connected header anchor pattern: requires anchor (Ans/Answer/Answering/Dans/Ass/Q/No) connected to a number
        # through valid header connectors (to, of, the, question, no, punctuation) or 1-edit OCR typos.
        header_m = re.match(
            r'^\s*(?:(?:\([^\)]*\)|Part[\s\-]*[A-Za-z0-9]+)\s*[:\-]\s*)?(?:Ansl?\.?\s*)?(?:Ans(?:i?to|i\s*to)?|Answers?|Answen|Answering|Dans|Ass|Am|Arn|Anis|Q(?:uestions?|ues|ue|n)?|due|dues|No|Number|Num)\b([^\n\r0-9]{0,40})([0-9]{1,3})\s*(?:[\(（\.\s]*([A-Za-z0-9])[\)）]?)?',
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

        # Struck-number recovery: If a line contains a struck question number (e.g. 'Ans to the Q no. [struck: 5] 1')
        # where the candidate number is an already-completed earlier question, recover the struck question.
        if valid_q_order and current_parent:
            struck_m = re.search(r'\[struck:\s*([0-9]{1,2})\s*\]', clean_lines[line_idx])
            if struck_m:
                struck_num = str(int(struck_m.group(1)))
                if struck_num in valid_q_order:
                    try:
                        curr_parent_base = re.sub(r'\(.*?\)', '', str(current_parent)).strip()
                        if curr_parent_base.isdigit() and int(curr_parent_base) >= int(raw_num) and int(struck_num) > int(curr_parent_base):
                            candidate = struck_num
                    except (ValueError, TypeError):
                        pass

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

            # The subpart letter is not a question on this paper (e.g. a list item "(i)" under the header):
            # the header still names the base question.
            if sub_letter and raw_num in valid_q_order:
                return raw_num

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
            if "flow" in combined:
                q_map["flowchart"] = sq_no
            elif "rearrang" in combined or "jumbled" in combined or "order" in combined:
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
            elif "match" in combined:
                q_map["matching"] = sq_no
            elif "table" in combined:
                q_map["table"] = sq_no

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

    # 1b. Matching / Substitution Table: e.g. (a+v) = ..., (b+iii) = ..., (a+iii), a) (a+iii)
    has_matching_notation = bool(re.search(r'[\(\[]?\s*[a-eA-E0-9]+\s*\+\s*[ivx0-9a-e]+\s*[\)\]]?', section_text, re.I)) and len(re.findall(r'[\(\[]?\s*[a-eA-E0-9]+\s*\+\s*[ivx0-9a-e]+\s*[\)\]]?', section_text, re.I)) >= 2
    has_matching_kw = bool(re.search(r'\b(?:match(?:ing)?|column\s*[ab])\b', first_lines, re.I))
    if has_matching_notation or has_matching_kw:
        match_q = _get_q("matching") or _get_q("table") or _get_q("rearrange")
        if match_q:
            return match_q

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
    if question_obj and question_obj.sub_questions:
        for sq in question_obj.sub_questions:
            sq_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
            sq_name = str(sq.get("name") or sq.get("title") or "").lower()
            if "story" in sq_name:
                title_words = [w for w in re.findall(r'[a-z]{4,}', sq_name) if w not in _KEYWORD_STOPWORDS and w != "story"]
                if title_words and sum(1 for w in set(title_words) if w in first_lines) >= 2:
                    if not valid_q_order or sq_no in valid_q_order:
                        return sq_no
    if has_story_opening or has_story_kw:
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
    valid_q_order: Optional[List[str]] = None,
    entity_index: Optional[Dict[str, Set[str]]] = None,
    completed_qs: Optional[Set[str]] = None
) -> Optional[str]:
    """
    Extract canonical question number from the leading lines of a text section.
    Supports English & Bengali headers, dynamic schema keywords, sub-parts,
    structural & semantic content fingerprinting, and sequence collision disambiguation.
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

    # 2. Bengali Question Header Matching (supports '১ নং উত্তর', '১(ক) নং', '১ নং প্রশ্নের উত্তর (ক)', 'উত্তর: ১', '৪ নং', '9 নং')
    beng_m = re.search(
        r'([0-9]{1,2})\s*(?:[\(（]([\u0980-\u09FFA-Za-z0-9]+)[\)）]\s*)?(?:নং|নম্বর)?\s*(?:প্রশ্নের?\s*)?উত্তর(?:\s*[\(（]([\u0980-\u09FFA-Za-z0-9]+)[\)）])?',
        arabic_lines
    )
    if not beng_m:
        beng_m = re.search(
            r'(?:(?:প্রশ্নের?\s*)?উত্তর)\s*(?:নং|নম্বর)?\s*[:\-]?\s*([0-9]{1,2})\s*(?:[\(（]([\u0980-\u09FFA-Za-z0-9]+)[\)）])?',
            arabic_lines
        )
    if not beng_m:
        # Bare Bengali header like '৪ নং', '9 নং', '৮ নং'
        beng_m = re.search(
            r'^\s*([0-9]{1,2})\s*(?:নং|নম্বর)\b',
            arabic_lines,
            re.MULTILINE
        )
    if beng_m:
        q_num = beng_m.group(1)
        sub_part = beng_m.group(2) if beng_m.lastindex and beng_m.lastindex >= 2 and beng_m.group(2) else None
        if not sub_part and beng_m.lastindex and beng_m.lastindex >= 3:
            sub_part = beng_m.group(3)
        if sub_part:
            ascii_sub = BENGALI_SUBPART_MAP.get(sub_part, sub_part.upper())
            return f"{q_num}({ascii_sub})"
        return q_num

    # 3. Dynamic Subpart (A/B/C/D) or (ক/খ/গ/ঘ) following active parent question
    if current_parent:
        base_parent = re.sub(r'\(.*?\)|[A-Za-z\u0980-\u09FF]+$', '', str(current_parent)).strip()
        if base_parent.isdigit():
            # Check if this line is a continuation sub-item like c), d), e)
            sub_continuation = re.match(r'^\s*([c-gC-G])[\)\.]', first_lines)
            if sub_continuation:
                # Continuation sub-item belonging to current_parent; do not switch questions
                return None

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
        top_line = first_lines.split("\n")[0].strip()
        is_heading_like = len(top_line.split()) <= 6 or bool(
            re.search(r'\b(?:ans|q|ques|question|no|title|theme|paragraph|topic)\b', top_line, re.IGNORECASE)
        )

        if is_heading_like:
            for sq in question_obj.sub_questions:
                sq_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
                sq_name = str(sq.get("name") or sq.get("title") or "").strip()
                if not sq_no or not sq_name:
                    continue
                # Monotonicity check: if current_parent is active, allow jump backward only with strong keyword match
                if valid_q_order and current_parent and current_parent in valid_q_order and sq_no in valid_q_order:
                    if valid_q_order.index(sq_no) < valid_q_order.index(current_parent):
                        words = [
                            w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', sq_name)
                            if w.lower() not in _KEYWORD_STOPWORDS
                        ]
                        if not words or sum(1 for w in set(words) if w in first_lines_lower) < 2:
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
                        if bool(re.search(r'\b(?:ans|q|ques|question|no|title|theme|paragraph|topic)\b', first_lines_lower, re.IGNORECASE)) or matched_w[0] in top_line.lower():
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

    # 6. Dynamic Schema-Driven Entity Fingerprinting (Headless / Unlabeled Answers)
    # Strictly applies to substantial headless answers (>= 25 words).
    # Never hijacks short sub-items like (a), (b), (c)...
    if question_obj and valid_q_order:
        is_sub_item = bool(re.match(r'^\s*[\(（]?[a-jA-J0-9][\)）\.]', first_lines))
        if len(section_text.split()) < 25 or is_sub_item:
            return None

        e_index = entity_index or build_question_entity_index(question_obj, valid_q_order)
        best_match_q = None
        best_match_overlap = 0
        if current_parent and current_parent in valid_q_order:
            curr_idx = valid_q_order.index(current_parent)
            search_qs = valid_q_order[curr_idx + 1:] + valid_q_order[:curr_idx]
        else:
            search_qs = list(valid_q_order)

        # Exclude questions that are already completed from headless entity matching
        if completed_qs:
            search_qs = [q for q in search_qs if q not in completed_qs]

        parent_overlap = compute_entity_overlap(section_text, e_index.get(current_parent, set())) if current_parent else 0

        for q in search_qs:
            q_ents = e_index.get(q, set())
            overlap = compute_entity_overlap(section_text, q_ents)
            if overlap >= 3 and overlap > parent_overlap and overlap > best_match_overlap:
                best_match_overlap = overlap
                best_match_q = q

        if best_match_q and best_match_overlap >= 3:
            return best_match_q

    return None


def audit_segmented_answers(
    answer_buckets: Dict[str, Dict[str, Any]],
    valid_q_order: List[str],
    question_obj: Optional[ExtractedQuestion] = None
) -> Dict[str, Any]:
    """
    Automated Segment Health & Completeness Auditor (Layer 2).
    Inspects question buckets for missing questions, embedded headers, and hanging syntax.
    """
    found_keys = set(answer_buckets.keys())
    missing_qs = [q for q in valid_q_order if q not in found_keys] if valid_q_order else []

    embedded_headers = []
    for q, data in answer_buckets.items():
        parts = data.get("text_parts", [])
        for pt in parts:
            lines = pt.split("\n")
            if len(lines) > 2:
                for line in lines[2:]:
                    cand = _extract_explicit_english_header(line, valid_q_order=valid_q_order)
                    if cand and str(cand) != str(q):
                        embedded_headers.append({
                            "parent_q": q,
                            "target_q": cand,
                            "line": line.strip()
                        })

    hanging_segments = []
    for q, data in answer_buckets.items():
        full_text = "\n\n".join(data.get("text_parts", [])).strip()
        if full_text:
            words = full_text.split()
            if words:
                last_w = words[-1].lower()
                if last_w in {"and", "because", "although", "which", "that", "to", "with", "or"}:
                    hanging_segments.append(q)
                elif last_w[-1] in {",", ";", "-", ":", "("}:
                    hanging_segments.append(q)

    content_misalignments = []
    paper_mismatch = False
    if question_obj and valid_q_order:
        e_index = build_question_entity_index(question_obj, valid_q_order)
        for q, data in answer_buckets.items():
            full_text = "\n\n".join(data.get("text_parts", [])).strip()
            if len(full_text.split()) >= 20:
                q_ents = e_index.get(q, set())
                self_overlap = compute_entity_overlap(full_text, q_ents)
                for mis_q in missing_qs:
                    mis_ents = e_index.get(mis_q, set())
                    mis_overlap = compute_entity_overlap(full_text, mis_ents)
                    if mis_overlap >= 3 and self_overlap == 0:
                        content_misalignments.append({
                            "labeled_q": q,
                            "likely_intended_q": mis_q,
                            "self_overlap": self_overlap,
                            "intended_overlap": mis_overlap
                        })
                        break

        # Check overall paper relevance across the entire script
        all_paper_ents = set().union(*e_index.values()) if e_index else set()
        all_script_text = " ".join("\n\n".join(d.get("text_parts", [])) for d in answer_buckets.values())
        overall_overlap = compute_entity_overlap(all_script_text, all_paper_ents)
        if len(all_script_text.split()) >= 150 and overall_overlap < 3:
            paper_mismatch = True

    return {
        "missing_questions": missing_qs,
        "embedded_headers": embedded_headers,
        "hanging_segments": hanging_segments,
        "content_misalignments": content_misalignments,
        "paper_mismatch": paper_mismatch,
        "is_clean": len(missing_qs) == 0 and len(embedded_headers) == 0 and len(content_misalignments) == 0 and not paper_mismatch
    }


def resolve_ambiguous_snippet(
    snippet: str,
    candidate_qs: List[Tuple[str, str]],
    engine: Optional[Any] = None
) -> Optional[str]:
    """
    Use engine (LLM) to classify which question an ambiguous or unheaded boundary snippet belongs to.
    Fast, focused micro-prompt (< 120 prompt tokens, max 8 new tokens).
    """
    if not engine or not candidate_qs or not snippet.strip():
        return None

    cand_lines = "\n".join(f"- Question {q}: {desc}" for q, desc in candidate_qs)
    prompt = (
        f"You are an exam layout boundary resolver.\n"
        f"Which exam question does the following student answer text belong to?\n\n"
        f"STUDENT TEXT:\n\"\"\"\n{snippet.strip()[:600]}\n\"\"\"\n\n"
        f"CANDIDATE QUESTIONS:\n{cand_lines}\n\n"
        f"Reply with ONLY the question number (e.g. {candidate_qs[0][0]}) and nothing else."
    )
    try:
        resp = ""
        if hasattr(engine, "generate_text"):
            resp = engine.generate_text(prompt, max_new_tokens=8, temperature=0.0)
        elif hasattr(engine, "chat"):
            resp = engine.chat(prompt)

        resp = resp.strip()
        m = re.search(r'\b([0-9]{1,3})\b', resp)
        if m:
            cand_q = m.group(1)
            valid_set = {str(c[0]) for c in candidate_qs}
            if cand_q in valid_set:
                return cand_q
    except Exception as e:
        print(f"[AnswerSegmenter] Micro-LLM boundary resolution failed: {e}")
    return None


def reconcile_delayed_headers(
    answer_buckets: Dict[str, Dict[str, Any]],
    valid_q_order: List[str],
    pages: Optional[List[Any]] = None,
    question_obj: Optional[ExtractedQuestion] = None,
    engine: Optional[Any] = None
) -> None:
    """
    Physical Exam Invariant Reconciler:
    If a bucket for Question Q contains an explicit header for Q at line L > 0
    preceded by non-header body lines, that prefix text belongs to a predecessor question.
    Uses Micro-LLM resolution if engine is provided; otherwise safely restores to the predecessor.
    """
    if not valid_q_order:
        return

    q_titles = {}
    if question_obj and question_obj.sub_questions:
        for sq in question_obj.sub_questions:
            sq_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
            if sq_no:
                q_titles[sq_no] = str(sq.get("name") or sq.get("title") or sq.get("prompt") or f"Question {sq_no}")

    for idx, q in enumerate(valid_q_order):
        data = answer_buckets.get(q)
        if not data or not data.get("text_parts"):
            continue
        full_text = "\n".join(data["text_parts"])
        lines = [l.strip() for l in full_text.split("\n") if l.strip()]
        if len(lines) <= 1:
            continue

        first_line = lines[0]
        m_first = re.match(
            r'^(?:\[struck:[^\]]*\]\s*)*(?:Ans(?:wers?|wer|ing)?\s+to\s+(?:the\s+)?(?:Q|Questions?|Ques)\s*(?:No\.?|Number)?\s*([0-9]{1,2}))',
            first_line,
            re.I
        )
        if m_first and m_first.group(1) == q:
            continue  # Already begins cleanly with explicit header

        for l_idx, line in enumerate(lines[1:], 1):
            m = re.match(
                r'^(?:\[struck:[^\]]*\]\s*)*(?:Ans(?:wers?|wer|ing)?\s+to\s+(?:the\s+)?(?:Q|Questions?|Ques)\s*(?:No\.?|Number)?\s*([0-9]{1,2}))',
                line,
                re.I
            )
            if m and m.group(1) == q:
                preceding_text = "\n".join(lines[:l_idx]).strip()
                post_text = "\n".join(lines[l_idx:]).strip()

                # Determine target predecessor question
                prev_q = valid_q_order[idx - 1] if idx > 0 else None
                dest_q = prev_q

                if engine and prev_q:
                    cand_list = [
                        (prev_q, q_titles.get(prev_q, f"Question {prev_q}")),
                        (q, q_titles.get(q, f"Question {q}"))
                    ]
                    llm_choice = resolve_ambiguous_snippet(preceding_text, cand_list, engine)
                    if llm_choice:
                        dest_q = llm_choice

                if dest_q and dest_q != q:
                    if dest_q not in answer_buckets:
                        answer_buckets[dest_q] = {"text_parts": [], "pages": set(), "errors": []}
                    answer_buckets[dest_q]["text_parts"].append(preceding_text)

                    # Re-attribute page numbers
                    if pages:
                        prec_snip = preceding_text[:40].strip()
                        post_snip = post_text[:40].strip()
                        for p in pages:
                            p_txt = (getattr(getattr(p, "stage2_verification", None), "verified_transcript", "") or
                                     getattr(getattr(p, "stage1_transcription", None), "raw_transcript", "") or "")
                            if prec_snip and prec_snip in p_txt:
                                answer_buckets[dest_q]["pages"].add(p.page_no)
                                if post_snip not in p_txt:
                                    data["pages"].discard(p.page_no)

                    print(f"[AnswerSegmenter] Reconciled delayed header in Q{q}: moved {len(preceding_text.split())} words -> Q{dest_q}")

                data["text_parts"] = [post_text]
                break


def segment_script_into_questions(
    extraction: ExtractionResult,
    question_obj: Optional[ExtractedQuestion] = None,
    engine: Optional[Any] = None
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
            q_title = str(sq.get("name") or sq.get("title") or sq.get("prompt") or "").lower()
            # Institutional assessment questions (e.g. "Q14: Class Test", "Class Test : 10") are teacher marks carried over,
            # not written on the student semester exam paper.
            task_type = str(sq.get("task_type") or "").lower()
            if task_type == "class_test" or re.match(r'^(?:q[0-9]+[:\s]*)?class[\s_-]*test(?:\s*assessment)?\s*[:0-9\s]*$', q_title.strip(), re.I):
                continue
            if q_no:
                valid_q_order.append(q_no)
                sub_q_names[q_no] = sq.get("name") or sq.get("title") or f"Question {q_no}"

    # Build dynamic entity index for semantic and gap-filling reconciliation
    entity_index = build_question_entity_index(question_obj, valid_q_order) if (question_obj and valid_q_order) else {}

    # Collect answers across all pages
    # Map: q_no -> {"text_parts": [], "pages": [], "errors": []}
    answer_buckets: Dict[str, Dict[str, Any]] = {}
    current_q_no: Optional[str] = None

    pages = extraction.pages or []

    # If pages are not available, segment the combined verified transcript directly
    if not pages:
        full_text = extraction.stage2_verification.verified_transcript or extraction.stage1_transcription.raw_transcript
        return _segment_raw_transcript(full_text, sub_q_names, valid_q_order, extraction.stage3_errors.errors, question_obj=question_obj, entity_index=entity_index)

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
                valid_q_order=valid_q_order,
                entity_index=entity_index,
                completed_qs=set(answer_buckets.keys())
            )
            is_explicit = bool(_extract_explicit_english_header(s, valid_q_order=valid_q_order, current_parent=current_q_no))
            if not is_explicit:
                beng_m = re.search(r'^\s*([0-9০-৯]{1,2})\s*(?:নং|নম্বর)\b', s, re.M)
                if beng_m:
                    is_explicit = True

            if header_q:
                if current_q_no and not is_explicit and header_q != current_q_no:
                    struct_fp = detect_structural_fingerprint(s, question_obj=question_obj, valid_q_order=valid_q_order, current_parent=current_q_no)
                    is_strong_struct = struct_fp and any(kw in s.lower() for kw in ["1 | 2", "1|2", "flow", "rearrange"])
                    if is_strong_struct:
                        current_q_no = struct_fp
                    else:
                        # Continuation of current active question
                        pass
                else:
                    current_q_no = header_q
            elif s_idx == 0 and not current_q_no:
                # Continuation from previous page if available
                prev_q = page_last_q.get(page_no - 1)
                if prev_q:
                    current_q_no = prev_q

            # If section is merely a structural paper divider (e.g. "Part-B : Writing Part", "P-B Writting", "Part - A", "Part-A: Reading Test")
            # and contains no actual answer text, do not append it to any question or default to Q1!
            s_clean = s.strip()
            is_part_divider = bool(re.match(r'^\s*(?:Part[\s\-]*[A-Z0-9]\b|P[\s\-]*[A-Z0-9]\s*(?:Writting|Writing|Reading)?)\b.*$', s_clean, re.I)) and len(s_clean.split()) <= 6
            if not header_q and is_part_divider:
                continue

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

    # -------------------------------------------------------------
    # Stage 2: Delayed-Header Semantic Reconciliation (Physical Exam Invariant)
    # -------------------------------------------------------------
    reconcile_delayed_headers(
        answer_buckets=answer_buckets,
        valid_q_order=valid_q_order,
        pages=pages,
        question_obj=question_obj,
        engine=engine
    )

    # -------------------------------------------------------------
    # Missing-Question Reconciliation (Gap-Filler Engine)
    # -------------------------------------------------------------
    if valid_q_order and question_obj and entity_index:
        found_qs = set(answer_buckets.keys())
        missing_qs = [q for q in valid_q_order if q not in found_qs]

        if missing_qs:
            for mis_q in missing_qs:
                mis_ents = entity_index.get(mis_q, set())
                if not mis_ents:
                    continue

                reallocated = False
                for parent_q, parent_data in list(answer_buckets.items()):
                    if parent_q == mis_q:
                        continue
                    parts = parent_data.get("text_parts", [])
                    if len(parts) <= 1:
                        # Check if single part actually contains embedded headers or section dividers
                        full_part = parts[0] if parts else ""
                        if full_part:
                            norm_fp = normalize_header_text(full_part)
                            sub_pieces = [sp for sp in HEADER_SPLIT_REGEX.split(norm_fp) if sp.strip()]
                            if len(sub_pieces) > 1:
                                parts = sub_pieces
                                parent_data["text_parts"] = parts
                            else:
                                continue
                        else:
                            continue

                    p_idx = 0
                    while p_idx < len(parts):
                        part_text = parts[p_idx]

                        # Explicit header shield: never steal a chunk that explicitly names another question!
                        part_hdr = _extract_explicit_english_header(part_text, valid_q_order=valid_q_order)
                        if part_hdr and str(part_hdr) != str(mis_q):
                            p_idx += 1
                            continue

                        overlap = compute_entity_overlap(part_text, mis_ents)
                        parent_overlap = compute_entity_overlap(part_text, entity_index.get(parent_q, set()))

                        # Strong match for missing question over parent
                        if overlap >= 2 and overlap > parent_overlap:
                            carved_text = parts.pop(p_idx)
                            if mis_q not in answer_buckets:
                                answer_buckets[mis_q] = {
                                    "text_parts": [],
                                    "pages": set(),
                                    "errors": []
                                }
                            answer_buckets[mis_q]["text_parts"].append(carved_text)

                            # Re-attribute page numbers
                            for p in pages:
                                p_txt = (p.stage2_verification.verified_transcript or p.stage1_transcription.raw_transcript or "")
                                if carved_text[:40].strip() and carved_text[:40].strip() in p_txt:
                                    answer_buckets[mis_q]["pages"].add(p.page_no)
                                    if not any(other_pt[:40].strip() in p_txt for other_pt in parts):
                                        parent_data["pages"].discard(p.page_no)
                            reallocated = True
                            continue
                        p_idx += 1

                    if reallocated:
                        break

    # -------------------------------------------------------------
    # Layer 3: Safe Neural Boundary & Transition Recovery (if engine provided)
    # -------------------------------------------------------------
    if engine and valid_q_order and question_obj:
        found_qs = set(answer_buckets.keys())
        still_missing = [q for q in valid_q_order if q not in found_qs]
        if still_missing:
            q_prompts = {}
            if question_obj and question_obj.sub_questions:
                for sq in question_obj.sub_questions:
                    sq_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
                    if sq_no:
                        p_txt = str(sq.get("prompt") or sq.get("question_prompt") or sq.get("name") or sq.get("title") or "")
                        q_prompts[sq_no] = p_txt

            for mis_q in still_missing:
                target_prompt_desc = q_prompts.get(mis_q, f"Question {mis_q}")
                # First: Check for inner unparsed header lines deterministically before calling LLM
                recovered_via_line = False
                for parent_q, parent_data in list(answer_buckets.items()):
                    if parent_q == mis_q:
                        continue
                    parts = parent_data.get("text_parts", [])
                    for pt_idx, pt in enumerate(parts):
                        pt_lines = pt.split("\n")
                        if len(pt_lines) >= 3:
                            for l_idx in range(2, len(pt_lines)):
                                line_cand = pt_lines[l_idx].strip()
                                cand_hq = extract_header_qno(
                                    line_cand,
                                    current_parent=parent_q,
                                    question_obj=question_obj,
                                    valid_q_order=valid_q_order,
                                    entity_index=entity_index
                                )
                                if cand_hq and str(cand_hq) == str(mis_q):
                                    # Safe boundary found at this line!
                                    pre_text = "\n".join(pt_lines[:l_idx]).strip()
                                    post_text = "\n".join(pt_lines[l_idx:]).strip()
                                    if pre_text:
                                        parts[pt_idx] = pre_text
                                    else:
                                        parts.pop(pt_idx)
                                    if mis_q not in answer_buckets:
                                        answer_buckets[mis_q] = {"text_parts": [], "pages": set(parent_data.get("pages", set())), "errors": []}
                                    answer_buckets[mis_q]["text_parts"].append(post_text)
                                    print(f"[AnswerSegmenter] Deterministic inner header recovered Question {mis_q} from Question {parent_q} at line: {line_cand[:30]}")
                                    recovered_via_line = True
                                    break
                        if recovered_via_line:
                            break
                    if recovered_via_line:
                        break

                if recovered_via_line:
                    continue

                # Second: Safe LLM transition detection (only on composite segments with >= 100 words)
                candidate_parents = sorted(
                    [(k, v) for k, v in answer_buckets.items() if sum(len(pt.split()) for pt in v.get("text_parts", [])) >= 100],
                    key=lambda x: sum(len(pt.split()) for pt in x[1].get("text_parts", [])),
                    reverse=True
                )
                if not candidate_parents:
                    continue

                for parent_q, parent_data in candidate_parents:
                    parent_text = "\n\n".join(parent_data.get("text_parts", []))
                    if len(parent_text.split()) < 80:
                        continue

                    # Require LLM to specify the exact starting transition line
                    llm_prompt = (
                        f"You are an exam layout boundary detector.\n"
                        f"We are looking for missing Question {mis_q} (Topic: '{target_prompt_desc}').\n"
                        f"The text below is filed under Question {parent_q}:\n"
                        f"\"\"\"\n{parent_text[:2500]}\n\"\"\"\n\n"
                        f"Does this text contain a distinct answer transition where Question {mis_q} begins?\n"
                        f"If YES, quote the EXACT starting line of Question {mis_q}.\n"
                        f"If NO (not attempted here or all of it belongs to Question {parent_q}), reply with NONE.\n"
                        f"Reply ONLY with the starting line or NONE."
                    )
                    try:
                        resp = ""
                        if hasattr(engine, "generate_text"):
                            resp = engine.generate_text(llm_prompt, max_new_tokens=64, temperature=0.0)
                        elif hasattr(engine, "chat"):
                            resp = engine.chat(llm_prompt)

                        if resp and resp.upper() != "NONE" and "\n" not in resp and len(resp) >= 4 and resp in parent_text:
                            # Safe split at transition line
                            split_pos = parent_text.find(resp)
                            pre_split = parent_text[:split_pos].strip()
                            post_split = parent_text[split_pos:].strip()
                            if len(pre_split.split()) >= 15 and len(post_split.split()) >= 10:
                                # Validation: Ensure post_split actually has relevance to mis_q!
                                e_idx = entity_index or (build_question_entity_index(question_obj, valid_q_order) if question_obj and valid_q_order else {})
                                mis_ents = e_idx.get(mis_q, set())
                                overlap = compute_entity_overlap(post_split, mis_ents)
                                cand_h = extract_header_qno(post_split, current_parent=parent_q, question_obj=question_obj, valid_q_order=valid_q_order, entity_index=e_idx)

                                prompt_words = [w.lower() for w in re.findall(r'[A-Za-z]{4,}', target_prompt_desc) if w.lower() not in _KEYWORD_STOPWORDS]
                                prompt_match = sum(1 for w in set(prompt_words) if w in post_split.lower())

                                is_verified = (cand_h and str(cand_h) == str(mis_q)) or overlap >= 2 or prompt_match >= 2
                                if not is_verified:
                                    print(f"[AnswerSegmenter] Layer 3 rejected unverified split for Question {mis_q} from Question {parent_q} (no entity/keyword support; overlap={overlap}, prompt_match={prompt_match})")
                                    continue

                                parent_data["text_parts"] = [pre_split]
                                if mis_q not in answer_buckets:
                                    answer_buckets[mis_q] = {
                                        "text_parts": [],
                                        "pages": set(parent_data.get("pages", set())),
                                        "errors": []
                                    }
                                answer_buckets[mis_q]["text_parts"].append(post_split)
                                print(f"[AnswerSegmenter] Layer 3 Safe Transition verified and recovered Question {mis_q} from Question {parent_q} at boundary: '{resp[:30]}'")
                                break
                    except Exception as ex:
                        print(f"[AnswerSegmenter] Note on neural recovery: {ex}")
                        break

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
    all_errors: List[Any],
    question_obj: Optional[ExtractedQuestion] = None,
    entity_index: Optional[Dict[str, Set[str]]] = None
) -> List[AlignedAnswerItem]:
    """Fallback segmenter when page-level checkpoint structures are absent."""
    normalized = normalize_header_text(full_text)
    sections = HEADER_SPLIT_REGEX.split(normalized)
    sections = [s for s in sections if s.strip()]

    e_index = entity_index or (build_question_entity_index(question_obj, valid_q_order) if (question_obj and valid_q_order) else {})

    items = []
    grouped: Dict[str, List[str]] = {}
    cur_q = valid_q_order[0] if valid_q_order else "1"
    for s in sections:
        hq = extract_header_qno(
            s,
            current_parent=cur_q,
            question_obj=question_obj,
            valid_q_order=valid_q_order,
            entity_index=e_index,
            completed_qs=set(grouped.keys())
        )
        if hq:
            cur_q = hq
        items.append((cur_q, s.strip()))
        grouped.setdefault(cur_q, []).append(s.strip())

    # Delayed-Header Semantic Reconciliation in raw fallback
    raw_buckets = {k: {"text_parts": list(v), "pages": set(), "errors": []} for k, v in grouped.items()}
    reconcile_delayed_headers(raw_buckets, valid_q_order, question_obj=question_obj, engine=None)
    grouped = {k: v["text_parts"] for k, v in raw_buckets.items()}

    # Missing question gap-filler in raw transcript fallback
    if valid_q_order and question_obj and e_index:
        found_qs = set(grouped.keys())
        missing_qs = [q for q in valid_q_order if q not in found_qs]
        for mis_q in missing_qs:
            mis_ents = e_index.get(mis_q, set())
            if not mis_ents:
                continue
            reallocated = False
            for parent_q, parts in list(grouped.items()):
                if parent_q == mis_q:
                    continue
                parts = list(parts)
                if len(parts) <= 1:
                    full_part = parts[0] if parts else ""
                    if full_part:
                        norm_fp = normalize_header_text(full_part)
                        sub_pieces = [sp for sp in HEADER_SPLIT_REGEX.split(norm_fp) if sp.strip()]
                        if len(sub_pieces) > 1:
                            parts = sub_pieces
                            grouped[parent_q] = parts
                        else:
                            continue
                    else:
                        continue

                p_idx = 0
                while p_idx < len(parts):
                    part_text = parts[p_idx]

                    # Explicit header shield: never steal a chunk that explicitly names another question!
                    part_hdr = _extract_explicit_english_header(part_text, valid_q_order=valid_q_order)
                    if part_hdr and str(part_hdr) != str(mis_q):
                        p_idx += 1
                        continue

                    overlap = compute_entity_overlap(part_text, mis_ents)
                    parent_overlap = compute_entity_overlap(part_text, e_index.get(parent_q, set()))
                    if overlap >= 2 and overlap > parent_overlap:
                        carved_text = parts.pop(p_idx)
                        grouped.setdefault(mis_q, []).append(carved_text)
                        reallocated = True
                        continue
                    p_idx += 1
                if reallocated:
                    break

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
