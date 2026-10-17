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
    (re.compile(r'([.?!;]\s+)(?:Ans(?:i?to|i\s*to)?|Answers?|Answen|Answering|Dans|Ass)\s+(?:to|of)\s+(?:the\s+)?(?:Questions?|Ques|Qus|Quse|due|dues|Qno|qn|Q\s*N)\s*(?:\b(?:number|num|no)\b\.?|[nN]o[\.\s]*ber\b)?\s*[\:\-]?\s*([0-9]{1,3})\b', re.IGNORECASE), r'\1\nAns to the Q No. \2\n'),
    (re.compile(r'(?im)^\s*(?:(?:\([^\)]*\)|Part[\s\-]*[A-Za-z0-9]+)\s*[:\-]\s*)?(?:Ans(?:i?to|i\s*to)?|Answers?|Answen|Answering|Dans|Ass|Am|Arn|Anis)\s+(?:to|of)\s+(?:the\s+)?(?:Questions?|Ques|Qus|Quse|due|dues|Qno|qn|Q\s*N(?:o\b\.?|umber\b|um\b)?)\s*(?:\b(?:number|num|no)\b\.?|[nN]o[\.\s]*ber\b)?\s*[\-:]?', re.IGNORECASE), 'Ans to the Q No. '),
    (re.compile(r'\bQus\b', re.IGNORECASE), 'Ques'),
    (re.compile(r'\bQno\b', re.IGNORECASE), 'Q No.'),
    (re.compile(r'\bQtn\b', re.IGNORECASE), 'Ques'),
    (re.compile(r'\bAnswen\b', re.IGNORECASE), 'Answer'),
    (re.compile(r'\bNo[\.\s]*ber\b', re.IGNORECASE), 'No.'),
    (re.compile(r'\bQ(?:uestion)?[\s\.\-]+ber\b', re.IGNORECASE), 'Q No.'),
    (re.compile(r'(?im)^\s*Dans\s+to\b'), 'Ans to'),
    (re.compile(r'(?im)^\s*([b-gB-G])(?:[\s\)\.]+ans[:\s]+|ans\s*:\s*)'), r'(\1) Ans: '),
    (re.compile(r'(?im)^\s*Dans[:\s]+'), '(D) Ans: '),
    (re.compile(r'(?im)^\s*(?:Am|Arn|Anis)[\.,\s]+(?:to|of|the|question|ques|no\b)(?=[^\n]*[0-9])', re.IGNORECASE), 'Ans to '),
    (re.compile(r'(?im)^\s*(?:Am|Arn|Anis)\.?\s+(?=[0-9])', re.IGNORECASE), 'Ans '),
    (re.compile(r'\b(?:Ass|Ansl)\b', re.IGNORECASE), 'Ans'),
    (re.compile(r'\bDans\b', re.IGNORECASE), 'Ans'),
    (re.compile(r'\bdo the [Qq]\b', re.IGNORECASE), 'to the Q'),
    (re.compile(r'Qhe o\.', re.IGNORECASE), 'Q. No.'),
    (re.compile(r'Anseve?r', re.IGNORECASE), 'Answer'),
    (re.compile(r'\bAns(?:i?to|i\s*to)\b', re.IGNORECASE), 'Ans to'),
    (re.compile(r'\bAnsi\b', re.IGNORECASE), 'Ans'),
    (re.compile(r'\b(?:Ans\s*[:\s]*(?:(?:to|of)\s+(?:the\s+)?)?)(?:ques?|qus?|dues?(?=[\s\.\:]*(?:no\b|number\b|[0-9])))\b', re.IGNORECASE), 'Ans to the Q'),
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
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:(?:\([^\)]*\)|Part[\s\-]*[A-Za-z0-9]+)\s*[:\-]\s*)?(?:Ansl?\.?\s*)?(?:Ans(?:i?to|i\s*to)?|Answers?|Answen|Answering|Dans|Ass|Am|Arn|Anis|Q(?:uestions?|ues|ue|us|use|u|n|no|tn)?|due|dues|No|Number|Num)\b[^\n\r0-9|]{0,40}[0-9০-৯]{1,3}(?:\s*[\(（][A-Za-z0-9\u0980-\u09FF][\)）])?|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*(?:Ans|Answers?|Answen|Answering|Am|Arn)[:\s]+(?:(?:to|of)\s+)?Qhe\s+o\.\s*No\.\s*[0-9A-Za-z]+|'
    # Leading number question headers: 1. No. Q. Ans, 2 NO.Q.Ans, 3 NO. Q. Ans, 4. NO. Q. Ans:-, 13.NO.Q.Ans, 12 NO. Q. Ans, 11 no, 13no, 4 নং, 9 নং, 1 Ans:, 2 Ans:, (5 Ans:), 11 Ans
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[\(（]?[0-9০-৯]{1,3}\s*[\.\:\-]?[ \t]*(?:(?:NO|No|no)\b[\.\:\s]*(?:Q|Ques|Question|qn)\b|(?:Q|Ques|Question|qn)\b[\.\:\s]*(?:(?:NO|No|no)\b[\.\:\s]*)?|(?:NO|No|no|নং)\b[\.\:\s]*|(?:Ans|Answers?|Answen|Answering)\b)\s*(?:Ans|Answers?|Answen|Answering)?\s*[\:\.\-\)]?|'
    # Standalone parenthesized question number: '(8)', '(5)'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[\(（][0-9০-৯]{1,2}[\)）]\s*[\:\.\-]?(?:\r?\n|$)|'
    r'(?:\n|\A)\s*(?:\[struck:[^\]]*\]\s*)*[\(（][0-9০-৯]{1,2}[\)）]\s*(?=[\(（]?[a-jA-J][\)）\.]\s)|'
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



# Lines that can carry a question header. Only these are rewritten by the normalisation rules, so the
# student's answer text reaches Stage 3 / Stage 4 unchanged.
_HEADER_CANDIDATE_RE = re.compile(
    r'^\s*(?:\[struck:[^\]]*\]\s*)*[:;\u2022\u00b7]*\s*[_*~]*\s*(?:'
    r'[\(\uff08]?[0-9\u09e6-\u09ef]{1,3}(?![0-9])'
    r'|(?:ans\w*|am|arn|anis|dans|ass|q\w*|no|num\w*|number|part|p[\s\-]*[ab])\b'
    r'|[\(\uff08][A-Za-z\u0980-\u09FF][\)\uff09]|[\u24b6-\u24e9]|[a-gA-G](?:[\s\)\.]+ans\b|ans\s*:)|[A-Za-z]\s*[\:\.\-]?\s*$'
    r'|\u09aa\u09cd\u09b0\u09b6\u09cd\u09a8|\u0989\u09a4\u09cd\u09a4\u09b0|summary|theme|flow|rearrange|the\s+(?:graph|chart)|once\s+upon'
    r'|\$'
    r')',
    re.IGNORECASE
)
_INLINE_HEADER_RE = re.compile(r'[.?!;]\s+(?:Ans|Answer|Answering|Dans|Ass)\w*\s+(?:to|of)\s', re.IGNORECASE)

# Formatting around a header written by the transcriber: '_Answer to ... 10_', '$\underline{7}$', '**Ans**'
_HEADER_MARKUP_RULES = [
    (re.compile(r'^(\s*(?:\[struck:[^\]]*\]\s*)*)[:;\u2022\u00b7]+\s*'), r'\1'),
    (re.compile(r'\$\s*\\(?:underline|textbf|textit|text|mathbf)\{([^{}]*)\}\s*\$'), r'\1'),
    (re.compile(r'\\(?:underline|textbf|textit|text|mathbf)\{([^{}]*)\}'), r'\1'),
    (re.compile(r'\*\*|__'), ''),
    (re.compile(r'^(\s*(?:\[struck:[^\]]*\]\s*)*)[_*~]+'), r'\1'),
    (re.compile(r'[_*~]+(\s*)$'), r'\1'),
]

# A number written before a full header restates the answer's position, e.g. "2. No Ans to the Qs No 9":
# the full header that follows names the question.
_LEADING_NUMBER_BEFORE_HEADER_RE = re.compile(
    r'^(\s*(?:\[struck:[^\]]*\]\s*)*)[\(\uff08]?[0-9]{1,3}[\)\uff09]?\s*[\.\:\-]?\s*(?:No\.?|Q\.?)?\s*'
    r'(?=(?:Ans|Answer)\w*\s+(?:to|of)\b[^\n]*?[0-9])',
    re.IGNORECASE
)


def is_header_candidate_line(line: str) -> bool:
    return bool(_HEADER_CANDIDATE_RE.match(line) or _INLINE_HEADER_RE.search(line))


def _normalize_header_line(line: str) -> str:
    s = line
    for pat, repl in _HEADER_MARKUP_RULES:
        s = pat.sub(repl, s)
    s = _LEADING_NUMBER_BEFORE_HEADER_RE.sub(r'\1', s)
    for pat, repl in HEADER_NORMALIZATION_RULES:
        s = pat.sub(repl, s)
    return s


def normalize_header_text(text: str) -> str:
    """Normalize OCR anomalies in question headings (header-candidate lines only; other lines are untouched)."""
    if not text:
        return text
    return "\n".join(
        _normalize_header_line(line) if is_header_candidate_line(line) else line
        for line in text.split("\n")
    )


_KEYWORD_STOPWORDS = {
    "question", "questions", "write", "writing", "answer", "answers", "following", "paragraph",
    "with", "that", "this", "from", "into", "those", "their", "thing", "there", "these", "about",
    "which", "what", "when", "your", "have", "been", "were", "will", "than", "then", "them", "they",
}


def _declared_passages(question_obj: Optional[ExtractedQuestion]) -> List[Tuple[Set[str], str, Set[str]]]:
    """Reading passages declared in the answer key: (questions served, whitespace-normalised text, terms)."""
    qid = getattr(question_obj, "question_id", None) if question_obj else None
    if not qid:
        return []
    try:
        from src.pipeline.stage4_modes import load_answer_key
        key = load_answer_key(qid)
    except Exception:
        return []
    out = []
    for psg in (key.get("passages") or []):
        text = " ".join(str((psg or {}).get("text") or "").split())
        qs = {str(q).strip() for q in ((psg or {}).get("questions") or [])}
        if text and qs:
            terms = {w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', text) if w.lower() not in _KEYWORD_STOPWORDS}
            out.append((qs, text, terms))
    return out


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

    declared = _declared_passages(question_obj)

    for q in q_order:
        clean_q = re.sub(r'\(.*?\)', '', str(q)).strip()
        entities: Set[str] = set()

        # 1. Source text from question_obj.question_text
        if getattr(question_obj, "question_text", None):
            try:
                from src.pipeline.stage4_modes import source_text_for_question
                src = source_text_for_question(clean_q, question_obj.question_text, getattr(question_obj, "question_id", None))
                if src and declared:
                    # a declared passage belongs only to the questions it serves, even when it is printed
                    # inside another question's block on the paper
                    src = " ".join(src.split())
                    for served, p_text, _terms in declared:
                        if q not in served and clean_q not in served:
                            src = src.replace(p_text, " ")
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

        # 3. Passages declared in the answer key for this question
        for served, _p_text, terms in declared:
            if q in served or clean_q in served:
                entities |= terms

        if entities:
            index[q] = entities

    # 4. Mine reading passages that introduce questions (e.g. 'Read the text and answer the questions 3 & 4 :')
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
        if not l_str or "|" in l_str or l_str.startswith("---"):  # table rows are never headers
            continue
        clean_lines.append(l_str)

    if not clean_lines:
        return None

    for line_idx in range(min(2, len(clean_lines))):
        top_line = re.sub(r'\[struck:[^\]]*\]', '', clean_lines[line_idx]).strip()
        if not top_line:
            continue

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
            # "4 (a) Beautiful - ugly": the letter is the first item of question 4, not a paper sub-part
            base = str(int(standalone_m.group(1)))
            if base in valid_q_order:
                return base

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
        if not leading_num_m:
            # Parenthesised question number followed by its first item on the same line: "(6) (a) True"
            leading_num_m = re.match(
                r'^\s*(?:\[struck:[^\]]*\]\s*)*[\(\uff08]([0-9]{1,2})[\)\uff09]\s*(?=[\(\uff08]?[a-jA-J][\)\uff09\.]\s)',
                top_line
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
            r'^\s*(?:(?:\([^\)]*\)|Part[\s\-]*[A-Za-z0-9]+)\s*[:\-]\s*)?(?:Ansl?\.?\s*)?(?:Ans(?:i?to|i\s*to)?|Answers?|Answen|Answering|Dans|Ass|Am|Arn|Anis|Q(?:uestions?|ues|ue|n)?|due|dues|No|Number|Num)\b([^\n\r0-9|]{0,40})([0-9]{1,3})\s*(?:[\(（\.\s]*([A-Za-z0-9])[\)）]?)?',
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
        # Several numbers in one header ("no. 01 13", a corrected number): the last valid one names the question
        tail_nums = re.match(r'^((?:\s*[\.,/\-]?\s*[0-9]{1,3}\b)+)', top_line[header_m.end(2):])
        if tail_nums and valid_q_order:
            for n in reversed(re.findall(r'[0-9]{1,3}', tail_nums.group(1))):
                if str(int(n)) in valid_q_order:
                    raw_num = str(int(n))
                    break
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
    current_parent: Optional[str] = None,
    use_source_overlap: bool = True
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
        fp_q = _get_q("flowchart")
        if fp_q:
            return fp_q

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
        fp_q = _get_q("rearrange")
        if fp_q:
            return fp_q

    # 3. Informal Letter / Email: Envelope box, [STAMP], salutation + sign-off
    has_envelope = bool(re.search(r'\b(?:\[?STAMP\]?|envelope)\b', section_text, re.I)) and bool(re.search(r'\b(?:from|to)\b', section_text, re.I))
    has_salutation = bool(re.search(r'^\s*(?:dear\s+[a-z]+|my\s+dear)\b', first_lines, re.I | re.MULTILINE))
    has_signoff = bool(re.search(r'\b(?:yours?\s+(?:ever|loving|faithfully|sincerely|truly|affectionately|friend)|lovingly\s+yours|your\s+(?:loving\s+)?friend)\b', sec_lower, re.I))
    if has_envelope or (has_salutation and (has_signoff or len(section_text.split()) >= 8)) or (has_signoff and current_parent in (q_map.get("letter"), None)):
        fp_q = _get_q("letter")
        if fp_q:
            return fp_q

    # 4. Data Chart / Graph: Statistical description with percentages or trend metrics
    has_chart_kw = bool(re.search(r'\b(?:graph|chart|pie\s*chart|bar\s*chart|diagram)\b', first_lines, re.I))
    has_stats = bool(re.search(r'\b(?:percent|percentage|\%)\b', sec_lower, re.I)) and bool(re.search(r'\b\d+(?:\.\d+)?\b', sec_lower))
    has_chart_desc = bool(re.search(r'\b(?:the\s+(?:given\s+)?(?:graph|chart|diagram|table)\s+shows|shows\s+that|increased|decreased|highest|lowest|fluctuat|rate\s+of)\b', sec_lower, re.I))
    if (has_chart_kw and (has_stats or has_chart_desc)) or (has_stats and has_chart_desc):
        fp_q = _get_q("graph")
        if fp_q:
            return fp_q

    # 5. Completing Story: Narrative openings or common story motifs
    has_story_opening = bool(re.search(r'\b(?:once\s+upon\s+a\s+time|once\s+(?:there\s+)?lived|once\s+there\s+was|there\s+(?:lived|was)\s+(?:a|an)|one\s+day\s+(?:a|an)?|long\s+(?:ago|time\s+ago)|many\s+days\s+ago)\b', first_lines, re.I))
    has_story_kw = bool(re.search(r'\b(?:completing\s+story|story\s*[:\-]|\btitle\s*[:\-])\b', first_lines, re.I))
    if question_obj and question_obj.sub_questions:
        for sq in question_obj.sub_questions:
            sq_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
            sq_name = str(sq.get("name") or sq.get("title") or "").lower()
            if "story" in sq_name:
                title_words = [w for w in re.findall(r'[a-z]{4,}', sq_name) if w not in _KEYWORD_STOPWORDS and w != "story"]
                first_words = set(re.findall(r'[a-z]{4,}', first_lines))
                if title_words and sum(1 for w in set(title_words) if w in first_words) >= 2:
                    if not valid_q_order or sq_no in valid_q_order:
                        return sq_no
    if has_story_opening or has_story_kw:
        fp_q = _get_q("story")
        if fp_q:
            return fp_q

    # 6. Poem Theme: Theme keyword, poem analysis phrasing
    has_theme_kw = bool(re.search(r'\b(?:theme\s*[:\-]|\bthe\s+(?:poem|stanza|verse|passage)\s+(?:deals\s+with|is\s+about|focuses\s+on)|central\s+theme\b)', first_lines, re.I))
    if has_theme_kw or ("theme:" in first_lines and any(w in first_lines for w in ["poem", "poet", "stanza", "verse", "central", "main", "message", "author"])):
        fp_q = _get_q("theme")
        if fp_q:
            return fp_q

    # 7. Summary: Explicit summary keyword
    has_summary_kw = bool(re.search(r'\b(?:summary\s*[:\-]|\bthe\s+passage\s+(?:deals\s+with|is\s+about|summarizes))\b', first_lines, re.I))
    if has_summary_kw:
        fp_q = _get_q("summary")
        if fp_q:
            return fp_q

    # 8. Dialogue / Conversation: Multiple speaker turns (e.g. "Afifa:", "Doctor:", "Myself:")
    speaker_turns = len(re.findall(r'(?im)^\s*[A-Z][a-z]{1,15}\s*:\s+', section_text))
    has_dialogue_kw = bool(re.search(r'\b(?:dialogue|conversation)\b', first_lines, re.I))
    if has_dialogue_kw or (speaker_turns >= 4 and len(section_text.split()) >= 20):
        fp_q = _get_q("dialogue")
        if fp_q:
            return fp_q

    # 9. Semantic Source-Text Fingerprint for Headless Answers (Summary & Theme)
    # When a student writes NO header (no "Ans 3", no "3.", no "Summary:"), we match the text
    # against the source poem/passage printed on the question paper.
    # Summaries and themes are continuous prose: an itemised list ("a) ...", "(ii) ...") is never one of them
    item_lines = [l for l in section_text.split("\n") if l.strip()]
    is_item_list = bool(item_lines) and sum(
        1 for l in item_lines if re.match(r'^\s*\(?(?:[a-jA-J]|[ivxIVX]{1,4}|[0-9]{1,2})[\)\.]', l)
    ) >= 0.5 * len(item_lines)
    if use_source_overlap and not is_item_list and question_obj and getattr(question_obj, "question_text", None):
        from src.pipeline.stage4_modes import source_text_for_question
        sec_words = section_text.split()
        sec_tokens = {
            w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', section_text)
            if w.lower() not in _KEYWORD_STOPWORDS
        }

        # Check Summary source passage overlap
        summary_q = _get_q("summary")
        if summary_q:
            q_source = source_text_for_question(summary_q, question_obj.question_text, getattr(question_obj, "question_id", None))
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
            q_source = source_text_for_question(theme_q, question_obj.question_text, getattr(question_obj, "question_id", None))
            if q_source and sec_tokens and (15 <= len(sec_words) <= 150):
                q_tokens = {
                    w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', q_source)
                    if w.lower() not in _KEYWORD_STOPWORDS
                }
                overlap = q_tokens & sec_tokens
                if len(overlap) >= 3:
                    return theme_q

    return None


def _classify_header_raw(
    section_text: str,
    current_parent: Optional[str] = None,
    question_obj: Optional[ExtractedQuestion] = None,
    valid_q_order: Optional[List[str]] = None,
    entity_index: Optional[Dict[str, Set[str]]] = None,
    completed_qs: Optional[Set[str]] = None
) -> Tuple[Optional[str], str]:
    """
    Extract canonical question number from the leading lines of a text section.
    Supports English & Bengali headers, dynamic schema keywords, sub-parts,
    structural & semantic content fingerprinting, and sequence collision disambiguation.
    """
    first_lines = "\n".join(section_text.strip().split("\n")[:4])
    normalized = normalize_header_text(first_lines)
    arabic_lines = to_arabic_digits(normalized)
    first_lines_lower = first_lines.lower()
    first_words = set(re.findall(r'[a-z\u0980-\u09ff]{4,}', first_lines_lower))

    # 1. Explicit numeric English header first (an explicit "Ans to the Q No 11" must never be
    #    overridden by topic keywords). Keyword matching (below) only runs when no header is present.
    explicit = _extract_explicit_english_header(arabic_lines, valid_q_order=valid_q_order, current_parent=current_parent)
    if explicit:
        return explicit, "explicit"

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
            return f"{q_num}({ascii_sub})", "bengali"
        return q_num, "bengali"

    # 3. Dynamic Subpart (A/B/C/D) or (ক/খ/গ/ঘ) following active parent question
    if current_parent:
        base_parent = re.sub(r'\(.*?\)|[A-Za-z\u0980-\u09FF]+$', '', str(current_parent)).strip()
        if base_parent.isdigit():
            # Check if this line is a continuation sub-item like c), d), e)
            sub_continuation = re.match(r'^\s*([c-gC-G])[\)\.]', first_lines)
            if sub_continuation:
                # Continuation sub-item belonging to current_parent; do not switch questions
                return None, "continuation"

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
                        return cand, "subpart"

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
                        if not words or sum(1 for w in set(words) if w in first_words) < 2:
                            continue
                words = [
                    w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', sq_name)
                    if w.lower() not in _KEYWORD_STOPWORDS
                ]
                if not words:
                    continue
                matches = sum(1 for w in set(words) if w in first_words)
                is_strong_match = False
                if matches >= 2:
                    is_strong_match = True
                elif matches == 1:
                    matched_w = [w for w in set(words) if w in first_words]
                    if matched_w and len(matched_w[0]) >= 6 and matched_w[0] not in {"english", "second", "paper", "write", "passage"}:
                        if bool(re.search(r'\b(?:ans|q|ques|question|no|title|theme|paragraph|topic)\b', first_lines_lower, re.IGNORECASE)) or matched_w[0] in set(re.findall(r'[a-z\u0980-\u09ff]{4,}', top_line.lower())):
                            is_strong_match = True
                if is_strong_match:
                    score = matches / len(set(words))
                    if score > best_score:
                        best_q, best_score = sq_no, score
            if best_q:
                return best_q, "keyword"

    # 5. Structural & Semantic Content Fingerprinting (Flowchart, Rearrange table, Letter, Graph, Story, Theme)
    fp = detect_structural_fingerprint(section_text, question_obj=question_obj, valid_q_order=valid_q_order, current_parent=current_parent)
    if fp:
        return fp, "structural"

    # 6. Dynamic Schema-Driven Entity Fingerprinting (Headless / Unlabeled Answers)
    # Strictly applies to substantial headless answers (>= 25 words).
    # Never hijacks short sub-items like (a), (b), (c)...
    if question_obj and valid_q_order:
        is_sub_item = bool(re.match(r'^\s*[\(（]?[a-jA-J0-9][\)）\.]', first_lines))
        if len(section_text.split()) < 25 or is_sub_item:
            return None, "none"

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
            return best_match_q, "content"

    return None, "none"


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


# ---------------------------------------------------------------------------
# Content evidence: how well a piece of text fits a question, measured from the question paper
# (distinctive terms of the printed question / passage), its answer key (accepted answers, the
# rearrangement letter sequence) and its layout (dialogue turns, letter, flow chart, ...).
# ---------------------------------------------------------------------------
# Evidence is a count of paper terms / answer-key items found in the text. These minimums apply to every
# script and paper alike; they were checked against scripts/audit_segmentation.py, not set per script.
MIN_REMAP_EVIDENCE = 2      # an alternative question needs at least this much evidence to override a header
STRUCTURAL_EVIDENCE = 3     # a matching layout fingerprint counts like three distinctive terms
MIN_CONTENT_WORDS = 8       # shorter text says too little about its content to override anything

EXPLICIT_SOURCES = ("explicit", "bengali", "subpart", "remap", "inner_header")

_BOUNDARY_LINE_RE = re.compile(
    r'^\s*(?:\[struck:[^\]]*\]\s*)*(?:Ans(?:wer)?\b[\s\.\:\-]|[0-9]{1,2}\s*[\.\)]?\s*[\(（]?[a-jA-J][\)）\.]\s)',
    re.IGNORECASE
)
# Headings that open an answer without naming its number; a content-based switch may happen here or at a page start
_STRUCTURAL_OPENING_RE = re.compile(
    r'^\s*(?:\[struck:[^\]]*\]\s*)*(?:Flow[\s\-]*chart|Theme|Summary|Rearrange|The\s+(?:graph|chart)\s+shows|Once\s+upon\s+a\s+time'
    r'|Ans(?:wer)?\s+(?:to|of)\s+(?:the\s+)?(?:Q|Ques\w*|Question\w*)\b[^0-9\n]*$)\b',
    re.IGNORECASE
)
_PART_DIVIDER_RE = re.compile(r'^\s*(?:Part[\s\-]*[A-Z0-9]\b|P[\s\-]*[A-Z0-9]\s*(?:Writting|Writing|Reading)?)\b.*$', re.IGNORECASE)
# An answer header whose number is unreadable or left out ("Qno: Ques: Ans:", "Ans to the Q/A-g", "Answer to the
# question"). It still opens a new answer, so the text below it may switch question by content (see _assign_sections).
_UNNUMBERED_HEADER_RE = re.compile(
    r'^\s*(?:\[struck:[^\]]*\]\s*)*(?:'
    r'(?:[^\s0-9]{1,8}\s*[:\.\-]?\s*){0,2}Q(?:ues|us|uestion)\w*\s*[:\.\-]?\s*Ans(?:wer)?s?\b[\s:\.\-]*'
    r'|Ans(?:wer)?s?\s+(?:to|of)\s+(?:the\s+)?Q\w*\b[^0-9\n]*'
    r')$',
    re.IGNORECASE
)


def build_answer_key_index(
    question_obj: Optional[ExtractedQuestion],
    valid_q_order: Optional[List[str]]
) -> Dict[str, Dict[str, Any]]:
    """Per question: one regex per answer-key item (its accepted answers) and the rearrangement sequence."""
    index: Dict[str, Dict[str, Any]] = {}
    qid = getattr(question_obj, "question_id", None) if question_obj else None
    if not qid or not valid_q_order:
        return index
    try:
        from src.pipeline.stage4_modes import load_answer_key, key_for_question
        key = load_answer_key(qid)
    except Exception:
        return index
    if not key:
        return index
    for q in valid_q_order:
        _section, entry = key_for_question(key, q)
        if not entry:
            continue
        info: Dict[str, Any] = {"items": [], "sequence": []}
        seq = entry.get("correct_sequence")
        if seq:
            info["sequence"] = [str(x).strip().lower() for x in seq if str(x).strip()]
        raw_items = entry.get("items") or []
        if isinstance(raw_items, dict):
            raw_items = list(raw_items.values())
        for it in raw_items:
            accepted = (it.get("accepted") or []) if isinstance(it, dict) else []
            phrases = []
            for a in accepted:
                a_norm = re.sub(r"\s+", " ", str(a).strip().lower())
                # option labels ("iv", "4", "b") identify nothing on their own
                if len(a_norm) < 4 or re.fullmatch(r"[ivx]+|[0-9]+|[a-z]", a_norm):
                    continue
                phrases.append(re.escape(a_norm))
            if phrases:
                info["items"].append(re.compile(r"(?<![a-z])(?:" + "|".join(phrases) + r")(?![a-z])"))
        if info["items"] or info["sequence"]:
            index[q] = info
    return index


_ITEM_LABEL_RE = re.compile(
    r'^\s*(?:(?:\(?[a-jA-J]\)|\(?[ivxIVX]{1,4}\)|\(?[0-9]{1,2}\)|[a-jA-J0-9]{1,2}\.|[\u24b6-\u24e9]|'
    r'ans(?:wer)?\s*[:;\.\-]*|[=\-:]+)\s*)+',
    re.IGNORECASE
)
_INLINE_ITEM_SPLIT_RE = re.compile(r'\s(?=(?:\(?[a-jA-J]\)|\(?[ivxIVX]{1,4}\)|[\u24b6-\u24e9])\s)')


def _item_answers(text: str) -> List[str]:
    """Answer strings written as items: one per line (or per inline label), with labels and 'Ans:' removed."""
    out = []
    for line in text.lower().split("\n"):
        for seg in _INLINE_ITEM_SPLIT_RE.split(line):
            body = _ITEM_LABEL_RE.sub("", seg).strip(" .;,:-")
            if body:
                out.append(body)
    return out


def _key_hits(text: str, info: Optional[Dict[str, Any]]) -> int:
    """
    Answer-key items answered in item form: a line (or labelled segment) that consists of an accepted
    answer plus at most a few words. Accepted answers that merely occur inside prose do not count:
    cloze and table answers are passage words and turn up in any essay about the passage.
    A written rearrangement sequence counts fully.
    """
    if not info or not text:
        return 0
    answers = _item_answers(text)
    hits = 0
    for rx in info.get("items", []):
        for body in answers:
            m = rx.search(body)
            if m and len(body.split()) <= len(m.group(0).split()) + 3:
                hits += 1
                break
    seq = info.get("sequence") or []
    if len(seq) >= 4 and all(re.fullmatch(r"[a-z]|[ivx]{1,4}", x) for x in seq):
        letters = set(seq)
        for line in text.lower().split("\n"):
            found = {t for t in re.findall(r"(?<![a-z])([a-z]|[ivx]{1,4})(?![a-z])", line) if t in letters}
            if len(found) >= len(letters) - 2:
                hits += len(seq)
                break
    return hits


_STRUCK_SPAN_RE = re.compile(r'\[struck:[^\]]*\]|struck\[:[^\]]*\]', re.IGNORECASE)


def _strip_struck(text: str) -> str:
    """Words the student crossed out are cancelled; they are no evidence of what a section answers."""
    return _STRUCK_SPAN_RE.sub(" ", text or "")


class QuestionContext:
    """Paper-derived lookups shared by every decision of one segmentation run."""

    def __init__(
        self,
        question_obj: Optional[ExtractedQuestion],
        valid_q_order: Optional[List[str]],
        entity_index: Optional[Dict[str, Set[str]]] = None
    ):
        self.question_obj = question_obj
        self.order: List[str] = list(valid_q_order or [])
        if entity_index is not None:
            self.entities = entity_index
        else:
            self.entities = build_question_entity_index(question_obj, self.order) if (question_obj and self.order) else {}
        self.keys = build_answer_key_index(question_obj, self.order) if question_obj else {}
        # A paper term shared by k questions (e.g. a reading passage behind several questions) says little about
        # which of them a text answers: it counts 1/k.
        df: Dict[str, int] = {}
        for q in self.order:
            for term in self.entities.get(q, set()):
                df[term] = df.get(term, 0) + 1
        self.term_weight = {t: 1.0 / n for t, n in df.items()}

    def scores(self, text: str) -> Dict[str, float]:
        """Evidence for every question on the paper (weighted paper terms + answer-key items + layout)."""
        out: Dict[str, float] = {}
        text = _strip_struck(text)
        if not text.strip() or not self.order:
            return out
        words = {w.lower() for w in re.findall(r'[A-Za-z\u0980-\u09FF]{4,}', text)}
        fp = None
        if self.question_obj is not None:
            fp = detect_structural_fingerprint(text, question_obj=self.question_obj, valid_q_order=self.order)
        for q in self.order:
            score = sum(self.term_weight.get(t, 0.0) for t in words & self.entities.get(q, set()))
            score += _key_hits(text, self.keys.get(q))
            if fp == q:
                score += STRUCTURAL_EVIDENCE
            out[q] = round(score, 3)
        return out


def _base_q(q: Optional[str]) -> str:
    return re.sub(r"\(.*?\)", "", str(q or "")).strip()


def resolve_header_collision(
    section_text: str,
    header_q: str,
    current_parent: Optional[str],
    completed_qs: Optional[Set[str]],
    ctx: QuestionContext
) -> Optional[str]:
    """
    An explicit header that names a question already answered, or jumps back in paper order, is checked
    against the content below it. If a question not yet answered fits that content clearly better
    (more evidence, and at least MIN_REMAP_EVIDENCE), the header is taken as a slip and that question
    is returned. Legitimate out-of-order answers keep their header: their content fits it.
    """
    order = ctx.order
    if not order or header_q not in order:
        return None
    completed = set(completed_qs or set()) - {current_parent}
    backward = bool(
        current_parent in order
        and order.index(header_q) < order.index(current_parent)
        and _base_q(current_parent) != _base_q(header_q)
    )
    if header_q not in completed and not backward:
        return None
    if len(section_text.split()) < MIN_CONTENT_WORDS:
        return None
    scores = ctx.scores(section_text)
    exclude = {header_q, current_parent} | completed
    candidates = [q for q in order if q not in exclude and _base_q(q) != _base_q(header_q)]
    if not candidates:
        return None
    alt = max(candidates, key=lambda q: scores.get(q, 0))
    if scores.get(alt, 0) >= MIN_REMAP_EVIDENCE and _is_unique_best(scores, alt):
        return alt
    return None


def _is_unique_best(scores: Dict[str, float], q: str) -> bool:
    """q has strictly more evidence than every other question on the paper."""
    return all(scores.get(q, 0) > v for k, v in scores.items() if k != q)


def classify_header(
    section_text: str,
    current_parent: Optional[str] = None,
    question_obj: Optional[ExtractedQuestion] = None,
    valid_q_order: Optional[List[str]] = None,
    entity_index: Optional[Dict[str, Set[str]]] = None,
    completed_qs: Optional[Set[str]] = None,
    context: Optional[QuestionContext] = None
) -> Tuple[Optional[str], str]:
    """
    Decide which question a section's leading lines name, and how:
      explicit / bengali / subpart  - a written header ("Ans to the Q No. 5", "৫ নং উত্তর", "(B)")
      remap                         - a written header overridden by content evidence (see resolve_header_collision)
      keyword / structural / content - no header; inferred from the question title, layout or paper terms
      continuation / none           - nothing that names a question
    """
    q, source = _classify_header_raw(
        section_text,
        current_parent=current_parent,
        question_obj=question_obj,
        valid_q_order=valid_q_order,
        entity_index=entity_index,
        completed_qs=completed_qs
    )
    if q and source in ("explicit", "bengali", "subpart") and valid_q_order and question_obj is not None:
        ctx = context or QuestionContext(question_obj, valid_q_order, entity_index)
        alt = resolve_header_collision(section_text, q, current_parent, completed_qs, ctx)
        if alt:
            return alt, "remap"
    return q, source


def extract_header_qno(
    section_text: str,
    current_parent: Optional[str] = None,
    question_obj: Optional[ExtractedQuestion] = None,
    valid_q_order: Optional[List[str]] = None,
    entity_index: Optional[Dict[str, Set[str]]] = None,
    completed_qs: Optional[Set[str]] = None
) -> Optional[str]:
    """Canonical question number named or implied by a section's leading lines (see classify_header)."""
    q, _source = classify_header(
        section_text,
        current_parent=current_parent,
        question_obj=question_obj,
        valid_q_order=valid_q_order,
        entity_index=entity_index,
        completed_qs=completed_qs
    )
    return q


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
class _Part:
    """One contiguous piece of a page's transcript and the question it was filed under."""
    __slots__ = ("text", "page", "q", "source")

    def __init__(self, text: str, page: int, q: str, source: str):
        self.text, self.page, self.q, self.source = text, page, q, source


def _split_page_sections(page_text: str) -> List[str]:
    normalized = normalize_header_text(page_text)
    sections = []
    for sec in (s for s in HEADER_SPLIT_REGEX.split(normalized) if s.strip()):
        # HEADER_SPLIT_REGEX needs a question number; an unnumbered header line also starts a section
        lines, start = sec.split("\n"), 0
        for i in range(1, len(lines)):
            if _UNNUMBERED_HEADER_RE.match(lines[i]) and "\n".join(lines[start:i]).strip():
                sections.append("\n".join(lines[start:i]))
                start = i
        sections.append("\n".join(lines[start:]))
    sections = [s for s in sections if s.strip()]
    return sections or ([normalized] if normalized.strip() else [])


def _explicit_header_of_line(line: str, order: List[str]) -> Optional[str]:
    return _extract_explicit_english_header(to_arabic_digits(normalize_header_text(line)), valid_q_order=order or None)


def _assign_sections(page_items: List[Tuple[int, str]], ctx: QuestionContext, log: List[Dict[str, Any]]) -> List[_Part]:
    """Reading-order pass: file every section under a question."""
    parts: List[_Part] = []
    current: Optional[str] = None
    default_q = ctx.order[0] if ctx.order else "1"
    for page_no, page_text in page_items:
        for s_idx, sec in enumerate(_split_page_sections(page_text)):
            completed = {p.q for p in parts}
            q, source = classify_header(
                sec,
                current_parent=current,
                question_obj=ctx.question_obj,
                valid_q_order=ctx.order or None,
                entity_index=ctx.entities,
                completed_qs=completed,
                context=ctx
            )
            sec_clean = sec.strip()
            first_line = sec_clean.split("\n", 1)[0]
            if not q and _PART_DIVIDER_RE.match(sec_clean) and len(sec_clean.split()) <= 6:
                continue  # "Part-B : Writing Part" carries no answer text

            if q and source in ("explicit", "bengali", "subpart", "remap"):
                target, how = q, source
                if source == "remap":
                    named = _classify_header_raw(sec, current_parent=current, question_obj=ctx.question_obj,
                                                 valid_q_order=ctx.order or None, entity_index=ctx.entities,
                                                 completed_qs=completed)[0]
                    log.append({"event": "header_remap", "page": page_no, "header_named": named, "assigned": q,
                                "line": sec_clean.split("\n", 1)[0][:80]})
                    print(f"[AnswerSegmenter] Header on page {page_no} names Q{named} but its content fits Q{q}: filed under Q{q}")
            elif q and q != current and source in ("keyword", "structural", "content"):
                if current is None:
                    target, how = q, source
                elif (q not in completed and len(sec_clean.split()) >= MIN_CONTENT_WORDS
                      and (s_idx == 0 or _STRUCTURAL_OPENING_RE.match(first_line)
                           or _UNNUMBERED_HEADER_RE.match(first_line))):
                    sc = ctx.scores(sec_clean)
                    if sc.get(q, 0) >= MIN_REMAP_EVIDENCE and _is_unique_best(sc, q):
                        target, how = q, source
                        log.append({"event": "content_switch", "page": page_no, "from": current, "to": q,
                                    "evidence": {q: sc.get(q, 0), current: sc.get(current, 0)}})
                    else:
                        target, how = current, "continuation"
                else:
                    target, how = current, "continuation"
            elif (not q and current is not None and _UNNUMBERED_HEADER_RE.match(first_line)
                  and len(sec_clean.split()) >= MIN_CONTENT_WORDS):
                # a header without a readable number opens a new answer: file it under the not-yet-answered
                # question its content fits best, by the same evidence rule as a content switch
                sc = ctx.scores(sec_clean)
                open_qs = [x for x in ctx.order if x not in completed]
                best = max(open_qs, key=lambda x: sc.get(x, 0)) if open_qs else None
                if best and sc.get(best, 0) >= MIN_REMAP_EVIDENCE and _is_unique_best(sc, best):
                    target, how = best, "unnumbered_header"
                    log.append({"event": "unnumbered_header", "page": page_no, "from": current, "to": best,
                                "line": first_line[:80], "evidence": {best: sc.get(best, 0), current: sc.get(current, 0)}})
                else:
                    target, how = current, "continuation"
            else:
                target = current or default_q
                how = "continuation" if current else "default"
            current = target
            parts.append(_Part(sec_clean, page_no, target, how))
    return parts


def _split_delayed_headers(
    parts: List[_Part],
    ctx: QuestionContext,
    engine: Optional[Any],
    llm_recovery: bool,
    q_titles: Dict[str, str],
    log: List[Dict[str, Any]]
) -> List[_Part]:
    """
    A part whose own header appears only after some body lines: those lines belong to the question that
    was being answered just before (reading order). Nothing is discarded; with no earlier question the
    lines stay where they are.
    """
    out: List[_Part] = []
    for part in parts:
        lines = part.text.split("\n")
        hdr_idx = None
        if part.source in EXPLICIT_SOURCES and len(lines) > 1:
            first = _explicit_header_of_line(lines[0], ctx.order)
            if not (first and _base_q(first) == _base_q(part.q)):
                for li in range(1, min(len(lines), 4)):
                    h = _explicit_header_of_line(lines[li], ctx.order)
                    if h and _base_q(h) == _base_q(part.q):
                        hdr_idx = li
                        break
        prev_q = out[-1].q if out else None
        if hdr_idx is None or not prev_q or prev_q == part.q:
            out.append(part)
            continue
        prefix = "\n".join(lines[:hdr_idx]).strip()
        if not prefix or (_PART_DIVIDER_RE.match(prefix) and len(prefix.split()) <= 6):
            out.append(part)
            continue
        dest = prev_q
        if llm_recovery and engine is not None:
            choice = resolve_ambiguous_snippet(
                prefix, [(prev_q, q_titles.get(prev_q, f"Question {prev_q}")), (part.q, q_titles.get(part.q, f"Question {part.q}"))], engine
            )
            dest = choice or prev_q
        if dest == part.q:
            out.append(part)
            continue
        out.append(_Part(prefix, part.page, dest, "delayed_header_prefix"))
        out.append(_Part("\n".join(lines[hdr_idx:]).strip(), part.page, part.q, part.source))
        log.append({"event": "delayed_header", "page": part.page, "moved_words": len(prefix.split()), "from": part.q, "to": dest})
        print(f"[AnswerSegmenter] Delayed header on page {part.page}: {len(prefix.split())} words before Q{part.q}'s header -> Q{dest}")
    return out


def _recover_inner_headers(parts: List[_Part], ctx: QuestionContext, log: List[Dict[str, Any]]) -> List[_Part]:
    """A missing question whose explicit header sits inside another part (not split earlier): split it out."""
    if not ctx.order:
        return parts
    for mis_q in [q for q in ctx.order if q not in {p.q for p in parts}]:
        for i, part in enumerate(parts):
            if part.q == mis_q:
                continue
            lines = part.text.split("\n")
            hit = next((li for li in range(1, len(lines)) if _explicit_header_of_line(lines[li], ctx.order) == mis_q), None)
            if hit is None:
                continue
            head, tail = "\n".join(lines[:hit]).strip(), "\n".join(lines[hit:]).strip()
            new_parts = ([_Part(head, part.page, part.q, part.source)] if head else []) + [_Part(tail, part.page, mis_q, "inner_header")]
            parts = parts[:i] + new_parts + parts[i + 1:]
            log.append({"event": "inner_header", "page": part.page, "from": part.q, "to": mis_q, "line": lines[hit].strip()[:60]})
            print(f"[AnswerSegmenter] Inner header for Q{mis_q} found inside Q{part.q} (page {part.page}): split out")
            break
    return parts


def _fill_gaps_by_content(parts: List[_Part], ctx: QuestionContext, log: List[Dict[str, Any]]) -> List[_Part]:
    """
    A question still missing takes the chunk of text that fits it clearly better than the question it is
    filed under (answer-key items / paper terms / layout). Chunks start at a part boundary or at a
    numberless 'Ans' / 'N (a)' line. A part that carries its own written header is only considered when
    the same header occurs twice (one of them is then a mislabel). A question is never emptied.
    """
    if not ctx.order or ctx.question_obj is None:
        return parts
    for mis_q in [q for q in ctx.order if q not in {p.q for p in parts}]:
        # written headers per question; a header repeated on consecutive parts is one header
        explicit_count: Dict[str, int] = {}
        for k, p in enumerate(parts):
            if p.source in EXPLICIT_SOURCES and not (k > 0 and parts[k - 1].q == p.q and parts[k - 1].source in EXPLICIT_SOURCES):
                explicit_count[p.q] = explicit_count.get(p.q, 0) + 1
        best = None  # (margin, part index, start line, end line, scores)
        for i, part in enumerate(parts):
            if part.q == mis_q:
                continue
            if part.source in EXPLICIT_SOURCES and explicit_count.get(part.q, 0) < 2:
                continue
            lines = part.text.split("\n")
            starts = [0] + [li for li in range(1, len(lines)) if _BOUNDARY_LINE_RE.match(lines[li])]
            for k, st in enumerate(starts):
                chunk = "\n".join(lines[st:]).strip()
                if len(chunk.split()) < 3:
                    continue
                sc = ctx.scores(chunk)
                a_m, a_p = sc.get(mis_q, 0), sc.get(part.q, 0)
                if a_m < MIN_REMAP_EVIDENCE or not _is_unique_best(sc, mis_q):
                    continue
                remaining = sum(len(p.text.split()) for j, p in enumerate(parts) if p.q == part.q and j != i)
                remaining += len("\n".join(lines[:st]).split())
                if remaining == 0:
                    continue
                margin = a_m - a_p
                if best is None or margin > best[0]:
                    best = (margin, i, st, sc)
        if best is None:
            continue
        _margin, i, st, sc = best
        part = parts[i]
        lines = part.text.split("\n")
        head, chunk = "\n".join(lines[:st]).strip(), "\n".join(lines[st:]).strip()
        new_parts = ([_Part(head, part.page, part.q, part.source)] if head else []) + [_Part(chunk, part.page, mis_q, "gap_fill")]
        parts = parts[:i] + new_parts + parts[i + 1:]
        # continuation parts were filed under the question before them; they follow the moved chunk unless
        # their own content fits the old question better
        j = i + len(new_parts)
        while j < len(parts) and parts[j].q == part.q and parts[j].source == "continuation":
            sc_j = ctx.scores(parts[j].text)
            if sc_j.get(mis_q, 0) >= sc_j.get(part.q, 0):
                parts[j] = _Part(parts[j].text, parts[j].page, mis_q, "gap_fill")
                j += 1
            else:
                break
        log.append({"event": "gap_fill", "page": part.page, "from": part.q, "to": mis_q,
                    "evidence": {mis_q: sc.get(mis_q, 0), part.q: sc.get(part.q, 0)}, "head": chunk[:60]})
        print(f"[AnswerSegmenter] Missing Q{mis_q} recovered from Q{part.q} on page {part.page} (evidence {sc.get(mis_q, 0)} vs {sc.get(part.q, 0)})")
    return parts


def _llm_transition_recovery(
    parts: List[_Part],
    ctx: QuestionContext,
    engine: Any,
    q_prompts: Dict[str, str],
    log: List[Dict[str, Any]]
) -> List[_Part]:
    """Ablation only (pipeline.segmentation_llm_recovery): ask the model where a missing answer starts."""
    for mis_q in [q for q in ctx.order if q not in {p.q for p in parts}]:
        target_desc = q_prompts.get(mis_q, f"Question {mis_q}")
        words_by_q: Dict[str, int] = {}
        for p in parts:
            words_by_q[p.q] = words_by_q.get(p.q, 0) + len(p.text.split())
        for parent_q in sorted((q for q, w in words_by_q.items() if w >= 100), key=lambda q: -words_by_q[q]):
            parent_text = "\n\n".join(p.text for p in parts if p.q == parent_q)
            prompt = (
                f"You are an exam layout boundary detector.\n"
                f"We are looking for missing Question {mis_q} (Topic: '{target_desc}').\n"
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
                    resp = engine.generate_text(prompt, max_new_tokens=64, temperature=0.0)
                elif hasattr(engine, "chat"):
                    resp = engine.chat(prompt)
                resp = (resp or "").strip()
            except Exception as ex:
                print(f"[AnswerSegmenter] Note on neural recovery: {ex}")
                break
            if not resp or resp.upper() == "NONE" or "\n" in resp or len(resp) < 4:
                continue
            idx = next((i for i, p in enumerate(parts) if p.q == parent_q and resp in p.text), None)
            if idx is None:
                continue
            post = parent_text[parent_text.find(resp):]
            sc = ctx.scores(post)
            if sc.get(mis_q, 0) < MIN_REMAP_EVIDENCE:
                print(f"[AnswerSegmenter] Layer 3 rejected unverified split for Question {mis_q} from Question {parent_q}")
                continue
            p = parts[idx]
            pos = p.text.find(resp)
            head, tail = p.text[:pos].strip(), p.text[pos:].strip()
            new_parts = ([_Part(head, p.page, parent_q, p.source)] if head else []) + [_Part(tail, p.page, mis_q, "llm_recovery")]
            parts = parts[:idx] + new_parts + parts[idx + 1:]
            k = idx + len(new_parts)
            while k < len(parts) and parts[k].q == parent_q and parts[k].source == "continuation":
                parts[k] = _Part(parts[k].text, parts[k].page, mis_q, "llm_recovery")
                k += 1
            log.append({"event": "llm_recovery", "page": p.page, "from": parent_q, "to": mis_q, "line": resp[:60]})
            print(f"[AnswerSegmenter] Layer 3 recovered Question {mis_q} from Question {parent_q} at boundary: '{resp[:30]}'")
            break
    return parts


def _attribute_errors(
    errors: List[Dict[str, Any]],
    parts: List[_Part],
    ordered_keys: List[str]
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    File each Stage 3 error under a question: restrict to the questions present on the error's page, then
    match its context sentence, then its erroneous text (4+ characters, whole words), then its recorded
    question number. Errors that match nothing are returned separately instead of being dropped.
    """
    texts = {q: "\n\n".join(p.text for p in parts if p.q == q).lower() for q in ordered_keys}
    pages = {q: {p.page for p in parts if p.q == q} for q in ordered_keys}
    out: Dict[str, List[Dict[str, Any]]] = {q: [] for q in ordered_keys}
    unattributed: List[Dict[str, Any]] = []
    for err in errors:
        page = err.get("page_no") if err.get("page_no") is not None else err.get("page_number")
        cands = [q for q in ordered_keys if page is not None and page in pages[q]] or list(ordered_keys)
        ctx_s = str(err.get("context_sentence") or "").strip().lower()
        needle = str(err.get("erroneous_text") or "").strip().lower()
        qn = re.sub(r"[()\s]", "", str(err.get("question_no") or "")).upper()
        matched = None
        if len(ctx_s) >= 12:
            matched = next((q for q in cands if ctx_s[:30] in texts[q] or (len(ctx_s) > 15 and ctx_s[-20:] in texts[q])), None)
        if not matched and len(needle) >= 4:
            rx = re.compile(r"\b" + re.escape(needle) + r"\b")
            matched = next((q for q in cands if rx.search(texts[q])), None)
        if not matched and qn:
            matched = next((q for q in cands if re.sub(r"[()\s]", "", q).upper() == qn), None)
        if not matched and page is not None and len(cands) == 1:
            matched = cands[0]
        if matched:
            err["question_no"] = matched
            out[matched].append(err)
        else:
            unattributed.append(err)
    return out, unattributed


def _paper_catalog(question_obj: Optional[ExtractedQuestion]) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """Questions written on the script (class-test marks carried over by the teacher are excluded)."""
    order: List[str] = []
    names: Dict[str, str] = {}
    prompts: Dict[str, str] = {}
    if question_obj and question_obj.sub_questions:
        for sq in question_obj.sub_questions:
            q_no = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
            q_title = str(sq.get("name") or sq.get("title") or sq.get("prompt") or "").lower()
            task_type = str(sq.get("task_type") or "").lower()
            if task_type == "class_test" or re.match(r'^(?:q[0-9]+[:\s]*)?class[\s_-]*test(?:\s*assessment)?\s*[:0-9\s]*$', q_title.strip(), re.I):
                continue
            if q_no and q_no not in order:
                order.append(q_no)
                names[q_no] = sq.get("name") or sq.get("title") or f"Question {q_no}"
                prompts[q_no] = str(sq.get("prompt") or sq.get("question_prompt") or sq.get("name") or sq.get("title") or "")
    return order, names, prompts


def _segment_pages(
    page_items: List[Tuple[int, str]],
    errors: List[Dict[str, Any]],
    question_obj: Optional[ExtractedQuestion],
    engine: Optional[Any],
    llm_recovery: bool,
    diagnostics: Optional[Dict[str, Any]]
) -> List[AlignedAnswerItem]:
    valid_q_order, sub_q_names, q_prompts = _paper_catalog(question_obj)
    ctx = QuestionContext(question_obj, valid_q_order)
    log: List[Dict[str, Any]] = []

    parts = _assign_sections(page_items, ctx, log)
    parts = _split_delayed_headers(parts, ctx, engine, llm_recovery, sub_q_names, log)
    parts = _recover_inner_headers(parts, ctx, log)
    parts = _fill_gaps_by_content(parts, ctx, log)
    if llm_recovery and engine is not None and valid_q_order and question_obj is not None:
        parts = _llm_transition_recovery(parts, ctx, engine, q_prompts, log)
    parts = [p for p in parts if p.text.strip()]

    all_keys = list(dict.fromkeys(p.q for p in parts))
    if valid_q_order:
        ordered_keys = [k for k in valid_q_order if k in all_keys] + [k for k in all_keys if k not in valid_q_order]
    else:
        ordered_keys = sorted(all_keys)

    question_errors, unattributed = _attribute_errors(errors, parts, ordered_keys)

    aligned_items: List[AlignedAnswerItem] = []
    for q in ordered_keys:
        q_parts = [p for p in parts if p.q == q]
        combined_ans = "\n\n".join(p.text for p in q_parts).strip()
        seen_errs = set()
        dedup_errors = []
        for err in question_errors[q]:
            key = (err.get("erroneous_text", ""), err.get("context_sentence", ""))
            if key not in seen_errs:
                seen_errs.add(key)
                dedup_errors.append(err)
        aligned_items.append(AlignedAnswerItem(
            q_no=q,
            q_name=sub_q_names.get(q, f"Question {q}"),
            answer_text=combined_ans,
            page_numbers=sorted({p.page for p in q_parts}),
            errors=dedup_errors,
            word_count=len(combined_ans.split()),
            character_count=len(combined_ans)
        ))

    if diagnostics is not None:
        diagnostics.update({
            "llm_recovery": bool(llm_recovery and engine is not None),
            "parts": [{"q": p.q, "page": p.page, "source": p.source, "words": len(p.text.split()),
                       "head": p.text[:60]} for p in parts],
            "events": log,
            "missing_questions": [q for q in valid_q_order if q not in all_keys],
            "unattributed_errors": unattributed,
        })
    return aligned_items


def segment_script_into_questions(
    extraction: ExtractionResult,
    question_obj: Optional[ExtractedQuestion] = None,
    engine: Optional[Any] = None,
    llm_recovery: bool = False,
    diagnostics: Optional[Dict[str, Any]] = None
) -> List[AlignedAnswerItem]:
    """
    Segment verified script text into aligned question answers.

    Pages are read in order; every section keeps its page number. Headers decide the question; a header
    that contradicts its content is corrected from paper evidence; missing questions are recovered from
    inner headers and content. With llm_recovery=True (ablation, needs `engine`) the model is also asked
    to place boundaries. Pass a dict as `diagnostics` to receive the part list, every remap/recovery
    event and errors that could not be attributed.
    """
    pages = extraction.pages or []
    errors: List[Dict[str, Any]] = []
    for p in pages:
        if p.stage3_errors and p.stage3_errors.errors:
            for e in p.stage3_errors.errors:
                d = e.model_dump() if hasattr(e, "model_dump") else dict(e)
                d.setdefault("page_no", p.page_no)
                errors.append(d)
    if not errors and extraction.stage3_errors and extraction.stage3_errors.errors:
        errors = [e.model_dump() if hasattr(e, "model_dump") else dict(e) for e in extraction.stage3_errors.errors]

    if pages:
        page_items = []
        for p in pages:
            text = (p.stage2_verification.verified_transcript or p.stage1_transcription.raw_transcript or "").strip()
            if text:
                page_items.append((p.page_no, text))
    else:
        full_text = (extraction.stage2_verification.verified_transcript or extraction.stage1_transcription.raw_transcript or "").strip()
        page_items = [(1, full_text)] if full_text else []

    return _segment_pages(page_items, errors, question_obj, engine, llm_recovery, diagnostics)
