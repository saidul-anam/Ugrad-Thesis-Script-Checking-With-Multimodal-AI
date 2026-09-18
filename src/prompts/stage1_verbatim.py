from typing import Optional, List, Dict, Any

STAGE1_SYSTEM_PROMPT = (
    "You are a strict, precise transcriber for handwritten exam scripts. "
    "Your ONLY job is to reproduce exactly what is written on the page, character for character — "
    "not to correct, improve, interpret, or normalize."
)

STAGE1_BASE_PROMPT = """You are transcribing a handwritten exam script. Your ONLY job is to reproduce exactly what is written on the page, character for character — not to correct, improve, or interpret it.

Transcribe only the student's original answer, written in the student's own ink (typically black or blue). Ignore and do not transcribe any red-ink teacher annotations, scores, ticks, or comments — those are not part of the student's response.

Rules:
1. Transcribe every word exactly as written, including spelling mistakes, grammar errors, and incorrect word choices. Do NOT fix them.
2. Preserve the student's original sentence structure and word order, even if grammatically incorrect.
3. Struck-Through / Crossed-Out Text: If a word or phrase has a horizontal line, diagonal slash, cross-out stroke, or scribble through it (e.g. started writing a word and crossed it out before writing the intended word, such as striking through 'grap' before 'pie-chart', or striking through 'possi' before 'positively'), transcribe it and mark it as [struck: original text] (e.g. '[struck: grap] pie-chart', '[struck: possi] positively'). NEVER transcribe struck-through words as plain active text.
4. If text is illegible, write [illegible] rather than guessing.
5. If ambiguous but you can make a plausible reading, write it as [unclear: your reading].
6. Preserve line breaks and paragraph structure as they appear.
7. Do not add punctuation, capitalization, or spacing not present in the original.
8. Do not translate — transcribe in the original script (Bangla or English).
9. Do not summarize, paraphrase, or omit any part.
10. Do not misinterpret normal cursive letterforms or minims:
    - Handwritten 'm' has 3 downward legs/humps: transcribe as single 'm', NOT double 'mm' (e.g., 'Storm', NOT 'Stormm').
    - Terminal horizontal/upward flourish on 'r' or 'w' is a pen exit stroke, NOT an added letter 'e' (e.g., 'over', NOT 'overe'; 'Dear', NOT 'Deare').
    - Rounded cursive 'v' in common words ('have', 'over', 'never', 'remove') is the letter 'v', NOT 'r' ('have', NOT 'hare'; 'remove', NOT 'remore').
    - Curvy or arched cursive 's' (e.g. in 'his', 'this', 'is', 'shows') has an entry curve and rounded base: transcribe as 's', NOT 'n' (e.g., 'with his sharp teeth', NOT 'with hin'; 'this chart', NOT 'thin chart'; 'it is divided', NOT 'it in divided').

Now transcribe the attached image."""


def build_stage1_prompt(
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    question_reference_vocab: Optional[List[str]] = None,
    question_syllabus: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Construct Stage 1 verbatim prompt incorporating strict rules, optional few-shot examples,
    question syllabus context for header digit disambiguation, and reference vocabulary.
    """
    prompt = STAGE1_BASE_PROMPT

    if question_syllabus:
        syllabus_lines = []
        for sq in question_syllabus[:15]:
            q_num = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
            q_title = str(sq.get("name") or sq.get("title") or "").strip()
            if q_num and q_title:
                syllabus_lines.append(f"- Q{q_num}: {q_title}")
        if syllabus_lines:
            prompt += (
                f"\n\n--- EXAM QUESTION SYLLABUS & HEADER DISAMBIGUATION ---\n"
                f"The target exam consists of the following questions:\n"
                + "\n".join(syllabus_lines)
                + "\nCRITICAL DIRECTIVE ON QUESTION HEADERS: When transcribing question headers (e.g. 'Ans to the Question No - ...'), "
                f"cross-reference ambiguous handwritten digits with the answer topic and syllabus (e.g., distinguishing a curved '09' for a story from '05' for a cloze test)."
            )

    if question_reference_vocab:
        vocab_preview = ", ".join(f"'{w}'" for w in question_reference_vocab[:250])
        prompt += (
            f"\n\n--- EXAM QUESTION REFERENCE VOCABULARY (STRICTLY NO AUTOCORRECTION) ---\n"
            f"Target exam question vocabulary (MCQ options, clue words, answer targets): [{vocab_preview}].\n"
            f"CRITICAL DIRECTIVE: Use this reference vocabulary ONLY to help decipher ambiguous cursive strokes or messy pen marks. "
            f"If the student made an actual spelling, grammatical, or word-choice error (e.g. wrote 'disasterre', 'succeded', 'corage'), "
            f"YOU MUST TRANSCRIBE THEIR EXACT MISSPELLING character-for-character so Stage 3 can penalize it. "
            f"Under no circumstances should you silently normalize or autocorrect student handwriting."
        )

    if few_shot_examples:
        prompt += "\n\n--- Few-Shot Demonstration Examples ---\n"
        for i, eg in enumerate(few_shot_examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Visual snippet description: {eg.get('description', 'Handwritten snippet')}\n"
            prompt += f"Ground Truth Verbatim Output:\n{eg.get('transcription', '')}\n"

    return prompt

