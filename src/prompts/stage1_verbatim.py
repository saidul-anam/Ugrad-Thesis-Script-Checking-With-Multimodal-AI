from typing import Optional, List, Dict, Any

STAGE1_SYSTEM_PROMPT = (
    "You are a strict, precise transcriber for handwritten exam scripts. "
    "Your ONLY job is to reproduce exactly what is written on the page, character for character — "
    "not to correct, improve, interpret, or normalize."
)

STAGE1_BASE_PROMPT = """You are a forensic paleographer transcribing handwritten student exam scripts. Your EXCLUSIVE duty is to transcribe the physical ink strokes character-by-character as written — NEVER correcting, normalizing, guessing, or improving the text.

Transcribe only the student's original answer written in student ink (black or blue). Ignore any red or green teacher grading marks, checkmarks, ticks, or margin scores.

CRITICAL RULES:
1. STRICT ZERO AUTOCORRECTION:
   - Transcribe all orthographic errors, phonetic misspellings, ungrammatical phrasing, and missing words EXACTLY as written.
   - Never replace non-standard spellings with standard dictionary words.
   - If a student omits a word or verb, transcribe only the words physically present; NEVER insert assumed words or missing parts of speech.

2. STRIKETHROUGHS & CROSS-OUTS:
   - When a student crosses out a word, syllable, or continuous multi-word phrase with a pen stroke or slash, enclose the entire crossed-out segment in [struck: ...].
   - If a cross-out spans an entire clause or multiple consecutive words, tag the full span (e.g. [struck: first word second word third word]).
   - For single-word or prefix cross-outs, enclose the exact struck token (e.g. [struck: word]).
   - Never omit struck text, and never transcribe struck text as plain active text.

3. UNREADABLE & AMBIGUOUS GLYPHS:
   - If a word or stroke is physically illegible, write [illegible].
   - If cursive strokes are genuinely ambiguous between two plausible readings, write [unclear: opt1 | opt2].

4. MARGIN CUT-OFF:
   - If a word is physically cut off at the edge of the scan or paper boundary, append [truncated] (e.g. incom[truncated]).

5. GHOST INK & EMPTY SPACE:
   - Stop transcribing when the student's answer ends; never hallucinate or invent text to fill empty page space.
   - Completely ignore faint, grayish, mirrored reverse-side bleed-through ink.

6. LAYOUT & CASING:
   - Preserve the student's line breaks and paragraph structure.
   - Transcribe student character casing as physically formed.

7. PURE OUTPUT:
   - Output ONLY the verbatim transcribed student text.
   - NEVER output disclaimers, conversational commentary, or meta-notes (e.g. never write 'The provided image contains...', 'There is no answer...'). If no student handwriting is present, output nothing.

Now transcribe the attached image."""


def build_stage1_prompt(
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    question_reference_vocab: Optional[List[str]] = None,
    question_reference_numerals: Optional[List[str]] = None,
    question_syllabus: Optional[List[Dict[str, Any]]] = None,
    strikethrough_detected: bool = False,
    strikethrough_region_count: int = 0,
    strikethrough_regions: Optional[List[Any]] = None,
) -> str:
    """
    Construct Stage 1 verbatim prompt incorporating strict rules, optional few-shot examples,
    question syllabus context for header digit disambiguation, reference vocabulary, reference numerals,
    and optical strikethrough directives with spatial coordinates.
    """
    prompt = STAGE1_BASE_PROMPT

    if strikethrough_detected or strikethrough_regions:
        regions = strikethrough_regions or []
        count = len(regions) if regions else strikethrough_region_count
        count_desc = f" ({count} candidate line segment(s))" if count > 0 else ""
        prompt += f"\n\n--- OPTICAL STRIKETHROUGH ADVISORY ---"
        if regions:
            prompt += f"\nImage preprocessing detected candidate cross-out stroke(s) at the following locations:\n"
            for idx, reg in enumerate(regions[:8], 1):
                y_p = getattr(reg, "y_pct", 0.0)
                y2_p = getattr(reg, "y2_pct", 0.0)
                x_p = getattr(reg, "x_pct", 0.0)
                x2_p = getattr(reg, "x2_pct", 0.0)
                w_px = getattr(reg, "w", 0)
                is_multi = getattr(reg, "is_multi_word", False)
                ang = getattr(reg, "angle", 0.0)
                type_desc = "multi-word clause strike" if is_multi else "word cross-out"
                angle_desc = f", angle: ~{ang:.0f}°" if abs(ang) >= 5.0 else ""
                prompt += f"- Region {idx}: near ~{y_p}% down the page (Y: ~{y_p}%-{y2_p}%, X: ~{x_p}%-{x2_p}%) [{type_desc}, ~{w_px}px wide{angle_desc}]\n"
            prompt += (
                "CRITICAL DIRECTIVE: Carefully inspect the handwriting at these locations. "
                "If the student drew a horizontal strike line, diagonal slash, or cross-out through text at these coordinates, "
                "you MUST enclose the crossed-out text in [struck: ...]. NEVER transcribe crossed-out words as plain active text.\n"
                "- If a line is simply an underline, table gridline, or faint reverse-side bleed-through/ghost ink, DO NOT tag it as struck."
            )
        else:
            prompt += (
                f"\nImage preprocessing flagged potential horizontal line stroke(s) on this page{count_desc} that may correspond to student cross-outs, underlines, or table borders.\n"
                f"- If a word, prefix, or phrase is genuinely crossed out by the student, enclose it in [struck: ...].\n"
                f"- If a line is simply an underline, table gridline, or faint reverse-side bleed-through/ghost ink, DO NOT tag it as struck.\n"
                f"- Transcribe strictly what the student intentionally wrote on the front of this page; NEVER invent or force struck text to match the candidate count."
            )

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

    if question_reference_numerals:
        num_preview = ", ".join(f"'{n}'" for n in question_reference_numerals[:50])
        prompt += (
            f"\n\n--- EXAM QUESTION REFERENCE NUMERALS (OPTICAL DISAMBIGUATION ONLY) ---\n"
            f"Target printed reference numbers & percentages from question prompt: [{num_preview}].\n"
            f"CRITICAL DIRECTIVE ON NUMERALS: When transcribing handwritten numbers and percentages, inspect ink topology closely: "
            f"distinguish open-top '6' from double-loop '8', '5' from '6', open '4' from closed '9', and straight '1' from angled '7'. "
            f"IMPORTANT: Students frequently make factual errors or deviate from figures in the question prompt. "
            f"The printed reference numerals above are provided strictly as an optical aid to resolve cursive stroke ambiguity. "
            f"ALWAYS transcribe what the student physically inked on the paper; NEVER alter or autocorrect student numbers to match the printed question prompt."
        )

    if few_shot_examples:
        prompt += "\n\n--- Few-Shot Demonstration Examples ---\n"
        for i, eg in enumerate(few_shot_examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Visual snippet description: {eg.get('description', 'Handwritten snippet')}\n"
            prompt += f"Ground Truth Verbatim Output:\n{eg.get('transcription', '')}\n"

    return prompt

