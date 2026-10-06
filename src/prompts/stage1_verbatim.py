from typing import Optional, List, Dict, Any

STAGE1_SYSTEM_PROMPT = (
    "You are a strict, precise transcriber for handwritten exam scripts. "
    "Your ONLY job is to reproduce exactly what is written on the page, character for character — "
    "not to correct, improve, interpret, or normalize."
)

STAGE1_BASE_PROMPT = """You are a forensic paleographer transcribing handwritten student exam scripts. Your EXCLUSIVE duty is to transcribe the physical ink strokes character-by-character as written — NEVER correcting, normalizing, guessing, or improving the text.

Transcribe only the student's original answer written in student ink (black or blue). Ignore any red or green teacher grading marks, checkmarks, ticks, or margin scores.

CRITICAL RULES:
1. STRICT ZERO AUTOCORRECTION (FORENSIC CHARACTER FIDELITY):
   - Transcribe all orthographic errors, phonetic misspellings, ungrammatical phrasing, and missing words EXACTLY as written.
   - You are a forensic paleographer transcribing handwritten student exam scripts, NOT a proofreader. In secondary school exams, student spelling mistakes are explicitly penalized. If you normalize, autocorrect, or improve a misspelled word into a standard dictionary word, you invalidate the student's exam score!
   - Faithfully transcribe character-by-character letter choices:
     * Vowel shifts & phonetic substitutions: If a student writes an unexpected vowel (e.g. 'u' instead of 'e' in a word ending in -ing, 'e' instead of 'a', 'i' instead of 'y'), transcribe the exact vowel inked.
     * Consonant omissions & simplifications: If letters are doubled, omitted, or simplified, transcribe the exact letters physically written.
     * Character order / metathesis: If letters are transposed (e.g. 'sl' vs 'ls'), transcribe the exact character sequence written.
   - NEVER replace non-standard spellings with standard dictionary words.
   - If a student omits a word or verb, transcribe only the words physically present; NEVER insert assumed words or missing parts of speech.

2. STRIKETHROUGHS & CROSS-OUTS:
   - When a student crosses out a word, syllable, or continuous multi-word phrase with a pen stroke or slash, enclose the entire crossed-out segment in [struck: ...].
   - If a cross-out spans an entire clause or multiple consecutive words, tag the full span (e.g. [struck: first word second word third word]).
   - For single-word or prefix cross-outs, enclose the exact struck token (e.g. [struck: word], or [struck: ab] abcdef when a false start is followed by the full word).
   - SINGLE-LETTER & SINGLE-WORD ABORTED STROKES: When a student starts writing a letter or word and strikes it out before writing their actual word, wrap the struck letter or word in [struck: ...] exactly where it stands (e.g. '[struck: x] word', 'word1 [struck: word2] word3', or 'wor[struck: x]d' for a struck letter inside a word). NEVER omit these single-letter or single-word cross-outs.
   - MULTI-LINE & PARAGRAPH BLOCKS: When an entire draft paragraph or multiple consecutive lines are crossed out (whether with parallel horizontal lines, a steep diagonal slash, or a large 'X' mark), wrap each cancelled line in [struck: ...] (or the entire block). NEVER omit the struck draft paragraph, and NEVER transcribe crossed-out paragraphs as active text.
   - In itemized lists (e.g. 'a)', 'b)', 'c)'): if a student crossed out an initial draft word or phrase before writing their replacement answer, enclose the crossed-out text in [struck: ...]. Never omit the struck token or transcribe it as active text.
   - CRITICAL NEGATIVE CONSTRAINTS (NO FALSE STRIKES):
      * Cursive 't' crossbars: In active words containing 't' (e.g. 'cut', 'not', 'it', 'to', 'at', 'that', 'plants', 'water'), the horizontal crossbar is a natural character stroke, NEVER a strikethrough. Do NOT tag active words as [struck: ...] simply because they have a 't' crossbar.
      * Underlines: A horizontal line running strictly beneath a word or question header (e.g. 'Answer to Question No-10') is an underline, NOT a strikethrough.
      * Question numbering, math symbols, and Roman numerals: Plus signs (+), dashes (-), brackets (), and Roman numerals (e.g. '(a + iv + ii)') in rearrangement/matching questions are NEVER strikethroughs. Never wrap question indices or table references in [struck: ...].
      * Interlinear corrections: When a student crosses out a word and writes a replacement above or beside it, tag the CROSSED-OUT word in [struck: ...], NEVER the active replacement.
    - Never omit struck text, and never transcribe struck text as plain active text.

3. CURSIVE LIGATURES & NATURAL LETTERFORMS:
   - Recognize natural English cursive ligatures and pen-lift variations in standard words. An uncrossed or softly-looped cursive 't' ligature is not a 'd', and a sweeping descender loop on cursive 'g' is not an 's'.
   - In rapid English cursive handwriting, lowercase 't' (and double 'tt') ascenders are frequently formed with a tall loop where the horizontal crossbar is faint or omitted in continuous pen flow. Distinguish these natural cursive ascender strokes from 'l' by cross-referencing lexical context and question vocabulary.
   - Distinguish cursive handwriting variations from genuine misspellings: transcribe standard English words when standard cursive letterforms are used, while strictly preserving authentic misspellings when non-standard letters are deliberately formed.

4. UNREADABLE & AMBIGUOUS GLYPHS:
   - If a word or stroke is physically illegible, write [illegible].
   - If cursive strokes are genuinely ambiguous between two plausible readings, write [unclear: opt1 | opt2].

5. MARGIN CUT-OFF:
   - If a word is physically cut off at the edge of the scan or paper boundary, append [truncated] (e.g. incom[truncated]).

6. GHOST INK & EMPTY SPACE:
   - Stop transcribing when the student's answer ends; never hallucinate or invent text to fill empty page space.
   - Completely ignore faint, grayish, mirrored reverse-side bleed-through ink.

7. LAYOUT & CASING:
   - Preserve the student's line breaks and paragraph structure.
   - Transcribe student character casing as physically formed.

8. PURE OUTPUT:
   - Output ONLY the verbatim transcribed student text.
   - NEVER output disclaimers, conversational commentary, or meta-notes (e.g. never write 'The image provided is too overexposed and lacks sufficient contrast...', 'The provided image contains...', 'There is no answer...'). If no student handwriting is present, output nothing.

Now transcribe the attached image."""


def build_stage1_prompt(
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    question_reference_vocab: Optional[List[str]] = None,
    question_reference_numerals: Optional[List[str]] = None,
    question_syllabus: Optional[List[Dict[str, Any]]] = None,
    strikethrough_detected: bool = False,
    strikethrough_region_count: int = 0,
    strikethrough_regions: Optional[List[Any]] = None,
    strikethrough_blocks: Optional[List[Any]] = None,
) -> str:
    """
    Construct Stage 1 verbatim prompt incorporating strict rules, optional few-shot examples,
    question syllabus context for header digit disambiguation, reference vocabulary, reference numerals,
    and optical strikethrough directives with spatial coordinates.
    """
    prompt = STAGE1_BASE_PROMPT

    # High-confidence optical strikethrough priors injection (blocks and multi-word strokes ONLY)
    # Never pass single-word/prefix strokes to avoid false positive hallucinations on plus signs, dashes, and question indices.
    optical_priors = []
    if strikethrough_blocks:
        for b in strikethrough_blocks[:5]:
            b_type = getattr(b, "stroke_type", "")
            if b_type == "x_cross":
                desc = f"Large 'X' cross-out across paragraph at vertical ~{b.y_pct}%-{b.y2_pct}%"
            elif b_type == "steep_diagonal":
                desc = f"Steep diagonal cross-out slash at vertical ~{b.y_pct}%-{b.y2_pct}%"
            elif getattr(b, "line_count", 0) >= 3:
                desc = f"Multi-line crossed-out block at vertical ~{b.y_pct}%-{b.y2_pct}% (approx {b.line_count} consecutive lines)"
            else:
                continue
            optical_priors.append(f"- {desc}")

    if strikethrough_regions:
        valid_strikes = [
            r for r in strikethrough_regions
            if not getattr(r, "is_underline", False)
            and getattr(r, "confidence", 0.0) >= 0.88
            and not getattr(r, "is_table_border", False)
        ]
        for r in valid_strikes[:6]:
            desc = "Multi-word clause cross-out" if getattr(r, "is_multi_word", False) else "Single-word/token cross-out stroke"
            y_val = round(getattr(r, "y_pct", 0.0), 1)
            x1_val = round(getattr(r, "x_pct", 0.0), 1)
            x2_val = round(getattr(r, "x2_pct", 100.0), 1)
            optical_priors.append(
                f"- {desc} at vertical ~{y_val}% (horizontal {x1_val}%-{x2_val}%)"
            )

    if optical_priors:
        priors_text = "\n".join(optical_priors)
        prompt += (
            f"\n\n--- VISUAL STRIKETHROUGH PRIORS (OPTICAL PRE-DETECTION) ---\n"
            f"The optical detector identified candidate cancellation block(s) on this page:\n"
            f"{priors_text}\n"
            f"DIRECTIVE: If the handwriting within these blocks is crossed out or cancelled by the student, "
            f"wrap the cancelled text in [struck: ...]. If it is active, normal text (or table borders), transcribe normally."
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
            f"If the student made an actual spelling, grammatical, or word-choice error (e.g. non-standard spellings or missing letters), "
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

