from typing import Optional, List, Dict, Any

STAGE2_SYSTEM_PROMPT = (
    "You are an expert handwriting verification and transcription auditor. "
    "Your objective is to cross-examine an initial transcription against the original handwritten image "
    "to ensure 100% fidelity: reverting silent autocorrection of student mistakes while eliminating "
    "any OCR ligature glitches, minim miscounts, misread question header digits, or hallucinated letters."
)

STAGE2_PROMPT_TEMPLATE = """You are performing Stage 2 Bi-Directional Transcription Auditing on a student's handwritten exam script.

TASK:
You are provided with:
1. The original handwritten exam image (attached).
2. The initial Stage 1 transcription output (provided below).
{syllabus_section}
Your job is to cross-examine the initial transcription line-by-line against the actual strokes in the image in TWO directions:

DIRECTION A — REVERT SILENT AUTOCORRECTIONS:
If Stage 1 "quietly corrected" a student's actual handwritten mistake to standard dictionary form (e.g., student wrote 'bortito' and Stage 1 wrote 'bornito', or student wrote 'seen' and Stage 1 wrote 'see'), revert it to the student's exact handwritten text so Stage 3 can properly evaluate it.

DIRECTION B — NORMALIZE VISUAL TRANSCRIPTION GLITCHES & MISREADS:
If Stage 1 misread standard cursive handwriting strokes or digits as an unnatural non-word or ligature artifact, check the image and restore the student's true intended writing:
- Minim Doubling: Lowercase 'm' naturally has 3 downward legs/humps. If Stage 1 misread it as 'mm' (e.g., 'Stormm'), verify against the image and restore 'Storm'.
- Exit Flourishes: A pen exit flick off a terminal 'r' or 'w' is NOT an added letter 'e'. If Stage 1 transcribed 'overe' for 'over' or 'Deare' for 'Dear', restore 'over' / 'Dear'.
- Looped / Curvy 's' vs 'n' (s <-> n): A cursive or rounded 's' whose bottom stroke loops inward at the baseline or whose entry arch resembles a hump is frequently misread by OCR as an 'n' (e.g. 'his' misread as 'hin', 'is' misread as 'in', 'this' misread as 'thin', 'shows' misread as 'shoun', 'sickness' misread as 'sick nen'), or as extra letters ('examis' for 'exams'). If the stroke morphology and surrounding sentence context support the standard word ('cut the net with his sharp teeth', 'it is divided', 'this chart'), verify against the ink and restore the true letter 's' ('his', 'is', 'this', 'shows'). DO NOT penalize cursive 's' as an 'n'.
- Struck-Through / Crossed-Out Text: If the student drew a horizontal line, diagonal slash, or cross-out stroke through a word, partial word, or phrase (e.g. started writing 'grap' and struck it through before writing 'pie-chart', or started 'possi' and struck it through before writing 'positively'), verify against the image ink and wrap the cancelled word in [struck: word] (e.g. '[struck: grap] pie-chart', '[struck: possi] positively'). Never leave struck-through words as plain active text, as this improperly exposes drafts to grading penalties.
- Asymmetric Cursive 'w' Splitting: When a student's 'w' has a wide first bowl and narrow second bowl, it is a single 'w', NOT the two letters 'cu'. If Stage 1 segmented 'w' into 'cu' (e.g. 'cuith' for 'with', 'cuould' for 'would', 'docun' for 'down', 'pocuer' for 'power', 'hocu' for 'how'), check the ink and restore 'w'.
- Blind Loop 'e' vs Hooked 'c': When ink fills the inner loop of 'e' ("blind e") or an entry hook on 'c' mimics a loop, cross-examine the word against the surrounding sentence and syllabus context (e.g. 'decicion' vs 'decision', 'villagc' vs 'village', 'rcality' vs 'reality') and restore the context-supported standard spelling.
- Cursive 'v' with Exit Ligature (v <-> r): When a cursive 'v' connects to an adjacent letter ('e', 'o', 'i'), its top rightward exit stroke frequently resembles an 'r' (e.g. 'remove' misread as 'remore' or 'rumore', 'have' misread as 'hare', 'every' misread as 'erery', 'river' misread as 'rirer', 'give' misread as 'gire', 'village' misread as 'rillage'). In natural English context (e.g., 'never remove hope', 'Many days have passed'), if the stroke dips and rises into a high connecting ligature, it is the letter 'v'. Restore the true intended word ('remove', 'have', 'every') and DO NOT mutate vowels (e.g. do NOT change 'remore' to non-word 'rumore').
- Ascender & Crossbar Ambiguities: Cursive ascender letters (f, t, l, d, b) and ligature joins are frequently misread as one another when a crossbar is faint or a loop is closed. If Stage 1 transcribed a non-word that differs from an obvious in-context standard word only by such a stroke, inspect the image: restore the standard word ONLY if the ink genuinely supports it; if the student's letters are clearly formed as written, keep the student's spelling.
- Question Header Digits: Verify handwritten question numbers after 'Ans to the Question No - ' against the image strokes, syllabus, and answer topic. If an answer corresponds to Question 9 (Story Completion: 'A lion and a mouse') and the handwritten digit was misread as '05' due to stroke curvature, restore 'Ans to the Question No - 09'.
- Objective Item Options: When verifying answers to objective questions (MCQs, cloze items with clues), cross-examine candidate answers against the exam vocabulary and question options (e.g. if Stage 1 transcribed an unnatural OCR slip like 'coastard' where the student wrote an MCQ choice like 'coward', check the image strokes and restore the student's true answer).
- Right-Edge / Margin Truncation: Check words at the right margin or image boundary. If a word was physically cut off by the edge of the page, binding fold, or camera framing where strokes exit the visible frame (e.g. 'renewabl', 'wor', 'pro'), ensure it is marked with [truncated] (e.g. 'renewabl[truncated]'). Do not hallucinate missing off-page letters, and do not strip [truncated] markers where ink runs off the boundary.

CRITICAL AUDITING CONSTRAINTS:
- Do NOT invent misspellings for correctly written words.
- Preserve actual student errors faithfully while ensuring OCR handwriting noise does not pollute the transcript.

INITIAL STAGE 1 TRANSCRIPTION:
\"\"\"
{stage1_transcript}
\"\"\"

OUTPUT FORMAT:
Return a valid JSON object matching this schema:
{{
  "verified_transcript": "The canonical verified text with all student errors strictly preserved and visual noise cleared",
  "silent_corrections_fixed": [
    {{
      "stage1_output": "word in stage 1",
      "actual_handwritten": "what student actually wrote",
      "reason": "explanation of why this was reverted to student's exact error",
      "context_snippet": "surrounding line or phrase"
    }}
  ],
  "total_corrections_count": 0,
  "verification_notes": "Summary of verification observations"
}}
"""


def build_stage2_prompt(
    stage1_transcript: str,
    question_syllabus: Optional[List[Dict[str, Any]]] = None,
    question_reference_vocab: Optional[List[str]] = None
) -> str:
    """Build the Stage 2 Autocorrection Verification prompt with optional syllabus and vocab context."""
    context_sections = []
    if question_syllabus:
        lines = []
        for sq in question_syllabus[:15]:
            q_num = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
            q_title = str(sq.get("name") or sq.get("title") or "").strip()
            if q_num and q_title:
                lines.append(f"- Q{q_num}: {q_title}")
        if lines:
            context_sections.append("EXAM SYLLABUS REFERENCE:\n" + "\n".join(lines))

    if question_reference_vocab:
        vocab_preview = ", ".join(f"'{w}'" for w in question_reference_vocab[:200])
        context_sections.append(f"TARGET EXAM QUESTION VOCABULARY & OPTIONS:\n[{vocab_preview}]")

    syllabus_str = ("\n" + "\n\n".join(context_sections) + "\n") if context_sections else ""

    return STAGE2_PROMPT_TEMPLATE.format(
        stage1_transcript=stage1_transcript,
        syllabus_section=syllabus_str
    )
