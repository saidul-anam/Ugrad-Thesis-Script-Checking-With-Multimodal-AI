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
- Cursive Loop Ambiguity: In standard English phrases (e.g., 'Many days have passed', 'do not have any knowledge'), if an open cursive 'v' was misread as 'r' ('hare'), verify against the sentence and image and restore 'have'.
- Ascender & Crossbar Ambiguities: Cursive 'f' often lacks a prominent crossbar and connects directly into following vowels (resembling 'd'), or 't' resembles 'd' or 'l' due to looped ascenders, or cursive 'c-t' ligatures stutter (resembling 'c-i-d-t'). In standard vocabulary (e.g. 'powerful sector', 'electricity supply', 'pie chart illustrates', 'new thoughts'), if Stage 1 transcribed an ambiguous stroke as a non-word (e.g. 'powerdul', 'electricidty', 'illustrodes', 'thouths'), inspect the image and restore the student's intended standard word ('powerful', 'electricity', 'illustrates', 'thoughts').
- Question Header Digits: Verify handwritten question numbers after 'Ans to the Question No - ' against the image strokes, syllabus, and answer topic. If an answer corresponds to Question 9 (Story Completion: 'A lion and a mouse') and the handwritten digit was misread as '05' due to stroke curvature, restore 'Ans to the Question No - 09'.

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
    question_syllabus: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build the Stage 2 Autocorrection Verification prompt with optional syllabus context."""
    syllabus_str = ""
    if question_syllabus:
        lines = []
        for sq in question_syllabus[:15]:
            q_num = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
            q_title = str(sq.get("name") or sq.get("title") or "").strip()
            if q_num and q_title:
                lines.append(f"- Q{q_num}: {q_title}")
        if lines:
            syllabus_str = "\nEXAM SYLLABUS REFERENCE:\n" + "\n".join(lines) + "\n"

    return STAGE2_PROMPT_TEMPLATE.format(
        stage1_transcript=stage1_transcript,
        syllabus_section=syllabus_str
    )
