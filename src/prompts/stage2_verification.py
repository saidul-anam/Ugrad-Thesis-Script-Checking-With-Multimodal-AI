from typing import Optional, List, Dict, Any

STAGE2_SYSTEM_PROMPT = (
    "You are an expert handwriting verification and transcription auditor. "
    "Your objective is to inspect an initial Stage 1 transcript against the original handwritten image "
    "and propose surgical JSON patches: reverting silent autocorrection of student mistakes, "
    "wrapping struck-through words in [struck: ...], and fixing OCR ligature glitches or misread digits, "
    "WITHOUT re-generating the full transcript."
)

STAGE2_PROMPT_TEMPLATE = """You are performing Stage 2 Surgical Transcription Auditing on a student's handwritten exam script.

TASK:
You are provided with:
1. The original handwritten exam image (attached).
2. The initial Stage 1 transcription output (provided below as the immutable reference base).
{pre_analysis_section}
{syllabus_section}
Your job is to cross-examine the initial transcription line-by-line against the actual strokes in the image and propose SURGICAL PATCHES ONLY.
DO NOT re-write or re-generate the full page transcript. The system will programmatically apply your proposed patches to the Stage 1 base.

DIRECTION A — REVERT SILENT AUTOCORRECTIONS:
If Stage 1 quietly corrected an authentic student handwritten mistake or non-standard spelling to standard dictionary form, propose a surgical patch reverting it to the student's exact physical handwriting so it can be evaluated fairly.

DIRECTION B — NORMALIZE VISUAL TRANSCRIPTION GLITCHES:
If Stage 1 misread standard cursive strokes, letter connections, or handwriting ligatures as an unnatural non-word or fractured token, inspect the image strokes and restore the intended writing:
- Cursive Ligatures: Verify letter connections (such as looping ascenders, joined bowls, or minim strokes) against surrounding sentence context and restore standard letters when visually justified by the ink.
- Cross-Outs & Strikethroughs: If the student physically crossed out words or syllables with a strike line or slash, enclose the cancelled text in [struck: ...]. NEVER mark text as struck based on underlines beneath words, notebook ruling lines, or red teacher grading marks.
- Margin Boundary Truncation: If a word is cut off at the edge of the scan or paper boundary, append [truncated] (e.g. incom[truncated]).
- Ambiguity: If a character or word is genuinely unreadable, tag it as [unclear: opt1 | opt2] or [illegible].

DIRECTION C — PRESERVE AUTHENTIC STUDENT ERRORS:
Students frequently make genuine phonetic, orthographic, and grammatical mistakes:
- Unconventional phonetic spellings and letter transpositions.
- Simplified consonant clusters, missing doubled letters, or wrong vowel choices.
- Incorrect verb tenses or broken subject-verb agreement.
CRITICAL DIRECTIVE: These are authentic student errors. You MUST NEVER autocorrect or "improve" genuine student mistakes to standard English. Only fix machine transcription artifacts (Direction B).

CRITICAL AUDITING CONSTRAINTS:
1. IMMUTABLE BASE: Do NOT output a full verified transcript. Propose ONLY surgical patches in 'proposed_patches'.
2. TARGET INTEGRITY: Every patch MUST have a 'stage1_target' that exists verbatim in the Stage 1 transcript.
3. CONTEXT ANCHORING: Provide a 3-6 word 'context_anchor' containing the target to disambiguate identical words.
4. PRESERVE STUDENT ERRORS: Never "fix" or autocorrect student misspellings to standard dictionary words.
5. STRIKETHROUGH DISCIPLINE: Tag text as [struck: ...] only if physical strike strokes cut through the ink. Never tag underlines, ruling lines, or teacher marks.
6. IMMUTABLE EXAM HEADERS: Never propose patches that modify structural exam navigation labels or abbreviations (e.g. 'Ans:', 'Ans to the', 'Question No', 'Q.'). These are exam structure markers, not student prose.
7. PATCH BUDGET: Propose at most 20 patches per page. If Stage 1 is accurate, return an empty proposed_patches list [].

INITIAL STAGE 1 TRANSCRIPTION (REFERENCE BASE):
\"\"\"
{stage1_transcript}
\"\"\"

OUTPUT FORMAT:
Return a valid JSON object matching this schema:
{{
  "proposed_patches": [
    {{
      "patch_type": "ligature_fix",
      "stage1_target": "exact word or phrase in Stage 1",
      "replacement": "what student wrote, or [struck: text], or [unclear: opt1 | opt2], or same as target if no_change",
      "confidence": "high",
      "reason": "explanation of correction (e.g. cursive w-split restored to writer, reverted autocorrection, wrapped struck-through draft)",
      "context_anchor": "surrounding 3-6 words containing stage1_target"
    }}
  ],
  "verification_notes": "Summary of verification observations"
}}
"""


def build_stage2_prompt(
    stage1_transcript: str,
    pre_analysis_report: Optional[str] = None,
    question_syllabus: Optional[List[Dict[str, Any]]] = None,
    question_reference_vocab: Optional[List[str]] = None,
    question_reference_numerals: Optional[List[str]] = None,
    strikethrough_regions: Optional[List[Any]] = None
) -> str:
    """Build the Stage 2 Autocorrection Verification prompt with pre-analysis hints and syllabus priors."""
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

    if question_reference_numerals:
        num_preview = ", ".join(f"'{n}'" for n in question_reference_numerals[:50])
        context_sections.append(
            f"TARGET EXAM QUESTION REFERENCE NUMERALS (OPTICAL DISAMBIGUATION ONLY):\n[{num_preview}]\n"
            f"CRITICAL DIRECTIVE ON NUMERALS: When auditing handwritten numbers and percentages, inspect ink topology closely: "
            f"distinguish open-top '6' from double-loop '8', '5' from '6', open '4' from closed '9', and straight '1' from angled '7'. "
            f"ALWAYS transcribe the exact physical digits written by the student. "
            f"NEVER alter or autocorrect student numbers to match the printed question prompt."
        )


    pre_analysis_str = ""
    if pre_analysis_report and pre_analysis_report.strip():
        pre_analysis_str = (
            "PRE-ANALYSIS ANOMALIES DETECTED ON THIS PAGE (VISUAL DISAMBIGUATION HINTS):\n"
            + pre_analysis_report.strip()
            + "\nAUDITING DIRECTIVE: Inspect each flagged anomaly above against the handwriting image. "
            "If an anomaly is an OCR artifact, propose a patch to fix it. "
            "If it is genuine student handwriting or an intentional spelling error, preserve it."
        )

    syllabus_str = ("\n" + "\n\n".join(context_sections) + "\n") if context_sections else ""

    return STAGE2_PROMPT_TEMPLATE.format(
        stage1_transcript=stage1_transcript,
        pre_analysis_section=pre_analysis_str,
        syllabus_section=syllabus_str
    )
