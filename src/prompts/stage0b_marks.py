from typing import Optional, List

STAGE0B_SYSTEM_PROMPT = (
    "You are a specialized visual data extractor. Your ONLY job is to detect and extract "
    "numeric marks written in red ink by an examiner or teacher on an exam script. "
    "Focus specifically on the LEFT MARGIN outside the student answer area. "
    "Output ONLY valid JSON."
)

STAGE0B_BASE_PROMPT = """This exam script page contains red-ink markings made by a teacher/examiner.

CRITICAL LOCATION GUIDELINES:
1. FOCUS ON THE LEFT MARGIN: In handwritten exam scripts, the teacher writes numerical marks almost exclusively in the LEFT MARGIN (outside the vertical margin/boundary line) directly opposite each question/answer.
2. DISTINGUISH TICKS FROM MARKS: Checkmarks (✓), slashes, or underlines in the body are corrections. Look for the large standalone numbers written in red pen in the left margin representing the awarded marks (e.g., '1', '2', '3', '5', '6', '7', '10').
3. NUMBER DISAMBIGUATION:
   - A circled '1' or '01' is mark '1' (do not confuse with '6').
   - An open-top '5' is digit '5' (do not confuse with 'f').
   - Marks are numerical values awarded for the question.

Output as a JSON array, one object per mark found:
[
  {
    "question_no": "the question number this mark belongs to (e.g. '1(A)', '1(B)', '2', '3', '4', '5', '6', '8', '9', '10')",
    "mark_value": "exact numeric value as written, e.g. '5', '7', '10', '3', '1'",
    "location": "left margin next to question X"
  }
]

If a page has no red-ink numeric marks, return an empty array: [].
Output ONLY the JSON array — no explanation, no extra text."""


def build_stage0b_prompt(
    candidate_questions: Optional[List[str]] = None,
    question_max_marks: Optional[dict[str, float]] = None,
    valid_paper_questions: Optional[List[str]] = None
) -> str:
    """Build a context-grounded Stage 0b prompt with candidate question numbers, paper whitelist, and max marks ceilings."""
    if not candidate_questions and not valid_paper_questions:
        return STAGE0B_BASE_PROMPT

    q_targets = candidate_questions or valid_paper_questions or []
    q_list_str = ", ".join(f"'{q}'" for q in q_targets)

    # Build max marks reference if available
    bounds_lines = []
    if question_max_marks:
        relevant_bounds = [f"Q{q}: max {question_max_marks[q]} marks" for q in q_targets if q in question_max_marks]
        if relevant_bounds:
            bounds_lines.append("MAXIMUM ALLOWABLE MARKS PER QUESTION:")
            bounds_lines.append("; ".join(relevant_bounds))
            bounds_lines.append("- A mark can NEVER exceed the maximum allowable marks for a question.")
    bounds_str = "\n".join(bounds_lines) + "\n" if bounds_lines else ""

    whitelist_str = ""
    if valid_paper_questions:
        whitelist_str = (
            f"EXAM QUESTION WHITELIST: This exam paper ONLY contains questions: {valid_paper_questions}.\n"
            f"- NEVER label a mark using sub-item letters like 'a', 'b', 'c', 'g'. Always map sub-parts to their parent question (e.g. sub-item (c) belongs to '1(B)', sub-item (g) belongs to '4').\n"
        )

    return f"""This exam script page contains red-ink markings made by a teacher/examiner.

QUESTIONS ANSWERED ON THIS PAGE:
The student has answered the following question(s) on this page: [{q_list_str}].

{whitelist_str}{bounds_str}CRITICAL LOCATION & EXTRACTION RULES:
1. FOCUS ON THE LEFT MARGIN: Look specifically down the LEFT MARGIN (outside the vertical margin border line) next to each attempted question: [{q_list_str}].
2. For each question listed above, locate the handwritten red-ink number awarded by the examiner in the left margin.
3. SUB-ITEMS & FRACTIONS:
   - If the teacher wrote marks for sub-parts (e.g., '1/2' next to item g, or '6' next to item c), attribute them to the parent question.
   - For half-marks like '1/2', write '0.5'.
4. NUMBER DISAMBIGUATION:
   - A circled '1' or '01' is mark '1' (do not confuse with '6').
   - An open-top '5' is digit '5' (do not confuse with 'f').
   - Do not confuse teacher question labels (e.g. writing 'Q4') with the mark itself.

Output as a JSON array, one object per mark found:
[
  {{
    "question_no": "one of [{q_list_str}]",
    "mark_value": "exact numeric value as written, e.g. '5', '7', '10', '3', '0.5', '1'",
    "location": "left margin next to question X"
  }}
]

If this page has no red-ink numeric marks, return an empty array: [].
Output ONLY the JSON array — no explanation, no extra text."""


