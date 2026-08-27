"""
Stage 0b: Teacher Mark Extraction Prompt.
Runs only on exam script pages flagged with has_red_ink: true.
"""

STAGE0B_SYSTEM_PROMPT = (
    "You are a specialized visual data extractor. Your ONLY job is to detect and extract "
    "numeric marks written in red ink by an examiner or teacher on an exam script. "
    "Output ONLY valid JSON."
)

STAGE0B_BASE_PROMPT = """This image may contain red-ink numeric marks written by a teacher next to individual answers, separate from the student's own writing.

Extract ONLY numeric marks written in red ink next to each answer — do not extract comments, remarks, or written feedback.

Output as a JSON array, one object per mark found:
[
  {
    "question_no": "the question number this mark belongs to, if identifiable, otherwise null",
    "mark_value": "exact value as written, e.g. '7/10', '4', 'VII'",
    "location": "brief description, e.g. 'margin next to answer 3'"
  }
]

If a page has no red-ink numeric marks, return an empty array: [].
Output ONLY the JSON array — no explanation, no extra text."""
