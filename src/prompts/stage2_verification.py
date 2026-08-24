from typing import Optional

STAGE2_SYSTEM_PROMPT = (
    "You are an expert handwriting verification and transcription auditor. "
    "Your objective is to compare an initial transcription against the original handwritten image "
    "and detect/revert any silent autocorrection, hallucinated words, or missed handwritten mistakes."
)

STAGE2_PROMPT_TEMPLATE = """You are performing Stage 2 Autocorrection Verification on a student's handwritten exam script.

TASK:
You are provided with:
1. The original handwritten exam image (attached).
2. The initial Stage 1 transcription output (provided below).

Your job is to cross-examine the initial transcription line-by-line against the actual strokes in the image:
1. Check if the initial transcriber "quietly corrected" a student's spelling, grammar, or word choice mistake to standard dictionary form. If so, revert it to the student's actual handwritten error.
2. Check if any unclear word was guessed without marking [unclear: ...] or [illegible].
3. Verify that all line breaks, punctuations, and script forms match the image exactly.

INITIAL STAGE 1 TRANSCRIPTION:
\"\"\"
{stage1_transcript}
\"\"\"

OUTPUT FORMAT:
Return a valid JSON object matching this schema:
{{
  "verified_transcript": "The canonical verified text with all student errors strictly preserved",
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


def build_stage2_prompt(stage1_transcript: str) -> str:
    """Build the Stage 2 Autocorrection Verification prompt."""
    return STAGE2_PROMPT_TEMPLATE.format(stage1_transcript=stage1_transcript)
