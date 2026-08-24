STAGE3_SYSTEM_PROMPT = (
    "You are a computational linguist and academic exam grader. "
    "Your task is to analyze verified student transcripts and identify all spelling, "
    "grammatical, syntactic, and structural errors systematically."
)

STAGE3_PROMPT_TEMPLATE = """You are performing Stage 3 Error Extraction on a verified exam script transcript.

VERIFIED TRANSCRIPT:
\"\"\"
{verified_transcript}
\"\"\"

TASK:
Analyze the transcript above and extract every distinct error. Categorize each error into:
- spelling (e.g., misspelled Bangla/English words, wrong vowel marks, Natwa-Satwa Bidhan violations)
- grammar (e.g., subject-verb agreement, tense inconsistency, preposition misuse)
- syntax (e.g., word order distortion, fragment sentence, run-on sentence)
- punctuation (e.g., missing dari/comma, improper quotation)

OUTPUT FORMAT:
Return a valid JSON object matching this schema:
{{
  "errors": [
    {{
      "error_type": "spelling | grammar | syntax | punctuation",
      "erroneous_text": "the exact word or phrase as written",
      "suggested_correction": "the correct standard form",
      "context_sentence": "the full sentence in which the error appears",
      "explanation": "concise grammatical or spelling rule explanation"
    }}
  ],
  "spelling_error_count": 0,
  "grammar_error_count": 0,
  "syntax_error_count": 0,
  "punctuation_error_count": 0,
  "total_error_count": 0,
  "linguistic_summary": "Brief pedagogical summary of the student's writing proficiency"
}}
"""


def build_stage3_prompt(verified_transcript: str) -> str:
    """Build the Stage 3 Linguistic Error Extraction prompt."""
    return STAGE3_PROMPT_TEMPLATE.format(verified_transcript=verified_transcript)
