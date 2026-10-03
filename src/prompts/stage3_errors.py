STAGE3_SYSTEM_PROMPT = (
    "You are an academic exam grader applying official National Curriculum and Cambridge marking doctrine. "
    "Your task is comprehensive candidate linguistic error extraction (orthographic, grammatical, syntactic) "
    "from student exam responses. Extract all genuine grammatical, syntactic, and spelling deviations."
)

STAGE3_PROMPT_TEMPLATE = """You are performing Stage 3 Error Extraction on a verified exam script transcript.

VERIFIED TRANSCRIPT:
\"\"\"
{verified_transcript}
\"\"\"

TASK:
Analyze the transcript above and extract all student linguistic errors representing grammatical violations, syntax errors, or spelling mistakes. Categorize each error into:
- spelling (unambiguous misspelled words, non-existent words, wrong vowel marks)
- grammar (subject-verb agreement, tense inconsistency, preposition misuse, wrong word form/part-of-speech)
- syntax (word order distortion, fragment sentence, run-on sentence)

OFFICIAL EXAMINER MARKING DOCTRINE:
1. CONSERVATIVE GRAMMAR & SPELLING EXTRACTION:
   Extract ONLY unambiguous grammatical violations, definite syntax failures, and genuine misspellings.
   - True spelling errors: Non-existent words, omitted/transposed letters, or phonetic misspellings that do not form a valid word.
   - True grammatical errors: Definite structural failures, broken subject-verb agreement, and incorrect word forms/parts of speech.
   - Narrative Verb Tense: In past-tense narrative prose, verbs describing past actions must use past tense. Base forms used where past tense is required must be extracted as grammar errors, except when occurring inside direct speech quotations.
2. REGIONAL & IDIOMATIC USAGE:
   Standard curriculum collocations, stylistic choices, and recognized vernacular idioms must NOT be penalized unless they represent a blatant grammatical breakdown.
3. ZERO PUNCTUATION & ZERO CAPITALIZATION PENALTIES:
   - PUNCTUATION: Completely ignore all punctuation marks (periods, commas, semicolons, hyphens, quotation marks). Never extract punctuation differences as errors.
   - CAPITALIZATION: Completely ignore letter casing. Do NOT extract or flag capitalization differences as grammatical, syntax, or spelling errors. Student handwriting casing is frequently ambiguous or stylistic; never penalize casing.
4. EXAM HEADERS:
   Do NOT flag structural question headings, subpart labels, or section identifiers.
5. PROPER NOUNS:
   Do NOT flag names of people, locations, organizations, or cultural entities.
6. SINGLE-WORD PRECISION:
   For "spelling", "erroneous_text" MUST be exactly one isolated word. If the student wrote a valid real word that is ungrammatical in context, classify it as "grammar" or "syntax", never spelling.
7. PHYSICAL MARGIN TRUNCATION:
   Words that terminate prematurely at the end of a line or right edge of the page due to scanning, photography boundaries, or tagged '[truncated]' must NOT be extracted as errors.
8. STRUCK-THROUGH / CANCELLED TEXT:
   Any text enclosed in [struck: ...] was crossed out and cancelled by the student. Never extract errors from struck-through words.

OUTPUT FORMAT:
Return a valid JSON object matching this schema:
{{
  "errors": [
    {{
      "error_type": "spelling | grammar | syntax",
      "erroneous_text": "the exact word or phrase as written",
      "suggested_correction": "the correct standard form",
      "context_sentence": "the full sentence in which the error appears",
      "explanation": "concise grammatical or spelling rule explanation"
    }}
  ],
  "spelling_error_count": 0,
  "grammar_error_count": 0,
  "syntax_error_count": 0,
  "total_error_count": 0,
  "linguistic_summary": "Brief pedagogical summary of the student's writing proficiency"
}}
"""


def build_stage3_prompt(verified_transcript: str) -> str:
    """Build the Stage 3 Linguistic Error Extraction prompt."""
    return STAGE3_PROMPT_TEMPLATE.format(verified_transcript=verified_transcript)

