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
1. CONSERVATIVE GRAMMAR & SPELLING EXTRACTION (BENEFIT OF THE DOUBT ON HANDWRITING AMBIGUITY DEFERRED TO STAGE 3B):
   Extract ONLY unambiguous grammatical violations, definite syntax failures, and genuine misspellings.
   - True spelling errors: Genuine orthographic misspellings of non-existent words (e.g. 'eingineer' for 'engineer', 'familyes' for 'families', 'inables' for 'enables', 'interduction' for 'introduction', 'powere' for 'power').
   - True grammatical errors: Definite structural errors, broken subject-verb agreement (e.g. 'he see' -> 'he sees', 'AI have' -> 'AI has'), and wrong word forms.
   - Narrative Past Tense: In narrative writing (e.g. Q9 Story Completion), actions set in the past MUST use past tense verbs. Base forms (e.g. 'wake' for 'woke', 'reply' for 'replied', 'live' for 'lived', 'be' for 'was') must be extracted as grammar errors. Do NOT flag present-tense verbs occurring inside direct speech quotation marks (e.g. "I will help you").
2. REGIONAL SOUTH ASIAN ENGLISH CALIBRATION:
   Standard South Asian / NCTB English collocations, stylistic choices, and vernacular idioms (e.g. 'take preparation', 'pass days', 'join with me', 'cope up with', 'discuss about', 'by this time') must NOT be penalized or extracted as errors unless they represent a blatant grammatical breakdown.
3. ZERO PUNCTUATION MARKS:
   Do NOT extract or flag punctuation marks (periods, commas, semicolons, quotation marks, hyphens, question marks, exclamation marks, or dari). Completely ignore all punctuation differences.
4. EXAM HEADERS:
   Do NOT flag question headers or labels (e.g. "Ans", "Ans to Q. No.", "Figure: Flow chart").
5. PROPER NOUNS:
   Do NOT flag names of people, places, or historical figures (e.g. "Pasteur", "Gaza", "Dhaka").
6. SINGLE-WORD PRECISION:
   For "spelling", "erroneous_text" MUST be exactly ONE isolated word. If the student wrote a valid real word that is grammatically incorrect in context, classify it as "grammar" or "syntax", NEVER spelling.
7. RIGHT-EDGE / MARGIN TRUNCATION:
   Words that end abruptly at the end of a line or right edge of the page (e.g. 'renewabl', 'wor', 'pro', 'co', 'villa', or any token tagged '[truncated]') due to margin cut-off, scanning boundaries, or camera photo framing MUST NOT be extracted as errors (neither spelling nor grammar). Do NOT penalize students for physical scanning, photo-clipping, or margin cut-off artifacts.

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

