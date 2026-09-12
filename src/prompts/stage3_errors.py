STAGE3_SYSTEM_PROMPT = (
    "You are an academic exam grader applying official National Curriculum and Cambridge marking doctrine. "
    "Your task is to identify genuine student linguistic errors (orthographic, grammatical, syntactic) "
    "while granting the candidate the benefit of the doubt on cursive handwriting ambiguities."
)

STAGE3_PROMPT_TEMPLATE = """You are performing Stage 3 Error Extraction on a verified exam script transcript.

VERIFIED TRANSCRIPT:
\"\"\"
{verified_transcript}
\"\"\"

TASK:
Analyze the transcript above and extract confirmed student linguistic errors representing genuine ignorance or violation of language rules. Categorize each error into:
- spelling (unambiguous misspelled words, non-existent words, wrong vowel marks)
- grammar (subject-verb agreement, tense inconsistency, preposition misuse, wrong word form/part-of-speech)
- syntax (word order distortion, fragment sentence, run-on sentence)

OFFICIAL EXAMINER MARKING DOCTRINE (CRITICAL):
1. BENEFIT OF THE DOUBT (HANDWRITING & COGNITIVE AMBIGUITY):
   Handwritten exam scripts contain natural cursive stroke variations. When a word's intended standard form is clear in context and the transcript differs only by an ambiguous cursive stroke (e.g. uncrossed 'f' or 't' resembling 'd' in 'powerdul' / 'illustrodes', ligature stutters in 'electricidty', missing descender loops in 'thouths', an open cursive loop on 'v' that resembles 'r' in 'remove' vs 'remore', 'have' vs 'hare', minim humps on 'm' in 'Storm', or terminal pen exit flicks on 'r'/'w' in 'over'), award the student the benefit of the doubt. Do NOT penalize handwriting stroke ambiguities as errors.
   - Distinguish Cognitive Spelling Errors from Graphemic Slips: Authentic student misspellings are phonologically or morphologically motivated (e.g. 'eingineer', 'familyes', 'accroding'). Phonetically absurd consonant substitutions in standard high-frequency vocabulary (e.g. /f/ -> /d/ in 'powerful' or /t/ -> /d/ in 'electricity') are handwriting stroke misreadings, NOT cognitive misspellings.
2. GENUINE ERRORS ONLY:
   Only extract confirmed, unambiguous errors:
   - True spelling errors: Genuine orthographic misspellings (e.g. 'eingineer' for 'engineer', 'familyes' for 'families', 'inables' for 'enables', 'interduction' for 'introduction').
   - True grammatical errors: Definite structural errors (e.g. 'he see' -> 'he sees', 'your are' -> 'you are', 'for going Cox\\'s Bazar' -> 'to go to Cox\\'s Bazar', tense inconsistency like 'I came' in future context).
3. ZERO PUNCTUATION MARKS:
   Do NOT extract or flag punctuation marks (periods, commas, semicolons, quotation marks, hyphens, question marks, exclamation marks, or dari). Completely ignore all punctuation differences.
4. EXAM HEADERS:
   Do NOT flag question headers or labels (e.g. "Ans", "Dans", "Q. No.", "Figure: Flow chart").
5. PROPER NOUNS:
   Do NOT flag names of people, places, or historical figures (e.g. "Pasteur", "Gaza", "Dhaka").
6. SINGLE-WORD PRECISION:
   For "spelling", "erroneous_text" MUST be exactly ONE isolated word. If the student wrote a valid real word that is grammatically incorrect in context, classify it as "grammar" or "syntax", NEVER spelling.

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

