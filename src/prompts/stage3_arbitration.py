"""
Stage 3b: Multimodal Visual Arbitration Prompts.

Instructs the VLM vision encoder to inspect the physical ink of candidate spelling errors
and determine whether an error is a genuine student cognitive misspelling or an ambiguous
cursive handwriting stroke (e.g. uncrossed 't'/'f', ascender curvature, ligature connection),
awarding the student the Benefit of the Doubt where appropriate.
"""

from typing import List, Dict, Any

STAGE3_ARBITRATION_SYSTEM_PROMPT = (
    "You are a senior academic handwriting examiner applying official National Curriculum "
    "and Cambridge marking doctrine. You inspect student handwriting on the attached image "
    "to distinguish genuine student cognitive misspellings from natural cursive stroke variations, "
    "uncrossed bars, and ligature ambiguities. You award the candidate the benefit of the doubt. "
    "Output valid JSON only."
)

STAGE3_ARBITRATION_PROMPT_TEMPLATE = """You are performing Stage 3b Multimodal Visual Arbitration on student handwriting.

Below are candidate spelling discrepancies flagged during OCR transcription from the attached page image.
For each candidate, inspect the physical handwritten ink strokes at its context location in the image.

CANDIDATES TO AUDIT:
{candidates_formatted}

EXAMINER MARKING DOCTRINE:
1. GENUINE COGNITIVE ERRORS:
   - Classify as "GENUINE_ERROR" ONLY if the student clearly, deliberately wrote an incorrect letter sequence indicating ignorance of orthography (e.g., 'eingineer' for 'engineer', 'familyes' for 'families', 'indelligence' for 'intelligence').
2. HANDWRITING STROKE AMBIGUITIES (BENEFIT OF THE DOUBT):
   - Classify as "HANDWRITING_AMBIGUITY" if the ink stroke represents an ambiguous cursive penmanship variation, such as:
     * Uncrossed or looped ascenders: cursive 'f' resembling 'd' in 'powerful' ('powerdul'), or 't' resembling 'd' in 'illustrates' ('illustrodes').
     * Ligature connections / pen stutters: cursive pen joins misread as an added stroke (e.g. 'electricidty' for 'electricity').
     * Missing descender loops: cursive 'g' resembling 'th' or 'h' (e.g. 'thouths' for 'thoughts').
     * Uncrossed 't' resembling 'l', minim counts on 'm'/'n', or open loops on 'v'/'r'.
   - When the student's intended word is obvious in context and the deviation is explainable by penmanship stroke ambiguity rather than phonological ignorance, award the Benefit of the Doubt.

OUTPUT FORMAT:
Output ONLY a valid JSON array of objects matching this schema:
```json
[
  {{
    "candidate": "the exact flagged word",
    "intended_word": "the intended standard word",
    "verdict": "GENUINE_ERROR | HANDWRITING_AMBIGUITY",
    "confidence": 0.95,
    "reason": "1 concise sentence explaining the physical ink observation"
  }}
]
```
"""


def build_stage3_arbitration_prompt(candidate_errors: List[Dict[str, Any]]) -> str:
    """Build the Stage 3b Multimodal Visual Arbitration prompt."""
    lines = []
    for idx, c in enumerate(candidate_errors, 1):
        err_w = c.get("erroneous_text", "")
        sug_w = c.get("suggested_correction", "")
        ctx = c.get("context_sentence", "")
        lines.append(f"{idx}. Flagged: \"{err_w}\" -> Intended in context: \"{sug_w}\"")
        if ctx:
            lines.append(f"   Context: \"{ctx}\"")
    candidates_formatted = "\n".join(lines)
    return STAGE3_ARBITRATION_PROMPT_TEMPLATE.format(candidates_formatted=candidates_formatted)
