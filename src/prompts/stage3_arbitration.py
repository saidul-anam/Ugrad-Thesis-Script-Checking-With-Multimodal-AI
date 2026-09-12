"""
Stage 3b (LEGACY mode): single whole-page Multimodal Visual Arbitration prompt.

Kept only for ablation (`arbitration.mode: legacy`). The evidence-based gate lives in
`src/pipeline/arbitration/` and uses the prompts in `src/prompts/stage3b_arbitration.py`.

NOTE: this prompt deliberately contains NO example words. Listing target words as examples is
label leakage: the model then "recognises" those words instead of inspecting the ink.
"""

from typing import List, Dict, Any

STAGE3_ARBITRATION_SYSTEM_PROMPT = (
    "You are a senior academic handwriting examiner. You inspect student handwriting on the attached "
    "image to distinguish genuine student misspellings from handwriting stroke ambiguities. "
    "You award the candidate the benefit of the doubt only when the ink itself is ambiguous. "
    "Output valid JSON only."
)

STAGE3_ARBITRATION_PROMPT_TEMPLATE = """You are performing Stage 3b Visual Arbitration on student handwriting.

Below are candidate discrepancies flagged during transcription of the attached page image.
For each candidate, find the word on the page and inspect the physical ink strokes.

CANDIDATES TO AUDIT:
{candidates_formatted}

EXAMINER MARKING DOCTRINE:
1. GENUINE_ERROR: the student clearly and deliberately wrote a letter sequence that differs from the
   standard spelling, i.e. the letters are well formed and unambiguous as written, and the deviation is the
   kind a learner produces from imperfect knowledge of the spelling (phonetically motivated substitutions,
   doubled or dropped letters, vowel confusions).
2. HANDWRITING_AMBIGUITY: the letters in question are ambiguous strokes (a faint or missing crossbar, a
   closed or open loop on an ascender or descender, a ligature join, a minim miscount) such that the
   intended standard word is an equally valid reading of the ink. Award the benefit of the doubt.

OUTPUT FORMAT:
Output ONLY a valid JSON array of objects matching this schema:
```json
[
  {{
    "candidate": "the exact flagged word",
    "intended_word": "the intended standard word",
    "verdict": "GENUINE_ERROR | HANDWRITING_AMBIGUITY",
    "confidence": 0.95,
    "reason": "1 concise sentence describing the physical ink observation"
  }}
]
```
"""


def build_stage3_arbitration_prompt(candidate_errors: List[Dict[str, Any]]) -> str:
    """Build the legacy Stage 3b whole-page arbitration prompt."""
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
