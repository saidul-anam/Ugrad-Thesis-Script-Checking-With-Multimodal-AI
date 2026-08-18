"""
Stage 3: Extractor B (Reference-primed OCR)
A second independent OCR pass utilizing reference solution and prompt context to resolve ambiguous handwriting strokes.
Must NEVER see Extractor A's output to maintain independence.
"""

from typing import Any, Dict, List, Optional
from PIL import Image

try:
    from src.schemas import Stage3Output, ResolvedViaReference
    from src.model_client.base import ModelClient
    from src.stages.base_stage import extract_json_from_response
except ImportError:
    try:
        from schemas import Stage3Output, ResolvedViaReference
        from model_client.base import ModelClient
        from stages.base_stage import extract_json_from_response
    except ImportError:
        from schemas import Stage3Output, ResolvedViaReference
        from model_client.base import ModelClient
        from .base_stage import extract_json_from_response


STAGE3_SYSTEM_PROMPT_TEMPLATE = """You are Extractor B, an expert Multimodal OCR specialist with access to domain reference context for NCTB HSC English 1st Paper scripts.

YOUR TASK:
Perform an independent transcription of the student's handwritten answer script image(s), using the provided reference solution / prompt context to help decipher ambiguous, faint, or illegible handwriting strokes.

CRITICAL HARD RULES:
1. BENEFIT OF THE DOUBT: You may use the reference solution to resolve ambiguous or messy handwriting strokes where multiple interpretations exist.
2. FIDELITY FIRST: You must NEVER override or correct content that is clearly written by the student, even if the student's answer contains spelling errors, factual inaccuracies, or differs from the expected answer.
3. TRANSPARENCY: Whenever you use the reference context to clarify an ambiguous stroke, you MUST log it in `RESOLVED_VIA_REFERENCE`.
4. PRESERVE PARAGRAPHS: Keep double newlines (\\n\\n) between paragraphs.

REFERENCE CONTEXT / PROMPT:
{applicable_solution}

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
```json
{{
  "QUESTION_TEXT": "<The question header or prompt number if written by student>",
  "STUDENT_ANSWER": "<Clean transcribed student answer text>",
  "RESOLVED_VIA_REFERENCE": [
    {{
      "location": "<line or paragraph description>",
      "resolved_text": "<text resolved via reference context>",
      "reference_cue": "<why reference supported this reading>"
    }}
  ],
  "STILL_UNCERTAIN": ["<unresolved illegible words/phrases>"]
}}
```
"""


def run_stage3_extractor_b(
    images: List[Image.Image],
    applicable_solution: str,
    model_client: ModelClient,
    question_images: Optional[List[Image.Image]] = None,
    temperature: float = 0.0,
    **kwargs: Any
) -> Stage3Output:
    """
    Execute Stage 3: Extractor B (Reference-primed OCR).
    """
    prompt = STAGE3_SYSTEM_PROMPT_TEMPLATE.format(
        applicable_solution=applicable_solution
    )
    if question_images:
        prompt += f"\n\nNOTE: The first {len(question_images)} image(s) are the Question Paper / Stimulus. The subsequent {len(images)} image(s) are the Student's Handwritten Answer Script."

    all_images = (question_images or []) + images

    raw_response = model_client.generate(
        prompt=prompt,
        images=all_images,
        temperature=temperature,
        **kwargs
    )

    data = extract_json_from_response(raw_response)

    resolved_list = [
        ResolvedViaReference.from_dict(r) if isinstance(r, dict) else r
        for r in data.get("RESOLVED_VIA_REFERENCE", [])
    ]

    return Stage3Output(
        QUESTION_TEXT=data.get("QUESTION_TEXT", ""),
        STUDENT_ANSWER=data.get("STUDENT_ANSWER", ""),
        RESOLVED_VIA_REFERENCE=resolved_list,
        STILL_UNCERTAIN=data.get("STILL_UNCERTAIN", [])
    )
