"""
Stage 4: OCR Supervisor
An impartial multimodal transcription referee.
Adjudicates discrepancies between Candidate A (Cold OCR) and Candidate B (Reference-primed OCR)
by inspecting physical image strokes directly, without reference solution bias.
"""

from typing import Any, Dict, List, Optional
import json
from PIL import Image

try:
    from src.schemas import Stage1Output, Stage3Output, Stage4Output
    from src.model_client.base import ModelClient
    from src.stages.base_stage import extract_json_from_response
except ImportError:
    try:
        from schemas import Stage1Output, Stage3Output, Stage4Output
        from model_client.base import ModelClient
        from stages.base_stage import extract_json_from_response
    except ImportError:
        from schemas import Stage1Output, Stage3Output, Stage4Output
        from model_client.base import ModelClient
        from .base_stage import extract_json_from_response


STAGE4_SYSTEM_PROMPT_TEMPLATE = """You are the OCR Supervisor, an impartial multimodal transcription referee for handwritten HSC English exam scripts.

YOUR TASK:
Compare Candidate A (independent cold OCR) and Candidate B (reference-assisted OCR) against the actual handwritten student script image(s).
Select or synthesize the true text that accurately represents what the student physically wrote on the page.

RULES:
1. IMAGE IS GROUND TRUTH: If Candidate B introduced words that are not physically present on the script, reject Candidate B in favor of Candidate A.
2. PRESERVE STUDENT ERRORS: Retain all authentic student misspellings, grammar slips, and syntax errors.
3. PRESERVE PARAGRAPHS: Ensure all paragraph boundaries (\\n\\n) are faithfully preserved.

CANDIDATE A (Cold OCR Read):
{candidate_a}
Uncertainty Areas: {candidate_a_uncertainties}

CANDIDATE B (Reference-primed OCR Read):
{candidate_b}
Resolved Via Reference: {candidate_b_resolved}

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
```json
{{
  "QUESTION_TEXT": "<Authoritative Question Header>",
  "STUDENT_ANSWER": "<Authoritative Final Student Answer Text>",
  "adjudication_notes": "<Brief explanation of key adjudication choices>"
}}
```
"""


def run_stage4_ocr_supervisor(
    images: List[Image.Image],
    stage1_output: Stage1Output,
    stage3_output: Optional[Stage3Output],
    model_client: ModelClient,
    question_images: Optional[List[Image.Image]] = None,
    temperature: float = 0.0,
    **kwargs: Any
) -> Stage4Output:
    """
    Execute Stage 4: OCR Supervisor.
    """
    candidate_b_text = stage3_output.STUDENT_ANSWER if stage3_output else "N/A (Stage 3 disabled)"
    candidate_b_resolved = (
        json.dumps([r.to_dict() for r in stage3_output.RESOLVED_VIA_REFERENCE], ensure_ascii=False)
        if stage3_output else "[]"
    )

    prompt = STAGE4_SYSTEM_PROMPT_TEMPLATE.format(
        candidate_a=stage1_output.STUDENT_ANSWER,
        candidate_a_uncertainties=json.dumps([u.to_dict() for u in stage1_output.UNCERTAINTY_AREAS], ensure_ascii=False),
        candidate_b=candidate_b_text,
        candidate_b_resolved=candidate_b_resolved
    )

    if question_images:
        prompt += f"\n\nNOTE: The first {len(question_images)} image(s) are the Question Paper. The subsequent {len(images)} image(s) are the Student's Handwritten Script."

    all_images = (question_images or []) + images

    raw_response = model_client.generate(
        prompt=prompt,
        images=all_images,
        temperature=temperature,
        **kwargs
    )

    data = extract_json_from_response(raw_response)

    return Stage4Output(
        QUESTION_TEXT=data.get("QUESTION_TEXT", stage1_output.QUESTION_TEXT),
        STUDENT_ANSWER=data.get("STUDENT_ANSWER", stage1_output.STUDENT_ANSWER),
        adjudication_notes=data.get("adjudication_notes")
    )
