"""
Stage 2: RubricAligner
Reconciles the official rubric_v2.txt specifications with the student's actual answer text before grading.
Runs in text-only mode (no image access).
"""

from typing import Any, Dict, List, Optional
from PIL import Image

try:
    from src.schemas import Stage1Output, Stage2Output, Stage2Decision
    from src.model_client.base import ModelClient
    from src.stages.base_stage import extract_json_from_response
except ImportError:
    try:
        from schemas import Stage1Output, Stage2Output, Stage2Decision
        from model_client.base import ModelClient
        from stages.base_stage import extract_json_from_response
    except ImportError:
        from ..schemas import Stage1Output, Stage2Output, Stage2Decision
        from ..model_client.base import ModelClient
        from .base_stage import extract_json_from_response


STAGE2_SYSTEM_PROMPT_TEMPLATE = """You are RubricAligner, an expert pedagogical assessment auditor for NCTB HSC English 1st Paper examinations.

YOUR ROLE:
Compare the transcribed student answer against the official rubric specification and reference solution/prompt context.

DECISION CRITERIA:
1. `KEEP`: The student followed the standard prompt and task format (e.g. Paragraph, Graph/Chart, Letter, Story, Summary, Theme). Apply the official rubric directly.
2. `REPAIR`: Minor mismatch (e.g. slight renumbering, omitted title, minor heading variance). Update `operative_rubric` with an explicit alignment note for the Examiner.
3. `ADAPT`: The student adopted an unconventional but valid creative/analytical structure. Formulate a `shadow_solution` detailing how the rubric criteria map to their response.

ORIGINAL RUBRIC SPECIFICATION:
{original_rubric}

REFERENCE SOLUTION / PROMPT CONTEXT:
{original_reference_solution}

TRANSCRIBED STUDENT ANSWER:
{student_answer}

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
```json
{{
  "decision": "KEEP | REPAIR | ADAPT",
  "operative_rubric": "<The full operative rubric text for the Examiner>",
  "examiner_note": "<Brief instruction for the Examiner, or null>",
  "shadow_solution": "<Shadow solution text if decision is ADAPT, else null>"
}}
```
"""


def run_stage2_rubric_aligner(
    stage1_output: Stage1Output,
    original_rubric: str,
    original_reference_solution: str,
    model_client: ModelClient,
    temperature: float = 0.0,
    **kwargs: Any
) -> Stage2Output:
    """
    Execute Stage 2: RubricAligner (Text-only).
    """
    prompt = STAGE2_SYSTEM_PROMPT_TEMPLATE.format(
        original_rubric=original_rubric,
        original_reference_solution=original_reference_solution,
        student_answer=stage1_output.STUDENT_ANSWER
    )

    raw_response = model_client.generate(
        prompt=prompt,
        images=None,  # Stage 2 is strictly text-only
        temperature=temperature,
        **kwargs
    )

    data = extract_json_from_response(raw_response)

    decision_str = data.get("decision", "KEEP").upper()
    try:
        decision = Stage2Decision(decision_str)
    except ValueError:
        decision = Stage2Decision.KEEP

    return Stage2Output(
        operative_rubric=data.get("operative_rubric", original_rubric),
        examiner_note=data.get("examiner_note"),
        shadow_solution=data.get("shadow_solution"),
        decision=decision
    )
