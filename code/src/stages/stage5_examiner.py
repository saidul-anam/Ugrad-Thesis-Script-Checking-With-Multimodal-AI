"""
Stage 5: Examiner
Implements the Chief Examiner Evaluation for NCTB HSC English 1st Paper (Rubric v2).
Scores answers across 6 task types (Paragraph, Story, Letter_Email, Graph_Chart, Summary, Theme),
evaluating the 4 standard sub-scores, performance bands, constraint hard caps, and qualitative error diagnostics.
"""

from typing import Any, Dict, List, Optional
import json
from PIL import Image

try:
    from src.schemas import (
        Stage5Output,
        ScoreBreakdown,
        StructuralAudit,
        AttemptStatus,
        ErrorAnalysis,
        TASK_MAX_MARKS
    )
    from src.model_client.base import ModelClient
    from src.stages.base_stage import extract_json_from_response
except ImportError:
    try:
        from schemas import (
            Stage5Output,
            ScoreBreakdown,
            StructuralAudit,
            AttemptStatus,
            ErrorAnalysis,
            TASK_MAX_MARKS
        )
        from model_client.base import ModelClient
        from stages.base_stage import extract_json_from_response
    except ImportError:
        from schemas import (
            Stage5Output,
            ScoreBreakdown,
            StructuralAudit,
            AttemptStatus,
            ErrorAnalysis,
            TASK_MAX_MARKS
        )
        from ..model_client.base import ModelClient
        from .base_stage import extract_json_from_response


STAGE5_SYSTEM_PROMPT_TEMPLATE = """You are an expert Chief Examiner for Higher Secondary Certificate (HSC) Class 11 English 1st Paper under the National Curriculum and Textbook Board (NCTB) in Bangladesh. Your objective is to objectively evaluate student writing samples across the 6 writing task types assessed in this examination.

TASK CONTEXT:
- Task Type: {task_type}
- MAX_MARK: {max_mark}
- The mark ceiling for the task you are scoring is {max_mark}. You must score out of {max_mark} and never out of any other number.

OPERATIVE RUBRIC & GUIDELINES:
{operative_rubric}

QUESTION / PROMPT:
{question_text}

REFERENCE SOLUTION / CONTEXT:
{reference_solution}

EXAMINER NOTE (from Stage 2 Alignment):
{examiner_note}

TRANSCRIBED STUDENT ANSWER:
{student_answer}

EVALUATION HARD RULES:
1. Zero Presentation Bias: Ignore penmanship or handwriting neatness. Evaluate raw text quality only.
2. No Per-Error Math: Do NOT subtract fixed points per error. Evaluate error density holistically within performance bands.
3. Scale Discipline: The 4 sub-scores must sum EXACTLY to `total_score` <= MAX_MARK ({max_mark}).
4. Hard Caps:
   - Paragraph: Capped at 5 if split into sub-paragraphs.
   - Letter/Email: Capped at 2 if missing 3+ structural layout parts.
   - Summary: Capped at 5 if exceeding 1/3rd length OR >50% direct copying.
   - Theme: Capped at 4 if reproducing poem lines instead of theme (>50% verbatim).
   - Graph/Chart: Capped at 6 if including personal opinions, moral judgments, or external facts not shown in the chart.
5. Performance Bands: Must match MAX_MARK ({max_mark}):
   - Max 10: Band 4 (8–10), Band 3 (6–7), Band 2 (4–5), Band 1 (1–3), Band 0 (0)
   - Max 8:  Band 4 (7–8),  Band 3 (5–6), Band 2 (3–4), Band 1 (1–2), Band 0 (0)
   - Max 7:  Band 4 (6–7),  Band 3 (4–5), Band 2 (3),   Band 1 (1–2), Band 0 (0)
   - Max 5:  Band 4 (4–5),  Band 3 (3),   Band 2 (2),   Band 1 (1),   Band 0 (0)

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
```json
{{
  "task_type": "{task_type}",
  "max_mark_applied": {max_mark},
  "attempt_status": {{
    "is_attempted": true,
    "is_blank_or_unintelligible": false
  }},
  "structural_audit": {{
    "cap_applied": false,
    "applied_cap_reason": "None | Paragraph_Subdivisions | Letter_Missing_Layout | Summary_Verbatim_Length | Theme_Verbatim_Copy | Graph_External_Facts"
  }},
  "score_breakdown": {{
    "context_content_data": 0.0,
    "structure_format_brevity": 0.0,
    "language_mechanics": 0.0,
    "originality_comparisons_paraphrase": 0.0,
    "total_score": 0.0
  }},
  "performance_band": "Band 4 | Band 3 | Band 2 | Band 1 | Band 0",
  "error_analysis": {{
    "frequent_errors": ["<Error 1>", "<Error 2>"],
    "positive_aspects": ["<Strength 1>", "<Strength 2>"]
  }},
  "feedback_summary": "<Consolidated qualitative reasoning for the score awarded>"
}}
```
"""


def run_stage5_examiner(
    final_ocr_question: str,
    final_ocr_answer: str,
    operative_rubric: str,
    reference_solution: str,
    examiner_note: Optional[str],
    model_client: ModelClient,
    task_type: str = "Paragraph",
    max_mark: float = 10.0,
    question_images: Optional[List[Image.Image]] = None,
    student_images: Optional[List[Image.Image]] = None,
    temperature: float = 0.0,
    **kwargs: Any
) -> Stage5Output:
    """
    Execute Stage 5: Examiner (rubric_v2 evaluation).
    """
    prompt = STAGE5_SYSTEM_PROMPT_TEMPLATE.format(
        task_type=task_type,
        max_mark=max_mark,
        operative_rubric=operative_rubric,
        question_text=final_ocr_question,
        reference_solution=reference_solution,
        examiner_note=examiner_note or "None",
        student_answer=final_ocr_answer
    )

    all_images = []
    if question_images:
        all_images.extend(question_images)
    if student_images:
        all_images.extend(student_images)

    raw_response = model_client.generate(
        prompt=prompt,
        images=all_images if all_images else None,
        temperature=temperature,
        **kwargs
    )

    data = extract_json_from_response(raw_response)

    attempt_stat = AttemptStatus.from_dict(data.get("attempt_status", {}))
    audit = StructuralAudit.from_dict(data.get("structural_audit", {}))
    breakdown = ScoreBreakdown.from_dict(data.get("score_breakdown", {}))
    err_analysis = ErrorAnalysis.from_dict(data.get("error_analysis", {}))
    
    root_total = data.get("total_score")
    if root_total is not None:
        try:
            total = float(root_total)
        except (ValueError, TypeError):
            total = breakdown.total_score
    else:
        total = breakdown.total_score

    if breakdown.total_score == 0.0 and total > 0.0:
        breakdown.total_score = total

    return Stage5Output(
        task_type=data.get("task_type", task_type),
        max_mark_applied=float(data.get("max_mark_applied", max_mark)),
        attempt_status=attempt_stat,
        structural_audit=audit,
        score_breakdown=breakdown,
        performance_band=data.get("performance_band", "Band 3"),
        error_analysis=err_analysis,
        feedback_summary=data.get("feedback_summary", ""),
        raw_cot=raw_response,
        stated_total=total
    )
