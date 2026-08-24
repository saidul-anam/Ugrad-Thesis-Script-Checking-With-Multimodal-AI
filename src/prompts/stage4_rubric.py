import json
from typing import Optional, Dict, Any

STAGE4_SYSTEM_PROMPT = (
    "You are an experienced senior examiner evaluating student exam scripts. "
    "You apply rubric criteria rigorously and fairly, providing transparent mark breakdowns, "
    "linguistic penalty deductions, and actionable feedback."
)

STAGE4_PROMPT_TEMPLATE = """You are performing Stage 4 Rubric-Based Evaluation on an exam script.

VERIFIED STUDENT TRANSCRIPT:
\"\"\"
{verified_transcript}
\"\"\"

EXTRACTED LINGUISTIC ERRORS (From Stage 3):
{error_list_json}

EVALUATION RUBRIC:
Subject: {subject}
Question Type: {question_type}
Total Marks: {total_marks}

Criteria:
{rubric_criteria_text}

Penalty Guidelines:
{penalty_guidelines_text}

{thematic_context_section}

TASK:
1. Evaluate the content against each criterion and assign appropriate marks.
2. Calculate content raw score (sum of criteria marks).
3. Apply language penalty deductions based on the extracted errors and penalty guidelines (subject to max linguistic deduction cap).
4. Compute final awarded score.
5. Provide detailed pedagogical feedback and actionable recommendations for improvement.

OUTPUT FORMAT:
Return a valid JSON object matching this schema:
{{
  "subject": "{subject}",
  "question_type": "{question_type}",
  "criteria_scores": [
    {{
      "criterion_id": "criterion id",
      "criterion_name": "criterion name",
      "max_marks": 0.0,
      "awarded_marks": 0.0,
      "justification": "Why this mark was awarded",
      "strengths": ["point 1"],
      "weaknesses": ["point 1"]
    }}
  ],
  "content_raw_score": 0.0,
  "linguistic_penalty": 0.0,
  "final_score": 0.0,
  "total_max_marks": {total_marks},
  "percentage": 0.0,
  "overall_feedback": "Detailed constructive teacher summary",
  "actionable_recommendations": [
    "Specific action step 1",
    "Specific action step 2"
  ]
}}
"""


def build_stage4_prompt(
    verified_transcript: str,
    error_list: Dict[str, Any],
    rubric_data: Dict[str, Any],
    thematic_context: Optional[str] = None
) -> str:
    """Build the Stage 4 Rubric Evaluation prompt."""
    criteria_lines = []
    for c in rubric_data.get("criteria", []):
        criteria_lines.append(f"- ID: {c.get('id')}, Name: {c.get('name')}, Max Marks: {c.get('max_marks')}")
        criteria_lines.append(f"  Description: {c.get('description', '')}")
    rubric_criteria_text = "\n".join(criteria_lines)

    penalties = rubric_data.get("penalties", {})
    penalty_lines = [
        f"- Spelling error deduction: {penalties.get('spelling_error_deduction', 0.25)} per error",
        f"- Grammar error deduction: {penalties.get('grammar_error_deduction', 0.5)} per error",
        f"- Max linguistic deduction cap: {penalties.get('max_linguistic_deduction', 2.0)}"
    ]
    penalty_guidelines_text = "\n".join(penalty_lines)

    thematic_context_section = ""
    if thematic_context:
        thematic_context_section = f"REFERENCE / THEMATIC TEXTBOOK CONTEXT (RAG):\n\"\"\"\n{thematic_context}\n\"\"\"\n"

    return STAGE4_PROMPT_TEMPLATE.format(
        verified_transcript=verified_transcript,
        error_list_json=json.dumps(error_list, ensure_ascii=False, indent=2),
        subject=rubric_data.get("subject", "General"),
        question_type=rubric_data.get("question_type", "Standard"),
        total_marks=rubric_data.get("total_marks", 10.0),
        rubric_criteria_text=rubric_criteria_text,
        penalty_guidelines_text=penalty_guidelines_text,
        thematic_context_section=thematic_context_section
    )
