"""
Stage 4 Modular Question-by-Question Evaluation Prompts.

Specialized prompt builder for focused, question-grounded rubric grading.
"""

import json
from typing import Dict, Any, List, Optional
from src.core.schemas import AlignedAnswerItem


STAGE4_MODULAR_SYSTEM_PROMPT = (
    "You are an experienced academic examiner grading handwritten exam answers. "
    "You evaluate each answer rigorously, objectively, and fairly according to the "
    "exact question prompt and mark ceiling. Output valid JSON only."
)


def is_objective_question(q_no: str, q_name: str, prompt_text: str) -> bool:
    """Detect if a question is an objective/structured question where linguistic penalties are forbidden."""
    q_str = f"{q_no} {q_name} {prompt_text[:200]}".lower()
    # 1. Match typical HSC English objective questions
    if q_no.strip() in ["1(A)", "1A", "1.A", "2", "4", "5", "6"]:
        return True
    
    # 2. Keyword detection for objective question types across subjects
    objective_keywords = [
        "mcq", "multiple choice", "flow chart", "flowchart", "cloze",
        "fill in the blank", "rearrang", "reorder", "table", "matching",
        "choose the best", "true or false", "one word"
    ]
    return any(kw in q_str for kw in objective_keywords)


def build_modular_question_prompt(
    answer: AlignedAnswerItem,
    question_prompt_text: str,
    max_marks: float,
    subject: str = "English",
    rubric_penalties: Optional[Dict[str, float]] = None
) -> str:
    """
    Build a focused evaluation prompt for a single question answer with question-type aware penalties.
    """
    is_obj = is_objective_question(answer.q_no, answer.q_name or "", question_prompt_text)

    if is_obj:
        max_pen = 0.0
        grading_rules = (
            f"1. This is an OBJECTIVE / STRUCTURED question (e.g. MCQ, flowchart, cloze, rearranging).\n"
            f"2. Grade SOLELY on factual correctness, slot completion, and prompt adherence up to {max_marks}.\n"
            f"3. ZERO LINGUISTIC PENALTY: linguistic_penalty MUST be 0.0. In official board exams, examiners do NOT "
            f"deduct language marks for notes in flowcharts, tables, or slot answers."
        )
    else:
        penalties = rubric_penalties or {
            "spelling_error_deduction": 0.25,
            "grammar_error_deduction": 0.25,
            "max_linguistic_deduction": min(0.5, max_marks * 0.1)
        }
        max_pen = min(float(penalties.get("max_linguistic_deduction", 0.5)), 0.5)
        high_min = round(max_marks * 0.8, 1)
        mid_min = round(max_marks * 0.5, 1)
        mid_max = round(max_marks * 0.79, 1)
        low_max = round(mid_min - 0.1, 1) if mid_min > 0.1 else 0.0

        grading_rules = (
            f"1. Match the student's answer against the prompt instructions. Award content marks up to {max_marks}.\n"
            f"2. NCTB ANCHOR SCORE BANDS (Counter central-tendency score compression):\n"
            f"   - HIGH BAND ({high_min} - {max_marks:.1f} marks / 80-100%): Addresses all required points with clear coherence, "
            f"appropriate vocabulary, and logical development. If the answer covers all required elements, award full or near-full marks ({high_min} - {max_marks:.1f}). "
            f"Do not artificially depress marks for well-written answers.\n"
            f"   - MID BAND ({mid_min} - {mid_max:.1f} marks / 50-79%): Main ideas present but noticeable gaps in development, repetition, or weak organization.\n"
            f"   - LOW BAND (0.0 - {low_max:.1f} marks / 0-49%): Fails to address central prompt, severe omissions, off-topic, or fragmentary.\n"
            f"3. LINGUISTIC DEDUCTION RULES:\n"
            f"   - Deduct ONLY if confirmed errors severely impair comprehensibility or meaning.\n"
            f"   - Strict maximum ceiling: {max_pen:.2f} marks.\n"
            f"   - Zero deduction for minor phonetic slips, British/American spelling variations, or handwriting ambiguities."
        )

    # Format errors compactly (omit errors entirely for objective questions to prevent bias)
    if is_obj or not answer.errors:
        errors_text = "No linguistic deductions applicable for this question type."
    else:
        err_snippets = []
        for e in answer.errors[:8]:
            t = e.get("error_type", "error")
            w = e.get("erroneous_text", "")
            c = e.get("suggested_correction", "")
            err_snippets.append(f"- [{t}] '{w}' -> '{c}'")
        if len(answer.errors) > 8:
            err_snippets.append(f"... and {len(answer.errors) - 8} more minor error(s)")
        errors_text = "\n".join(err_snippets)

    return f"""You are evaluating an individual question answer from an exam script.

EXAM QUESTION CONTEXT:
Subject: {subject}
Question Number: {answer.q_no}
Question Title: {answer.q_name or f'Question {answer.q_no}'}
Maximum Marks: {max_marks}
Question Nature: {'OBJECTIVE / STRUCTURED' if is_obj else 'SUBJECTIVE / CONTINUOUS COMPOSITION'}

OFFICIAL QUESTION PROMPT & INSTRUCTIONS:
\"\"\"
{question_prompt_text.strip()}
\"\"\"

STUDENT'S HANDWRITTEN ANSWER (Verified Transcription):
\"\"\"
{answer.answer_text.strip()}
\"\"\"

RELEVANT LINGUISTIC OBSERVATIONS:
{errors_text}

GRADING GUIDELINES:
{grading_rules}
4. The final awarded marks must NEVER exceed {max_marks}, and must NOT be negative.

Output ONLY a JSON object:
{{
  "q_no": "{answer.q_no}",
  "max_marks": {max_marks},
  "content_raw_score": 0.0,
  "linguistic_penalty": {0.0 if is_obj else 0.0},
  "awarded_marks": 0.0,
  "justification": "1-2 concise sentences explaining the mark awarded",
  "strengths": ["Key strength or accurate element"],
  "weaknesses": ["Key weakness or missing point"],
  "examiner_feedback": "Concise actionable advice for the student"
}}"""

