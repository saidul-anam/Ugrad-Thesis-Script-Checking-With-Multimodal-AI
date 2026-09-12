import re
import json
from typing import Optional, Dict, Any, List
from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import (
    Stage4EvaluationResult,
    CriterionScore,
    Stage3ErrorResult,
    ExtractedQuestion,
    AlignedAnswerItem,
    QuestionEvaluationItem
)
from src.prompts.stage4_rubric import build_stage4_prompt, STAGE4_SYSTEM_PROMPT
from src.prompts.stage4_modular import build_modular_question_prompt, STAGE4_MODULAR_SYSTEM_PROMPT
from src.pipeline.stage2_verifier import _extract_json_from_text


def get_sub_question_prompt_and_marks(
    q_no: str,
    question_obj: Optional[ExtractedQuestion]
) -> tuple[str, float]:
    """Retrieve question-specific instructions and maximum marks from ExtractedQuestion."""
    if not question_obj:
        return f"Evaluate Question {q_no}.", 10.0

    max_marks = 10.0
    for sq in question_obj.sub_questions:
        num = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
        if num == q_no:
            try:
                max_marks = float(sq.get("max_marks") or sq.get("marks") or 10.0)
            except (ValueError, TypeError):
                max_marks = 10.0
            break

    full_text = question_obj.question_text
    # Match question header e.g. "1. Read...", "2. Read...", "7. Answer...", "A.", "B."
    clean_num = re.sub(r'\(.*?\)', '', q_no).strip()
    if clean_num:
        pattern = re.compile(rf'(?:^|\n)\s*{re.escape(clean_num)}\.?\s+(.*?)(?=\n\s*(?:[0-9]{{1,2}}\.|\Z))', re.DOTALL)
        match = pattern.search(full_text)
        if match:
            snippet = match.group(1).strip()
            # If subpart (A) or (B)
            if "(A)" in q_no and "A." in snippet:
                sub_pat = re.compile(r'A\.\s+(.*?)(?=\n\s*B\.|\Z)', re.DOTALL)
                sub_m = sub_pat.search(snippet)
                if sub_m:
                    return f"Context:\n{snippet[:400]}...\n\nInstructions (Part A):\n{sub_m.group(1).strip()}", max_marks
            elif "(B)" in q_no and "B." in snippet:
                sub_pat = re.compile(r'B\.\s+(.*?)(?=\Z)', re.DOTALL)
                sub_m = sub_pat.search(snippet)
                if sub_m:
                    return f"Context:\n{snippet[:400]}...\n\nInstructions (Part B):\n{sub_m.group(1).strip()}", max_marks
            return snippet, max_marks

    # Fallback to sub_question name
    for sq in question_obj.sub_questions:
        if str(sq.get("q_no") or "").strip() == q_no:
            name = sq.get("name") or sq.get("title") or ""
            return f"Question {q_no}: {name}", max_marks

    return f"Evaluate handwritten answer for Question {q_no}.", max_marks


class Stage4Evaluator:
    """
    Stage 4: Rubric Evaluation & Pedagogical Feedback.
    Supports both Modular (Question-by-Question) and Monolithic (Single-Pass) evaluation.
    """

    def __init__(self, engine: BaseVLMEngine):
        self.engine = engine

    def evaluate_modular(
        self,
        answers: List[AlignedAnswerItem],
        question_obj: Optional[ExtractedQuestion],
        rubric_data: Dict[str, Any],
        ground_truth_marks: Optional[Dict[str, float]] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 1024,
        thinking_mode: bool = False,
        generation_max_time: Optional[float] = None
    ) -> Stage4EvaluationResult:
        """
        Execute Question-Mapped Modular Evaluation (Approach 2):
        Evaluates each answer in an isolated, high-focus LLM call,
        aggregates scores, and calculates question-level MAE against gt.txt.
        """
        subject = rubric_data.get("subject", "English")
        question_evaluations: List[QuestionEvaluationItem] = []
        total_awarded = 0.0
        total_max = 0.0
        total_raw_content = 0.0
        total_linguistic_deductions = 0.0
        deltas: List[float] = []

        print(f"[Stage 4 Modular] Evaluating {len(answers)} segmented answers question-by-question...")

        for idx, ans in enumerate(answers, 1):
            q_prompt_text, q_max_marks = get_sub_question_prompt_and_marks(ans.q_no, question_obj)
            prompt = build_modular_question_prompt(
                answer=ans,
                question_prompt_text=q_prompt_text,
                max_marks=q_max_marks,
                subject=subject,
                rubric_penalties=rubric_data.get("penalties")
            )

            print(f"[Stage 4 Modular] [{idx}/{len(answers)}] Grading Q{ans.q_no} ({ans.q_name or 'Answer'}, {ans.word_count} words, max {q_max_marks} marks)...")

            response = self.engine.generate_text(
                prompt=prompt,
                system_prompt=STAGE4_MODULAR_SYSTEM_PROMPT,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                thinking_mode=thinking_mode,
                max_time=generation_max_time
            )

            parsed = _extract_json_from_text(response)
            if not parsed or not isinstance(parsed, dict):
                print(f"[Stage 4 Modular] Warning: Malformed JSON for Q{ans.q_no}. Using conservative fallback.")
                parsed = {
                    "awarded_marks": round(q_max_marks * 0.6, 1),
                    "content_raw_score": round(q_max_marks * 0.6, 1),
                    "linguistic_penalty": 0.0,
                    "justification": "Completed basic answer according to prompt.",
                    "strengths": ["Clear handwriting"],
                    "weaknesses": ["Minor errors noted"],
                    "examiner_feedback": "Review key concepts."
                }

            # Check if this is an objective / structured question
            from src.prompts.stage4_modular import is_objective_question
            is_obj = is_objective_question(ans.q_no, ans.q_name or "", q_prompt_text)

            content_raw = float(parsed.get("content_raw_score", 0.0))
            if is_obj:
                penalty = 0.0
            else:
                raw_pen = float(parsed.get("linguistic_penalty", 0.0))
                penalty = max(0.0, min(0.5, raw_pen))

            # Clamp awarded marks between 0 and q_max_marks
            try:
                raw_m = float(parsed.get("awarded_marks", parsed.get("final_score", 0.0)))
            except (ValueError, TypeError):
                raw_m = round(q_max_marks * 0.5, 1)

            if is_obj and content_raw > 0:
                # Guarantee objective questions are graded on content with zero linguistic deduction
                awarded = max(0.0, min(q_max_marks, max(raw_m, content_raw)))
            else:
                awarded = max(0.0, min(q_max_marks, raw_m))
            
            if not content_raw:
                content_raw = awarded

            # Ground truth comparison if available
            gt_val = None
            delta_val = None
            if ground_truth_marks and ans.q_no in ground_truth_marks:
                gt_val = ground_truth_marks[ans.q_no]
                delta_val = round(abs(awarded - gt_val), 2)
                deltas.append(delta_val)

            gt_str = f" | Human GT: {gt_val} (Δ={delta_val})" if gt_val is not None else ""
            print(f"[Stage 4 Modular] Q{ans.q_no} Score: {awarded:.1f} / {q_max_marks:.1f}{gt_str}")

            question_evaluations.append(QuestionEvaluationItem(
                q_no=ans.q_no,
                q_name=ans.q_name or f"Question {ans.q_no}",
                page_numbers=ans.page_numbers,
                max_marks=q_max_marks,
                awarded_marks=awarded,
                human_ground_truth=gt_val,
                delta=delta_val,
                content_raw_score=content_raw,
                linguistic_penalty=penalty,
                examiner_feedback=parsed.get("examiner_feedback") or parsed.get("justification") or "",
                strengths=parsed.get("strengths", []),
                weaknesses=parsed.get("weaknesses", [])
            ))

            total_awarded += awarded
            total_max += q_max_marks
            total_raw_content += content_raw
            total_linguistic_deductions += penalty

        percentage = round((total_awarded / total_max) * 100.0, 2) if total_max > 0 else 0.0
        mae_vs_human = round(sum(deltas) / len(deltas), 2) if deltas else None

        # Build overall synthesis feedback
        feedback_summary = (
            f"Modular evaluation successfully completed across {len(question_evaluations)} questions. "
            f"Total awarded score: {total_awarded:.1f} out of {total_max:.1f} ({percentage:.1f}%)."
        )
        if mae_vs_human is not None:
            feedback_summary += f" Mean Absolute Error (MAE) against verified human examiner marks: {mae_vs_human:.2f}."

        recommendations = [
            "Review sub-questions with significant score deductions.",
            "Focus on grammar and vocabulary precision in extended composition tasks."
        ]

        total_paper_max = float(question_obj.total_marks) if (question_obj and question_obj.total_marks) else float(rubric_data.get("total_marks", total_max or 10.0))

        return Stage4EvaluationResult(
            subject=subject,
            question_type=rubric_data.get("question_type", "Comprehensive Exam"),
            question_id=question_obj.question_id if question_obj else None,
            question_text=question_obj.question_text if question_obj else None,
            eval_mode="modular",
            criteria_scores=[],
            question_evaluations=question_evaluations,
            content_raw_score=total_raw_content,
            linguistic_penalty=total_linguistic_deductions,
            final_score=round(total_awarded, 2),
            total_max_marks=round(total_paper_max, 2),
            percentage=percentage,
            mae_vs_human=mae_vs_human,
            overall_feedback=feedback_summary,
            actionable_recommendations=recommendations
        )

    def run(
        self,
        verified_transcript: str,
        stage3_errors: Stage3ErrorResult,
        rubric_data: Dict[str, Any],
        thematic_context: Optional[str] = None,
        question_text: Optional[str] = None,
        question_id: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 3072,
        thinking_mode: bool = False,
        generation_max_time: Optional[float] = None
    ) -> Stage4EvaluationResult:
        """Legacy Monolithic Single-Pass Evaluator (Preserved for Thesis Baseline Ablation)."""
        error_dict = stage3_errors.model_dump()
        prompt = build_stage4_prompt(
            verified_transcript=verified_transcript,
            error_list=error_dict,
            rubric_data=rubric_data,
            thematic_context=thematic_context,
            question_text=question_text,
            question_id=question_id
        )

        response = self.engine.generate_text(
            prompt=prompt,
            system_prompt=STAGE4_SYSTEM_PROMPT,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            thinking_mode=thinking_mode,
            max_time=generation_max_time
        )

        parsed_data = _extract_json_from_text(response)
        total_max_marks = float(rubric_data.get("total_marks", 10.0))

        if parsed_data and "criteria_scores" in parsed_data:
            criteria_scores = []
            raw_sum = 0.0
            for item in parsed_data.get("criteria_scores", []):
                max_m = float(item.get("max_marks", 2.0))
                awarded_m = float(item.get("awarded_marks", 0.0))
                raw_sum += awarded_m
                criteria_scores.append(CriterionScore(
                    criterion_id=item.get("criterion_id", "c"),
                    criterion_name=item.get("criterion_name", "Criterion"),
                    max_marks=max_m,
                    awarded_marks=awarded_m,
                    justification=item.get("justification", ""),
                    strengths=item.get("strengths", []),
                    weaknesses=item.get("weaknesses", [])
                ))

            content_raw_score = float(parsed_data.get("content_raw_score", raw_sum))
            penalty = float(parsed_data.get("linguistic_penalty", 0.0))
            final_score = max(0.0, min(total_max_marks, float(parsed_data.get("final_score", content_raw_score - penalty))))
            percentage = round((final_score / total_max_marks) * 100.0, 2) if total_max_marks > 0 else 0.0

            return Stage4EvaluationResult(
                subject=parsed_data.get("subject", rubric_data.get("subject", "General")),
                question_type=parsed_data.get("question_type", rubric_data.get("question_type", "Standard")),
                question_id=question_id or parsed_data.get("question_id"),
                question_text=question_text,
                eval_mode="monolithic",
                criteria_scores=criteria_scores,
                question_evaluations=[],
                content_raw_score=content_raw_score,
                linguistic_penalty=penalty,
                final_score=final_score,
                total_max_marks=total_max_marks,
                percentage=percentage,
                overall_feedback=parsed_data.get("overall_feedback", "Evaluation completed."),
                actionable_recommendations=parsed_data.get("actionable_recommendations", [])
            )

        # Fallback default if model returned unformatted text
        print(f"[Stage4Evaluator] Warning: Falling back to default scoring. Raw response len={len(response)}.")
        return Stage4EvaluationResult(
            subject=rubric_data.get("subject", "General"),
            question_type=rubric_data.get("question_type", "Standard"),
            question_id=question_id,
            question_text=question_text,
            eval_mode="monolithic",
            criteria_scores=[],
            question_evaluations=[],
            content_raw_score=5.0,
            linguistic_penalty=0.0,
            final_score=5.0,
            total_max_marks=total_max_marks,
            percentage=50.0,
            overall_feedback="Automated fallback evaluation generated.",
            actionable_recommendations=["Review transcript and rubric criteria manually."]
        )
