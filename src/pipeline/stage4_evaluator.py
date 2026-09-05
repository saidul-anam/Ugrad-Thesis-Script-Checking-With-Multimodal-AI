import json
from typing import Optional, Dict, Any, List
from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import Stage4EvaluationResult, CriterionScore, Stage3ErrorResult
from src.prompts.stage4_rubric import build_stage4_prompt, STAGE4_SYSTEM_PROMPT
from src.pipeline.stage2_verifier import _extract_json_from_text


class Stage4Evaluator:
    """Stage 4: Rubric Evaluation & Feedback (Verified Text + Error List + RAG -> Final Score)."""

    def __init__(self, engine: BaseVLMEngine):
        self.engine = engine

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
                criteria_scores=criteria_scores,
                content_raw_score=content_raw_score,
                linguistic_penalty=penalty,
                final_score=final_score,
                total_max_marks=total_max_marks,
                percentage=percentage,
                overall_feedback=parsed_data.get("overall_feedback", "Evaluation completed."),
                actionable_recommendations=parsed_data.get("actionable_recommendations", [])
            )

        # Fallback default if model returned unformatted text
        print(f"[Stage4Evaluator] Warning: Falling back to default scoring. Raw response (len={len(response)}). End of response:\n{response[-500:]}")
        return Stage4EvaluationResult(
            subject=rubric_data.get("subject", "General"),
            question_type=rubric_data.get("question_type", "Standard"),
            question_id=question_id,
            question_text=question_text,
            criteria_scores=[],
            content_raw_score=5.0,
            linguistic_penalty=0.0,
            final_score=5.0,
            total_max_marks=total_max_marks,
            percentage=50.0,
            overall_feedback="Automated fallback evaluation generated.",
            actionable_recommendations=["Review transcript and rubric criteria manually."]
        )
