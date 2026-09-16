import re
import json
from typing import Optional, Dict, Any, List, Tuple
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
from src.pipeline.stage4_modes import (
    load_rubric_specs, load_answer_key, key_for_question, source_text_for_question,
    build_mode_a_prompt, build_mode_b_prompt, build_mode_c_prompt,
    score_mode_a, score_mode_b, score_mode_c, snap_half, STAGE4_MODES_SYSTEM_PROMPT,
)
from src.utils.ground_truth import canonicalize_question_key


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

    def __init__(self, engine: BaseVLMEngine, answer_keys_dir: str = "configs/answer_keys"):
        self.engine = engine
        self.answer_keys_dir = answer_keys_dir

    # ------------------------------------------------------------------ helpers
    def _generate_json(self, prompt: str, system_prompt: str, temperature: float, top_p: float,
                       max_new_tokens: int, thinking_mode: bool, generation_max_time: Optional[float],
                       attempts: int = 2) -> Tuple[Optional[Dict[str, Any]], str]:
        """Call the model and parse JSON; one retry with a strict reminder. Returns (parsed_or_None, raw)."""
        raw = ""
        for attempt in range(attempts):
            p = prompt if attempt == 0 else prompt + "\n\nREMINDER: Output ONLY the JSON object described above. No prose, no markdown."
            try:
                raw = self.engine.generate_text(
                    prompt=p, system_prompt=system_prompt, temperature=temperature, top_p=top_p,
                    max_new_tokens=max_new_tokens, thinking_mode=thinking_mode, max_time=generation_max_time,
                )
            except Exception as ex:
                raw = f"<generation failed: {ex}>"
                continue
            parsed = _extract_json_from_text(raw)
            if isinstance(parsed, dict):
                return parsed, raw
        return None, raw

    def _evaluate_rubric_driven(
        self,
        ans: AlignedAnswerItem,
        spec,
        key_section: str,
        key_entry: Dict[str, Any],
        question_prompt_text: str,
        q_max_marks: float,
        question_text: str,
        temperature: float, top_p: float, max_new_tokens: int, thinking_mode: bool,
        generation_max_time: Optional[float],
    ) -> QuestionEvaluationItem:
        """One question under the mode-based rubric. The model judges, the code scores."""
        mode = spec.mode
        max_mark = float(q_max_marks or spec.max_mark)
        spec.max_mark = max_mark
        errors_text = "No confirmed linguistic errors."
        if ans.errors:
            from src.pipeline.token_guard import filter_protected_errors
            clean_errors, _ = filter_protected_errors(ans.errors, ans.answer_text)
            if clean_errors:
                errors_text = "\n".join(f"- [{e.get('error_type', 'error')}] '{e.get('erroneous_text', '')}' -> '{e.get('suggested_correction', '')}'" for e in clean_errors[:12])
            else:
                errors_text = "No confirmed linguistic errors."
        source_text = source_text_for_question(ans.q_no, question_text) if spec.task_type.lower() in ("summary", "theme") else ""

        if mode == "A":
            prompt = build_mode_a_prompt(ans, spec, key_entry, question_prompt_text)
        elif mode == "B":
            prompt = build_mode_b_prompt(ans, spec, key_entry, question_prompt_text)
        else:
            prompt = build_mode_c_prompt(ans, spec, question_prompt_text, errors_text, source_text)

        parsed, raw = self._generate_json(prompt, STAGE4_MODES_SYSTEM_PROMPT, temperature, top_p, max_new_tokens, thinking_mode, generation_max_time)
        base = dict(q_no=ans.q_no, q_name=ans.q_name or f"Question {ans.q_no}", page_numbers=ans.page_numbers,
                    max_marks=max_mark, task_mode={"A": "A_ITEM", "B": "B_POINT", "C": "C_BAND"}[mode], task_type=spec.task_type,
                    key_source=("answer_key" if key_entry else "none"))
        if parsed is None:
            return QuestionEvaluationItem(awarded_marks=0.0, scoring_status="unscored",
                                          examiner_feedback="Model output could not be parsed; question left unscored.",
                                          scoring_notes=[f"raw: {raw[:200]}"], **base)
        try:
            if mode == "A":
                sr = score_mode_a(parsed, spec, key_entry)
            elif mode == "B":
                sr = score_mode_b(parsed, spec, key_entry)
            else:
                sr = score_mode_c(parsed, spec, ans.answer_text, source_text)
        except Exception as ex:
            return QuestionEvaluationItem(awarded_marks=0.0, scoring_status="unscored",
                                          examiner_feedback=f"Scoring failed: {ex}", scoring_notes=[f"raw: {raw[:200]}"], **base)
        awarded = max(0.0, min(max_mark, snap_half(sr.awarded)))
        feedback = sr.feedback or str(parsed.get("notes") or "")
        if mode in ("A", "B") and not feedback:
            n_ok = sum(1 for it in sr.items if it.get("status") == "correct")
            feedback = f"{n_ok}/{len(sr.items)} items fully correct."
        return QuestionEvaluationItem(
            awarded_marks=awarded, content_raw_score=float(sr.raw_total if sr.raw_total is not None else awarded),
            linguistic_penalty=0.0, examiner_feedback=feedback, strengths=sr.strengths, weaknesses=sr.weaknesses,
            scoring_status="scored", items=sr.items, subscores=sr.subscores, raw_total=sr.raw_total,
            cap_applied=sr.cap_applied, cap_reason=sr.cap_reason, capped_from=sr.capped_from,
            performance_band=sr.band, scoring_notes=sr.notes, **base,
        )

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
        Question-by-question evaluation. With a mode-based rubric (English NCTB framework) every
        question is scored by `stage4_modes` (model judges, code scores); otherwise the generic
        modular prompt is used. Unparseable outputs are `unscored`; questions expected by the rubric
        or the ground truth but absent from the segmentation are `missing`. Both are excluded from
        `mae_vs_human` and counted as 0 in `mae_including_missing`.
        """
        subject = rubric_data.get("subject", "English")
        question_evaluations: List[QuestionEvaluationItem] = []
        total_awarded = 0.0
        total_max = 0.0
        total_raw_content = 0.0
        total_linguistic_deductions = 0.0
        deltas: List[float] = []
        deltas_incl: List[float] = []

        specs = load_rubric_specs(rubric_data)
        rubric_driven = bool(specs)
        answer_key = load_answer_key(question_obj.question_id if question_obj else None, self.answer_keys_dir) if rubric_driven else {}
        gt_canon = {canonicalize_question_key(str(k)): float(v) for k, v in (ground_truth_marks or {}).items()}
        print(f"[Stage 4 Modular] Evaluating {len(answers)} segmented answers question-by-question "
              f"({'rubric-driven modes A/B/C' if rubric_driven else 'generic prompt'}; answer key: {'yes' if answer_key else 'none'})...")

        answered: set = set()
        for idx, ans in enumerate(answers, 1):
            q_prompt_text, q_max_marks = get_sub_question_prompt_and_marks(ans.q_no, question_obj)
            q_canon = canonicalize_question_key(ans.q_no)
            answered.add(q_canon)
            spec = specs.get(q_canon) if rubric_driven else None
            if spec is not None:
                key_section, key_entry = key_for_question(answer_key, q_canon)
                print(f"[Stage 4 Modular] [{idx}/{len(answers)}] Q{ans.q_no} mode {spec.mode} ({spec.task_type}, {ans.word_count} words, max {q_max_marks:g}, key: {'yes' if key_entry else 'no'})...")
                qe = self._evaluate_rubric_driven(
                    ans, spec, key_section, key_entry, q_prompt_text, q_max_marks,
                    question_obj.question_text if question_obj else "",
                    temperature, top_p, max_new_tokens, thinking_mode, generation_max_time,
                )
                gt_val = gt_canon.get(q_canon)
                if gt_val is not None:
                    qe.human_ground_truth = gt_val
                    qe.delta = round(abs(qe.awarded_marks - gt_val), 2)
                    deltas_incl.append(qe.delta)
                    if qe.scoring_status == "scored":
                        deltas.append(qe.delta)
                gt_str = f" | Human GT: {gt_val} (Δ={qe.delta})" if gt_val is not None else ""
                cap_str = f" [cap {qe.cap_reason}: {qe.capped_from}->{qe.awarded_marks}]" if qe.cap_applied else ""
                print(f"[Stage 4 Modular] Q{ans.q_no} {qe.scoring_status}: {qe.awarded_marks:.1f} / {qe.max_marks:.1f}{cap_str}{gt_str}")
                question_evaluations.append(qe)
                total_awarded += qe.awarded_marks
                total_max += qe.max_marks
                total_raw_content += qe.content_raw_score
                continue

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
                # one strict retry, then leave the question explicitly unscored (never fabricate a mark)
                parsed, response = self._generate_json(prompt, STAGE4_MODULAR_SYSTEM_PROMPT, temperature, top_p,
                                                       max_new_tokens, thinking_mode, generation_max_time, attempts=1)
            if not parsed or not isinstance(parsed, dict):
                print(f"[Stage 4 Modular] Warning: Malformed JSON for Q{ans.q_no} after retry. Marked UNSCORED.")
                qe = QuestionEvaluationItem(
                    q_no=ans.q_no, q_name=ans.q_name or f"Question {ans.q_no}", page_numbers=ans.page_numbers,
                    max_marks=q_max_marks, awarded_marks=0.0, scoring_status="unscored", task_mode="generic",
                    examiner_feedback="Model output could not be parsed; question left unscored.",
                    scoring_notes=[f"raw: {str(response)[:200]}"],
                )
                gt_val = gt_canon.get(q_canon)
                if gt_val is not None:
                    qe.human_ground_truth = gt_val
                    qe.delta = round(abs(0.0 - gt_val), 2)
                    deltas_incl.append(qe.delta)
                question_evaluations.append(qe)
                total_max += q_max_marks
                continue

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

            awarded = snap_half(awarded)

            # Ground truth comparison if available
            gt_val = gt_canon.get(q_canon)
            delta_val = None
            if gt_val is not None:
                delta_val = round(abs(awarded - gt_val), 2)
                deltas.append(delta_val)
                deltas_incl.append(delta_val)

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
                weaknesses=parsed.get("weaknesses", []),
                task_mode="generic",
                scoring_status="scored",
            ))

            total_awarded += awarded
            total_max += q_max_marks
            total_raw_content += content_raw
            total_linguistic_deductions += penalty

        # Questions expected by the rubric or the ground truth but absent from the segmentation
        expected = set(specs.keys()) | set(gt_canon.keys())
        missing_qs: List[str] = []
        for q in sorted(expected, key=lambda x: (len(x), x)):
            if q in answered:
                continue
            spec = specs.get(q)
            _, q_max = get_sub_question_prompt_and_marks(q, question_obj)
            max_mark = float(q_max or (spec.max_mark if spec else 0.0))
            gt_val = gt_canon.get(q)
            qe = QuestionEvaluationItem(
                q_no=q, q_name=f"Question {q}", page_numbers=[], max_marks=max_mark, awarded_marks=0.0,
                scoring_status="missing", task_mode=({"A": "A_ITEM", "B": "B_POINT", "C": "C_BAND"}[spec.mode] if spec else None),
                task_type=(spec.task_type if spec else None), human_ground_truth=gt_val,
                examiner_feedback="No answer segment found for this question (unattempted or segmentation miss).",
            )
            if gt_val is not None:
                qe.delta = round(abs(0.0 - gt_val), 2)
                deltas_incl.append(qe.delta)
            question_evaluations.append(qe)
            missing_qs.append(q)
            total_max += max_mark
        unscored_qs = [qe.q_no for qe in question_evaluations if qe.scoring_status == "unscored"]
        if missing_qs:
            print(f"[Stage 4 Modular] Missing answer segments for: {', '.join(missing_qs)}")
        if unscored_qs:
            print(f"[Stage 4 Modular] Unscored (unparseable model output): {', '.join(unscored_qs)}")

        percentage = round((total_awarded / total_max) * 100.0, 2) if total_max > 0 else 0.0
        mae_vs_human = round(sum(deltas) / len(deltas), 2) if deltas else None
        mae_incl = round(sum(deltas_incl) / len(deltas_incl), 2) if deltas_incl else None

        # Build overall synthesis feedback
        feedback_summary = (
            f"Modular evaluation completed across {len(question_evaluations)} questions "
            f"({len(question_evaluations) - len(missing_qs) - len(unscored_qs)} scored, {len(missing_qs)} missing, {len(unscored_qs)} unscored). "
            f"Total awarded score: {total_awarded:.1f} out of {total_max:.1f} ({percentage:.1f}%)."
        )
        if mae_vs_human is not None:
            feedback_summary += f" MAE vs human examiner over scored questions: {mae_vs_human:.2f} (including missing/unscored as 0: {mae_incl})."

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
            mae_including_missing=mae_incl,
            scored_with_gt=len(deltas),
            missing_questions=missing_qs,
            unscored_questions=unscored_qs,
            rubric_driven=rubric_driven,
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
