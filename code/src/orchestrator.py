"""
Pipeline Orchestrator for Modular Answer-Script OCR & Grading.
Implements the 6-stage execution flow, deterministic fallback matrix,
multimodal question image support, multi-page script routing (extraction.csv),
NCTB HSC English 1st Paper (Rubric v2) scoring, and evaluation.csv generation.
"""

from __future__ import annotations
import os
import sys
import json
import time
import re
import csv
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from src.schemas import (
        PipelineConfig,
        PipelineResult,
        Stage1Output,
        Stage2Output,
        Stage3Output,
        Stage4Output,
        Stage5Output,
        Stage6Output,
        ScoreBreakdown,
        StructuralAudit,
        AttemptStatus,
        ErrorAnalysis,
        QuestionSegment,
        Stage2Decision,
        TASK_MAX_MARKS,
        TASK_QUESTION_NUMBERS
    )
    from src.model_client.base import ModelClient
    from src.model_client.factory import get_model_client
    from src.stages import (
        run_stage1_extractor_a,
        run_stage2_rubric_aligner,
        run_stage3_extractor_b,
        run_stage4_ocr_supervisor,
        run_stage5_examiner,
        run_stage6_compressor
    )
    from src.ingestion.page_router import PageRouter, load_manifest_from_csv
    from src.ingestion.pdf_loader import load_images_from_file
    from src.ingestion.image_normalizer import normalize_images
except ImportError:
    try:
        from schemas import (
            PipelineConfig,
            PipelineResult,
            Stage1Output,
            Stage2Output,
            Stage3Output,
            Stage4Output,
            Stage5Output,
            Stage6Output,
            ScoreBreakdown,
            StructuralAudit,
            AttemptStatus,
            ErrorAnalysis,
            QuestionSegment,
            Stage2Decision,
            TASK_MAX_MARKS,
            TASK_QUESTION_NUMBERS
        )
        from model_client.base import ModelClient
        from model_client.factory import get_model_client
        from stages import (
            run_stage1_extractor_a,
            run_stage2_rubric_aligner,
            run_stage3_extractor_b,
            run_stage4_ocr_supervisor,
            run_stage5_examiner,
            run_stage6_compressor
        )
        from ingestion.page_router import PageRouter, load_manifest_from_csv
        from ingestion.pdf_loader import load_images_from_file
        from ingestion.image_normalizer import normalize_images
    except ImportError:
        from .schemas import (
            PipelineConfig,
            PipelineResult,
            Stage1Output,
            Stage2Output,
            Stage3Output,
            Stage4Output,
            Stage5Output,
            Stage6Output,
            ScoreBreakdown,
            StructuralAudit,
            AttemptStatus,
            ErrorAnalysis,
            QuestionSegment,
            Stage2Decision,
            TASK_MAX_MARKS,
            TASK_QUESTION_NUMBERS
        )
        from .model_client.base import ModelClient
        from .model_client.factory import get_model_client
        from .stages import (
            run_stage1_extractor_a,
            run_stage2_rubric_aligner,
            run_stage3_extractor_b,
            run_stage4_ocr_supervisor,
            run_stage5_examiner,
            run_stage6_compressor
        )
        from .ingestion.page_router import PageRouter, load_manifest_from_csv
        from .ingestion.pdf_loader import load_images_from_file
        from .ingestion.image_normalizer import normalize_images


def resolve_final_ocr(
    config: PipelineConfig,
    stage1_output: Optional[Stage1Output],
    stage3_output: Optional[Stage3Output],
    stage4_output: Optional[Stage4Output]
) -> Tuple[str, str]:
    """
    Deterministic resolution of final OCR question and answer text based on enabled stages.
    """
    if config.enabled_stages.ocr_supervisor and stage4_output is not None:
        return stage4_output.QUESTION_TEXT, stage4_output.STUDENT_ANSWER

    if config.enabled_stages.extractor_b and stage3_output is not None:
        return stage3_output.QUESTION_TEXT, stage3_output.STUDENT_ANSWER

    if stage1_output is not None:
        return stage1_output.QUESTION_TEXT, stage1_output.STUDENT_ANSWER

    return "", ""


def resolve_final_marks(
    config: PipelineConfig,
    stage5_output: Optional[Stage5Output],
    stage6_output: Optional[Stage6Output],
    default_max_mark: float = 10.0
) -> Tuple[float, ScoreBreakdown, StructuralAudit, str, ErrorAnalysis, str, bool, bool, Optional[str]]:
    """
    Deterministic resolution of final marks and criterion breakdown.
    """
    if config.enabled_stages.compressor and stage6_output is not None:
        return (
            stage6_output.final_marks,
            stage6_output.score_breakdown,
            stage6_output.structural_audit,
            stage6_output.performance_band,
            stage6_output.error_analysis,
            stage6_output.feedback_summary,
            stage6_output.sum_check_passed,
            stage6_output.band_check_passed,
            stage6_output.error_detection
        )

    if stage5_output is not None:
        marks = stage5_output.stated_total
        bd = stage5_output.score_breakdown
        if marks == 0.0 and bd.total_score > 0.0:
            marks = bd.total_score

        return (
            marks,
            stage5_output.score_breakdown,
            stage5_output.structural_audit,
            stage5_output.performance_band,
            stage5_output.error_analysis,
            stage5_output.feedback_summary,
            True,
            True,
            "not verified (Stage 6 disabled)"
        )

    return 0.0, ScoreBreakdown(), StructuralAudit(), "Band 0", ErrorAnalysis(), "", False, False, "No grading stages executed"


def save_stage_json(output_dir: str, student_id: str, question_id: str, filename: str, data: Dict[str, Any]) -> None:
    """Helper to save per-stage JSON output to disk."""
    stage_dir = os.path.join(output_dir, student_id, question_id)
    os.makedirs(stage_dir, exist_ok=True)
    file_path = os.path.join(stage_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_pipeline(
    pages: List[Image.Image],
    rubric: str,
    reference_solution: str,
    config: PipelineConfig,
    task_type: str = "Paragraph",
    max_mark: Optional[float] = None,
    question_no: str = "7",
    task_id: str = "PARA_001",
    student_id: str = "SE_11_Q1_0001",
    teacher_mark: Optional[float] = None,
    page_range_str: str = "",
    question_images: Optional[List[Image.Image]] = None,
    model_client: Optional[ModelClient] = None,
    verbose: bool = False
) -> PipelineResult:
    """
    Execute the 6-stage modular pipeline for a single student question answer.
    """
    start_time = time.time()
    client = model_client or get_model_client(config.model)
    enabled = config.enabled_stages
    save_logs = config.logging.save_per_stage_json
    out_dir = config.logging.output_dir
    applied_max_mark = max_mark if max_mark is not None else TASK_MAX_MARKS.get(task_type, 10.0)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Grading Pipeline | Task: {task_id} ({task_type}) | Script: {student_id} | QNo: {question_no}")
        print(f"Max Mark: {applied_max_mark} | Pages: {len(pages)} (Range: {page_range_str or 'all'})")
        print(f"Active Stages: {[k for k, v in enabled.to_dict().items() if v]}")
        print(f"{'='*70}")

    # STAGE 1: Extractor A (Cold OCR Read)
    stage1_out: Optional[Stage1Output] = None
    if enabled.extractor_a:
        if verbose:
            print("[Stage 1/6] Running Extractor A (Cold OCR Read)...")
        stage1_out = run_stage1_extractor_a(
            images=pages,
            model_client=client,
            question_images=question_images,
            temperature=config.model.temperature
        )
        if save_logs:
            save_stage_json(out_dir, student_id, task_id, "stage_1_extractor_a.json", stage1_out.to_dict())
    else:
        if verbose:
            print("[Stage 1/6] Extractor A DISABLED.")

    # STAGE 2: RubricAligner
    stage2_out: Optional[Stage2Output] = None
    operative_rubric = rubric
    applicable_solution = reference_solution
    examiner_note: Optional[str] = None

    if enabled.rubric_aligner and stage1_out is not None:
        if verbose:
            print("[Stage 2/6] Running RubricAligner...")
        stage2_out = run_stage2_rubric_aligner(
            stage1_output=stage1_out,
            original_rubric=rubric,
            original_reference_solution=reference_solution,
            model_client=client,
            temperature=config.model.temperature
        )
        operative_rubric = stage2_out.operative_rubric or rubric
        applicable_solution = stage2_out.shadow_solution if stage2_out.shadow_solution else reference_solution
        examiner_note = stage2_out.examiner_note
        if save_logs:
            save_stage_json(out_dir, student_id, task_id, "stage_2_rubric_aligner.json", stage2_out.to_dict())
    else:
        if verbose:
            print("[Stage 2/6] RubricAligner DISABLED.")
        stage2_out = Stage2Output(
            operative_rubric=rubric,
            examiner_note=None,
            shadow_solution=None,
            decision=Stage2Decision.KEEP
        )

    # STAGE 3: Extractor B (Reference-primed OCR)
    stage3_out: Optional[Stage3Output] = None
    if enabled.extractor_b:
        if verbose:
            print("[Stage 3/6] Running Extractor B (Reference-primed OCR)...")
        stage3_out = run_stage3_extractor_b(
            images=pages,
            applicable_solution=applicable_solution,
            model_client=client,
            question_images=question_images,
            temperature=config.model.temperature
        )
        if save_logs:
            save_stage_json(out_dir, student_id, task_id, "stage_3_extractor_b.json", stage3_out.to_dict())
    else:
        if verbose:
            print("[Stage 3/6] Extractor B DISABLED.")

    # STAGE 4: OCR Supervisor
    stage4_out: Optional[Stage4Output] = None
    if enabled.ocr_supervisor and stage1_out is not None:
        if verbose:
            print("[Stage 4/6] Running OCR Supervisor...")
        stage4_out = run_stage4_ocr_supervisor(
            images=pages,
            stage1_output=stage1_out,
            stage3_output=stage3_out,
            model_client=client,
            question_images=question_images,
            temperature=config.model.temperature
        )
        if save_logs:
            save_stage_json(out_dir, student_id, task_id, "stage_4_ocr_supervisor.json", stage4_out.to_dict())
    else:
        if verbose:
            print("[Stage 4/6] OCR Supervisor DISABLED.")

    # Authoritative OCR Resolution
    final_ocr_q, final_ocr_ans = resolve_final_ocr(
        config=config,
        stage1_output=stage1_out,
        stage3_output=stage3_out,
        stage4_output=stage4_out
    )

    # STAGE 5: Examiner (Rubric v2 Evaluation)
    stage5_out: Optional[Stage5Output] = None
    if enabled.examiner:
        if verbose:
            print(f"[Stage 5/6] Running Chief Examiner ({task_type}, Max: {applied_max_mark})...")
        stage5_out = run_stage5_examiner(
            final_ocr_question=final_ocr_q,
            final_ocr_answer=final_ocr_ans,
            operative_rubric=operative_rubric,
            reference_solution=applicable_solution,
            examiner_note=examiner_note,
            model_client=client,
            task_type=task_type,
            max_mark=applied_max_mark,
            question_images=question_images,
            student_images=pages,
            temperature=config.model.temperature
        )
        if save_logs:
            save_stage_json(out_dir, student_id, task_id, "stage_5_examiner.json", stage5_out.to_dict())
    else:
        if verbose:
            print("[Stage 5/6] Examiner DISABLED.")

    # STAGE 6: Compressor (Audit & Sanity-Check)
    stage6_out: Optional[Stage6Output] = None
    if enabled.compressor and stage5_out is not None:
        if verbose:
            print("[Stage 6/6] Running Compressor (Sanity Audit & Hard Cap Check)...")
        stage6_out = run_stage6_compressor(
            stage5_output=stage5_out,
            model_client=client,
            temperature=config.model.temperature
        )
        if save_logs:
            save_stage_json(out_dir, student_id, task_id, "stage_6_compressor.json", stage6_out.to_dict())
    else:
        if verbose:
            print("[Stage 6/6] Compressor DISABLED.")

    # Resolve Final Marks & Diagnostics
    final_score, breakdown, audit, band, err_analysis, feedback_summary, sum_ok, band_ok, error_det = resolve_final_marks(
        config=config,
        stage5_output=stage5_out,
        stage6_output=stage6_out,
        default_max_mark=applied_max_mark
    )

    elapsed_time = time.time() - start_time
    now_iso = datetime.now(timezone.utc).isoformat()
    grader_model = config.model.checkpoint or "easyocr-local-pipeline"

    frequent_errors_str = " | ".join(err_analysis.frequent_errors)
    positive_aspects_str = " | ".join(err_analysis.positive_aspects)

    result = PipelineResult(
        evaluation_id=f"{task_id}__{grader_model}",
        task_id=task_id,
        script_id=student_id,
        question_no=question_no,
        question_type=task_type,
        max_mark=applied_max_mark,
        teacher_mark=teacher_mark,
        total_score=final_score,
        performance_band=band,
        context_content_data=breakdown.context_content_data,
        structure_format_brevity=breakdown.structure_format_brevity,
        language_mechanics=breakdown.language_mechanics,
        originality_comparisons_paraphrase=breakdown.originality_comparisons_paraphrase,
        is_attempted=True,
        cap_applied=audit.cap_applied,
        cap_reason=audit.applied_cap_reason,
        frequent_errors=frequent_errors_str,
        positive_aspects=positive_aspects_str,
        feedback_summary=feedback_summary,
        sum_check_passed=sum_ok,
        band_check_passed=band_ok,
        error_detection=error_det,
        final_ocr_question=final_ocr_q,
        final_ocr_answer=final_ocr_ans,
        page_range=page_range_str,
        stage1_output=stage1_out,
        stage2_output=stage2_out,
        stage3_output=stage3_out,
        stage4_output=stage4_out,
        stage5_output=stage5_out,
        stage6_output=stage6_out,
        grader=grader_model,
        prompt_version="rubric_v2",
        temperature=config.model.temperature,
        raw_json_path=f"evaluations/raw/{task_id}__{grader_model}.json",
        eval_timestamp=now_iso,
        execution_time_seconds=elapsed_time
    )

    if save_logs:
        save_stage_json(out_dir, student_id, task_id, "final_result.json", result.to_dict())

    if verbose:
        print(f"Completed in {elapsed_time:.2f}s | Score: {final_score}/{applied_max_mark} ({band}) | Cap: {audit.applied_cap_reason if audit.cap_applied else 'None'}")

    return result


def run_script_pipeline(
    pdf_path: str,
    config: PipelineConfig,
    manifest_csv: Optional[str] = "extraction.csv",
    rubric_text: Optional[str] = None,
    auto_route: bool = False,
    verbose: bool = True
) -> List[PipelineResult]:
    """
    Process an entire multi-page student script PDF (e.g. SE_11_Q1_0001.pdf).
    Segments all constituent questions (Q3, Q7, Q8, Q9, Q10, Q11) and grades each.
    """
    script_id = os.path.splitext(os.path.basename(pdf_path))[0]
    pages = load_images_from_file(pdf_path, dpi=config.ingestion.pdf_dpi)
    pages = normalize_images(pages, max_side=config.ingestion.max_image_side)

    # Segment script into questions
    segments = PageRouter.segment_script_into_questions(
        pages=pages,
        script_id=script_id,
        manifest_csv=manifest_csv,
        auto_detect=auto_route
    )

    client = get_model_client(config.model)
    rubric = rubric_text or ""
    if not rubric and config.rubric_path and os.path.exists(config.rubric_path):
        with open(config.rubric_path, "r", encoding="utf-8") as f:
            rubric = f.read()

    results: List[PipelineResult] = []

    for task_key, (q_pages, seg) in segments.items():
        res = run_pipeline(
            pages=q_pages,
            rubric=rubric,
            reference_solution=seg.question_prompt,
            config=config,
            task_type=seg.question_type,
            max_mark=seg.max_mark,
            question_no=seg.question_no,
            task_id=seg.task_id or task_key,
            student_id=script_id,
            teacher_mark=seg.teacher_mark,
            page_range_str=seg.page_range_str,
            model_client=client,
            verbose=verbose
        )
        results.append(res)

    return results


def export_evaluation_csv(results: List[PipelineResult], output_csv_path: str) -> None:
    """Export pipeline results into exact format matching evaluation.csv."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    fieldnames = [
        "evaluation_id", "task_id", "script_id", "grader", "max_mark", "teacher_mark",
        "total_score", "performance_band", "context_content_data", "structure_format_brevity",
        "language_mechanics", "originality_comparisons_paraphrase", "is_attempted", "cap_applied",
        "cap_reason", "frequent_errors", "positive_aspects", "feedback_summary",
        "sum_check_passed", "band_check_passed", "model_version", "prompt_version",
        "temperature", "raw_json_path", "eval_timestamp"
    ]

    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_evaluation_csv_row())
    except PermissionError:
        try:
            import time
            time.sleep(0.5)
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(r.to_evaluation_csv_row())
        except Exception:
            pass


def run_batch_pipeline(
    items: List[Dict[str, Any]],
    config: PipelineConfig,
    summary_csv_path: Optional[str] = None
) -> List[PipelineResult]:
    """Run batch pipeline over a list of items and export CSV."""
    results: List[PipelineResult] = []
    client = get_model_client(config.model)

    for item in items:
        pages = item.get("pages")
        if not pages and "file_path" in item:
            pages = load_images_from_file(item["file_path"], dpi=config.ingestion.pdf_dpi)
            pages = normalize_images(pages, max_side=config.ingestion.max_image_side)

        res = run_pipeline(
            pages=pages,
            rubric=item.get("rubric", ""),
            reference_solution=item.get("reference_solution", ""),
            config=config,
            task_type=item.get("task_type", "Paragraph"),
            max_mark=item.get("max_mark", 10.0),
            question_no=item.get("question_no", "7"),
            task_id=item.get("task_id", "TASK_001"),
            student_id=item.get("student_id", "script_01"),
            teacher_mark=item.get("teacher_mark"),
            page_range_str=item.get("page_range", ""),
            model_client=client,
            verbose=False
        )
        results.append(res)

    if summary_csv_path:
        export_evaluation_csv(results, summary_csv_path)

    return results
