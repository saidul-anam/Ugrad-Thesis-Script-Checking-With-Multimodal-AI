"""
Stage 6: Compressor & Assessment Auditor
Performs mathematical integrity verification, hard-cap ceiling enforcement,
performance band validation, and structured compression of Examiner output.
"""

from typing import Any, Dict, List, Optional
import json

try:
    from src.schemas import (
        Stage5Output,
        Stage6Output,
        ScoreBreakdown,
        StructuralAudit,
        AttemptStatus,
        ErrorAnalysis
    )
    from src.model_client.base import ModelClient
    from src.stages.base_stage import extract_json_from_response
except ImportError:
    try:
        from schemas import (
            Stage5Output,
            Stage6Output,
            ScoreBreakdown,
            StructuralAudit,
            AttemptStatus,
            ErrorAnalysis
        )
        from model_client.base import ModelClient
        from stages.base_stage import extract_json_from_response
    except ImportError:
        from schemas import (
            Stage5Output,
            Stage6Output,
            ScoreBreakdown,
            StructuralAudit,
            AttemptStatus,
            ErrorAnalysis
        )
        from model_client.base import ModelClient
        from .base_stage import extract_json_from_response


def determine_performance_band(score: float, max_mark: float) -> str:
    """Determine official performance band based on score and max_mark from rubric_v2.txt."""
    if score <= 0.0:
        return "Band 0"

    if max_mark == 10.0:
        if score >= 8.0:
            return "Band 4"
        elif score >= 6.0:
            return "Band 3"
        elif score >= 4.0:
            return "Band 2"
        else:
            return "Band 1"
    elif max_mark == 8.0:
        if score >= 7.0:
            return "Band 4"
        elif score >= 5.0:
            return "Band 3"
        elif score >= 3.0:
            return "Band 2"
        else:
            return "Band 1"
    elif max_mark == 7.0:
        if score >= 6.0:
            return "Band 4"
        elif score >= 4.0:
            return "Band 3"
        elif score >= 3.0:
            return "Band 2"
        else:
            return "Band 1"
    elif max_mark == 5.0:
        if score >= 4.0:
            return "Band 4"
        elif score >= 3.0:
            return "Band 3"
        elif score >= 2.0:
            return "Band 2"
        else:
            return "Band 1"
    else:
        ratio = score / max_mark
        if ratio >= 0.8:
            return "Band 4"
        elif ratio >= 0.6:
            return "Band 3"
        elif ratio >= 0.4:
            return "Band 2"
        else:
            return "Band 1"


def get_hard_cap_limit(task_type: str, max_mark: float, cap_reason: str) -> Optional[float]:
    """Calculate hard cap ceiling from rubric_v2.txt."""
    if cap_reason == "None" or not cap_reason:
        return None
    if cap_reason == "Graph_External_Facts":
        return 6.0  # 60% of 10
    if cap_reason == "Paragraph_Subdivisions":
        return 5.0
    if cap_reason == "Summary_Verbatim_Length":
        return 5.0
    if cap_reason == "Theme_Verbatim_Copy":
        return 4.0
    if cap_reason == "Letter_Missing_Layout":
        return 2.0
    
    # Standard 50% cap
    return int(max_mark * 0.5)


def run_stage6_compressor(
    stage5_output: Stage5Output,
    model_client: Optional[ModelClient] = None,
    temperature: float = 0.0,
    **kwargs: Any
) -> Stage6Output:
    """
    Execute Stage 6: Compressor (Audit & Sanity-Check).
    Validates arithmetic sum, enforces hard cap ceilings, and checks performance band integrity.
    """
    bd = stage5_output.score_breakdown
    max_m = stage5_output.max_mark_applied
    audit = stage5_output.structural_audit
    
    # 1. Sum Check
    calculated_sum = round(
        bd.context_content_data +
        bd.structure_format_brevity +
        bd.language_mechanics +
        bd.originality_comparisons_paraphrase,
        2
    )
    
    reported_total = round(stage5_output.stated_total, 2)
    sum_check_passed = abs(calculated_sum - reported_total) < 0.01

    final_score = calculated_sum if calculated_sum > 0.0 else reported_total
    
    # 2. Max Mark Enforcement
    final_score = min(final_score, max_m)
    final_score = max(final_score, 0.0)

    # 3. Hard Cap Enforcement
    error_flags = []
    if not sum_check_passed:
        error_flags.append(f"Sum mismatch: sub-scores sum to {calculated_sum}, but stated total was {reported_total}.")

    cap_limit = get_hard_cap_limit(stage5_output.task_type, max_m, audit.applied_cap_reason)
    if audit.cap_applied and cap_limit is not None:
        if final_score > cap_limit:
            error_flags.append(f"Hard Cap Enforced: Score capped at {cap_limit} (was {final_score}) due to {audit.applied_cap_reason}.")
            final_score = cap_limit

    bd.total_score = final_score

    # 4. Performance Band Check
    expected_band = determine_performance_band(final_score, max_m)
    band_check_passed = (expected_band == stage5_output.performance_band)
    if not band_check_passed:
        error_flags.append(f"Band mismatch: Expected {expected_band} for score {final_score}, got {stage5_output.performance_band}.")

    err_detection_str = " | ".join(error_flags) if error_flags else None

    return Stage6Output(
        task_type=stage5_output.task_type,
        max_mark_applied=max_m,
        final_marks=final_score,
        performance_band=expected_band,
        score_breakdown=bd,
        structural_audit=audit,
        attempt_status=stage5_output.attempt_status,
        error_analysis=stage5_output.error_analysis,
        feedback_summary=stage5_output.feedback_summary,
        sum_check_passed=sum_check_passed,
        band_check_passed=band_check_passed,
        error_detection=err_detection_str
    )
