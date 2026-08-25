"""Pydantic models mirroring the rubric_v2 output schema, extended with
raw_subscores, per-criterion reasoning+evidence, and capped_from.

Also holds the rubric's fixed data tables (sub-score ceilings, caps, bands) used
by validation (Stage 4) and to synthesise not_attempted records.
"""

from __future__ import annotations

from pydantic import BaseModel

FOUR_CRITERIA = [
    "context_content_data",
    "structure_format_brevity",
    "language_mechanics",
    "originality_comparisons_paraphrase",
]

# Per-task sub-score ceilings (from the TASK-SPECIFIC RUBRICS section).
SUBSCORE_CEILINGS: dict[str, dict[str, int]] = {
    "Paragraph":   {"context_content_data": 3, "structure_format_brevity": 3,
                    "language_mechanics": 2, "originality_comparisons_paraphrase": 2},
    "Summary":     {"context_content_data": 3, "structure_format_brevity": 2,
                    "language_mechanics": 2, "originality_comparisons_paraphrase": 3},
    "Graph_Chart": {"context_content_data": 3, "structure_format_brevity": 3,
                    "language_mechanics": 2, "originality_comparisons_paraphrase": 2},
    "Theme":       {"context_content_data": 3, "structure_format_brevity": 2,
                    "language_mechanics": 1, "originality_comparisons_paraphrase": 2},
    "Story":       {"context_content_data": 2, "structure_format_brevity": 2,
                    "language_mechanics": 1, "originality_comparisons_paraphrase": 2},
    "Letter_Email":{"context_content_data": 1, "structure_format_brevity": 2,
                    "language_mechanics": 1, "originality_comparisons_paraphrase": 1},
}

# Standard cap = 50% of MAX_MARK rounded down.
STANDARD_CAP = {10: 5, 8: 4, 7: 3, 5: 2}

# Cap value by structural_audit.applied_cap_reason.
CAP_VALUES: dict[str, int | None] = {
    "None": None,
    "Paragraph_Subdivisions": 5,
    "Letter_Missing_Layout": 2,
    "Summary_Verbatim_Length": 5,
    "Theme_Verbatim_Copy": 4,
    "Graph_External_Facts": 6,
}

# Performance bands per MAX_MARK: (band_name, low, high) inclusive.
BANDS: dict[int, list[tuple[str, int, int]]] = {
    10: [("Band 4", 8, 10), ("Band 3", 6, 7), ("Band 2", 4, 5), ("Band 1", 1, 3), ("Band 0", 0, 0)],
    8:  [("Band 4", 7, 8), ("Band 3", 5, 6), ("Band 2", 3, 4), ("Band 1", 1, 2), ("Band 0", 0, 0)],
    7:  [("Band 4", 6, 7), ("Band 3", 4, 5), ("Band 2", 3, 3), ("Band 1", 1, 2), ("Band 0", 0, 0)],
    5:  [("Band 4", 4, 5), ("Band 3", 3, 3), ("Band 2", 2, 2), ("Band 1", 1, 1), ("Band 0", 0, 0)],
}


def band_for(total: int, max_mark: int) -> str | None:
    """Return the band name containing `total` for this MAX_MARK, or None."""
    for name, lo, hi in BANDS.get(max_mark, []):
        if lo <= total <= hi:
            return name
    return None


# ---------------------------------------------------------------------------
# Output schema (rubric base + spec extensions)
# ---------------------------------------------------------------------------
class AttemptStatus(BaseModel):
    is_attempted: bool
    is_blank_or_unintelligible: bool


class StructuralAudit(BaseModel):
    cap_applied: bool
    applied_cap_reason: str


# Scores are float: some graders (e.g. gemma) award half-marks. Capturing the
# real value beats coercing to int and losing genuine model behaviour.
class _Criteria(BaseModel):
    context_content_data: float
    structure_format_brevity: float
    language_mechanics: float
    originality_comparisons_paraphrase: float


class RawSubscores(_Criteria):
    raw_total: float


class ScoreBreakdown(_Criteria):
    total_score: float
    capped_from: float | None = None


class CriterionReason(BaseModel):
    rationale: str
    evidence: str = ""


class CriterionReasoning(BaseModel):
    context_content_data: CriterionReason
    structure_format_brevity: CriterionReason
    language_mechanics: CriterionReason
    originality_comparisons_paraphrase: CriterionReason


class ErrorAnalysis(BaseModel):
    frequent_errors: list[str] = []
    positive_aspects: list[str] = []


class Evaluation(BaseModel):
    task_type: str
    max_mark_applied: int
    attempt_status: AttemptStatus
    structural_audit: StructuralAudit
    raw_subscores: RawSubscores
    criterion_reasoning: CriterionReasoning
    score_breakdown: ScoreBreakdown
    performance_band: str
    error_analysis: ErrorAnalysis
    feedback_summary: str


def check_evidence(ev: Evaluation, extracted_text: str) -> list[str]:
    """Return criteria whose non-empty evidence span is NOT verbatim in the text.

    A fabricated span is a hallucination signal worth counting; empty evidence is
    not checked.
    """
    missing: list[str] = []
    for crit in FOUR_CRITERIA:
        span = getattr(ev.criterion_reasoning, crit).evidence
        if span and span not in extracted_text:
            missing.append(crit)
    return missing


def synthesise_blank(task_type: str, max_mark: int) -> Evaluation:
    """Directly build the record for a verified-blank (not_attempted) answer.

    No API call: all sub-scores 0, total 0, Band 0, blank/unintelligible true.
    """
    def blank_reason() -> CriterionReason:  # a fresh instance per criterion
        return CriterionReason(rationale="Not attempted; verified blank.", evidence="")

    return Evaluation(
        task_type=task_type,
        max_mark_applied=max_mark,
        attempt_status=AttemptStatus(is_attempted=False, is_blank_or_unintelligible=True),
        structural_audit=StructuralAudit(cap_applied=False, applied_cap_reason="None"),
        raw_subscores=RawSubscores(
            context_content_data=0, structure_format_brevity=0,
            language_mechanics=0, originality_comparisons_paraphrase=0, raw_total=0,
        ),
        criterion_reasoning=CriterionReasoning(
            context_content_data=blank_reason(), structure_format_brevity=blank_reason(),
            language_mechanics=blank_reason(), originality_comparisons_paraphrase=blank_reason(),
        ),
        score_breakdown=ScoreBreakdown(
            context_content_data=0, structure_format_brevity=0,
            language_mechanics=0, originality_comparisons_paraphrase=0,
            total_score=0, capped_from=0,
        ),
        performance_band="Band 0",
        error_analysis=ErrorAnalysis(frequent_errors=[], positive_aspects=[]),
        feedback_summary="Not attempted; synthesised Band 0 record (no API call).",
    )
