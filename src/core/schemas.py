from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage 1: Verbatim Transcription Schemas
# ---------------------------------------------------------------------------
class Stage1TranscriptionResult(BaseModel):
    """Output of Stage 1: Strict verbatim transcription from script image to text."""
    raw_transcript: str = Field(..., description="Exact verbatim transcript preserving all errors, layout, and tags.")
    illegible_count: int = Field(0, description="Count of [illegible] occurrences.")
    unclear_count: int = Field(0, description="Count of [unclear: ...] annotations.")
    character_count: int = Field(0, description="Total characters transcribed.")
    word_count: int = Field(0, description="Total words transcribed.")
    detected_script: str = Field("unknown", description="Bangla, English, or Mixed script detected.")


# ---------------------------------------------------------------------------
# Stage 2: Autocorrection Verification Schemas
# ---------------------------------------------------------------------------
class AutocorrectionDiffItem(BaseModel):
    """An individual instance where the model silently corrected an error, now reverted."""
    stage1_output: str = Field(..., description="What Stage 1 output wrote.")
    actual_handwritten: str = Field(..., description="What was actually written in handwriting on the image.")
    reason: str = Field(..., description="Why this was flagged as a silent correction or misread.")
    context_snippet: str = Field("", description="Surrounding sentence or line snippet.")


class Stage2VerificationResult(BaseModel):
    """Output of Stage 2: Reconciled transcript with silent autocorrection audit."""
    verified_transcript: str = Field(..., description="Canonical verified transcript.")
    silent_corrections_fixed: List[AutocorrectionDiffItem] = Field(default_factory=list)
    total_corrections_count: int = Field(0, description="Number of silent corrections reverted.")
    verification_notes: str = Field("", description="Observations during visual cross-check.")


# ---------------------------------------------------------------------------
# Stage 3: Error Extraction Schemas
# ---------------------------------------------------------------------------
class LinguisticErrorItem(BaseModel):
    """A linguistic or structural error extracted from the verified transcript."""
    error_type: str = Field(..., description="spelling, grammar, syntax, punctuation, or word_choice")
    erroneous_text: str = Field(..., description="Exact word or phrase containing the error.")
    suggested_correction: str = Field(..., description="Standard or grammatically correct form.")
    context_sentence: str = Field(..., description="Full sentence where error occurred.")
    explanation: str = Field(..., description="Linguistic explanation of the rule violated.")


class Stage3ErrorResult(BaseModel):
    """Output of Stage 3: Complete catalog of linguistic errors from verified text."""
    errors: List[LinguisticErrorItem] = Field(default_factory=list)
    spelling_error_count: int = Field(0)
    grammar_error_count: int = Field(0)
    syntax_error_count: int = Field(0)
    punctuation_error_count: int = Field(0)
    total_error_count: int = Field(0)
    linguistic_summary: str = Field("", description="Summary analysis of student's language proficiency.")


# ---------------------------------------------------------------------------
# Stage 4: Rubric Evaluation Schemas
# ---------------------------------------------------------------------------
class CriterionScore(BaseModel):
    """Grading score and breakdown for an individual rubric criterion."""
    criterion_id: str
    criterion_name: str
    max_marks: float
    awarded_marks: float
    justification: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class Stage4EvaluationResult(BaseModel):
    """Output of Stage 4: Rubric-based marks, deductions, and pedagogical feedback."""
    subject: str = Field(...)
    question_type: str = Field(...)
    criteria_scores: List[CriterionScore] = Field(default_factory=list)
    content_raw_score: float = Field(..., description="Sum of criterion marks before penalties.")
    linguistic_penalty: float = Field(0.0, description="Deductions based on Stage 3 errors.")
    final_score: float = Field(..., description="Final marks awarded (content_raw_score - penalty).")
    total_max_marks: float = Field(10.0, description="Maximum total marks.")
    percentage: float = Field(0.0, description="Final score percentage.")
    overall_feedback: str = Field(..., description="Constructive teacher-level feedback.")
    actionable_recommendations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# End-to-End Complete Evaluation Report
# ---------------------------------------------------------------------------
class CompleteEvaluationReport(BaseModel):
    """Comprehensive artifact containing outputs from all 4 pipeline stages."""
    script_id: str
    image_path: str
    model_id: str = "google/gemma-4-31b-it"
    timestamp: str
    stage1_transcription: Stage1TranscriptionResult
    stage2_verification: Stage2VerificationResult
    stage3_errors: Stage3ErrorResult
    stage4_evaluation: Stage4EvaluationResult
    metadata: Dict[str, Any] = Field(default_factory=dict)
