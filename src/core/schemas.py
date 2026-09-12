from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage 0: Red-Ink Detection & Stage 0b: Teacher Mark Schemas
# ---------------------------------------------------------------------------
class TeacherMarkItem(BaseModel):
    """An individual red-ink numeric mark extracted from a script page."""
    question_no: Optional[str] = Field(None, description="The question number this mark belongs to, if identifiable, otherwise null.")
    mark_value: str = Field(..., description="Exact value as written, e.g. '7/10', '4', 'VII'.")
    location: str = Field("", description="Brief description, e.g. 'margin next to answer 3'.")
    y_position: Optional[str] = Field(None, description="Vertical position on page: 'top', 'mid', 'bottom'.")


# ---------------------------------------------------------------------------
# Stage 1: Verbatim Transcription Schemas
# ---------------------------------------------------------------------------
class Stage1TranscriptionResult(BaseModel):
    """Output of Stage 1: Strict verbatim transcription from script image to text."""
    raw_transcript: str = Field(..., description="Exact verbatim transcript preserving all errors, layout, and tags.")
    illegible_count: int = Field(0, description="Count of [illegible] occurrences.")
    unclear_count: int = Field(0, description="Count of [unclear: ...] annotations.")
    struck_count: int = Field(0, description="Count of [struck: ...] annotations.")
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
    error_type: str = Field(..., description="spelling, grammar, or syntax")
    erroneous_text: str = Field(..., description="Exact word or phrase containing the error.")
    suggested_correction: str = Field(..., description="Standard or grammatically correct form.")
    context_sentence: str = Field(..., description="Full sentence where error occurred.")
    explanation: str = Field(..., description="Linguistic explanation of the rule violated.")
    question_no: Optional[str] = Field(None, description="Exam question number where error occurred.")


class Stage3ErrorResult(BaseModel):
    """Output of Stage 3: Complete catalog of linguistic errors from verified text."""
    errors: List[LinguisticErrorItem] = Field(default_factory=list)
    spelling_error_count: int = Field(0)
    grammar_error_count: int = Field(0)
    syntax_error_count: int = Field(0)
    punctuation_error_count: Optional[int] = Field(0)
    total_error_count: int = Field(0)
    linguistic_summary: str = Field("", description="Summary analysis of student's language proficiency.")


# ---------------------------------------------------------------------------
# Stage 3b: Evidence-Fused Handwriting Ambiguity Arbitration Schemas
# ---------------------------------------------------------------------------
class ArbitrationCandidate(BaseModel):
    """A single (read token, intended token) pair extracted from a Stage 3 error for arbitration."""
    candidate_id: str = Field(..., description="Unique id: '<q_no>:<error_index>:<pair_index>'")
    error_index: int = Field(..., description="Index into the question's Stage 3 error list")
    error_type: str = Field("spelling")
    erroneous_text: str = Field("", description="Full erroneous span from Stage 3")
    suggested_correction: str = Field("", description="Full suggested correction from Stage 3")
    candidate_token: str = Field(..., description="The differing token as transcribed, e.g. 'do'")
    intended_token: str = Field(..., description="The differing token as intended, e.g. 'to'")
    context_sentence: str = Field("")
    question_no: Optional[str] = None
    page_no: Optional[int] = None


class ArbitrationEvidence(BaseModel):
    """All raw signals gathered for one candidate. Every signal is in [0,1] where 1 = ambiguity (perceptual)."""
    phonetic_plausibility: Optional[float] = Field(None, description="Similarity of phonetic codes (1 = sounds identical => cognitive error likely)")
    phonetic_signal: Optional[float] = Field(None, description="1 - phonetic_plausibility")
    edit_ops: List[str] = Field(default_factory=list, description="Character edit path, e.g. ['sub:f>d']")
    writer_prior: Optional[float] = None
    writer_pair_count: int = 0
    consensus_signal: Optional[float] = None
    consensus_agreement_candidate: Optional[float] = None
    consensus_agreement_intended: Optional[float] = None
    consensus_word_confidence: Optional[float] = None
    consensus_samples: List[str] = Field(default_factory=list)
    consensus_token: Optional[str] = None
    consensus_token_agreement: Optional[float] = None
    forced_choice_signal: Optional[float] = None
    forced_choice: Optional[str] = Field(None, description="candidate | intended | neither | invalid")
    forced_choice_confidence: Optional[float] = None
    forced_choice_reason: str = ""
    forced_choice_order: str = ""
    localization_method: str = Field("none", description="bbox | projection | none")
    localization_match_ratio: float = 0.0
    crop_path: Optional[str] = None
    model_calls: int = 0
    notes: List[str] = Field(default_factory=list)


class ArbitrationRecord(BaseModel):
    candidate: ArbitrationCandidate
    evidence: ArbitrationEvidence
    ambiguity_score: float = 0.5
    verdict: str = Field("UNCERTAIN", description="GENUINE_ERROR | HANDWRITING_AMBIGUITY | UNCERTAIN")
    mode: str = "evidence"


class Stage3bArbitrationResult(BaseModel):
    mode: str = "evidence"
    records: List[ArbitrationRecord] = Field(default_factory=list)
    cleared: List[Dict[str, Any]] = Field(default_factory=list, description="Legacy-compatible cleared list")
    uncertain: List[ArbitrationRecord] = Field(default_factory=list)
    total_candidates: int = 0
    total_model_calls: int = 0
    token_usage: Dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-Page Extraction Container
# ---------------------------------------------------------------------------
class PageExtractionResult(BaseModel):
    """Extraction results for a single page of an exam script."""
    page_no: int = Field(1, description="1-indexed page number in the exam script.")
    image_path: str = Field("")
    has_red_ink: bool = Field(False, description="Stage 0: Whether red ink was detected on this page.")
    red_pixel_count: int = Field(0)
    stage1_transcription: Stage1TranscriptionResult
    stage2_verification: Stage2VerificationResult
    stage3_errors: Stage3ErrorResult
    teacher_marks: List[TeacherMarkItem] = Field(default_factory=list, description="Stage 0b: Extracted red-ink teacher marks.")
    token_usage: Dict[str, Any] = Field(default_factory=dict, description="Context tokens consumed by this page (in, out, total).")


# ---------------------------------------------------------------------------
# Script-Level Extraction Stage Output (Stages 0, 0b, 1 - 3)
# ---------------------------------------------------------------------------
class ExtractionResult(BaseModel):
    """Output of Stages 0 to 3: Verified transcript, errors, and teacher marks across all pages."""
    script_id: str
    image_path: str
    model_id: str = "google/gemma-4-31b-it"
    timestamp: str
    has_red_ink: bool = Field(False, description="Whether any page in the script contained red ink.")
    stage1_transcription: Stage1TranscriptionResult
    stage2_verification: Stage2VerificationResult
    stage3_errors: Stage3ErrorResult
    teacher_marks: List[TeacherMarkItem] = Field(default_factory=list)
    pages: List[PageExtractionResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Raw-Tier Research Dataset Record Schema
# ---------------------------------------------------------------------------
class RawTierRecord(BaseModel):
    """Single row representing a script/page/question in the raw-tier research CSV."""
    script_id: str
    page_no: int = 1
    question_no: Optional[str] = None
    paper: str = "bangla"  # bangla / english
    task_type: str = "creative_question"
    transcript_text: str
    ocr_flags: str = "none"  # e.g. illegible: 0, unclear: 1
    error_list: str = "[]"  # JSON string of errors
    teacher_mark: str = ""  # isolated teacher mark
    has_red_ink: bool = False
    original_marker_id: str = "unknown"
    school_id: str = "default"
    region: str = "default"


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


# ---------------------------------------------------------------------------
# Question Paper Schemas
# ---------------------------------------------------------------------------
class ExtractedQuestion(BaseModel):
    """Structured question paper artifact extracted from question PDF/image."""
    question_id: str = Field(..., description="Unique question identifier, e.g. 'SE_11_Q1' or 'SB_11_Q1'")
    language: str = Field("english", description="Language/subject: 'english' or 'bangla'")
    title: Optional[str] = Field(None, description="Title or header of the question paper")
    question_text: str = Field(..., description="Full text and prompt instructions of the question")
    total_marks: Optional[float] = Field(None, description="Total marks allocated to this question")
    sub_questions: List[Dict[str, Any]] = Field(default_factory=list, description="Sub-questions or criteria breakdown")
    source_file: Optional[str] = Field(None, description="Path to original question PDF or image")
    extracted_at: Optional[str] = Field(None, description="ISO timestamp of extraction")


class AlignedAnswerItem(BaseModel):
    """An individual answer segment matched to a specific question on the paper."""
    q_no: str = Field(..., description="Canonical question identifier (e.g. '1(A)', '2', '7')")
    q_name: Optional[str] = Field(None, description="Name or topic of question (e.g. 'Paragraph on Artificial Intelligence')")
    answer_text: str = Field(..., description="Student handwritten answer text for this question")
    page_numbers: List[int] = Field(default_factory=list, description="Script page numbers where this answer appears")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Linguistic errors located within this answer")
    word_count: int = 0
    character_count: int = 0


class QuestionEvaluationItem(BaseModel):
    """Grading breakdown and feedback for an individual question answer."""
    q_no: str = Field(..., description="Question number, e.g. '1(A)', '2', '7'")
    q_name: Optional[str] = Field(None, description="Question topic / title")
    page_numbers: List[int] = Field(default_factory=list, description="Page numbers where this answer is located")
    max_marks: float = Field(..., description="Maximum marks for this question")
    awarded_marks: float = Field(..., description="Predicted marks awarded by AI")
    human_ground_truth: Optional[float] = Field(None, description="Verified human examiner mark (from gt.txt)")
    delta: Optional[float] = Field(None, description="Absolute difference |awarded - human|")
    criteria_scores: List[CriterionScore] = Field(default_factory=list)
    content_raw_score: float = 0.0
    linguistic_penalty: float = 0.0
    examiner_feedback: str = ""
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    # Rubric-driven scoring (mode-based rubrics)
    task_mode: Optional[str] = Field(None, description="A_ITEM | B_POINT | C_BAND | generic")
    task_type: Optional[str] = None
    scoring_status: str = Field("scored", description="scored | unscored (model output unusable) | missing (no answer segment found)")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Per-item results (Mode A/B)")
    subscores: Dict[str, float] = Field(default_factory=dict, description="Criterion subscores (Mode C)")
    raw_total: Optional[float] = None
    cap_applied: bool = False
    cap_reason: str = "None"
    capped_from: Optional[float] = None
    performance_band: Optional[str] = None
    key_source: Optional[str] = Field(None, description="answer_key | none")
    scoring_notes: List[str] = Field(default_factory=list)

    @property
    def human_gt(self) -> Optional[float]:
        return self.human_ground_truth

    @property
    def feedback(self) -> str:
        return self.examiner_feedback

    @property
    def linguistic_deductions(self) -> float:
        return self.linguistic_penalty


class Stage4EvaluationResult(BaseModel):
    """Output of Stage 4: Rubric-based marks, deductions, and pedagogical feedback."""
    subject: str = Field(...)
    question_type: str = Field(...)
    question_id: Optional[str] = Field(None, description="Matched question identifier, e.g. 'SE_11_Q1' or 'SB_11_Q1'")
    question_text: Optional[str] = Field(None, description="Prompt text of the matched question")
    eval_mode: str = Field("modular", description="'modular' (per-question) or 'monolithic' (single-pass)")
    criteria_scores: List[CriterionScore] = Field(default_factory=list)
    question_evaluations: List[QuestionEvaluationItem] = Field(default_factory=list, description="Per-question evaluation breakdown")
    content_raw_score: float = Field(..., description="Sum of criterion marks before penalties.")
    linguistic_penalty: float = Field(0.0, description="Deductions based on Stage 3 errors.")
    final_score: float = Field(..., description="Final marks awarded (content_raw_score - penalty).")
    total_max_marks: float = Field(10.0, description="Maximum total marks.")
    percentage: float = Field(0.0, description="Final score percentage.")
    mae_vs_human: Optional[float] = Field(None, description="Mean Absolute Error over SCORED questions with ground truth")
    mae_including_missing: Optional[float] = Field(None, description="MAE counting missing/unscored questions as 0 marks")
    scored_with_gt: int = Field(0, description="Number of scored questions that had a ground-truth mark")
    missing_questions: List[str] = Field(default_factory=list, description="Rubric/GT questions with no answer segment")
    unscored_questions: List[str] = Field(default_factory=list, description="Questions whose model output could not be scored")
    rubric_driven: bool = Field(False, description="True when marks were computed from the mode-based rubric")
    overall_feedback: str = Field(..., description="Constructive teacher-level feedback.")
    actionable_recommendations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# End-to-End Complete Evaluation Report
# ---------------------------------------------------------------------------
class CompleteEvaluationReport(BaseModel):
    """Comprehensive artifact containing outputs from all pipeline stages."""
    script_id: str
    image_path: str
    model_id: str = "google/gemma-4-31b-it"
    timestamp: str
    has_red_ink: bool = False
    question_id: Optional[str] = Field(None, description="Matched question identifier, e.g. 'SE_11_Q1' or 'SB_11_Q1'")
    question_text: Optional[str] = Field(None, description="Prompt text of the matched question")
    stage1_transcription: Stage1TranscriptionResult
    stage2_verification: Stage2VerificationResult
    stage3_errors: Stage3ErrorResult
    teacher_marks: List[TeacherMarkItem] = Field(default_factory=list)
    stage4_evaluation: Stage4EvaluationResult
    metadata: Dict[str, Any] = Field(default_factory=dict)
