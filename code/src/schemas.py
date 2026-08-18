"""
Data schemas and Pydantic models for the Modular Answer-Script OCR & Grading Pipeline.
Defines strict, typed input/output contracts for all 6 stages, configuration dataclasses,
task types for NCTB HSC English 1st Paper (Rubric v2), and multi-page question segmentation.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class Stage2Decision(str, Enum):
    KEEP = "KEEP"
    REPAIR = "REPAIR"
    ADAPT = "ADAPT"


class TaskType(str, Enum):
    PARAGRAPH = "Paragraph"
    STORY = "Story"
    LETTER_EMAIL = "Letter_Email"
    GRAPH_CHART = "Graph_Chart"
    SUMMARY = "Summary"
    THEME = "Theme"


class PerformanceBand(str, Enum):
    BAND_4 = "Band 4"
    BAND_3 = "Band 3"
    BAND_2 = "Band 2"
    BAND_1 = "Band 1"
    BAND_0 = "Band 0"


class CapReason(str, Enum):
    NONE = "None"
    PARAGRAPH_SUBDIVISIONS = "Paragraph_Subdivisions"
    LETTER_MISSING_LAYOUT = "Letter_Missing_Layout"
    SUMMARY_VERBATIM_LENGTH = "Summary_Verbatim_Length"
    THEME_VERBATIM_COPY = "Theme_Verbatim_Copy"
    GRAPH_EXTERNAL_FACTS = "Graph_External_Facts"


# Standard Task Mark Allocations based on rubric_v2.txt
TASK_MAX_MARKS: Dict[str, float] = {
    "Summary": 10.0,
    "Paragraph": 10.0,
    "Graph_Chart": 10.0,
    "Story": 7.0,
    "Letter_Email": 5.0,
    "Theme": 8.0
}

# Standard Task Question Numbers based on NCTB HSC English 1st Paper
TASK_QUESTION_NUMBERS: Dict[str, str] = {
    "Summary": "3",
    "Paragraph": "7",
    "Graph_Chart": "8",
    "Story": "9",
    "Letter_Email": "10",
    "Theme": "11"
}


@dataclass
class UncertaintyArea:
    """Represents a low-confidence or ambiguous OCR region in Stage 1."""
    location: str
    text_guess: str
    reason: str
    bbox: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UncertaintyArea:
        return cls(**data)


@dataclass
class ResolvedViaReference:
    """Represents a handwriting stroke disambiguated using reference solution in Stage 3."""
    location: str
    resolved_text: str
    reference_cue: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResolvedViaReference:
        return cls(**data)


@dataclass
class ScoreBreakdown:
    """
    Standard 4-criterion sub-scores from rubric_v2.txt.
    sub-scores must sum exactly to total_score <= max_mark.
    """
    context_content_data: float = 0.0
    structure_format_brevity: float = 0.0
    language_mechanics: float = 0.0
    originality_comparisons_paraphrase: float = 0.0
    total_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScoreBreakdown:
        return cls(
            context_content_data=float(data.get("context_content_data", 0.0)),
            structure_format_brevity=float(data.get("structure_format_brevity", 0.0)),
            language_mechanics=float(data.get("language_mechanics", 0.0)),
            originality_comparisons_paraphrase=float(data.get("originality_comparisons_paraphrase", 0.0)),
            total_score=float(data.get("total_score", 0.0))
        )


@dataclass
class StructuralAudit:
    """Structural constraint and hard-cap audit from rubric_v2.txt."""
    cap_applied: bool = False
    applied_cap_reason: str = "None"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StructuralAudit:
        return cls(
            cap_applied=bool(data.get("cap_applied", False)),
            applied_cap_reason=str(data.get("applied_cap_reason", "None"))
        )


@dataclass
class ErrorAnalysis:
    """Frequent errors and positive aspects identified during evaluation."""
    frequent_errors: List[str] = field(default_factory=list)
    positive_aspects: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ErrorAnalysis:
        fe = data.get("frequent_errors", [])
        if isinstance(fe, str):
            fe = [s.strip() for s in fe.split("|") if s.strip()]
        pa = data.get("positive_aspects", [])
        if isinstance(pa, str):
            pa = [s.strip() for s in pa.split("|") if s.strip()]
        return cls(
            frequent_errors=fe,
            positive_aspects=pa
        )


@dataclass
class AttemptStatus:
    """Records whether the script was attempted or left blank / unintelligible."""
    is_attempted: bool = True
    is_blank_or_unintelligible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AttemptStatus:
        return cls(
            is_attempted=bool(data.get("is_attempted", True)),
            is_blank_or_unintelligible=bool(data.get("is_blank_or_unintelligible", False))
        )


# ----------------------------------------------------------------------
# Stage Outputs
# ----------------------------------------------------------------------

@dataclass
class Stage1Output:
    """Output contract for Stage 1: Extractor A (Cold OCR Read)."""
    QUESTION_TEXT: str
    STUDENT_ANSWER: str
    UNCERTAINTY_AREAS: List[UncertaintyArea] = field(default_factory=list)
    extracted_text_raw: Optional[str] = None
    struck_tokens: List[str] = field(default_factory=list)
    word_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["UNCERTAINTY_AREAS"] = [u.to_dict() if isinstance(u, UncertaintyArea) else u for u in self.UNCERTAINTY_AREAS]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stage1Output:
        uncertainties = [
            UncertaintyArea.from_dict(u) if isinstance(u, dict) else u
            for u in data.get("UNCERTAINTY_AREAS", [])
        ]
        return cls(
            QUESTION_TEXT=data.get("QUESTION_TEXT", ""),
            STUDENT_ANSWER=data.get("STUDENT_ANSWER", ""),
            UNCERTAINTY_AREAS=uncertainties,
            extracted_text_raw=data.get("extracted_text_raw"),
            struck_tokens=data.get("struck_tokens", []),
            word_count=data.get("word_count", 0)
        )


@dataclass
class Stage2Output:
    """Output contract for Stage 2: RubricAligner."""
    operative_rubric: str
    examiner_note: Optional[str] = None
    shadow_solution: Optional[str] = None
    decision: Stage2Decision = Stage2Decision.KEEP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operative_rubric": self.operative_rubric,
            "examiner_note": self.examiner_note,
            "shadow_solution": self.shadow_solution,
            "decision": self.decision.value if isinstance(self.decision, Stage2Decision) else self.decision
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stage2Output:
        dec_val = data.get("decision", Stage2Decision.KEEP.value)
        try:
            decision = Stage2Decision(dec_val)
        except ValueError:
            decision = Stage2Decision.KEEP
        return cls(
            operative_rubric=data.get("operative_rubric", ""),
            examiner_note=data.get("examiner_note"),
            shadow_solution=data.get("shadow_solution"),
            decision=decision
        )


@dataclass
class Stage3Output:
    """Output contract for Stage 3: Extractor B (Reference-primed OCR)."""
    QUESTION_TEXT: str
    STUDENT_ANSWER: str
    RESOLVED_VIA_REFERENCE: List[ResolvedViaReference] = field(default_factory=list)
    STILL_UNCERTAIN: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["RESOLVED_VIA_REFERENCE"] = [
            r.to_dict() if isinstance(r, ResolvedViaReference) else r
            for r in self.RESOLVED_VIA_REFERENCE
        ]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stage3Output:
        resolved = [
            ResolvedViaReference.from_dict(r) if isinstance(r, dict) else r
            for r in data.get("RESOLVED_VIA_REFERENCE", [])
        ]
        return cls(
            QUESTION_TEXT=data.get("QUESTION_TEXT", ""),
            STUDENT_ANSWER=data.get("STUDENT_ANSWER", ""),
            RESOLVED_VIA_REFERENCE=resolved,
            STILL_UNCERTAIN=data.get("STILL_UNCERTAIN", [])
        )


@dataclass
class Stage4Output:
    """Output contract for Stage 4: OCR Supervisor."""
    QUESTION_TEXT: str
    STUDENT_ANSWER: str
    adjudication_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stage4Output:
        return cls(
            QUESTION_TEXT=data.get("QUESTION_TEXT", ""),
            STUDENT_ANSWER=data.get("STUDENT_ANSWER", ""),
            adjudication_notes=data.get("adjudication_notes")
        )


@dataclass
class Stage5Output:
    """Output contract for Stage 5: Examiner (Rubric v2 Chief Examiner Evaluation)."""
    task_type: str = "Paragraph"
    max_mark_applied: float = 10.0
    attempt_status: AttemptStatus = field(default_factory=AttemptStatus)
    structural_audit: StructuralAudit = field(default_factory=StructuralAudit)
    score_breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    performance_band: str = "Band 3"
    error_analysis: ErrorAnalysis = field(default_factory=ErrorAnalysis)
    feedback_summary: str = ""
    raw_cot: Optional[str] = None
    stated_total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "max_mark_applied": self.max_mark_applied,
            "attempt_status": self.attempt_status.to_dict(),
            "structural_audit": self.structural_audit.to_dict(),
            "score_breakdown": self.score_breakdown.to_dict(),
            "performance_band": self.performance_band,
            "error_analysis": self.error_analysis.to_dict(),
            "feedback_summary": self.feedback_summary,
            "raw_cot": self.raw_cot,
            "stated_total": self.stated_total
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stage5Output:
        att = AttemptStatus.from_dict(data.get("attempt_status", {}))
        audit = StructuralAudit.from_dict(data.get("structural_audit", {}))
        breakdown = ScoreBreakdown.from_dict(data.get("score_breakdown", {}))
        errs = ErrorAnalysis.from_dict(data.get("error_analysis", {}))
        total = float(data.get("stated_total", breakdown.total_score))
        return cls(
            task_type=data.get("task_type", "Paragraph"),
            max_mark_applied=float(data.get("max_mark_applied", 10.0)),
            attempt_status=att,
            structural_audit=audit,
            score_breakdown=breakdown,
            performance_band=data.get("performance_band", "Band 3"),
            error_analysis=errs,
            feedback_summary=data.get("feedback_summary", ""),
            raw_cot=data.get("raw_cot"),
            stated_total=total
        )


@dataclass
class Stage6Output:
    """Output contract for Stage 6: Compressor (Audit & Sanity-Check)."""
    task_type: str = "Paragraph"
    max_mark_applied: float = 10.0
    final_marks: float = 0.0
    performance_band: str = "Band 3"
    score_breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    structural_audit: StructuralAudit = field(default_factory=StructuralAudit)
    attempt_status: AttemptStatus = field(default_factory=AttemptStatus)
    error_analysis: ErrorAnalysis = field(default_factory=ErrorAnalysis)
    feedback_summary: str = ""
    sum_check_passed: bool = True
    band_check_passed: bool = True
    error_detection: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "max_mark_applied": self.max_mark_applied,
            "final_marks": self.final_marks,
            "performance_band": self.performance_band,
            "score_breakdown": self.score_breakdown.to_dict(),
            "structural_audit": self.structural_audit.to_dict(),
            "attempt_status": self.attempt_status.to_dict(),
            "error_analysis": self.error_analysis.to_dict(),
            "feedback_summary": self.feedback_summary,
            "sum_check_passed": self.sum_check_passed,
            "band_check_passed": self.band_check_passed,
            "error_detection": self.error_detection
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stage6Output:
        breakdown = ScoreBreakdown.from_dict(data.get("score_breakdown", {}))
        audit = StructuralAudit.from_dict(data.get("structural_audit", {}))
        att = AttemptStatus.from_dict(data.get("attempt_status", {}))
        errs = ErrorAnalysis.from_dict(data.get("error_analysis", {}))
        final_m = float(data.get("final_marks", breakdown.total_score))
        return cls(
            task_type=data.get("task_type", "Paragraph"),
            max_mark_applied=float(data.get("max_mark_applied", 10.0)),
            final_marks=final_m,
            performance_band=data.get("performance_band", "Band 3"),
            score_breakdown=breakdown,
            structural_audit=audit,
            attempt_status=att,
            error_analysis=errs,
            feedback_summary=data.get("feedback_summary", ""),
            sum_check_passed=bool(data.get("sum_check_passed", True)),
            band_check_passed=bool(data.get("band_check_passed", True)),
            error_detection=data.get("error_detection")
        )


# ----------------------------------------------------------------------
# Multi-Page Question Segmentation & Manifest Schemas
# ----------------------------------------------------------------------

@dataclass
class QuestionSegment:
    """Represents a recognized question and its associated pages in a multi-page PDF."""
    task_id: str
    script_id: str
    question_no: str
    question_type: str
    max_mark: float
    page_indices: List[int] = field(default_factory=list)  # 0-indexed page numbers
    page_range_str: str = ""                               # e.g. "3-4" or "15"
    teacher_mark: Optional[float] = None
    question_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuestionSegment:
        return cls(**data)


@dataclass
class ScriptManifest:
    """Full question manifest for a single multi-page student script PDF."""
    script_id: str
    file_path: Optional[str] = None
    total_pages: int = 0
    questions: Dict[str, QuestionSegment] = field(default_factory=dict)  # keyed by task_id or question_no

    def to_dict(self) -> Dict[str, Any]:
        return {
            "script_id": self.script_id,
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "questions": {k: q.to_dict() for k, q in self.questions.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScriptManifest:
        qs = {
            k: QuestionSegment.from_dict(v) if isinstance(v, dict) else v
            for k, v in data.get("questions", {}).items()
        }
        return cls(
            script_id=data.get("script_id", ""),
            file_path=data.get("file_path"),
            total_pages=data.get("total_pages", 0),
            questions=qs
        )


# ----------------------------------------------------------------------
# Pipeline Config & Pipeline Result
# ----------------------------------------------------------------------

@dataclass
class EnabledStagesConfig:
    extractor_a: bool = True
    rubric_aligner: bool = True
    extractor_b: bool = True
    ocr_supervisor: bool = True
    examiner: bool = True
    compressor: bool = True

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnabledStagesConfig:
        return cls(
            extractor_a=data.get("extractor_a", True),
            rubric_aligner=data.get("rubric_aligner", True),
            extractor_b=data.get("extractor_b", True),
            ocr_supervisor=data.get("ocr_supervisor", True),
            examiner=data.get("examiner", True),
            compressor=data.get("compressor", True),
        )


@dataclass
class ModelConfig:
    backend: str = "vllm"  # vllm | transformers | mock
    checkpoint: str = "google/gemma-3-27b-it"
    quantization: str = "w4a16"
    kv_cache: str = "fp8"
    temperature: float = 0.0
    max_tokens: int = 4096

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        return cls(**data)


@dataclass
class IngestionConfig:
    pdf_dpi: int = 300
    max_image_side: int = 2048
    page_router: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IngestionConfig:
        return cls(**data)


@dataclass
class LoggingConfig:
    output_dir: str = "outputs"
    save_per_stage_json: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LoggingConfig:
        return cls(**data)


@dataclass
class PipelineConfig:
    enabled_stages: EnabledStagesConfig = field(default_factory=EnabledStagesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    supported_languages: List[str] = field(default_factory=lambda: ["bn", "en"])
    rubric_path: Optional[str] = "rubric_v2.txt"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline": {"enabled_stages": self.enabled_stages.to_dict()},
            "model": self.model.to_dict(),
            "ingestion": self.ingestion.to_dict(),
            "language": {"supported": self.supported_languages},
            "logging": self.logging.to_dict(),
            "rubric_path": self.rubric_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        pipe_data = data.get("pipeline", {})
        enabled = EnabledStagesConfig.from_dict(pipe_data.get("enabled_stages", {}))
        model = ModelConfig.from_dict(data.get("model", {}))
        ingest = IngestionConfig.from_dict(data.get("ingestion", {}))
        logs = LoggingConfig.from_dict(data.get("logging", {}))
        langs = data.get("language", {}).get("supported", ["bn", "en"])
        rubric_p = data.get("rubric_path", "rubric_v2.txt")
        return cls(
            enabled_stages=enabled,
            model=model,
            ingestion=ingest,
            logging=logs,
            supported_languages=langs,
            rubric_path=rubric_p
        )


@dataclass
class PipelineResult:
    """
    Consolidated end-to-end output for a single graded question.
    Conforms to evaluation.csv schema and provides full audit traceability.
    """
    evaluation_id: str = ""
    task_id: str = ""
    script_id: str = "SE_11_Q1_0001"
    question_no: str = "8"
    question_type: str = "Graph_Chart"
    max_mark: float = 10.0
    teacher_mark: Optional[float] = None
    total_score: float = 0.0
    performance_band: str = "Band 3"
    
    # 4 Universal Sub-scores
    context_content_data: float = 0.0
    structure_format_brevity: float = 0.0
    language_mechanics: float = 0.0
    originality_comparisons_paraphrase: float = 0.0
    
    # Status & Caps
    is_attempted: bool = True
    cap_applied: bool = False
    cap_reason: str = "None"
    
    # Qualitative Diagnostics
    frequent_errors: str = ""
    positive_aspects: str = ""
    feedback_summary: str = ""
    
    # Verification Flags
    sum_check_passed: bool = True
    band_check_passed: bool = True
    error_detection: Optional[str] = None
    
    # Authoritative OCR
    final_ocr_question: str = ""
    final_ocr_answer: str = ""
    page_range: str = ""
    
    # Per-Stage Outputs
    stage1_output: Optional[Stage1Output] = None
    stage2_output: Optional[Stage2Output] = None
    stage3_output: Optional[Stage3Output] = None
    stage4_output: Optional[Stage4Output] = None
    stage5_output: Optional[Stage5Output] = None
    stage6_output: Optional[Stage6Output] = None
    
    grader: str = "easyocr-local-pipeline"
    prompt_version: str = "rubric_v2"
    temperature: float = 0.0
    raw_json_path: Optional[str] = None
    eval_timestamp: Optional[str] = None
    execution_time_seconds: float = 0.0

    @property
    def final_marks(self) -> float:
        return self.total_score

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.stage1_output:
            d["stage1_output"] = self.stage1_output.to_dict()
        if self.stage2_output:
            d["stage2_output"] = self.stage2_output.to_dict()
        if self.stage3_output:
            d["stage3_output"] = self.stage3_output.to_dict()
        if self.stage4_output:
            d["stage4_output"] = self.stage4_output.to_dict()
        if self.stage5_output:
            d["stage5_output"] = self.stage5_output.to_dict()
        if self.stage6_output:
            d["stage6_output"] = self.stage6_output.to_dict()
        return d

    def to_evaluation_csv_row(self) -> Dict[str, Any]:
        """Convert result into exact row dictionary matching evaluation.csv."""
        return {
            "evaluation_id": self.evaluation_id or f"{self.task_id}__{self.grader}",
            "task_id": self.task_id,
            "script_id": self.script_id,
            "grader": self.grader,
            "max_mark": int(self.max_mark) if self.max_mark.is_integer() else self.max_mark,
            "teacher_mark": int(self.teacher_mark) if self.teacher_mark is not None and self.teacher_mark.is_integer() else self.teacher_mark,
            "total_score": int(self.total_score) if self.total_score.is_integer() else self.total_score,
            "performance_band": self.performance_band,
            "context_content_data": self.context_content_data,
            "structure_format_brevity": self.structure_format_brevity,
            "language_mechanics": self.language_mechanics,
            "originality_comparisons_paraphrase": self.originality_comparisons_paraphrase,
            "is_attempted": self.is_attempted,
            "cap_applied": self.cap_applied,
            "cap_reason": self.cap_reason,
            "frequent_errors": self.frequent_errors,
            "positive_aspects": self.positive_aspects,
            "feedback_summary": self.feedback_summary,
            "sum_check_passed": self.sum_check_passed,
            "band_check_passed": self.band_check_passed,
            "model_version": self.grader,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "raw_json_path": self.raw_json_path or f"evaluations/raw/{self.task_id}__{self.grader}.json",
            "eval_timestamp": self.eval_timestamp or ""
        }
