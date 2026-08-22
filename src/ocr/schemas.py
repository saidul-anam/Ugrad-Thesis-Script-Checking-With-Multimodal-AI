"""
Strict, typed Pydantic schemas for the Single-Pass English Handwritten Exam OCR Subsystem.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class InputMetadata(BaseModel):
    """Metadata regarding the input image file."""
    image_path: str = Field(..., description="Absolute or relative path to the image file.")
    image_hash: str = Field(..., description="Stable SHA-256 hash of the input image.")
    image_size: Optional[List[int]] = Field(default=None, description="Image dimensions as [width, height].")
    page_number: Optional[int] = Field(default=1, description="Page number if extracted from a multi-page document.")


class OCRSegment(BaseModel):
    """Sub-region, line, or token level extraction with confidence."""
    text: str = Field(..., description="Extracted line or token text.")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Segment-level confidence score [0.0, 1.0].")
    bbox: Optional[List[List[int]]] = Field(default=None, description="Optional bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].")


class OCRContent(BaseModel):
    """Extracted text and confidence metrics from the single OCR pass."""
    backend: str = Field(..., description="Identifier of the OCR engine used (e.g., infinite_ocr, trocr, easyocr).")
    model: str = Field(..., description="Model checkpoint or architecture name.")
    raw_text: str = Field(..., description="Unaltered raw output directly from the OCR engine.")
    normalized_text: str = Field(..., description="Safe normalized text (Unicode NFC, standardized whitespace).")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Overall aggregated or direct confidence score in range [0.0, 1.0].")
    confidence_type: Literal["raw", "derived", "calibrated", "none"] = Field(
        default="none",
        description="Whether confidence is raw from model, derived from signals, calibrated, or unavailable."
    )
    confidence_available: bool = Field(default=False, description="Explicit boolean stating if a reliable confidence signal exists.")
    segments: List[OCRSegment] = Field(default_factory=list, description="Optional segment or line-level extractions.")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v


class ExecutionMetadata(BaseModel):
    """Performance and environment telemetry for reproducibility."""
    processing_time_seconds: float = Field(..., ge=0.0, description="End-to-end OCR latency in seconds.")
    device: str = Field(default="cpu", description="Compute device used ('cuda', 'cpu').")
    gpu_memory_peak_mb: Optional[float] = Field(default=None, description="Peak GPU memory allocated in MB.")
    cpu_memory_mb: Optional[float] = Field(default=None, description="Process RSS memory usage in MB.")
    timestamp: str = Field(..., description="ISO 8601 execution timestamp.")


class OCRResult(BaseModel):
    """Complete, self-contained output schema for an OCR execution."""
    script_id: str = Field(..., description="Unique anonymized identifier for the examination script.")
    input: InputMetadata = Field(..., description="Input image metadata.")
    ocr: OCRContent = Field(..., description="OCR text, confidence, and segments.")
    metadata: ExecutionMetadata = Field(..., description="Execution telemetry.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)


class GroundTruthSample(BaseModel):
    """Ground truth reference sample for OCR benchmarking."""
    script_id: str = Field(..., description="Script identifier matching OCRResult.")
    image_path: str = Field(..., description="Path to handwritten exam image.")
    ground_truth_text: str = Field(..., description="Human-verified transcription text.")
    image_hash: Optional[str] = Field(default=None, description="SHA-256 hash of the ground truth image.")


class BenchmarkSampleResult(BaseModel):
    """Evaluation result comparing an OCR result against human ground truth."""
    script_id: str
    image_path: str
    raw_ocr_text: str
    normalized_ocr_text: str
    ground_truth_text: str
    confidence: Optional[float]
    confidence_available: bool
    raw_cer: float
    normalized_cer: float
    raw_wer: float
    normalized_wer: float
    is_acceptable: bool
    processing_time_seconds: float
