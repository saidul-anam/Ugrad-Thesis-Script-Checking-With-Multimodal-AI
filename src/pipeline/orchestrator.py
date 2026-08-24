import os
import time
import yaml
from datetime import datetime
from typing import Optional, Dict, Any, Union, List
from PIL import Image

from src.core.config import PipelineConfig
from src.core.schemas import (
    CompleteEvaluationReport,
    Stage1TranscriptionResult,
    Stage2VerificationResult,
    Stage3ErrorResult,
    Stage4EvaluationResult
)
from src.engine.base_engine import BaseVLMEngine
from src.utils.image_loader import load_and_preprocess_image
from src.utils.pdf_processor import is_pdf, extract_images_from_pdf
from src.utils.export_utils import (
    export_stage1_artifacts,
    export_stage2_artifacts,
    export_stage3_artifacts,
    export_stage4_artifacts,
    export_report_json,
    export_report_markdown
)
from src.rag.context_provider import RAGContextProvider

from src.pipeline.stage1_transcriber import Stage1Transcriber
from src.pipeline.stage2_verifier import Stage2Verifier
from src.pipeline.stage3_error_analyzer import Stage3ErrorAnalyzer
from src.pipeline.stage4_evaluator import Stage4Evaluator


class ScriptCheckingPipeline:
    """
    End-to-end 4-Stage Multimodal Exam Script Evaluation Orchestrator
    powered by Gemma 4 31B IT with PDF and Image input support and stage-by-stage persistence.
    """

    def __init__(
        self,
        engine: BaseVLMEngine,
        config: PipelineConfig,
        rubric_path: Optional[str] = None
    ):
        self.engine = engine
        self.config = config
        self.rubric_path = rubric_path or "configs/rubrics/bangla_creative_question.yaml"
        self.rubric_data = self._load_rubric(self.rubric_path)

        # Stage Processors
        self.stage1 = Stage1Transcriber(self.engine)
        self.stage2 = Stage2Verifier(self.engine)
        self.stage3 = Stage3ErrorAnalyzer(self.engine)
        self.stage4 = Stage4Evaluator(self.engine)

        # RAG Context
        self.rag_provider = RAGContextProvider(
            context_dir=self.config.pipeline.rag.thematic_context_dir
        ) if self.config.pipeline.rag.enabled else None

    def _load_rubric(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {
            "subject": "General",
            "question_type": "Descriptive",
            "total_marks": 10.0,
            "criteria": [
                {"id": "content", "name": "Content Quality", "max_marks": 5.0, "description": "Relevance"},
                {"id": "accuracy", "name": "Accuracy", "max_marks": 5.0, "description": "Correctness"}
            ],
            "penalties": {"spelling_error_deduction": 0.25, "grammar_error_deduction": 0.5, "max_linguistic_deduction": 2.0}
        }

    def evaluate_script(
        self,
        input_source: Union[str, Image.Image],
        script_id: Optional[str] = None,
        thinking_mode: Optional[bool] = None,
        thematic_topic: Optional[str] = None,
        pdf_samples_dir: str = "data/samples"
    ) -> CompleteEvaluationReport:
        """
        Execute full 4-stage pipeline on an exam script (PDF or Image)
        and save stage-by-stage outputs in real-time.
        """
        start_time = time.time()
        
        # 1. Resolve identifier and load image(s)
        source_str = input_source if isinstance(input_source, str) else "in_memory_image.jpg"
        if not script_id:
            base = os.path.splitext(os.path.basename(source_str))[0]
            script_id = base

        # Handle PDF input vs Image input
        if isinstance(input_source, str) and is_pdf(input_source):
            print(f"[Pipeline] Input is PDF '{input_source}'. Rendering pages to image(s)...")
            pdf_pages = extract_images_from_pdf(
                input_source,
                output_dir=os.path.join(pdf_samples_dir, script_id),
                dpi=200
            )
            if not pdf_pages:
                raise ValueError(f"No pages extracted from PDF: {input_source}")
            # Primary page or stitched representation for evaluation
            pil_image = pdf_pages[0][1]
        else:
            pil_image = load_and_preprocess_image(input_source)

        decoding = self.config.decoding
        active_thinking = decoding.thinking_mode if thinking_mode is None else thinking_mode

        # Prepare dedicated script output directory for stage-by-stage persistence
        script_output_dir = os.path.join(self.config.pipeline.output_dir, script_id)
        os.makedirs(script_output_dir, exist_ok=True)

        print(f"\n[Pipeline] === Starting Evaluation for '{script_id}' ===")
        print(f"[Pipeline] Output directory: {script_output_dir}")
        print(f"[Pipeline] Thinking Mode: {active_thinking} | Temperature: {decoding.temperature}")

        # -------------------------------------------------------------
        # STAGE 1: Verbatim Transcription
        # -------------------------------------------------------------
        print("\n[Pipeline] [1/4] Executing Stage 1: Verbatim Transcription...")
        stage1_result = self.stage1.run(
            image=pil_image,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            max_new_tokens=decoding.max_new_tokens,
            thinking_mode=active_thinking
        )
        export_stage1_artifacts(stage1_result, script_output_dir)
        print(f"[Pipeline] [1/4] Stage 1 Saved -> {script_output_dir}/stage1_transcription.json")

        # -------------------------------------------------------------
        # STAGE 2: Autocorrection Verification
        # -------------------------------------------------------------
        print("\n[Pipeline] [2/4] Executing Stage 2: Autocorrection Verification...")
        stage2_result = self.stage2.run(
            image=pil_image,
            stage1_transcript=stage1_result.raw_transcript,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            max_new_tokens=decoding.max_new_tokens,
            thinking_mode=active_thinking
        )
        export_stage2_artifacts(stage2_result, script_output_dir)
        print(f"[Pipeline] [2/4] Stage 2 Saved -> {script_output_dir}/stage2_verification.json ({stage2_result.total_corrections_count} silent corrections reverted)")

        # -------------------------------------------------------------
        # STAGE 3: Error Extraction
        # -------------------------------------------------------------
        print("\n[Pipeline] [3/4] Executing Stage 3: Linguistic Error Extraction...")
        stage3_result = self.stage3.run(
            verified_transcript=stage2_result.verified_transcript,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            max_new_tokens=decoding.max_new_tokens,
            thinking_mode=active_thinking
        )
        export_stage3_artifacts(stage3_result, script_output_dir)
        print(f"[Pipeline] [3/4] Stage 3 Saved -> {script_output_dir}/stage3_errors.json & stage3_errors.csv ({stage3_result.total_error_count} errors)")

        # -------------------------------------------------------------
        # STAGE 4: Rubric Evaluation
        # -------------------------------------------------------------
        print("\n[Pipeline] [4/4] Executing Stage 4: Rubric Evaluation & Feedback...")
        thematic_context = None
        if self.rag_provider:
            lookup_topic = thematic_topic or self.rubric_data.get("subject", "bangla")
            thematic_context = self.rag_provider.get_context(lookup_topic)

        stage4_result = self.stage4.run(
            verified_transcript=stage2_result.verified_transcript,
            stage3_errors=stage3_result,
            rubric_data=self.rubric_data,
            thematic_context=thematic_context,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            max_new_tokens=decoding.max_new_tokens,
            thinking_mode=active_thinking
        )
        export_stage4_artifacts(stage4_result, script_output_dir)
        print(f"[Pipeline] [4/4] Stage 4 Saved -> {script_output_dir}/stage4_evaluation.json")

        # -------------------------------------------------------------
        # Complete Report Compilation
        # -------------------------------------------------------------
        elapsed = round(time.time() - start_time, 2)
        print(f"\n[Pipeline] Evaluation Complete in {elapsed}s | Final Marks: {stage4_result.final_score}/{stage4_result.total_max_marks}")

        report = CompleteEvaluationReport(
            script_id=script_id,
            image_path=source_str,
            model_id=self.config.model.model_id,
            timestamp=datetime.now().isoformat(),
            stage1_transcription=stage1_result,
            stage2_verification=stage2_result,
            stage3_errors=stage3_result,
            stage4_evaluation=stage4_result,
            metadata={
                "elapsed_seconds": elapsed,
                "thinking_mode": active_thinking,
                "temperature": decoding.temperature,
                "rubric_used": self.rubric_path,
                "engine_info": self.engine.get_engine_info(),
                "output_dir": script_output_dir
            }
        )

        # Export consolidated JSON & Markdown
        json_path = os.path.join(script_output_dir, "complete_report.json")
        md_path = os.path.join(script_output_dir, "evaluation_report.md")
        export_report_json(report, json_path)
        export_report_markdown(report, md_path)

        return report
