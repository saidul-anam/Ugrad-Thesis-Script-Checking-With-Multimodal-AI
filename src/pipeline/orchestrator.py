import os
import time
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from PIL import Image

from src.core.config import PipelineConfig
from src.core.schemas import (
    CompleteEvaluationReport,
    ExtractionResult,
    PageExtractionResult,
    TeacherMarkItem,
    RawTierRecord,
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
    export_teacher_marks_artifacts,
    export_extraction_artifacts,
    export_raw_tier_csv,
    load_extraction_artifacts,
    export_report_json,
    export_report_markdown
)
from src.rag.context_provider import RAGContextProvider

from src.pipeline.stage0_red_ink_detector import RedInkDetector, RedInkDetectionResult
from src.pipeline.stage0b_teacher_marks import Stage0bTeacherMarkExtractor, Stage0bResult
from src.pipeline.stage1_transcriber import Stage1Transcriber
from src.pipeline.stage2_verifier import Stage2Verifier
from src.pipeline.stage3_error_analyzer import Stage3ErrorAnalyzer
from src.pipeline.stage4_evaluator import Stage4Evaluator


class ScriptCheckingPipeline:
    """
    End-to-end Multimodal Exam Script Extraction & Evaluation Orchestrator
    incorporating Stage 0 (OpenCV Red Ink), Stage 0b (Teacher Marks),
    Stage 1 (Verbatim), Stage 2 (Autocorrection Audit), Stage 3 (Linguistic Errors),
    Stage 4 (Rubric Grading), and Raw-Tier CSV export.
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
        self.stage0 = RedInkDetector()
        self.stage0b = Stage0bTeacherMarkExtractor(self.engine)
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

    def extract_script(
        self,
        input_source: Union[str, Image.Image],
        script_id: Optional[str] = None,
        thinking_mode: Optional[bool] = None,
        pdf_samples_dir: str = "data/samples",
        output_dir: Optional[str] = None,
        paper: str = "bangla",
        task_type: str = "creative_question",
        original_marker_id: str = "unknown",
        school_id: str = "default",
        region: str = "default"
    ) -> ExtractionResult:
        """
        Execute full multimodal extraction on all pages of an exam script:
        - Stage 0: OpenCV Red-Ink Detection
        - Stage 1: Verbatim Transcription (ignoring red-ink teacher notes)
        - Stage 2: Autocorrection Verification (reverting silent LLM fixes)
        - Stage 3: Text-only Linguistic Error Extraction
        - Stage 0b: Red-Ink Teacher Mark Extraction (conditionally run if Stage 0 detected red ink)
        - Raw-Tier Dataset CSV generation
        """
        start_time = time.time()

        # 1. Resolve identifier
        source_str = input_source if isinstance(input_source, str) else "in_memory_image.jpg"
        if not script_id:
            base = os.path.splitext(os.path.basename(source_str))[0]
            script_id = base

        # 2. Extract or Load Pages
        page_images: List[tuple[int, Image.Image, str]] = []  # (page_no, PIL Image, image_path)

        if isinstance(input_source, str) and is_pdf(input_source):
            print(f"[Extraction] Input is PDF '{input_source}'. Rendering pages to image(s)...")
            pdf_pages = extract_images_from_pdf(
                input_source,
                output_dir=os.path.join(pdf_samples_dir, script_id),
                dpi=200
            )
            if not pdf_pages:
                raise ValueError(f"No pages extracted from PDF: {input_source}")
            for p_no, p_img, saved_path in pdf_pages:
                p_path = saved_path or os.path.join(pdf_samples_dir, script_id, f"page_{p_no}.png")
                page_images.append((p_no, p_img, p_path))
        else:
            pil_image = load_and_preprocess_image(input_source)
            page_images.append((1, pil_image, source_str))

        decoding = self.config.decoding
        active_thinking = decoding.thinking_mode if thinking_mode is None else thinking_mode

        # Dedicated output directory for this script
        base_out = output_dir or self.config.pipeline.output_dir
        script_output_dir = os.path.join(base_out, script_id)
        os.makedirs(script_output_dir, exist_ok=True)

        print(f"\n[Extraction] === Starting Extraction for '{script_id}' ({len(page_images)} page(s)) ===")
        print(f"[Extraction] Output directory: {script_output_dir}")
        print(f"[Extraction] Thinking Mode: {active_thinking} | Temperature: {decoding.temperature}")

        page_results: List[PageExtractionResult] = []
        all_teacher_marks: List[TeacherMarkItem] = []
        any_red_ink = False

        for page_no, p_img, p_path in page_images:
            print(f"\n--- Processing Page {page_no}/{len(page_images)} ---")

            # ---------------------------------------------------------
            # STAGE 0: OpenCV Red-Ink Detection
            # ---------------------------------------------------------
            print(f"[Extraction] [0/3] Stage 0: Running OpenCV HSV Red-Ink Detection (Page {page_no})...")
            stage0_res = self.stage0.detect(p_img)
            print(f"[Extraction] [0/3] Stage 0 Result -> has_red_ink={stage0_res.has_red_ink} ({stage0_res.red_pixel_count} px, {stage0_res.red_pixel_ratio*100:.3f}%)")
            if stage0_res.has_red_ink:
                any_red_ink = True

            # ---------------------------------------------------------
            # STAGE 1: Verbatim Transcription (Ignoring Teacher Red Ink)
            # ---------------------------------------------------------
            print(f"[Extraction] [1/3] Stage 1: Verbatim Transcription (Page {page_no})...")
            stage1_result = self.stage1.run(
                image=p_img,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                max_new_tokens=decoding.max_new_tokens,
                thinking_mode=active_thinking
            )
            print(f"[Extraction] [1/3] Stage 1 Transcribed -> {stage1_result.word_count} words (illegible: {stage1_result.illegible_count}, unclear: {stage1_result.unclear_count}, struck: {stage1_result.struck_count})")

            # ---------------------------------------------------------
            # STAGE 2: Autocorrection Verification & Audit
            # ---------------------------------------------------------
            print(f"[Extraction] [2/3] Stage 2: Autocorrection Verification (Page {page_no})...")
            stage2_result = self.stage2.run(
                image=p_img,
                stage1_transcript=stage1_result.raw_transcript,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                max_new_tokens=decoding.max_new_tokens,
                thinking_mode=active_thinking
            )
            print(f"[Extraction] [2/3] Stage 2 Verified -> {stage2_result.total_corrections_count} silent corrections reverted")

            # ---------------------------------------------------------
            # STAGE 3: Linguistic Error Extraction (Text-Only)
            # ---------------------------------------------------------
            print(f"[Extraction] [3/3] Stage 3: Linguistic Error Extraction (Page {page_no})...")
            stage3_result = self.stage3.run(
                verified_transcript=stage2_result.verified_transcript,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                max_new_tokens=decoding.max_new_tokens,
                thinking_mode=active_thinking
            )
            print(f"[Extraction] [3/3] Stage 3 Errors -> {stage3_result.total_error_count} errors (spelling: {stage3_result.spelling_error_count}, grammar: {stage3_result.grammar_error_count})")

            # ---------------------------------------------------------
            # STAGE 0b: Teacher Mark Extraction (Conditional on Stage 0)
            # ---------------------------------------------------------
            page_marks: List[TeacherMarkItem] = []
            if stage0_res.has_red_ink:
                print(f"[Extraction] [0b/3] Stage 0b: Red ink detected -> Running Gemma 4 Teacher Mark Extraction (Page {page_no})...")
                stage0b_res = self.stage0b.run(
                    image=p_img,
                    temperature=decoding.temperature,
                    top_p=decoding.top_p,
                    max_new_tokens=1024,
                    thinking_mode=active_thinking
                )
                page_marks = stage0b_res.teacher_marks
                all_teacher_marks.extend(page_marks)
                print(f"[Extraction] [0b/3] Stage 0b Found -> {len(page_marks)} numeric teacher mark(s) on Page {page_no}")
            else:
                print(f"[Extraction] [0b/3] Stage 0b: has_red_ink=False -> Skipping teacher mark extraction for Page {page_no}.")

            page_results.append(PageExtractionResult(
                page_no=page_no,
                image_path=p_path,
                has_red_ink=stage0_res.has_red_ink,
                red_pixel_count=stage0_res.red_pixel_count,
                stage1_transcription=stage1_result,
                stage2_verification=stage2_result,
                stage3_errors=stage3_result,
                teacher_marks=page_marks
            ))

        # -------------------------------------------------------------
        # Aggregate Multi-Page Transcripts & Errors
        # -------------------------------------------------------------
        if len(page_results) == 1:
            aggregated_stage1 = page_results[0].stage1_transcription
            aggregated_stage2 = page_results[0].stage2_verification
            aggregated_stage3 = page_results[0].stage3_errors
        else:
            combined_raw = "\n\n--- Page Break ---\n\n".join(p.stage1_transcription.raw_transcript for p in page_results)
            combined_verified = "\n\n--- Page Break ---\n\n".join(p.stage2_verification.verified_transcript for p in page_results)
            combined_diffs = [d for p in page_results for d in p.stage2_verification.silent_corrections_fixed]
            combined_errors = [e for p in page_results for e in p.stage3_errors.errors]

            aggregated_stage1 = Stage1TranscriptionResult(
                raw_transcript=combined_raw,
                illegible_count=sum(p.stage1_transcription.illegible_count for p in page_results),
                unclear_count=sum(p.stage1_transcription.unclear_count for p in page_results),
                struck_count=sum(p.stage1_transcription.struck_count for p in page_results),
                character_count=sum(p.stage1_transcription.character_count for p in page_results),
                word_count=sum(p.stage1_transcription.word_count for p in page_results),
                detected_script=page_results[0].stage1_transcription.detected_script
            )

            aggregated_stage2 = Stage2VerificationResult(
                verified_transcript=combined_verified,
                silent_corrections_fixed=combined_diffs,
                total_corrections_count=len(combined_diffs),
                verification_notes="; ".join(p.stage2_verification.verification_notes for p in page_results if p.stage2_verification.verification_notes)
            )

            aggregated_stage3 = Stage3ErrorResult(
                errors=combined_errors,
                spelling_error_count=sum(p.stage3_errors.spelling_error_count for p in page_results),
                grammar_error_count=sum(p.stage3_errors.grammar_error_count for p in page_results),
                syntax_error_count=sum(p.stage3_errors.syntax_error_count for p in page_results),
                punctuation_error_count=sum(p.stage3_errors.punctuation_error_count for p in page_results),
                total_error_count=len(combined_errors),
                linguistic_summary="; ".join(p.stage3_errors.linguistic_summary for p in page_results if p.stage3_errors.linguistic_summary)
            )

        elapsed = round(time.time() - start_time, 2)

        extraction_result = ExtractionResult(
            script_id=script_id,
            image_path=source_str,
            model_id=self.config.model.model_id,
            timestamp=datetime.now().isoformat(),
            has_red_ink=any_red_ink,
            stage1_transcription=aggregated_stage1,
            stage2_verification=aggregated_stage2,
            stage3_errors=aggregated_stage3,
            teacher_marks=all_teacher_marks,
            pages=page_results,
            metadata={
                "elapsed_seconds": elapsed,
                "total_pages": len(page_results),
                "thinking_mode": active_thinking,
                "temperature": decoding.temperature,
                "engine_info": self.engine.get_engine_info(),
                "output_dir": script_output_dir,
                "paper": paper,
                "task_type": task_type,
                "original_marker_id": original_marker_id,
                "school_id": school_id,
                "region": region
            }
        )

        # -------------------------------------------------------------
        # Save Extraction Artifacts & Raw-Tier CSV Dataset
        # -------------------------------------------------------------
        export_extraction_artifacts(extraction_result, script_output_dir)

        # Build Raw-Tier CSV records per page
        raw_tier_records = []
        for p in page_results:
            ocr_flag_str = f"illegible: {p.stage1_transcription.illegible_count}, unclear: {p.stage1_transcription.unclear_count}, struck: {p.stage1_transcription.struck_count}"
            error_json_str = json.dumps([e.model_dump() for e in p.stage3_errors.errors], ensure_ascii=False)
            marks_str = "; ".join(f"Q{m.question_no or '?'}:{m.mark_value} ({m.location})" for m in p.teacher_marks) if p.teacher_marks else ""

            raw_tier_records.append(RawTierRecord(
                script_id=script_id,
                page_no=p.page_no,
                question_no=p.teacher_marks[0].question_no if p.teacher_marks else None,
                paper=paper,
                task_type=task_type,
                transcript_text=p.stage2_verification.verified_transcript,
                ocr_flags=ocr_flag_str,
                error_list=error_json_str,
                teacher_mark=marks_str,
                has_red_ink=p.has_red_ink,
                original_marker_id=original_marker_id,
                school_id=school_id,
                region=region
            ))

        # Save per-script raw-tier CSV and root dataset CSV
        per_script_csv = os.path.join(script_output_dir, "raw_tier_records.csv")
        export_raw_tier_csv(raw_tier_records, per_script_csv)

        root_dataset_csv = os.path.join(base_out, "raw_tier_dataset.csv")
        export_raw_tier_csv(raw_tier_records, root_dataset_csv)

        print(f"[Extraction] Saved Raw-Tier CSV -> {root_dataset_csv}")
        print(f"[Extraction] Extraction Complete for '{script_id}' in {elapsed}s.")
        return extraction_result

    def evaluate_extracted_script(
        self,
        extraction_input: Union[str, ExtractionResult],
        rubric_path: Optional[str] = None,
        thematic_topic: Optional[str] = None,
        thinking_mode: Optional[bool] = None,
        output_dir: Optional[str] = None
    ) -> CompleteEvaluationReport:
        """
        Execute Stage 4 Rubric Evaluation on pre-extracted script transcripts and errors.
        Teacher marks and original marker IDs remain strictly isolated from grading inputs.
        """
        start_time = time.time()

        # 1. Resolve ExtractionResult
        if isinstance(extraction_input, ExtractionResult):
            extraction = extraction_input
        else:
            print(f"[Evaluation] Loading pre-extracted artifacts from: {extraction_input}")
            extraction = load_extraction_artifacts(extraction_input)

        script_id = extraction.script_id

        # Update rubric if custom path provided
        active_rubric_path = rubric_path or self.rubric_path
        rubric_data = self._load_rubric(active_rubric_path) if rubric_path else self.rubric_data

        decoding = self.config.decoding
        active_thinking = decoding.thinking_mode if thinking_mode is None else thinking_mode

        if output_dir:
            script_output_dir = os.path.join(output_dir, script_id)
        elif isinstance(extraction_input, (str, Path)) and os.path.isdir(str(extraction_input)):
            script_output_dir = str(extraction_input)
        elif extraction.metadata.get("output_dir"):
            script_output_dir = extraction.metadata.get("output_dir")
        else:
            script_output_dir = os.path.join(self.config.pipeline.output_dir, script_id)
        os.makedirs(script_output_dir, exist_ok=True)

        print(f"\n[Evaluation] === Starting Rubric Evaluation for '{script_id}' ===")
        print(f"[Evaluation] Output directory: {script_output_dir}")
        print(f"[Evaluation] Rubric: {active_rubric_path}")

        # -------------------------------------------------------------
        # STAGE 4: Rubric Evaluation (Verified text + errors only)
        # -------------------------------------------------------------
        print("\n[Evaluation] [4/4] Executing Stage 4: Rubric Evaluation & Pedagogical Feedback...")
        thematic_context = None
        if self.rag_provider:
            lookup_topic = thematic_topic or rubric_data.get("subject", "bangla")
            thematic_context = self.rag_provider.get_context(lookup_topic)

        stage4_result = self.stage4.run(
            verified_transcript=extraction.stage2_verification.verified_transcript,
            stage3_errors=extraction.stage3_errors,
            rubric_data=rubric_data,
            thematic_context=thematic_context,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            max_new_tokens=decoding.max_new_tokens,
            thinking_mode=active_thinking
        )
        export_stage4_artifacts(stage4_result, script_output_dir)
        print(f"[Evaluation] [4/4] Stage 4 Saved -> {script_output_dir}/stage4_evaluation.json")

        # -------------------------------------------------------------
        # Complete Report Compilation
        # -------------------------------------------------------------
        elapsed = round(time.time() - start_time, 2)
        print(f"\n[Evaluation] Evaluation Complete for '{script_id}' in {elapsed}s | Final Marks: {stage4_result.final_score}/{stage4_result.total_max_marks} ({stage4_result.percentage:.1f}%)")

        report = CompleteEvaluationReport(
            script_id=script_id,
            image_path=extraction.image_path,
            model_id=self.config.model.model_id,
            timestamp=datetime.now().isoformat(),
            has_red_ink=extraction.has_red_ink,
            stage1_transcription=extraction.stage1_transcription,
            stage2_verification=extraction.stage2_verification,
            stage3_errors=extraction.stage3_errors,
            teacher_marks=extraction.teacher_marks,
            stage4_evaluation=stage4_result,
            metadata={
                "extraction_elapsed": extraction.metadata.get("elapsed_seconds", 0),
                "evaluation_elapsed": elapsed,
                "total_pages": len(extraction.pages),
                "thinking_mode": active_thinking,
                "temperature": decoding.temperature,
                "rubric_used": active_rubric_path,
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

    def evaluate_script(
        self,
        input_source: Union[str, Image.Image],
        script_id: Optional[str] = None,
        thinking_mode: Optional[bool] = None,
        thematic_topic: Optional[str] = None,
        pdf_samples_dir: str = "data/samples",
        output_dir: Optional[str] = None
    ) -> CompleteEvaluationReport:
        """
        Execute full end-to-end pipeline (Extraction Stages 0, 0b, 1-3 -> Evaluation Stage 4).
        """
        # Step 1: Extraction
        extraction = self.extract_script(
            input_source=input_source,
            script_id=script_id,
            thinking_mode=thinking_mode,
            pdf_samples_dir=pdf_samples_dir,
            output_dir=output_dir
        )

        # Step 2: Evaluation
        return self.evaluate_extracted_script(
            extraction_input=extraction,
            rubric_path=self.rubric_path,
            thematic_topic=thematic_topic,
            thinking_mode=thinking_mode,
            output_dir=output_dir
        )
