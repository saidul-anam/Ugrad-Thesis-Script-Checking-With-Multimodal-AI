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
from src.utils.question_utils import load_question_for_script, ExtractedQuestion


def _attribute_errors_to_pages(
    errors: List[Any],
    page_results: List[PageExtractionResult]
) -> None:
    """Attribute script-level linguistic errors to their originating pages."""
    if not page_results:
        return
    if len(page_results) == 1:
        p = page_results[0]
        p.stage3_errors = Stage3ErrorResult(
            errors=errors,
            spelling_error_count=sum(1 for e in errors if "spell" in getattr(e, "error_type", "").lower()),
            grammar_error_count=sum(1 for e in errors if "gram" in getattr(e, "error_type", "").lower()),
            syntax_error_count=sum(1 for e in errors if "synt" in getattr(e, "error_type", "").lower()),
            punctuation_error_count=sum(1 for e in errors if "punct" in getattr(e, "error_type", "").lower()),
            total_error_count=len(errors),
            linguistic_summary=""
        )
        return

    page_map: Dict[int, List[Any]] = {p.page_no: [] for p in page_results}
    for err in errors:
        needle = getattr(err, "erroneous_text", "").strip().lower()
        context = getattr(err, "context_sentence", "").strip().lower()
        matched_page = None
        for p in page_results:
            p_text = p.stage2_verification.verified_transcript.lower()
            if needle and needle in p_text:
                matched_page = p.page_no
                break
            elif context and (context[:30] in p_text or (len(context) > 15 and context[-20:] in p_text)):
                matched_page = p.page_no
                break
        if matched_page is None:
            matched_page = page_results[0].page_no
        page_map[matched_page].append(err)

    for p in page_results:
        p_errs = page_map.get(p.page_no, [])
        p.stage3_errors = Stage3ErrorResult(
            errors=p_errs,
            spelling_error_count=sum(1 for e in p_errs if "spell" in getattr(e, "error_type", "").lower()),
            grammar_error_count=sum(1 for e in p_errs if "gram" in getattr(e, "error_type", "").lower()),
            syntax_error_count=sum(1 for e in p_errs if "synt" in getattr(e, "error_type", "").lower()),
            punctuation_error_count=sum(1 for e in p_errs if "punct" in getattr(e, "error_type", "").lower()),
            total_error_count=len(p_errs),
            linguistic_summary=""
        )


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
        region: str = "default",
        skip_stage2: bool = False,
        force_extract: bool = False
    ) -> ExtractionResult:
        """
        Execute optimized multimodal extraction on all pages of an exam script:
        - Page-level checkpoint caching & instant resume
        - Stage 0: OpenCV Red-Ink Detection (with noise suppression)
        - Stage 1: Verbatim Transcription (ignoring red-ink teacher notes)
        - Stage 2: Autocorrection Verification (or fast bypass when skip_stage2=True)
        - Stage 0b: Red-Ink Teacher Mark Extraction (conditionally run if Stage 0 detected red ink)
        - Stage 3: Script-Level Linguistic Error Extraction (single global call on full transcript)
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
        checkpoint_dir = os.path.join(script_output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"\n[Extraction] === Starting Extraction for '{script_id}' ({len(page_images)} page(s)) ===")
        print(f"[Extraction] Output directory: {script_output_dir}")
        print(f"[Extraction] Fast Mode (skip Stage 2): {skip_stage2} | Thinking Mode: {active_thinking}")

        page_results: List[PageExtractionResult] = []
        all_teacher_marks: List[TeacherMarkItem] = []
        any_red_ink = False

        for page_no, p_img, p_path in page_images:
            ckpt_path = os.path.join(checkpoint_dir, f"page_{page_no}.json")

            # Check for page-level checkpoint
            if not force_extract and os.path.exists(ckpt_path):
                try:
                    with open(ckpt_path, "r", encoding="utf-8") as f:
                        cached_p = PageExtractionResult.model_validate_json(f.read())
                    print(f"\n--- Page {page_no}/{len(page_images)} (Resumed from checkpoint) ---")
                    print(f"[Extraction] Loaded Page {page_no} from checkpoint ({len(cached_p.stage1_transcription.raw_transcript.split())} words)")
                    page_results.append(cached_p)
                    if cached_p.has_red_ink:
                        any_red_ink = True
                    all_teacher_marks.extend(cached_p.teacher_marks)
                    continue
                except Exception as e:
                    print(f"[Extraction] Note: Checkpoint for page {page_no} invalid ({e}). Re-extracting.")

            print(f"\n--- Processing Page {page_no}/{len(page_images)} ---")

            # ---------------------------------------------------------
            # STAGE 0: OpenCV Red-Ink Detection
            # ---------------------------------------------------------
            print(f"[Extraction] [0/3] Stage 0: Running OpenCV HSV Red-Ink Detection (Page {page_no})...")
            stage0_res = self.stage0.detect(p_img)
            print(f"[Extraction] [0/3] Stage 0 Result -> has_red_ink={stage0_res.has_red_ink} ({stage0_res.red_pixel_count} px, {stage0_res.red_pixel_ratio*100:.3f}%) [Context: 0/4,096 tokens (0.0%)]")
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
            u1 = self.engine.get_last_usage()
            ctx1 = self.engine.format_last_usage()
            print(f"[Extraction] [1/3] Stage 1 Transcribed -> {stage1_result.word_count} words (illegible: {stage1_result.illegible_count}, unclear: {stage1_result.unclear_count}, struck: {stage1_result.struck_count}) {ctx1}")

            # ---------------------------------------------------------
            # STAGE 2: Autocorrection Verification & Audit
            # ---------------------------------------------------------
            if skip_stage2:
                print(f"[Extraction] [2/3] Stage 2: Skipped (Fast Mode enabled). Preserving Stage 1 verbatim. [Context: 0 tokens (bypassed)]")
                stage2_result = Stage2VerificationResult(
                    verified_transcript=stage1_result.raw_transcript,
                    silent_corrections_fixed=[],
                    total_corrections_count=0,
                    verification_notes="Fast mode: Stage 2 verification skipped; Stage 1 verbatim preserved."
                )
                u2 = {}
            else:
                print(f"[Extraction] [2/3] Stage 2: Autocorrection Verification (Page {page_no})...")
                stage2_result = self.stage2.run(
                    image=p_img,
                    stage1_transcript=stage1_result.raw_transcript,
                    temperature=decoding.temperature,
                    top_p=decoding.top_p,
                    max_new_tokens=decoding.max_new_tokens,
                    thinking_mode=active_thinking
                )
                u2 = self.engine.get_last_usage()
                ctx2 = self.engine.format_last_usage()
                print(f"[Extraction] [2/3] Stage 2 Verified -> {stage2_result.total_corrections_count} silent corrections reverted {ctx2}")

            # ---------------------------------------------------------
            # STAGE 0b: Teacher Mark Extraction (Conditional on Stage 0)
            # ---------------------------------------------------------
            page_marks: List[TeacherMarkItem] = []
            u0b = {}
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
                u0b = self.engine.get_last_usage()
                ctx0b = self.engine.format_last_usage()
                print(f"[Extraction] [0b/3] Stage 0b Found -> {len(page_marks)} numeric teacher mark(s) on Page {page_no} {ctx0b}")
            else:
                print(f"[Extraction] [0b/3] Stage 0b: has_red_ink=False -> Skipping teacher mark extraction for Page {page_no}.")

            max_ctx = getattr(self.engine, "context_window", getattr(self.engine, "max_context_window", 4096))
            page_prompt_t = u1.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0) + u0b.get("prompt_tokens", 0)
            page_comp_t = u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0) + u0b.get("completion_tokens", 0)
            page_total_t = u1.get("total_tokens", 0) + u2.get("total_tokens", 0) + u0b.get("total_tokens", 0)
            page_pct = round((page_total_t / max_ctx) * 100, 1) if max_ctx > 0 else 0.0

            page_token_usage = {
                "stage1": u1,
                "stage2": u2,
                "stage0b": u0b,
                "prompt_tokens": page_prompt_t,
                "completion_tokens": page_comp_t,
                "total_tokens": page_total_t,
                "max_context": max_ctx,
                "pct_context": page_pct
            }

            p_res = PageExtractionResult(
                page_no=page_no,
                image_path=p_path,
                has_red_ink=stage0_res.has_red_ink,
                red_pixel_count=stage0_res.red_pixel_count,
                stage1_transcription=stage1_result,
                stage2_verification=stage2_result,
                stage3_errors=Stage3ErrorResult(),
                teacher_marks=page_marks,
                token_usage=page_token_usage
            )
            page_results.append(p_res)

            # Checkpoint this page to disk immediately
            try:
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    f.write(p_res.model_dump_json(indent=2))
            except Exception as e:
                print(f"[Extraction] Note: Could not save checkpoint for Page {page_no} ({e})")

        # -------------------------------------------------------------
        # Aggregate Multi-Page Transcripts & Run Global Script-Level Stage 3
        # -------------------------------------------------------------
        if len(page_results) == 1:
            combined_raw = page_results[0].stage1_transcription.raw_transcript
            combined_verified = page_results[0].stage2_verification.verified_transcript
            aggregated_stage1 = page_results[0].stage1_transcription
            aggregated_stage2 = page_results[0].stage2_verification
        else:
            combined_raw = "\n\n--- Page Break ---\n\n".join(p.stage1_transcription.raw_transcript for p in page_results)
            combined_verified = "\n\n--- Page Break ---\n\n".join(p.stage2_verification.verified_transcript for p in page_results)
            combined_diffs = [d for p in page_results for d in p.stage2_verification.silent_corrections_fixed]

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

        # ---------------------------------------------------------
        # STAGE 3: Script-Level Linguistic Error Extraction (Single Global Call)
        # ---------------------------------------------------------
        print(f"\n[Extraction] [3/3] Stage 3: Script-Level Linguistic Error Extraction ({len(combined_verified)} chars, {aggregated_stage1.word_count} words)...")
        aggregated_stage3 = self.stage3.run(
            verified_transcript=combined_verified,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            max_new_tokens=decoding.max_new_tokens,
            thinking_mode=active_thinking
        )
        u3 = self.engine.get_last_usage()
        ctx3 = self.engine.format_last_usage()
        print(f"[Extraction] [3/3] Stage 3 Errors -> {aggregated_stage3.total_error_count} errors (spelling: {aggregated_stage3.spelling_error_count}, grammar: {aggregated_stage3.grammar_error_count}, syntax: {aggregated_stage3.syntax_error_count}, punctuation: {aggregated_stage3.punctuation_error_count}) {ctx3}")

        # Attribute errors to pages and sync checkpoints
        _attribute_errors_to_pages(aggregated_stage3.errors, page_results)
        for p in page_results:
            ckpt_p = os.path.join(checkpoint_dir, f"page_{p.page_no}.json")
            try:
                with open(ckpt_p, "w", encoding="utf-8") as f:
                    f.write(p.model_dump_json(indent=2))
            except Exception:
                pass

        elapsed = round(time.time() - start_time, 2)

        max_ctx = getattr(self.engine, "context_window", getattr(self.engine, "max_context_window", 4096))
        total_ext_prompt = sum(p.token_usage.get("prompt_tokens", 0) for p in page_results) + u3.get("prompt_tokens", 0)
        total_ext_completion = sum(p.token_usage.get("completion_tokens", 0) for p in page_results) + u3.get("completion_tokens", 0)
        total_ext_tokens = sum(p.token_usage.get("total_tokens", 0) for p in page_results) + u3.get("total_tokens", 0)
        total_ext_pct = round((total_ext_tokens / max_ctx) * 100, 1) if max_ctx > 0 else 0.0

        extraction_token_usage = {
            "max_context_window": max_ctx,
            "stage3": u3,
            "total_prompt_tokens": total_ext_prompt,
            "total_completion_tokens": total_ext_completion,
            "total_tokens": total_ext_tokens,
            "pct_context": total_ext_pct,
            "pages": {f"page_{p.page_no}": p.token_usage for p in page_results}
        }

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
                "token_usage": extraction_token_usage,
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
            page_tok = p.token_usage.get("total_tokens", 0) if p.token_usage else 0
            page_max_ctx = p.token_usage.get("max_context", 4096) if p.token_usage else 4096
            page_pct = p.token_usage.get("pct_context", 0.0) if p.token_usage else 0.0
            base_ocr = f"illegible: {p.stage1_transcription.illegible_count}, unclear: {p.stage1_transcription.unclear_count}, struck: {p.stage1_transcription.struck_count}"
            if page_tok > 0:
                ocr_flag_str = f"{base_ocr} | tokens: {page_tok}/{page_max_ctx} ({page_pct}%)"
            else:
                ocr_flag_str = base_ocr

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
        output_dir: Optional[str] = None,
        question_input: Optional[Union[str, ExtractedQuestion]] = None,
        questions_root: str = "outputs/questions"
    ) -> CompleteEvaluationReport:
        """
        Execute Stage 4 Rubric Evaluation on pre-extracted script transcripts and errors.
        Teacher marks and original marker IDs remain strictly isolated from grading inputs.
        Matches the extracted script to its corresponding question prompt.
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

        # Resolve Question Paper Matching
        question_obj: Optional[ExtractedQuestion] = None
        lang = extraction.metadata.get("paper") or rubric_data.get("subject", "english")
        if isinstance(question_input, ExtractedQuestion):
            question_obj = question_input
        elif isinstance(question_input, str):
            question_obj = load_question_for_script(
                script_id_or_path=script_id,
                lang=lang,
                question_override=question_input,
                questions_root=questions_root
            )
        else:
            question_obj = load_question_for_script(
                script_id_or_path=script_id,
                lang=lang,
                questions_root=questions_root
            )

        decoding = self.config.decoding
        active_thinking = decoding.thinking_mode if thinking_mode is None else thinking_mode

        if output_dir:
            if Path(output_dir).name == script_id:
                script_output_dir = str(output_dir)
            else:
                script_output_dir = os.path.join(output_dir, script_id)
        elif isinstance(extraction_input, (str, Path)) and "outputs/extracted" in str(extraction_input):
            script_output_dir = str(extraction_input).replace("outputs/extracted", "outputs/evaluated")
        elif extraction.metadata.get("output_dir") and "outputs/extracted" in extraction.metadata.get("output_dir", ""):
            script_output_dir = extraction.metadata["output_dir"].replace("outputs/extracted", "outputs/evaluated")
        elif isinstance(extraction_input, (str, Path)) and os.path.isdir(str(extraction_input)):
            script_output_dir = str(extraction_input)
        elif extraction.metadata.get("output_dir"):
            script_output_dir = extraction.metadata.get("output_dir")
        else:
            lang = extraction.metadata.get("paper") or rubric_data.get("subject", "english")
            script_output_dir = os.path.join("outputs", "evaluated", lang, script_id)
        os.makedirs(script_output_dir, exist_ok=True)

        print(f"\n[Evaluation] === Starting Rubric Evaluation for '{script_id}' ===")
        print(f"[Evaluation] Output directory: {script_output_dir}")
        print(f"[Evaluation] Rubric: {active_rubric_path}")
        if question_obj:
            print(f"[Evaluation] Matched Question: '{question_obj.question_id}' ({len(question_obj.question_text.split())} words prompt)")
        else:
            print(f"[Evaluation] Note: No question paper matched for script '{script_id}'. Proceeding with rubric criteria.")

        stage4_max_tokens = max(256, min(decoding.max_new_tokens, self.config.pipeline.stage4_max_new_tokens))
        stage4_timeout_sec = max(30.0, float(self.config.pipeline.stage4_generation_timeout_sec))
        print(f"[Evaluation] Stage 4 Token Budget: max_new_tokens={stage4_max_tokens}")
        print(f"[Evaluation] Stage 4 Timeout: {stage4_timeout_sec:.0f}s")

        # -------------------------------------------------------------
        # STAGE 4: Rubric Evaluation (Verified text + errors only)
        # -------------------------------------------------------------
        print("\n[Evaluation] [4/4] Executing Stage 4: Rubric Evaluation & Pedagogical Feedback...")
        thematic_context = None
        if self.rag_provider:
            lookup_topic = thematic_topic or rubric_data.get("subject", "bangla")
            thematic_context = self.rag_provider.get_context(lookup_topic)

        if hasattr(self.engine, "clear_cuda_cache"):
            self.engine.clear_cuda_cache()

        stage4_result = self.stage4.run(
            verified_transcript=extraction.stage2_verification.verified_transcript,
            stage3_errors=extraction.stage3_errors,
            rubric_data=rubric_data,
            thematic_context=thematic_context,
            question_text=question_obj.question_text if question_obj else None,
            question_id=question_obj.question_id if question_obj else None,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            max_new_tokens=stage4_max_tokens,
            thinking_mode=active_thinking,
            generation_max_time=stage4_timeout_sec
        )
        export_stage4_artifacts(stage4_result, script_output_dir)
        u4 = self.engine.get_last_usage()
        ctx4 = self.engine.format_last_usage()
        print(f"[Evaluation] [4/4] Stage 4 Saved -> {script_output_dir}/stage4_evaluation.json {ctx4}")

        # -------------------------------------------------------------
        # Complete Report Compilation
        # -------------------------------------------------------------
        elapsed = round(time.time() - start_time, 2)
        print(f"\n[Evaluation] Evaluation Complete for '{script_id}' in {elapsed}s | Final Marks: {stage4_result.final_score}/{stage4_result.total_max_marks} ({stage4_result.percentage:.1f}%) {ctx4}")

        report = CompleteEvaluationReport(
            script_id=script_id,
            image_path=extraction.image_path,
            model_id=self.config.model.model_id,
            timestamp=datetime.now().isoformat(),
            has_red_ink=extraction.has_red_ink,
            question_id=question_obj.question_id if question_obj else None,
            question_text=question_obj.question_text if question_obj else None,
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
                "question_id": question_obj.question_id if question_obj else None,
                "engine_info": self.engine.get_engine_info(),
                "stage4_token_usage": u4,
                "extraction_token_usage": extraction.metadata.get("token_usage", {}),
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
        output_dir: Optional[str] = None,
        skip_stage2: bool = False,
        force_extract: bool = False,
        question_input: Optional[Union[str, ExtractedQuestion]] = None
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
            output_dir=output_dir,
            skip_stage2=skip_stage2,
            force_extract=force_extract
        )

        # Step 2: Evaluation
        return self.evaluate_extracted_script(
            extraction_input=extraction,
            rubric_path=self.rubric_path,
            thematic_topic=thematic_topic,
            thinking_mode=thinking_mode,
            output_dir=output_dir,
            question_input=question_input
        )
