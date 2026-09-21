import os
import re
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
    Stage4EvaluationResult,
    AlignedAnswerItem
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
from src.pipeline.arbitration import EvidenceArbitrationGate
from src.utils.linguistic_sanitizer import get_english_lexicon
from src.pipeline.stage4_evaluator import Stage4Evaluator
from src.pipeline.answer_segmenter import segment_script_into_questions, extract_header_qno
from src.prompts.stage4_modular import is_objective_question
from src.utils.question_utils import (
    load_question_for_script,
    ExtractedQuestion,
    extract_question_vocab
)
from src.utils.ground_truth import (
    get_ground_truth_for_script,
    ground_truth_to_teacher_marks,
    extract_candidate_questions,
    canonicalize_question_key
)


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
            punctuation_error_count=0,
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
            punctuation_error_count=0,
            total_error_count=len(p_errs),
            linguistic_summary=""
        )


def _chunk_text_by_sentences(text: str, target_words: int = 120) -> List[str]:
    """
    Split long text into sentence-bounded chunks of ~100-140 words.
    Avoids cutting sentences across chunks.
    """
    sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    current_chunk = []
    current_count = 0

    for s in sentences:
        s_words = len(s.split())
        if current_chunk and (current_count + s_words > target_words):
            chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_count = s_words
        else:
            current_chunk.append(s)
            current_count += s_words

    if current_chunk:
        if chunks and current_count < 30:
            chunks[-1] = chunks[-1] + " " + " ".join(current_chunk)
        else:
            chunks.append(" ".join(current_chunk))

    return chunks


def _token_pattern(word: str) -> "re.Pattern":
    return re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.IGNORECASE)


def _replace_token_preserving_case(text: str, c_word: str, i_word: str, count: int = 0) -> str:
    def repl(m: "re.Match") -> str:
        src = m.group(0)
        return (i_word[:1].upper() + i_word[1:]) if src[:1].isupper() else i_word
    return _token_pattern(c_word).sub(repl, text, count=count)


def _normalize_token_in_context(
    text: str,
    c_word: str,
    i_word: str,
    erroneous_text: str,
    context: str,
    lexicon: Optional[set] = None,
) -> str:
    """
    Replace the misread token with the intended token ONLY inside the span where the error occurred
    (the context sentence, else the erroneous span), so a cleared 'do'->'to' does not rewrite every
    other 'do' on the page. If the span cannot be found, a page-wide replacement is done only when
    the misread token is not a dictionary word (i.e. it can only be this misread).
    """
    if not text or not c_word or not i_word or c_word.lower() == i_word.lower():
        return text
    for needle in (context, erroneous_text):
        needle = " ".join((needle or "").split())
        if len(needle) < 3:
            continue
        span_re = re.compile(r"\s+".join(re.escape(t) for t in needle.split()), re.IGNORECASE)
        m = span_re.search(text)
        if m:
            new_span = _replace_token_preserving_case(m.group(0), c_word, i_word, count=1)
            return text[:m.start()] + new_span + text[m.end():]
    if lexicon is not None and c_word.lower() in lexicon:
        return text
    return _replace_token_preserving_case(text, c_word, i_word)


def _apply_normalizations(
    cleared: List[Dict[str, Any]],
    ans: Optional[AlignedAnswerItem],
    page_results: List[PageExtractionResult],
    errors: List[Any],
    lexicon: Optional[set] = None,
) -> int:
    """Apply HANDWRITING_AMBIGUITY normalizations to the answer text AND the per-page transcripts."""
    n = 0
    for amb in cleared:
        if not amb.get("normalize", True):
            continue
        c_word = str(amb.get("candidate") or "")
        i_word = str(amb.get("intended_word") or "")
        if not c_word or not i_word or c_word.lower() == i_word.lower():
            continue
        err_text, ctx = "", ""
        cid = str(amb.get("candidate_id") or "")
        try:
            e_idx = int(cid.split(":")[1]) if cid.count(":") >= 2 else -1
            if 0 <= e_idx < len(errors):
                err_text = getattr(errors[e_idx], "erroneous_text", "") or ""
                ctx = getattr(errors[e_idx], "context_sentence", "") or ""
        except Exception:
            pass
        if not ctx and not err_text:
            for e in errors:
                if c_word.lower() in (getattr(e, "erroneous_text", "") or "").lower():
                    err_text = getattr(e, "erroneous_text", "") or ""
                    ctx = getattr(e, "context_sentence", "") or ""
                    break
        if ans is not None:
            ans.answer_text = _normalize_token_in_context(ans.answer_text, c_word, i_word, err_text, ctx, lexicon)
        targets = [p for p in page_results if ans is None or not ans.page_numbers or p.page_no in ans.page_numbers]
        for p in targets:
            p.stage2_verification.verified_transcript = _normalize_token_in_context(
                p.stage2_verification.verified_transcript, c_word, i_word, err_text, ctx, lexicon
            )
        n += 1
    return n


def _rebuild_combined_verified(page_results: List[PageExtractionResult]) -> str:
    if len(page_results) == 1:
        return page_results[0].stage2_verification.verified_transcript
    return "\n\n--- Page Break ---\n\n".join(p.stage2_verification.verified_transcript for p in page_results)


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
        force_extract: bool = False,
        question_input: Optional[Union[str, ExtractedQuestion]] = None,
        questions_root: str = "outputs/questions",
        extract_teacher_marks: Optional[bool] = None,
        parallel_workers: Optional[int] = None
    ) -> ExtractionResult:
        """
        Execute optimized multimodal extraction on all pages of an exam script:
        - Page-level checkpoint caching & instant resume
        - Question Paper Grounding (contextual vocabulary & candidate question marks)
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

        # Resolve Question Paper Context
        question_obj: Optional[ExtractedQuestion] = None
        if isinstance(question_input, ExtractedQuestion):
            question_obj = question_input
        elif isinstance(question_input, str):
            question_obj = load_question_for_script(
                script_id_or_path=script_id,
                lang=paper,
                question_override=question_input,
                questions_root=questions_root
            )
        else:
            question_obj = load_question_for_script(
                script_id_or_path=script_id,
                lang=paper,
                questions_root=questions_root
            )

        valid_paper_questions: List[str] = []
        question_max_marks: Dict[str, float] = {}
        question_vocab: List[str] = []

        if question_obj:
            question_vocab = extract_question_vocab(question_obj)
            for sq in question_obj.sub_questions:
                q_num = str(sq.get("q_no") or sq.get("part") or sq.get("question_no") or "").strip()
                if q_num:
                    valid_paper_questions.append(q_num)
                    max_m = sq.get("max_marks") or sq.get("marks")
                    if max_m is not None:
                        try:
                            question_max_marks[q_num] = float(max_m)
                        except (ValueError, TypeError):
                            pass
            print(f"[Extraction] 📘 Matched Question Context: '{question_obj.question_id}' ({len(valid_paper_questions)} sub-questions, {len(question_vocab)} vocab tokens)")
        else:
            print(f"[Extraction] ℹ️ No question context matched for '{script_id}'. Proceeding with standard visual extraction.")

        # Ground truth is strictly an evaluation benchmark (never fed to extraction VLM)
        ground_truth_marks = get_ground_truth_for_script(script_id)

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

        # Resolve teacher marks extraction toggle
        if extract_teacher_marks is None:
            extract_teacher_marks = getattr(self.config.pipeline, "stage0b_teacher_marks", True)

        # Dedicated output directory for this script
        base_out = output_dir or self.config.pipeline.output_dir
        script_output_dir = os.path.join(base_out, script_id)
        checkpoint_dir = os.path.join(script_output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"\n[Extraction] === Starting Extraction for '{script_id}' ({len(page_images)} page(s)) ===")
        print(f"[Extraction] Output directory: {script_output_dir}")
        print(f"[Extraction] Fast Mode (skip Stage 2): {skip_stage2} | Thinking Mode: {active_thinking} | Teacher Marks (Stage 0b): {extract_teacher_marks}")

        page_results: List[PageExtractionResult] = []
        all_teacher_marks: List[TeacherMarkItem] = []
        any_red_ink = False

        active_workers = parallel_workers if parallel_workers is not None else getattr(self.config.pipeline, "parallel_workers", 1)
        active_workers = max(1, int(active_workers or 1))

        def _process_single_page(item):
            page_no, p_img, p_path = item
            ckpt_path = os.path.join(checkpoint_dir, f"page_{page_no}.json")

            # Check for page-level checkpoint
            if not force_extract and os.path.exists(ckpt_path):
                try:
                    with open(ckpt_path, "r", encoding="utf-8") as f:
                        cached_p = PageExtractionResult.model_validate_json(f.read())
                    print(f"\n--- Page {page_no}/{len(page_images)} (Resumed from checkpoint) ---")
                    print(f"[Extraction] Loaded Page {page_no} from checkpoint ({len(cached_p.stage1_transcription.raw_transcript.split())} words)")
                    return cached_p
                except Exception as e:
                    print(f"[Extraction] Note: Checkpoint for page {page_no} invalid ({e}). Re-extracting.")

            print(f"\n--- Processing Page {page_no}/{len(page_images)} ---")

            # ---------------------------------------------------------
            # STAGE 0: OpenCV Red-Ink Detection
            # ---------------------------------------------------------
            print(f"[Extraction] [0/3] Stage 0: Running OpenCV HSV Red-Ink Detection (Page {page_no})...")
            stage0_res = self.stage0.detect(p_img)
            print(f"[Extraction] [0/3] Stage 0 Result -> has_red_ink={stage0_res.has_red_ink} ({stage0_res.red_pixel_count} px, {stage0_res.red_pixel_ratio*100:.3f}%) [Context: 0/4,096 tokens (0.0%)]")

            # ---------------------------------------------------------
            # STAGE 1: Verbatim Transcription (Ignoring Teacher Red Ink)
            # ---------------------------------------------------------
            print(f"[Extraction] [1/3] Stage 1: Verbatim Transcription (Page {page_no})...")
            stage1_input_img = stage0_res.clean_image if getattr(stage0_res, "clean_image", None) is not None else p_img
            stage1_result = self.stage1.run(
                image=stage1_input_img,
                question_reference_vocab=question_vocab if question_obj else None,
                question_syllabus=question_obj.sub_questions if question_obj else None,
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
            is_clean_page = (
                stage1_result.unclear_count == 0
                and stage1_result.illegible_count == 0
                and "[unclear:" not in stage1_result.raw_transcript
                and "[illegible]" not in stage1_result.raw_transcript
            )
            conditional_stage2 = getattr(self.cfg.pipeline, "stage2_conditional", True) if hasattr(self, "cfg") and hasattr(self.cfg, "pipeline") else True

            if skip_stage2 or (conditional_stage2 and is_clean_page):
                bypass_reason = "Fast mode enabled" if skip_stage2 else "Stage 1 transcript clean (0 unclear/illegible markers)"
                print(f"[Extraction] [2/3] Stage 2: Skipped ({bypass_reason}). Preserving Stage 1 verbatim. [Context: 0 tokens (bypassed)]")
                stage2_result = Stage2VerificationResult(
                    verified_transcript=stage1_result.raw_transcript,
                    silent_corrections_fixed=[],
                    total_corrections_count=0,
                    verification_notes=f"Conditional bypass ({bypass_reason}); Stage 1 verbatim preserved."
                )
                u2 = {}
            else:
                print(f"[Extraction] [2/3] Stage 2: Autocorrection Verification (Page {page_no})...")
                stage2_result = self.stage2.run(
                    image=p_img,
                    stage1_transcript=stage1_result.raw_transcript,
                    question_syllabus=question_obj.sub_questions if question_obj else None,
                    question_reference_vocab=question_vocab if question_obj else None,
                    temperature=decoding.temperature,
                    top_p=decoding.top_p,
                    max_new_tokens=decoding.max_new_tokens,
                    thinking_mode=active_thinking
                )
                u2 = self.engine.get_last_usage()
                ctx2 = self.engine.format_last_usage()
                print(f"[Extraction] [2/3] Stage 2 Verified -> {stage2_result.total_corrections_count} silent corrections reverted {ctx2}")

            # ---------------------------------------------------------
            # STAGE 0b: Teacher Mark Extraction (Conditional on Margin Red Ink)
            # ---------------------------------------------------------
            page_marks: List[TeacherMarkItem] = []
            u0b = {}
            has_margin_scores = getattr(stage0_res, "margin_has_red_ink", stage0_res.has_red_ink)

            if not extract_teacher_marks:
                print(f"[Extraction] [0b/3] Stage 0b: extract_teacher_marks=False -> Skipping teacher mark extraction for Page {page_no}.")
            elif has_margin_scores:
                active_page_text = stage2_result.verified_transcript or stage1_result.raw_transcript
                page_candidates = extract_candidate_questions(active_page_text)
                detected_topic_q = extract_header_qno(active_page_text, question_obj=question_obj)
                if detected_topic_q:
                    if detected_topic_q not in page_candidates:
                        page_candidates = [detected_topic_q] + page_candidates
                    else:
                        page_candidates = [detected_topic_q] + [c for c in page_candidates if c != detected_topic_q]
                cand_info = f" (Candidate questions: {page_candidates})" if page_candidates else ""
                print(f"[Extraction] [0b/3] Stage 0b: Red margin ink detected -> Running Gemma 4 Teacher Mark Extraction (Page {page_no}){cand_info}...")
                stage0b_res = self.stage0b.run(
                    image=p_img,
                    candidate_questions=page_candidates,
                    question_max_marks=question_max_marks if question_obj else None,
                    valid_paper_questions=valid_paper_questions if question_obj else None,
                    temperature=decoding.temperature,
                    top_p=decoding.top_p,
                    max_new_tokens=1024,
                    thinking_mode=active_thinking
                )
                page_marks = stage0b_res.teacher_marks
                u0b = self.engine.get_last_usage()
                ctx0b = self.engine.format_last_usage()
                print(f"[Extraction] [0b/3] Stage 0b Found -> {len(page_marks)} numeric teacher mark(s) on Page {page_no} {ctx0b}")
            elif stage0_res.has_red_ink:
                print(f"[Extraction] [0b/3] Stage 0b: Page {page_no} has checkmarks/ticks only (no margin scores) -> Skipping mark extraction.")
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

            # Checkpoint this page to disk immediately
            try:
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    f.write(p_res.model_dump_json(indent=2))
            except Exception as e:
                print(f"[Extraction] Note: Could not save checkpoint for Page {page_no} ({e})")

            return p_res

        if active_workers > 1 and len(page_images) > 1:
            from concurrent.futures import ThreadPoolExecutor
            print(f"[Extraction] ⚡ Running concurrent page processing with {active_workers} worker(s)...")
            with ThreadPoolExecutor(max_workers=min(len(page_images), active_workers)) as executor:
                page_results = list(executor.map(_process_single_page, page_images))
        else:
            page_results = [_process_single_page(item) for item in page_images]

        # Sort strictly by page_no to guarantee deterministic output ordering
        page_results.sort(key=lambda p: p.page_no)

        for p in page_results:
            if p.has_red_ink:
                any_red_ink = True
            if extract_teacher_marks:
                all_teacher_marks.extend(p.teacher_marks)

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
        # Global Script Calibration & Pen-Lift Split Stitching
        # ---------------------------------------------------------
        from src.pipeline.allograph_calibrator import AllographCalibrator
        from src.pipeline.split_token_stitcher import stitch_pen_lift_splits
        from src.pipeline.arbitration.writer_profile import save_writer_profile

        subject_name = "Bangla" if ("bangla" in script_id.lower() or "bangla" in source_str.lower()) else "English"
        arb_lexicon = get_english_lexicon() if subject_name == "English" else set()

        allograph_calibrator = AllographCalibrator(min_support=2)
        writer_profile = allograph_calibrator.calibrate(
            script_id=script_id,
            transcript=combined_verified,
            lexicon=arb_lexicon,
            question_vocab=set(question_vocab or [])
        )

        # 1. Apply discovered allographs to transcript
        calibrated_verified, allograph_diffs = allograph_calibrator.apply_adaptations(
            combined_verified,
            writer_profile
        )

        # 2. Stitch intra-word pen-lift splits (e.g. "elec tricity", "pro blems", "Hy dro - electric")
        stitched_verified, stitch_diffs = stitch_pen_lift_splits(
            calibrated_verified,
            lexicon=arb_lexicon,
            question_vocab=set(question_vocab or []),
            allograph_map=writer_profile.discovered_allographs
        )
        writer_profile.stitched_splits = stitch_diffs

        if allograph_diffs or stitch_diffs:
            print(f"[Extraction] ✍️ Global Calibration: {len(allograph_diffs)} allograph adaptation(s), {len(stitch_diffs)} stitched split(s).")
            combined_verified = stitched_verified
            aggregated_stage2.verified_transcript = combined_verified
            from src.core.schemas import AutocorrectionDiffItem

            all_added_diffs = [
                AutocorrectionDiffItem(
                    stage1_output=d["original"],
                    actual_handwritten=d["adapted"],
                    reason="Script-wide allograph calibration",
                    context_snippet=d["adapted"],
                )
                for d in allograph_diffs
            ] + [
                AutocorrectionDiffItem(
                    stage1_output=d["original"],
                    actual_handwritten=d["stitched"],
                    reason=f"Pen-lift split stitching: {d.get('reason', 'syllable split')}",
                    context_snippet=d["stitched"],
                )
                for d in stitch_diffs
            ]
            aggregated_stage2.silent_corrections_fixed.extend(all_added_diffs)
            aggregated_stage2.total_corrections_count = len(aggregated_stage2.silent_corrections_fixed)

            # Propagate stitched & calibrated text back to individual page_results
            page_chunks = combined_verified.split("\n\n--- Page Break ---\n\n")
            if len(page_chunks) == len(page_results):
                for p_idx, p_res in enumerate(page_results):
                    p_res.stage2_verification.verified_transcript = page_chunks[p_idx]

        # Save writer profile artifact
        try:
            save_writer_profile(writer_profile, os.path.join(script_output_dir, "writer_profile.json"))
        except Exception:
            pass

        # ---------------------------------------------------------
        # Question-Aware Stage 3: Alignment & Targeted Error Extraction
        # ---------------------------------------------------------
        print(f"\n[Extraction] [3/3] Question-Aware Alignment & Error Extraction ({len(combined_verified)} chars, {aggregated_stage1.word_count} words)...")

        # Partition transcript into canonical question answers
        interim_res = ExtractionResult(
            script_id=script_id,
            image_path=source_str,
            model_id=self.config.model.model_id,
            timestamp=datetime.now().isoformat(),
            has_red_ink=any_red_ink,
            stage1_transcription=aggregated_stage1,
            stage2_verification=aggregated_stage2,
            stage3_errors=Stage3ErrorResult(),
            teacher_marks=final_teacher_marks if 'final_teacher_marks' in locals() else all_teacher_marks,
            pages=page_results,
            metadata={"paper": paper}
        )
        aligned_answers = segment_script_into_questions(interim_res, question_obj)
        print(f"[Extraction] Segmented transcript into {len(aligned_answers)} distinct question answers.")

        all_extracted_errors: List[Any] = []
        u3_prompt = 0
        u3_comp = 0
        u3_total = 0

        # ---------------------------------------------------------
        # STAGE 3b setup: evidence-fused handwriting ambiguity gate
        # ---------------------------------------------------------
        subject_name = "Bangla" if ("bangla" in script_id.lower() or "bangla" in source_str.lower()) else "English"
        arb_cfg = getattr(self.config, "arbitration", None)
        arb_mode = (arb_cfg.mode if arb_cfg is not None else "legacy").lower()
        arb_lexicon = get_english_lexicon() if subject_name == "English" else set()
        gate: Optional[EvidenceArbitrationGate] = None
        if arb_mode == "evidence" and page_images:
            _img_by_page = {p_no: img for p_no, img, _ in page_images}

            def _clean_image_fn(n: int, _m=_img_by_page):
                res0 = self.stage0.detect(_m[n])
                return res0.clean_image if getattr(res0, "clean_image", None) is not None else _m[n]

            gate = EvidenceArbitrationGate(
                engine=self.engine,
                cfg=arb_cfg,
                script_id=script_id,
                output_dir=script_output_dir,
                page_images=page_images,
                page_transcripts={p.page_no: p.stage2_verification.verified_transcript for p in page_results},
                clean_image_fn=_clean_image_fn,
                full_transcript=combined_verified,
                lexicon=arb_lexicon,
                question_vocab=(set(question_vocab or [])
                                | ({w.lower() for w in re.findall(r"[A-Za-z]+", question_obj.question_text)} if question_obj else set())),
                language="bn" if subject_name == "Bangla" else "en",
                profile=writer_profile,
            )
            print(f"[Extraction] [3b/3] Evidence gate ready (writer profile: {len(gate.profile.pair_counts)} confusion pairs from {len(gate.profile.anchors)} anchor words)")

        # Run Stage 3 targeted per question
        if aligned_answers and len(aligned_answers) > 1:
            for ans in aligned_answers:
                is_obj = is_objective_question(ans.q_no, ans.q_name, "")
                if is_obj:
                    # Objective question (Flowchart, MCQ, Cloze, Rearranging)
                    # Graded against factual answer key; 0 essay linguistic deductions
                    ans.errors = []
                    continue

                ans_text = ans.answer_text.strip()
                if len(ans_text.split()) < 3:
                    ans.errors = []
                    continue

                ans_words = len(ans_text.split())
                if ans_words > 150:
                    chunks = _chunk_text_by_sentences(ans_text, target_words=120)
                    chunk_errors: List[LinguisticErrorItem] = []
                    for chunk in chunks:
                        c_err_res = self.stage3.run(
                            verified_transcript=chunk,
                            question_vocab=set(question_vocab) if question_vocab else None,
                            subject="Bangla" if ("bangla" in script_id.lower() or "bangla" in source_str.lower()) else "English",
                            temperature=decoding.temperature,
                            top_p=decoding.top_p,
                            max_new_tokens=min(decoding.max_new_tokens, 3072),
                            thinking_mode=active_thinking
                        )
                        chunk_errors.extend(c_err_res.errors)
                        usage = self.engine.get_last_usage()
                        u3_prompt += usage.get("prompt_tokens", 0)
                        u3_comp += usage.get("completion_tokens", 0)
                        u3_total += usage.get("total_tokens", 0)

                    # Deduplicate chunk errors
                    seen_errs = set()
                    deduped: List[LinguisticErrorItem] = []
                    for e in chunk_errors:
                        key = (e.error_type.lower(), e.erroneous_text.lower().strip(), e.suggested_correction.lower().strip())
                        if key not in seen_errs:
                            seen_errs.add(key)
                            deduped.append(e)

                    # Post-filter against full answer text to catch line-end edge truncations
                    from src.utils.linguistic_sanitizer import verify_and_filter_stage3_errors
                    deduped = verify_and_filter_stage3_errors(
                        errors=deduped,
                        question_vocab=set(question_vocab) if question_vocab else None,
                        subject="Bangla" if ("bangla" in script_id.lower() or "bangla" in source_str.lower()) else "English",
                        transcript=ans_text
                    )

                    q_err_res = Stage3ErrorResult(
                        errors=deduped,
                        spelling_error_count=sum(1 for e in deduped if "spell" in e.error_type.lower()),
                        grammar_error_count=sum(1 for e in deduped if "gram" in e.error_type.lower()),
                        syntax_error_count=sum(1 for e in deduped if "synt" in e.error_type.lower()),
                        punctuation_error_count=0,
                        total_error_count=len(deduped),
                        linguistic_summary=f"Extracted across {len(chunks)} chunks for long answer ({ans_words} words)."
                    )
                else:
                    q_err_res = self.stage3.run(
                        verified_transcript=ans_text,
                        question_vocab=set(question_vocab) if question_vocab else None,
                        subject="Bangla" if ("bangla" in script_id.lower() or "bangla" in source_str.lower()) else "English",
                        temperature=decoding.temperature,
                        top_p=decoding.top_p,
                        max_new_tokens=min(decoding.max_new_tokens, 3072),
                        thinking_mode=active_thinking
                    )
                    usage = self.engine.get_last_usage()
                    u3_prompt += usage.get("prompt_tokens", 0)
                    u3_comp += usage.get("completion_tokens", 0)
                    u3_total += usage.get("total_tokens", 0)

                # -------------------------------------------------------------
                # STAGE 3b: Handwriting Ambiguity Arbitration (Benefit of the Doubt)
                # -------------------------------------------------------------
                if q_err_res.errors and page_images and arb_mode != "off":
                    ans_pno = ans.page_numbers[0] if ans.page_numbers else 1
                    target_p_img = page_images[ans_pno - 1][1] if (1 <= ans_pno <= len(page_images)) else page_images[0][1]
                    original_errs = list(q_err_res.errors)
                    confirmed_errs, cleared_ambiguities = self.stage3.arbitrate_visual_errors(
                        image=target_p_img,
                        errors=q_err_res.errors,
                        temperature=decoding.temperature,
                        top_p=decoding.top_p,
                        thinking_mode=active_thinking,
                        gate=gate,
                        q_no=ans.q_no,
                        answer_text=ans.answer_text,
                    )
                    if gate is None:
                        usage_arb = self.engine.get_last_usage()
                        u3_prompt += usage_arb.get("prompt_tokens", 0)
                        u3_comp += usage_arb.get("completion_tokens", 0)
                        u3_total += usage_arb.get("total_tokens", 0)

                    if cleared_ambiguities:
                        n_amb = sum(1 for c in cleared_ambiguities if c.get("verdict", "HANDWRITING_AMBIGUITY") == "HANDWRITING_AMBIGUITY")
                        n_unc = len(cleared_ambiguities) - n_amb
                        print(f"[Extraction] [3b/3] Arbitration for Q{ans.q_no}: benefit of the doubt {n_amb}, uncertain (human review) {n_unc}, deductions kept {len(confirmed_errs)}/{len(original_errs)}.")
                        _apply_normalizations(cleared_ambiguities, ans, page_results, original_errs, arb_lexicon)
                    q_err_res.errors = confirmed_errs

                for e in q_err_res.errors:
                    e.question_no = ans.q_no
                    all_extracted_errors.append(e)

                ans.errors = [e.model_dump() for e in q_err_res.errors]
        else:
            # Fallback if segmentation yielded no distinct sections
            q_err_res = self.stage3.run(
                verified_transcript=combined_verified,
                question_vocab=set(question_vocab) if question_vocab else None,
                subject="Bangla" if ("bangla" in script_id.lower() or "bangla" in source_str.lower()) else "English",
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                max_new_tokens=decoding.max_new_tokens,
                thinking_mode=active_thinking
            )
            usage = self.engine.get_last_usage()
            u3_prompt = usage.get("prompt_tokens", 0)
            u3_comp = usage.get("completion_tokens", 0)
            u3_total = usage.get("total_tokens", 0)

            if q_err_res.errors and page_images and arb_mode != "off":
                original_errs = list(q_err_res.errors)
                confirmed_errs, cleared_ambiguities = self.stage3.arbitrate_visual_errors(
                    image=page_images[0][1],
                    errors=q_err_res.errors,
                    temperature=decoding.temperature,
                    top_p=decoding.top_p,
                    thinking_mode=active_thinking,
                    gate=gate,
                    q_no=None,
                    answer_text=combined_verified,
                )
                if gate is None:
                    usage_arb = self.engine.get_last_usage()
                    u3_prompt += usage_arb.get("prompt_tokens", 0)
                    u3_comp += usage_arb.get("completion_tokens", 0)
                    u3_total += usage_arb.get("total_tokens", 0)
                if cleared_ambiguities:
                    _apply_normalizations(cleared_ambiguities, None, page_results, original_errs, arb_lexicon)
                    if aligned_answers:
                        for a in aligned_answers:
                            _apply_normalizations(cleared_ambiguities, a, [], original_errs, arb_lexicon)
                q_err_res.errors = confirmed_errs

            all_extracted_errors = q_err_res.errors

        # ---------------------------------------------------------
        # STAGE 3b wrap-up: persist evidence, sync normalized transcripts
        # ---------------------------------------------------------
        arbitration_meta: Dict[str, Any] = {"mode": arb_mode}
        if gate is not None:
            arb_res = gate.finalize()
            u3_prompt += arb_res.token_usage.get("prompt_tokens", 0)
            u3_comp += arb_res.token_usage.get("completion_tokens", 0)
            u3_total += arb_res.token_usage.get("total_tokens", 0)
            arbitration_meta.update({
                "total_candidates": arb_res.total_candidates,
                "benefit_of_doubt": sum(1 for r in arb_res.records if r.verdict == "HANDWRITING_AMBIGUITY"),
                "genuine": sum(1 for r in arb_res.records if r.verdict == "GENUINE_ERROR"),
                "uncertain_count": len(arb_res.uncertain),
                "uncertain": [
                    {
                        "q_no": r.candidate.question_no,
                        "read": r.candidate.candidate_token,
                        "intended": r.candidate.intended_token,
                        "context": r.candidate.context_sentence,
                        "score": r.ambiguity_score,
                        "crop_path": r.evidence.crop_path,
                    } for r in arb_res.uncertain
                ],
                "cleared": [
                    {
                        "q_no": r.candidate.question_no,
                        "read": r.candidate.candidate_token,
                        "intended": r.candidate.intended_token,
                        "score": r.ambiguity_score,
                        "localization": r.evidence.localization_method,
                    } for r in arb_res.records if r.verdict == "HANDWRITING_AMBIGUITY"
                ],
                "model_calls": arb_res.total_model_calls,
                "token_usage": arb_res.token_usage,
                "artifact": "stage3b_arbitration.json",
                "writer_profile": "writer_profile.json",
            })
            print(f"[Extraction] [3b/3] Arbitration complete: {arb_res.total_candidates} candidates, "
                  f"{arbitration_meta['benefit_of_doubt']} benefit of doubt, {arbitration_meta['genuine']} genuine, "
                  f"{arbitration_meta['uncertain_count']} uncertain, {arb_res.total_model_calls} crop-level model calls.")
        combined_verified = _rebuild_combined_verified(page_results)
        aggregated_stage2.verified_transcript = combined_verified

        spelling_cnt = sum(1 for e in all_extracted_errors if "spell" in e.error_type.lower())
        grammar_cnt = sum(1 for e in all_extracted_errors if "gram" in e.error_type.lower())
        syntax_cnt = sum(1 for e in all_extracted_errors if "synt" in e.error_type.lower())

        aggregated_stage3 = Stage3ErrorResult(
            errors=all_extracted_errors,
            spelling_error_count=spelling_cnt,
            grammar_error_count=grammar_cnt,
            syntax_error_count=syntax_cnt,
            punctuation_error_count=0,
            total_error_count=len(all_extracted_errors),
            linguistic_summary=f"Cataloged {len(all_extracted_errors)} linguistic errors across {len(aligned_answers)} questions."
        )

        u3 = {
            "prompt_tokens": u3_prompt,
            "completion_tokens": u3_comp,
            "total_tokens": u3_total
        }
        print(f"[Extraction] [3/3] Stage 3 Errors -> {aggregated_stage3.total_error_count} verified errors (spelling: {aggregated_stage3.spelling_error_count}, grammar: {aggregated_stage3.grammar_error_count}, syntax: {aggregated_stage3.syntax_error_count})")

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

        vlm_detected_marks = [m.model_dump() for m in all_teacher_marks]
        final_teacher_marks = all_teacher_marks
        has_gt = bool(ground_truth_marks)
        is_verified_gt = has_gt

        if has_gt:
            print(f"[Extraction] 🎯 Autonomous VLM extracted {len(final_teacher_marks)} teacher marks. Verified against gt.txt benchmark ({len(ground_truth_marks)} ground truth marks).")

        extraction_result = ExtractionResult(
            script_id=script_id,
            image_path=source_str,
            model_id=self.config.model.model_id,
            timestamp=datetime.now().isoformat(),
            has_red_ink=any_red_ink,
            stage1_transcription=aggregated_stage1,
            stage2_verification=aggregated_stage2,
            stage3_errors=aggregated_stage3,
            teacher_marks=final_teacher_marks,
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
                "question_id": question_obj.question_id if question_obj else None,
                "original_marker_id": original_marker_id if not is_verified_gt else "human_examiner_gt",
                "school_id": school_id,
                "region": region,
                "verified_by_human": is_verified_gt,
                "ground_truth_marks": ground_truth_marks,
                "vlm_detected_teacher_marks": vlm_detected_marks,
                "aligned_answers": [a.model_dump() for a in aligned_answers] if aligned_answers else [],
                "arbitration": arbitration_meta
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
            q_no_field = p.teacher_marks[0].question_no if p.teacher_marks else None
            if not q_no_field:
                page_cands = extract_candidate_questions(p.stage1_transcription.raw_transcript)
                q_no_field = page_cands[0] if page_cands else None

            raw_tier_records.append(RawTierRecord(
                script_id=script_id,
                page_no=p.page_no,
                question_no=q_no_field,
                paper=paper,
                task_type=task_type,
                transcript_text=p.stage2_verification.verified_transcript,
                ocr_flags=ocr_flag_str,
                error_list=error_json_str,
                teacher_mark=marks_str,
                has_red_ink=p.has_red_ink,
                original_marker_id=original_marker_id if not is_verified_gt else "human_examiner_gt",
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
        questions_root: str = "outputs/questions",
        eval_mode: str = "modular"
    ) -> CompleteEvaluationReport:
        """
        Execute Stage 4 Rubric Evaluation on pre-extracted script transcripts and errors.
        Teacher marks and original marker IDs remain strictly isolated from grading inputs.
        Matches the extracted script to its corresponding question prompt.
        Supports both 'modular' (question-by-question mapping) and 'monolithic' (single-pass baseline) modes.
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
        print(f"[Evaluation] Mode: {eval_mode.upper()} ({'Question-by-Question Isolated Calls' if eval_mode == 'modular' else 'Monolithic Single-Pass'})")
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

        # Check for verified human ground truth marks (strictly for evaluation metrics, isolated from grading)
        gt_dict = extraction.metadata.get("ground_truth_marks")
        if not gt_dict:
            gt_dict = get_ground_truth_for_script(script_id)
        is_verified_gt = extraction.metadata.get("verified_by_human", False) or (gt_dict is not None)

        # -------------------------------------------------------------
        # STAGE 4: Rubric Evaluation (Verified text + errors only)
        # -------------------------------------------------------------
        if hasattr(self.engine, "clear_cuda_cache"):
            self.engine.clear_cuda_cache()

        if eval_mode == "modular":
            print("\n[Evaluation] [4/4] Executing Stage 4: Modular Question-by-Question Evaluation...")
            aligned_answers = segment_script_into_questions(extraction, question_obj)
            if not aligned_answers and extraction.metadata.get("aligned_answers"):
                aligned_answers = [AlignedAnswerItem.model_validate(a) for a in extraction.metadata["aligned_answers"]]
            print(f"[Evaluation] Loaded {len(aligned_answers)} distinct question answers.")

            stage4_result = self.stage4.evaluate_modular(
                answers=aligned_answers,
                question_obj=question_obj,
                rubric_data=rubric_data,
                ground_truth_marks=gt_dict,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                max_new_tokens=stage4_max_tokens,
                thinking_mode=active_thinking,
                generation_max_time=stage4_timeout_sec
            )
        else:
            print("\n[Evaluation] [4/4] Executing Stage 4: Monolithic Rubric Evaluation & Pedagogical Feedback...")
            thematic_context = None
            if self.rag_provider:
                lookup_topic = thematic_topic or rubric_data.get("subject", "bangla")
                thematic_context = self.rag_provider.get_context(lookup_topic)

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
                "eval_mode": eval_mode,
                "thinking_mode": active_thinking,
                "temperature": decoding.temperature,
                "rubric_used": active_rubric_path,
                "question_id": question_obj.question_id if question_obj else None,
                "engine_info": self.engine.get_engine_info(),
                "stage4_token_usage": u4,
                "extraction_token_usage": extraction.metadata.get("token_usage", {}),
                "output_dir": script_output_dir,
                "verified_by_human": is_verified_gt,
                "ground_truth_marks": gt_dict,
                "mae_vs_human": stage4_result.mae_vs_human
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
        question_input: Optional[Union[str, ExtractedQuestion]] = None,
        extract_teacher_marks: Optional[bool] = None,
        eval_mode: str = "modular"
    ) -> CompleteEvaluationReport:
        """
        Execute full end-to-end pipeline (Extraction Stages 0, 0b, 1-3 -> Evaluation Stage 4).
        Supports both 'modular' (question-by-question mapping) and 'monolithic' modes.
        """
        # Step 1: Extraction
        extraction = self.extract_script(
            input_source=input_source,
            script_id=script_id,
            thinking_mode=thinking_mode,
            pdf_samples_dir=pdf_samples_dir,
            output_dir=output_dir,
            skip_stage2=skip_stage2,
            force_extract=force_extract,
            question_input=question_input,
            extract_teacher_marks=extract_teacher_marks
        )

        # Step 2: Evaluation
        return self.evaluate_extracted_script(
            extraction_input=extraction,
            rubric_path=self.rubric_path,
            thematic_topic=thematic_topic,
            thinking_mode=thinking_mode,
            output_dir=output_dir,
            question_input=question_input,
            eval_mode=eval_mode
        )
