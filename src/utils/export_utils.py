import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from src.core.schemas import (
    Stage1TranscriptionResult,
    Stage2VerificationResult,
    Stage3ErrorResult,
    Stage4EvaluationResult,
    TeacherMarkItem,
    PageExtractionResult,
    ExtractionResult,
    RawTierRecord,
    CompleteEvaluationReport
)


def export_stage1_artifacts(result: Stage1TranscriptionResult, script_dir: str):
    """Save Stage 1 transcription outputs."""
    os.makedirs(script_dir, exist_ok=True)
    json_path = os.path.join(script_dir, "stage1_transcription.json")
    txt_path = os.path.join(script_dir, "stage1_raw_transcript.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result.raw_transcript)


def export_stage2_artifacts(result: Stage2VerificationResult, script_dir: str):
    """Save Stage 2 verification and autocorrection audit outputs."""
    os.makedirs(script_dir, exist_ok=True)
    json_path = os.path.join(script_dir, "stage2_verification.json")
    txt_path = os.path.join(script_dir, "stage2_verified_transcript.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result.verified_transcript)


def export_stage3_artifacts(result: Stage3ErrorResult, script_dir: str):
    """Save Stage 3 linguistic error extraction outputs (JSON + CSV)."""
    os.makedirs(script_dir, exist_ok=True)
    json_path = os.path.join(script_dir, "stage3_errors.json")
    csv_path = os.path.join(script_dir, "stage3_errors.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["error_type", "erroneous_text", "suggested_correction", "context_sentence", "explanation"])
        for err in result.errors:
            writer.writerow([err.error_type, err.erroneous_text, err.suggested_correction, err.context_sentence, err.explanation])


def export_teacher_marks_artifacts(marks: List[TeacherMarkItem], script_dir: str):
    """Save Stage 0b extracted teacher marks."""
    os.makedirs(script_dir, exist_ok=True)
    json_path = os.path.join(script_dir, "stage0b_teacher_marks.json")
    data = [m.model_dump() for m in marks]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_stage4_artifacts(result: Stage4EvaluationResult, script_dir: str):
    """Save Stage 4 rubric scoring outputs."""
    os.makedirs(script_dir, exist_ok=True)
    json_path = os.path.join(script_dir, "stage4_evaluation.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)


STATUS_BADGES = {
    "scored": "Scored",
    "unscored": "⚠️ Unscored",
    "blank_answer": "Blank answer",
    "missing_no_segment": "⚠️ No answer found",
    "missing": "⚠️ Missing",  # evaluations written before the two cases were separated
}


def save_segmentation_diagnostics(diagnostics: Dict[str, Any], answers: List[Any], script_dir: str) -> str:
    """Save how the transcript was split into questions: every part with its page and the rule that placed it,
    each header correction or content recovery, questions left without text, and unattributed Stage 3 errors."""
    os.makedirs(script_dir, exist_ok=True)
    path = os.path.join(script_dir, "segmentation.json")
    data = dict(diagnostics or {})
    data["answers"] = [{"q_no": a.q_no, "words": a.word_count, "pages": list(a.page_numbers)} for a in answers or []]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path


def export_extraction_artifacts(result: ExtractionResult, script_dir: str):
    """Save all extraction stage artifacts (Stage 0, 0b, 1 to 3, JSON, CSV, transcripts, and summary)."""
    os.makedirs(script_dir, exist_ok=True)

    # 1. Save individual stage outputs
    export_stage1_artifacts(result.stage1_transcription, script_dir)
    export_stage2_artifacts(result.stage2_verification, script_dir)
    export_stage3_artifacts(result.stage3_errors, script_dir)
    export_teacher_marks_artifacts(result.teacher_marks, script_dir)

    # 2. Save consolidated extraction JSON
    json_path = os.path.join(script_dir, "extraction_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    # 3. Save readable extraction summary Markdown
    summary_md_path = os.path.join(script_dir, "extraction_summary.md")
    export_extraction_summary_markdown(result, summary_md_path)


def export_extraction_summary_markdown(result: ExtractionResult, output_path: str) -> str:
    """Save human-readable Markdown summary of the extraction phase."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    md_lines = [
        f"# Script Extraction Report: `{result.script_id}`",
        f"**Extracted At**: {result.timestamp} | **Engine**: `{result.model_id}`",
        f"**Input Source**: `{result.image_path}`",
        "",
        "---",
        "",
        "## Extraction Overview",
        f"- **Stage 0 (Red Ink Detected)**: `{'Yes' if result.has_red_ink else 'No'}`",
        f"- **Transcribed Words**: {result.stage1_transcription.word_count}",
        f"- **OCR Markers**: Illegible: {result.stage1_transcription.illegible_count}, Unclear: {result.stage1_transcription.unclear_count}, Struck: {result.stage1_transcription.struck_count}",
        f"- **Silent Corrections Reverted (Stage 2)**: {result.stage2_verification.total_corrections_count}",
        f"- **Total Linguistic Errors Found (Stage 3)**: {result.stage3_errors.total_error_count}",
        f"  - Spelling: {result.stage3_errors.spelling_error_count}",
        f"  - Grammar: {result.stage3_errors.grammar_error_count}",
        f"  - Syntax: {result.stage3_errors.syntax_error_count}",
        f"- **Teacher Marks Extracted (Stage 0b)**: {len(result.teacher_marks)} mark(s)",
        "",
        "---",
        "",
        "## Verified Canonical Transcript (Handwriting Preserved)",
        "```text",
        result.stage2_verification.verified_transcript,
        "```",
        ""
    ]

    if result.teacher_marks:
        md_lines.extend([
            "### Stage 0b Extracted Red-Ink Teacher Marks",
            "| Question No | Mark Value | Location |",
            "| --- | --- | --- |"
        ])
        for tm in result.teacher_marks:
            md_lines.append(f"| {tm.question_no or 'N/A'} | `{tm.mark_value}` | {tm.location} |")
        md_lines.append("")

    if result.stage2_verification.silent_corrections_fixed:
        md_lines.extend([
            "### Silent Corrections Audited & Reverted",
            "| Stage 1 Output | Actual Handwriting | Reason |",
            "| --- | --- | --- |"
        ])
        for diff in result.stage2_verification.silent_corrections_fixed:
            if hasattr(diff, "stage1_output"):
                s1, act, rsn = diff.stage1_output, diff.actual_handwritten, diff.reason
            elif isinstance(diff, dict):
                s1, act, rsn = diff.get("stage1_output", ""), diff.get("actual_handwritten", ""), diff.get("reason", "")
            elif isinstance(diff, str) and "->" in diff:
                parts = diff.split("->", 1)
                s1, act, rsn = parts[0].strip(), parts[1].strip(), "Global handwriting calibration"
            else:
                s1, act, rsn = str(diff), "", "Global handwriting calibration"
            md_lines.append(f"| `{s1}` | `{act}` | {rsn} |")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## Cataloged Linguistic & Structural Errors (Question-Aligned)",
        f"> {result.stage3_errors.linguistic_summary}",
        ""
    ])

    aligned_answers = result.metadata.get("aligned_answers", [])
    if aligned_answers:
        from src.prompts.stage4_modular import is_objective_question
        for a in aligned_answers:
            q_no = a.get("q_no", "")
            q_name = a.get("q_name", f"Question {q_no}")
            q_errors = a.get("errors", [])
            is_obj = is_objective_question(q_no, q_name, "")

            if is_obj:
                md_lines.extend([
                    f"### Question {q_no}: {q_name} `[Objective]`",
                    "> *Objective Question: Student answer is evaluated on factual content against the answer key. Subjective grammar/syntax deductions are bypassed, but orthographic spelling in fill-in-the-blank tokens is audited.*",
                    ""
                ])
                if q_errors:
                    md_lines.extend([
                        "| Type | Erroneous Text | Suggested Correction | Context Sentence | Explanation |",
                        "| --- | --- | --- | --- | --- |"
                    ])
                    for err in q_errors:
                        etype = err.get("error_type", "spelling")
                        etext = err.get("erroneous_text", "")
                        ecorr = err.get("suggested_correction", "")
                        ectx = err.get("context_sentence", "")
                        eexpl = err.get("explanation", "")
                        md_lines.append(f"| **{etype}** | `{etext}` | `{ecorr}` | *\"{ectx}\"* | {eexpl} |")
            else:
                md_lines.append(f"### Question {q_no}: {q_name} `[Subjective Writing]`")
                if q_errors:
                    md_lines.extend([
                        "| Type | Erroneous Text | Suggested Correction | Context Sentence | Explanation |",
                        "| --- | --- | --- | --- | --- |"
                    ])
                    for err in q_errors:
                        etype = err.get("error_type", "grammar")
                        etext = err.get("erroneous_text", "")
                        ecorr = err.get("suggested_correction", "")
                        ectx = err.get("context_sentence", "")
                        eexpl = err.get("explanation", "")
                        md_lines.append(f"| **{etype}** | `{etext}` | `{ecorr}` | *\"{ectx}\"* | {eexpl} |")
                    md_lines.append("")
                else:
                    md_lines.extend([
                        "> *No linguistic errors detected in this answer.*",
                        ""
                    ])
    elif result.stage3_errors.errors:
        error_page_map = {}
        if result.pages:
            for p in result.pages:
                for e in (p.stage3_errors.errors if p.stage3_errors else []):
                    error_page_map[(e.erroneous_text, e.context_sentence)] = p.page_no

        md_lines.extend([
            "| Page | Type | Erroneous Text | Suggested Correction | Explanation |",
            "| --- | --- | --- | --- | --- |"
        ])
        for err in result.stage3_errors.errors:
            p_no = error_page_map.get((err.erroneous_text, err.context_sentence))
            if p_no is None and result.pages:
                needle = err.erroneous_text.lower().strip()
                ctx = err.context_sentence.lower().strip()
                for p in result.pages:
                    p_text = (p.stage2_verification.verified_transcript or p.stage1_transcription.raw_transcript or "").lower()
                    if needle and needle in p_text:
                        p_no = p.page_no
                        break
                    elif ctx and len(ctx) > 8 and ctx[:20] in p_text:
                        p_no = p.page_no
                        break
            page_str = f"Page {p_no}" if p_no is not None else "Page 1"
            md_lines.append(f"| `{page_str}` | **{err.error_type}** | `{err.erroneous_text}` | `{err.suggested_correction}` | {err.explanation} |")
        md_lines.append("")

    arb = result.metadata.get("arbitration") or {}
    if arb:
        md_lines.extend([
            "---",
            "",
            "## Stage 3b Handwriting Ambiguity Arbitration",
            f"- **Mode**: `{arb.get('mode', 'legacy')}`",
        ])
        if "total_candidates" in arb:
            md_lines.extend([
                f"- **Candidates gated**: {arb.get('total_candidates', 0)} | **Benefit of the doubt**: {arb.get('benefit_of_doubt', 0)} | "
                f"**Genuine**: {arb.get('genuine', 0)} | **Uncertain**: {arb.get('uncertain_count', 0)}",
                f"- **Crop-level model calls**: {arb.get('model_calls', 0)} | **Evidence file**: `{arb.get('artifact', '')}` | **Writer profile**: `{arb.get('writer_profile', '')}`",
                "",
            ])
            cleared = arb.get("cleared") or []
            if cleared:
                md_lines.extend([
                    "### Benefit of the Doubt Granted (normalized in transcript)",
                    "| Q | Read as | Intended | Ambiguity score | Localization |",
                    "| --- | --- | --- | --- | --- |",
                ])
                for c in cleared:
                    md_lines.append(f"| {c.get('q_no') or '-'} | `{c.get('read')}` | `{c.get('intended')}` | {c.get('score', 0):.2f} | {c.get('localization', '-')} |")
                md_lines.append("")
            uncertain = arb.get("uncertain") or []
            if uncertain:
                md_lines.extend([
                    "### Flagged for Human Review (UNCERTAIN, no deduction applied)",
                    "| Q | Read as | Intended | Ambiguity score | Context | Crop |",
                    "| --- | --- | --- | --- | --- | --- |",
                ])
                for u in uncertain:
                    ctx = (u.get("context") or "")[:80]
                    md_lines.append(f"| {u.get('q_no') or '-'} | `{u.get('read')}` | `{u.get('intended')}` | {u.get('score', 0):.2f} | *\"{ctx}\"* | `{u.get('crop_path') or '-'}` |")
                md_lines.append("")
        else:
            md_lines.append("")

    tok_usage = result.metadata.get("token_usage", {})
    if tok_usage:
        max_ctx = tok_usage.get("max_context_window", 4096)
        tot_all = tok_usage.get("total_tokens", 0)
        tot_pct = tok_usage.get("pct_context", (tot_all / max_ctx * 100) if max_ctx > 0 else 0)
        tot_in = tok_usage.get("total_prompt_tokens", 0)
        tot_out = tok_usage.get("total_completion_tokens", 0)

        md_lines.extend([
            "---",
            "",
            "## Context & Token Usage Breakdown",
            f"- **Context Window Limit**: `{max_ctx:,}` tokens",
            f"- **Cumulative Extraction Tokens**: **{tot_all:,}** ({tot_pct:.1f}% of window)",
            f"- **Prompt Tokens (In)**: {tot_in:,} | **Completion Tokens (Out)**: {tot_out:,}",
            "",
            "| Component / Stage | Prompt (In) | Completion (Out) | Total Tokens | Context Window | Usage % |",
            "| --- | --- | --- | --- | --- | --- |"
        ])
        pages_dict = tok_usage.get("pages", {})
        for page_key, p_data in pages_dict.items():
            p_label = page_key.replace("page_", "Page ")
            s1 = p_data.get("stage1", {})
            if s1 and s1.get("total_tokens"):
                p1_pct = (s1["total_tokens"] / max_ctx * 100) if max_ctx > 0 else 0
                md_lines.append(f"| {p_label}: Stage 1 (Verbatim OCR) | {s1.get('prompt_tokens', 0):,} | {s1.get('completion_tokens', 0):,} | {s1.get('total_tokens', 0):,} | {max_ctx:,} | {p1_pct:.1f}% |")
            s2 = p_data.get("stage2", {})
            if s2 and s2.get("total_tokens"):
                p2_pct = (s2["total_tokens"] / max_ctx * 100) if max_ctx > 0 else 0
                md_lines.append(f"| {p_label}: Stage 2 (Verification) | {s2.get('prompt_tokens', 0):,} | {s2.get('completion_tokens', 0):,} | {s2.get('total_tokens', 0):,} | {max_ctx:,} | {p2_pct:.1f}% |")
            s0b = p_data.get("stage0b", {})
            if s0b and s0b.get("total_tokens"):
                p0b_pct = (s0b["total_tokens"] / max_ctx * 100) if max_ctx > 0 else 0
                md_lines.append(f"| {p_label}: Stage 0b (Teacher Marks) | {s0b.get('prompt_tokens', 0):,} | {s0b.get('completion_tokens', 0):,} | {s0b.get('total_tokens', 0):,} | {max_ctx:,} | {p0b_pct:.1f}% |")

        s3 = tok_usage.get("stage3", {})
        if s3 and s3.get("total_tokens"):
            p3_pct = (s3["total_tokens"] / max_ctx * 100) if max_ctx > 0 else 0
            md_lines.append(f"| Script: Stage 3 (Linguistic Errors) | {s3.get('prompt_tokens', 0):,} | {s3.get('completion_tokens', 0):,} | {s3.get('total_tokens', 0):,} | {max_ctx:,} | {p3_pct:.1f}% |")

        md_lines.extend([
            f"| **Cumulative Extraction Context** | **{tot_in:,}** | **{tot_out:,}** | **{tot_all:,}** | **{max_ctx:,}** | **{tot_pct:.1f}%** |",
            ""
        ])

    content = "\n".join(md_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


def export_raw_tier_csv(
    records: List[RawTierRecord],
    output_path: str,
    overwrite: bool = False
) -> str:
    """
    Save the standardized 13-column Raw-Tier dataset CSV for research and analysis:
    Columns: script_id, page_no, question_no, paper, task_type, transcript_text,
             ocr_flags, error_list, teacher_mark, has_red_ink, original_marker_id, school_id, region

    If overwrite is True, overwrites the output file.
    If writing to an existing shared file, replaces/upserts rows with matching (script_id, page_no, question_no)
    to prevent duplicate accumulations across re-runs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    fieldnames = [
        "script_id",
        "page_no",
        "question_no",
        "paper",
        "task_type",
        "transcript_text",
        "ocr_flags",
        "error_list",
        "teacher_mark",
        "has_red_ink",
        "original_marker_id",
        "school_id",
        "region"
    ]

    new_dict_records = [r.model_dump() for r in records]

    if overwrite or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in new_dict_records:
                writer.writerow(r)
        return output_path

    # Upsert into existing CSV: read existing rows and replace matching (script_id, page_no, question_no)
    existing_rows = []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
    except Exception:
        existing_rows = []

    incoming_keys = {(r.get("script_id"), str(r.get("page_no")), str(r.get("question_no"))) for r in new_dict_records}
    retained_rows = [
        row for row in existing_rows
        if (row.get("script_id"), str(row.get("page_no")), str(row.get("question_no"))) not in incoming_keys
    ]

    all_rows = retained_rows + new_dict_records

    def _csv_sort_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
        s_id = str(row.get("script_id") or "")
        p_no = int(row.get("page_no") or 0) if str(row.get("page_no", "0")).isdigit() else 0
        q_no = str(row.get("question_no") or "")
        return (s_id, p_no, q_no)

    all_rows.sort(key=_csv_sort_key)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    return output_path


def load_extraction_artifacts(target_path: Union[str, Path]) -> ExtractionResult:
    """
    Load ExtractionResult from a directory path, extraction_result.json, or complete_report.json.
    """
    target = Path(target_path)

    if target.is_dir():
        extraction_json = target / "extraction_result.json"
        complete_json = target / "complete_report.json"
        stage1_json = target / "stage1_transcription.json"
        stage2_json = target / "stage2_verification.json"
        stage3_json = target / "stage3_errors.json"
        marks_json = target / "stage0b_teacher_marks.json"

        if extraction_json.exists():
            with open(extraction_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ExtractionResult(**data)

        if complete_json.exists():
            with open(complete_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ExtractionResult(
                script_id=data.get("script_id", target.name),
                image_path=data.get("image_path", ""),
                model_id=data.get("model_id", "google/gemma-4-31b-it"),
                timestamp=data.get("timestamp", ""),
                has_red_ink=data.get("has_red_ink", False),
                stage1_transcription=data["stage1_transcription"],
                stage2_verification=data["stage2_verification"],
                stage3_errors=data["stage3_errors"],
                teacher_marks=data.get("teacher_marks", []),
                metadata=data.get("metadata", {})
            )

        if stage1_json.exists() and stage2_json.exists() and stage3_json.exists():
            with open(stage1_json, "r", encoding="utf-8") as f1, \
                 open(stage2_json, "r", encoding="utf-8") as f2, \
                 open(stage3_json, "r", encoding="utf-8") as f3:
                s1 = json.load(f1)
                s2 = json.load(f2)
                s3 = json.load(f3)

            teacher_marks = []
            if marks_json.exists():
                with open(marks_json, "r", encoding="utf-8") as fm:
                    marks_data = json.load(fm)
                    teacher_marks = [TeacherMarkItem(**m) for m in marks_data]

            return ExtractionResult(
                script_id=target.name,
                image_path="",
                model_id="google/gemma-4-31b-it",
                timestamp="",
                has_red_ink=len(teacher_marks) > 0,
                stage1_transcription=s1,
                stage2_verification=s2,
                stage3_errors=s3,
                teacher_marks=teacher_marks,
                metadata={"loaded_from_stage_jsons": str(target)}
            )

        raise FileNotFoundError(
            f"Could not find extraction artifacts in directory '{target}'. Expected 'extraction_result.json' or stage 1-3 JSONs."
        )

    elif target.is_file():
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "stage1_transcription" in data and "stage2_verification" in data and "stage3_errors" in data:
            return ExtractionResult(
                script_id=data.get("script_id", target.stem),
                image_path=data.get("image_path", ""),
                model_id=data.get("model_id", "google/gemma-4-31b-it"),
                timestamp=data.get("timestamp", ""),
                has_red_ink=data.get("has_red_ink", False),
                stage1_transcription=data["stage1_transcription"],
                stage2_verification=data["stage2_verification"],
                stage3_errors=data["stage3_errors"],
                teacher_marks=data.get("teacher_marks", []),
                metadata=data.get("metadata", {})
            )
        raise ValueError(f"JSON file '{target}' does not contain expected extraction stages.")

    raise FileNotFoundError(f"Target path does not exist: {target_path}")


def export_report_json(report: CompleteEvaluationReport, output_path: str) -> str:
    """Save complete evaluation report to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)
    return output_path


def export_report_markdown(report: CompleteEvaluationReport, output_path: str) -> str:
    """Save complete evaluation report as formatted GitHub Markdown document."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    gt_dict = report.metadata.get("ground_truth_marks")
    gt_source = report.metadata.get("ground_truth_source")
    if not gt_dict:
        from src.utils.ground_truth import get_ground_truth_info_for_script
        gt_dict, gt_source = get_ground_truth_info_for_script(report.script_id)
    elif not gt_source:
        from src.utils.ground_truth import get_ground_truth_info_for_script
        _, gt_source = get_ground_truth_info_for_script(report.script_id)

    from src.utils.ground_truth import get_teacher_marks_total, canonicalize_question_key
    teacher_total_gt = get_teacher_marks_total(gt_dict) if gt_dict else None
    gt_source_name = os.path.basename(gt_source) if gt_source else "teacher_marks directory"
    has_examiner_gt = bool(gt_dict)

    md_lines = [
        f"# Exam Script Evaluation Report: `{report.script_id}`",
        f"**Evaluated At**: {report.timestamp} | **Engine**: `{report.model_id}`",
        f"**Input Source**: `{report.image_path}`",
        "",
        "---",
        "",
        "## Executive Summary",
        f"- **Subject**: {report.stage4_evaluation.subject}",
        f"- **Question Type**: {report.stage4_evaluation.question_type}",
        f"- **Evaluation Architecture**: `{'Modular (Question-Mapped)' if getattr(report.stage4_evaluation, 'eval_mode', None) == 'modular' else 'Monolithic (Single-Pass)'}`",
        f"- **Matched Question**: `{report.question_id or 'General'}`",
        f"- **Final Score**: **{report.stage4_evaluation.final_score:.2f} / {report.stage4_evaluation.total_max_marks:.2f}** ({report.stage4_evaluation.percentage:.1f}%)",
        f"- **Raw Content Score**: {report.stage4_evaluation.content_raw_score:.2f}",
        f"- **Linguistic Penalty**: -{report.stage4_evaluation.linguistic_penalty:.2f}",
        f"- **Red Ink Detected (Stage 0)**: `{'Yes' if report.has_red_ink else 'No'}`",
        f"- **Model-Detected Red-Ink Marks (Stage 0b)**: {len(report.teacher_marks)} visual detection(s) *(autonomous VLM, unverified)*",
        f"- **Silent Autocorrections Reverted (Stage 2)**: {report.stage2_verification.total_corrections_count}",
        f"- **Linguistic Errors Found (Stage 3)**: {report.stage3_errors.total_error_count}"
    ]

    if has_examiner_gt and teacher_total_gt is not None:
        md_lines.append(f"- **Official Examiner Ground Truth (`{gt_source_name}`)**: **{teacher_total_gt:.1f} / {report.stage4_evaluation.total_max_marks:.1f} marks**")

    if getattr(report.stage4_evaluation, "mae_vs_human", None) is not None:
        md_lines.append(f"- **MAE vs. Teacher Marks (`{gt_source_name}`)**: **{report.stage4_evaluation.mae_vs_human:.2f} marks**")

    if report.question_text:
        md_lines.extend([
            f"- **Question Prompt Preview**: {report.question_text.strip()[:120]}..."
        ])

    stage4_tok = report.metadata.get("stage4_token_usage", {})
    if stage4_tok and stage4_tok.get("total_tokens"):
        s4_max = stage4_tok.get("context_window", 4096)
        s4_tot = stage4_tok.get("total_tokens", 0)
        s4_pct = (s4_tot / s4_max * 100) if s4_max > 0 else 0
        s4_in = stage4_tok.get("prompt_tokens", 0)
        s4_out = stage4_tok.get("completion_tokens", 0)
        md_lines.append(f"- **Stage 4 Token Usage**: **{s4_tot:,} / {s4_max:,} tokens** ({s4_pct:.1f}%) [In: {s4_in:,}, Out: {s4_out:,}]")

    s4 = report.stage4_evaluation
    tot_max = s4.total_max_marks
    tot_awd = s4.final_score
    tot_pct = s4.percentage
    grade = "Grade A+ (Outstanding)" if tot_pct >= 80 else ("Grade A (Very Good)" if tot_pct >= 70 else ("Grade B (Good)" if tot_pct >= 60 else "Passed"))

    md_lines.extend([
        "",
        "---",
        "",
        "## 📊 Consolidated Exam Marks Scorecard",
        "| Question | Task / Topic | Mode | Max | Awarded | Score % | Teacher GT | Δ (AI - GT) | Status |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    # Gather segmented writing per question
    writing_by_q: Dict[str, Dict[str, Any]] = {}
    aligned_meta = report.metadata.get("aligned_answers") or []
    for a in aligned_meta:
        q_k = str(a.get("q_no") if isinstance(a, dict) else getattr(a, "q_no", "")).strip()
        ans_t = a.get("answer_text") if isinstance(a, dict) else getattr(a, "answer_text", "")
        p_nums = a.get("page_numbers") if isinstance(a, dict) else getattr(a, "page_numbers", [])
        writing_by_q[q_k] = {
            "writing": ans_t,
            "pages": p_nums,
            "word_count": len(ans_t.split()) if ans_t else 0
        }

    q_evals = getattr(s4, "question_evaluations", []) or []
    for qe in q_evals:
        q_k = str(qe.q_no).strip()
        if q_k not in writing_by_q:
            writing = getattr(qe, "student_writing", "") or ""
            p_nums = getattr(qe, "page_numbers", []) or []
            writing_by_q[q_k] = {
                "writing": writing,
                "pages": p_nums,
                "word_count": len(writing.split()) if writing else 0
            }

        # Scorecard Row
        pct = (qe.awarded_marks / qe.max_marks * 100.0) if qe.max_marks > 0 else 0.0
        gt_val = qe.human_gt
        if gt_val is None and gt_dict:
            q_c = canonicalize_question_key(qe.q_no)
            gt_val = gt_dict.get(q_c) or gt_dict.get(qe.q_no)
        gt_str = f"{gt_val:.1f}" if gt_val is not None else "N/A"
        delta_str = f"{(qe.awarded_marks - gt_val):+.1f}" if gt_val is not None else "N/A"
        mode_str = getattr(qe, "task_mode", None) or "-"
        status = getattr(qe, "scoring_status", "scored")
        status_badge = STATUS_BADGES.get(status, status.capitalize())

        md_lines.append(
            f"| **Q{qe.q_no}** | {qe.q_name} | `{mode_str}` | {qe.max_marks:.1f} | **{qe.awarded_marks:.1f}** | {pct:.1f}% | {gt_str} | {delta_str} | `{status_badge}` |"
        )

    tot_gt_str = f"{teacher_total_gt:.1f}" if teacher_total_gt is not None else "N/A"
    tot_delta_str = f"{(tot_awd - teacher_total_gt):+.1f}" if teacher_total_gt is not None else "N/A"
    md_lines.extend([
        f"| **TOTAL** | **All {len(q_evals)} Questions** | — | **{tot_max:.1f}** | **{tot_awd:.1f}** | **{tot_pct:.1f}%** | **{tot_gt_str}** | **{tot_delta_str}** | **{grade}** |",
        "",
        "---",
        "",
        "## 📝 Question-by-Question Evaluation & Student Answers",
        "> Each question below pairs the student's transcribed answer directly with the rubric evaluation, criteria judgments, and examiner rationale.",
        ""
    ])

    if q_evals:
        for qe in q_evals:
            q_k = str(qe.q_no).strip()
            ans_info = writing_by_q.get(q_k, {})
            writing_text = (ans_info.get("writing") or "").strip()
            pages_list = ans_info.get("pages") or getattr(qe, "page_numbers", []) or []
            pages_str = ", ".join(f"Page {p}" for p in sorted(pages_list)) if pages_list else "—"
            w_cnt = ans_info.get("word_count") or (len(writing_text.split()) if writing_text else 0)

            gt_val = qe.human_gt
            if gt_val is None and gt_dict:
                q_c = canonicalize_question_key(qe.q_no)
                gt_val = gt_dict.get(q_c) or gt_dict.get(qe.q_no)
            gt_info = f" | **Teacher GT**: {gt_val:.1f} (Δ {(qe.awarded_marks - gt_val):+.1f})" if gt_val is not None else ""
            pct = (qe.awarded_marks / qe.max_marks * 100.0) if qe.max_marks > 0 else 0.0
            mode_str = getattr(qe, "task_mode", None) or "-"
            status = getattr(qe, "scoring_status", "scored")
            status_badge = STATUS_BADGES.get(status, status.capitalize())

            md_lines.append(f"### Question {qe.q_no}: {qe.q_name}")
            md_lines.append(f"> **Mode**: `{mode_str}` | **Pages**: {pages_str} | **Word Count**: {w_cnt} words")
            md_lines.append(f"> **Marks Awarded**: **{qe.awarded_marks:.1f} / {qe.max_marks:.1f}** ({pct:.1f}%){gt_info} | **Status**: `{status_badge}`")
            md_lines.append("")

            # 1. Student Answer
            md_lines.append("#### 📝 Student's Answer:")
            if writing_text:
                md_lines.extend([
                    "```text",
                    writing_text,
                    "```",
                    ""
                ])
            else:
                notice = {
                    "blank_answer": "The answer segment for this question is blank (unattempted).",
                    "missing_no_segment": "No student answer segment identified for this question (unattempted, "
                                          "or not recognised by the segmenter; see segmentation.json).",
                    "missing": "No student answer segment identified for this question (unattempted).",
                }.get(status, "The answer text was not stored with this evaluation.")
                md_lines.extend([f"> ⚠️ *{notice}* ", ""])

            # 2. Evaluation & Marks Breakdown
            md_lines.append("#### 📊 Evaluation & Marks Breakdown:")
            if qe.examiner_feedback:
                md_lines.append(f"- **Examiner Feedback**: {qe.examiner_feedback}")

            # Mode A / B Items table
            if getattr(qe, "items", None):
                has_rationale = any(bool(it.get("rationale")) for it in qe.items)
                if has_rationale:
                    md_lines.extend([
                        "",
                        "| Item | Student Answer | Expected / Cue | Status | Marks | Examiner Rationale |",
                        "| :---: | :--- | :--- | :---: | :---: | :--- |"
                    ])
                    for it in qe.items:
                        exp = it.get("expected")
                        exp_str = ", ".join(str(x) for x in exp) if isinstance(exp, list) else (str(exp) if exp else "-")
                        cand = str(it.get("candidate_answer", "")).replace("\n", " ").strip()
                        rat = str(it.get("rationale", "")).replace("|", "/").strip()
                        md_lines.append(
                            f"| **{it.get('item_label', '')}** | `{cand[:60]}` | {exp_str[:50]} | `{it.get('status', '')}` | **{it.get('marks_awarded', 0):.1f} / {it.get('marks_available', 1):.1f}** | {rat[:100]} |"
                        )
                else:
                    md_lines.extend([
                        "",
                        "| Item | Student Answer | Expected Answer | Status | Marks Awarded |",
                        "| :---: | :--- | :--- | :---: | :---: |"
                    ])
                    for it in qe.items:
                        exp = it.get("expected")
                        exp_str = ", ".join(str(x) for x in exp) if isinstance(exp, list) else (str(exp) if exp else "-")
                        cand = str(it.get("candidate_answer", "")).replace("\n", " ").strip()
                        md_lines.append(
                            f"| **{it.get('item_label', '')}** | `{cand[:60]}` | {exp_str[:60]} | `{it.get('status', '')}` | **{it.get('marks_awarded', 0):.1f} / {it.get('marks_available', 1):.1f}** |"
                        )
                md_lines.append("")

            # Mode C Writing Criteria table
            if getattr(qe, "subscores", None) or getattr(qe, "criteria_reasoning", None):
                band_str = qe.performance_band or "N/A"
                md_lines.append(f"- **Performance Band**: `{band_str}` | **Raw Content Score**: {qe.content_raw_score:.1f} | **Linguistic Penalty**: -{qe.linguistic_penalty:.2f}")
                if qe.criteria_reasoning:
                    md_lines.extend([
                        "",
                        "| Criterion Dimension | Ceiling | Awarded | Examiner Justification | Evidence Quote |",
                        "| :--- | :---: | :---: | :--- | :--- |"
                    ])
                    for crit_k, crit_v in qe.criteria_reasoning.items():
                        crit_name = crit_k.replace("_", " ").title()
                        sub_val = f"{qe.subscores.get(crit_k, 0.0):.1f}" if qe.subscores else "-"
                        if isinstance(crit_v, dict):
                            just = str(crit_v.get("justification", "")).replace("|", "/")
                            ev = str(crit_v.get("evidence", "")).replace("|", "/")
                            md_lines.append(f"| **{crit_name}** | — | **{sub_val}** | {just} | `{ev[:60]}` |")
                        else:
                            md_lines.append(f"| **{crit_name}** | — | **{sub_val}** | {str(crit_v).replace('|', '/')} | — |")
                    md_lines.append("")

                if qe.student_feedback:
                    md_lines.append(f"- **Actionable Writing Advice**: {qe.student_feedback}")
                if qe.penalty_breakdown:
                    pb = qe.penalty_breakdown
                    cat_ded = pb.get("category_deductions", {})
                    ded_strs = [f"{k}: -{v:.2f}" for k, v in cat_ded.items() if v > 0]
                    if ded_strs:
                        cap_desc = f" ({pb.get('cap_reason')})" if pb.get("is_capped") else ""
                        md_lines.append(f"- **Linguistic Deductions**: {', '.join(ded_strs)}{cap_desc} (Total: -{qe.linguistic_penalty:.2f} marks)")
                md_lines.append("")

            md_lines.append("---")
            md_lines.append("")
    elif getattr(s4, "criteria_scores", None):
        md_lines.extend([
            "### Rubric Marks Breakdown (Monolithic)",
            "| Criterion | Max Marks | Awarded | Justification |",
            "| --- | --- | --- | --- |"
        ])
        for c in s4.criteria_scores:
            md_lines.append(f"| **{c.criterion_name}** | {c.max_marks:.2f} | **{c.awarded_marks:.2f}** | {c.justification} |")
        md_lines.append("")

    # Overall Teacher Feedback & Recommendations
    if s4.overall_feedback:
        md_lines.extend([
            "## 👨‍🏫 Overall Teacher Feedback & Recommendations",
            f"> {s4.overall_feedback}",
            ""
        ])
    if s4.actionable_recommendations:
        md_lines.append("**Key Actionable Improvements:**")
        for rec in s4.actionable_recommendations:
            md_lines.append(f"- {rec}")
        md_lines.append("")

    # Appendix: Linguistic Errors & Script Transcript
    md_lines.extend([
        "---",
        "",
        "## 📁 Appendix: Linguistic Errors & Transcript Reference"
    ])

    if report.stage3_errors and report.stage3_errors.errors:
        md_lines.extend([
            f"### Stage 3: Cataloged Linguistic Errors ({report.stage3_errors.total_error_count} total)",
            "| Type | Erroneous Text | Suggested Correction | Context / Explanation |",
            "| :---: | :--- | :--- | :--- |"
        ])
        for err in report.stage3_errors.errors[:30]:
            clean_exp = (err.explanation or "").replace("|", "/").replace("\n", " ")
            clean_err = str(err.erroneous_text or "").replace("|", "/")
            clean_corr = str(err.suggested_correction or "").replace("|", "/")
            md_lines.append(f"| `{err.error_type}` | `{clean_err}` | `{clean_corr}` | {clean_exp[:80]} |")
        md_lines.append("")

    if getattr(report, "stage2_verification", None) and report.stage2_verification.verified_transcript:
        md_lines.extend([
            "<details>",
            "<summary>📄 <strong>Click to expand full unsegmented script transcript (Reference)</strong></summary>",
            "",
            "```text",
            report.stage2_verification.verified_transcript,
            "```",
            "</details>",
            ""
        ])

    # Context & Token Usage Summary across Extraction and Stage 4
    ext_tok = report.metadata.get("extraction_token_usage", {})
    if stage4_tok or ext_tok:
        max_ctx = stage4_tok.get("context_window") or ext_tok.get("max_context_window", 4096)
        ext_in = ext_tok.get("total_prompt_tokens", 0)
        ext_out = ext_tok.get("total_completion_tokens", 0)
        ext_total = ext_tok.get("total_tokens", 0)
        ext_pct = (ext_total / max_ctx * 100) if max_ctx > 0 else 0

        s4_in = stage4_tok.get("prompt_tokens", 0)
        s4_out = stage4_tok.get("completion_tokens", 0)
        s4_total = stage4_tok.get("total_tokens", 0)
        s4_pct = (s4_total / max_ctx * 100) if max_ctx > 0 else 0

        pipe_in = ext_in + s4_in
        pipe_out = ext_out + s4_out
        pipe_total = ext_total + s4_total

        md_lines.extend([
            "---",
            "",
            "### Pipeline Context & Token Usage Breakdown",
            f"- **Context Window Limit**: `{max_ctx:,}` tokens | **Total Tokens**: **{pipe_total:,}** (Prompt: {pipe_in:,}, Completion: {pipe_out:,})",
            "",
            "| Pipeline Phase | Prompt (In) | Completion (Out) | Total Tokens | Context Window | Phase Usage % |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| Extraction (Stages 0-3) | {ext_in:,} | {ext_out:,} | {ext_total:,} | {max_ctx:,} | {ext_pct:.1f}% |",
            f"| Evaluation (Stage 4 Rubric) | {s4_in:,} | {s4_out:,} | {s4_total:,} | {max_ctx:,} | {s4_pct:.1f}% |",
            f"| **Full Pipeline Total** | **{pipe_in:,}** | **{pipe_out:,}** | **{pipe_total:,}** | **{max_ctx:,}** | - |",
            ""
        ])

    content = "\n".join(md_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


def export_mistakes_markdown(report: CompleteEvaluationReport, output_path: str) -> str:
    """Save dedicated standalone linguistic and structural error audit report as Markdown."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    errors = report.stage3_errors.errors if report.stage3_errors else []
    total_count = report.stage3_errors.total_error_count if report.stage3_errors else len(errors)
    spelling_count = report.stage3_errors.spelling_error_count if report.stage3_errors else sum(1 for e in errors if "spell" in str(e.error_type).lower())
    grammar_count = report.stage3_errors.grammar_error_count if report.stage3_errors else sum(1 for e in errors if "gramm" in str(e.error_type).lower())
    syntax_count = report.stage3_errors.syntax_error_count if report.stage3_errors else sum(1 for e in errors if "synt" in str(e.error_type).lower())
    total_pen = report.stage4_evaluation.linguistic_penalty if getattr(report, "stage4_evaluation", None) else 0.0

    # Recurring error pattern frequency
    word_freq = {}
    for e in errors:
        w = str(e.erroneous_text or "").lower().strip()
        if w:
            word_freq[w] = word_freq.get(w, 0) + 1

    top_repeated = [f"`{w}` ({c}x)" for w, c in sorted(word_freq.items(), key=lambda x: x[1], reverse=True) if c > 1][:4]
    repeated_str = ", ".join(top_repeated) if top_repeated else "None (all detected errors are singular occurrences)"

    md_lines = [
        f"# Dedicated Mistakes & Linguistic Error Audit: `{report.script_id}`",
        f"**Audit Generated**: {report.timestamp} | **Engine**: `{report.model_id}`",
        f"**Input Source**: `{report.image_path}`",
        "",
        "---",
        "",
        "## 🔍 Executive Error Diagnostic Summary",
        f"- **Total Detected Errors**: **{total_count}**",
        f"  - Spelling / Orthographic: `{spelling_count}`",
        f"  - Grammar & Tense: `{grammar_count}`",
        f"  - Syntax & Word Order: `{syntax_count}`",
        f"- **Total Linguistic Deduction Applied to Script**: **-{total_pen:.2f} marks**",
        f"- **Recurrent Misspelled Words**: {repeated_str}",
        f"- **NCTB Recurrent Error Policy Applied**: Identical misspelled words penalized once; objective question types exempted.",
        "",
        "---",
        "",
        "## Detailed Error Catalog",
        "| Page | Error Type | Student's Text | Suggested Correction | Linguistic Explanation & Rule Violated |",
        "| --- | --- | --- | --- | --- |"
    ]

    pages_list = getattr(report, "pages", None)
    error_page_map = {}
    if pages_list:
        for p in pages_list:
            for e in (p.stage3_errors.errors if p.stage3_errors else []):
                error_page_map[(e.erroneous_text, e.context_sentence)] = p.page_no

    for err in errors:
        p_no = error_page_map.get((err.erroneous_text, err.context_sentence))
        if p_no is None and pages_list:
            needle = str(err.erroneous_text or "").lower().strip()
            ctx = str(err.context_sentence or "").lower().strip()
            for p in pages_list:
                p_text = (p.stage2_verification.verified_transcript or p.stage1_transcription.raw_transcript or "").lower()
                if needle and needle in p_text:
                    p_no = p.page_no
                    break
                elif ctx and len(ctx) > 8 and ctx[:20] in p_text:
                    p_no = p.page_no
                    break
        page_str = f"Page {p_no}" if p_no is not None else "Page 1"
        clean_exp = (err.explanation or "").replace("|", "/").replace("\n", " ")
        clean_err = str(err.erroneous_text or "").replace("|", "/")
        clean_corr = str(err.suggested_correction or "").replace("|", "/")
        md_lines.append(f"| `{page_str}` | **{err.error_type}** | `{clean_err}` | `{clean_corr}` | {clean_exp} |")

    md_lines.append("")
    content = "\n".join(md_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path
