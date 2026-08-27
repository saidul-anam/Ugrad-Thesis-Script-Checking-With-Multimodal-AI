import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
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
        f"  - Punctuation: {result.stage3_errors.punctuation_error_count}",
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
            md_lines.append(f"| `{diff.stage1_output}` | `{diff.actual_handwritten}` | {diff.reason} |")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## Cataloged Linguistic & Structural Errors",
        f"> {result.stage3_errors.linguistic_summary}",
        ""
    ])

    if result.stage3_errors.errors:
        md_lines.extend([
            "| Type | Erroneous Text | Suggested Correction | Explanation |",
            "| --- | --- | --- | --- |"
        ])
        for err in result.stage3_errors.errors:
            md_lines.append(f"| **{err.error_type}** | `{err.erroneous_text}` | `{err.suggested_correction}` | {err.explanation} |")
        md_lines.append("")

    content = "\n".join(md_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


def export_raw_tier_csv(records: List[RawTierRecord], output_path: str) -> str:
    """
    Save the standardized 13-column Raw-Tier dataset CSV for research and analysis:
    Columns: script_id, page_no, question_no, paper, task_type, transcript_text,
             ocr_flags, error_list, teacher_mark, has_red_ink, original_marker_id, school_id, region
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

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    with open(output_path, "w" if not file_exists else "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for rec in records:
            writer.writerow(rec.model_dump())

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
        f"- **Final Score**: **{report.stage4_evaluation.final_score:.2f} / {report.stage4_evaluation.total_max_marks:.2f}** ({report.stage4_evaluation.percentage:.1f}%)",
        f"- **Raw Content Score**: {report.stage4_evaluation.content_raw_score:.2f}",
        f"- **Linguistic Penalty**: -{report.stage4_evaluation.linguistic_penalty:.2f}",
        f"- **Red Ink Detected (Stage 0)**: `{'Yes' if report.has_red_ink else 'No'}`",
        f"- **Extracted Teacher Marks (Stage 0b)**: {len(report.teacher_marks)} mark(s)",
        f"- **Silent Autocorrections Reverted (Stage 2)**: {report.stage2_verification.total_corrections_count}",
        f"- **Linguistic Errors Found (Stage 3)**: {report.stage3_errors.total_error_count}",
        "",
        "---",
        "",
        "## Stage 1 & Stage 2: Transcripts & Autocorrection Audit",
        "",
        "### Verified Canonical Transcript (Stage 2)",
        "```text",
        report.stage2_verification.verified_transcript,
        "```",
        ""
    ]

    if report.teacher_marks:
        md_lines.extend([
            "### Teacher Red-Ink Marks (Isolated)",
            "| Question | Mark | Location |",
            "| --- | --- | --- |"
        ])
        for tm in report.teacher_marks:
            md_lines.append(f"| {tm.question_no or 'N/A'} | `{tm.mark_value}` | {tm.location} |")
        md_lines.append("")

    if report.stage2_verification.silent_corrections_fixed:
        md_lines.extend([
            "### Reverted Silent Autocorrections",
            "| Stage 1 Output | Actual Handwriting | Reason |",
            "| --- | --- | --- |"
        ])
        for diff in report.stage2_verification.silent_corrections_fixed:
            md_lines.append(f"| `{diff.stage1_output}` | `{diff.actual_handwritten}` | {diff.reason} |")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## Stage 3: Linguistic & Structural Errors",
        f"**Total Errors**: {report.stage3_errors.total_error_count} (Spelling: {report.stage3_errors.spelling_error_count}, Grammar: {report.stage3_errors.grammar_error_count}, Syntax: {report.stage3_errors.syntax_error_count}, Punctuation: {report.stage3_errors.punctuation_error_count})",
        "",
        f"> {report.stage3_errors.linguistic_summary}",
        ""
    ])

    if report.stage3_errors.errors:
        md_lines.extend([
            "| Type | Written Text | Correct Form | Explanation |",
            "| --- | --- | --- | --- |"
        ])
        for err in report.stage3_errors.errors:
            md_lines.append(f"| **{err.error_type}** | `{err.erroneous_text}` | `{err.suggested_correction}` | {err.explanation} |")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## Stage 4: Rubric Marks Breakdown",
        "| Criterion | Max Marks | Awarded | Justification |",
        "| --- | --- | --- | --- |"
    ])

    for c in report.stage4_evaluation.criteria_scores:
        md_lines.append(f"| **{c.criterion_name}** | {c.max_marks:.2f} | **{c.awarded_marks:.2f}** | {c.justification} |")

    md_lines.extend([
        "",
        "### Teacher Feedback & Recommendations",
        f"> {report.stage4_evaluation.overall_feedback}",
        "",
        "**Actionable Improvements**:"
    ])

    for rec in report.stage4_evaluation.actionable_recommendations:
        md_lines.append(f"- {rec}")

    md_lines.append("")

    content = "\n".join(md_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path
