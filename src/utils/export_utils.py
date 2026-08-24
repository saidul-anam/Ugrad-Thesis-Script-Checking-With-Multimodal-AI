import os
import json
import csv
from typing import Dict, Any, List
from src.core.schemas import (
    Stage1TranscriptionResult,
    Stage2VerificationResult,
    Stage3ErrorResult,
    Stage4EvaluationResult,
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


def export_stage4_artifacts(result: Stage4EvaluationResult, script_dir: str):
    """Save Stage 4 rubric scoring outputs."""
    os.makedirs(script_dir, exist_ok=True)
    json_path = os.path.join(script_dir, "stage4_evaluation.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)


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
        f"- **Silent Autocorrections Reverted**: {report.stage2_verification.total_corrections_count}",
        f"- **Linguistic Errors Found**: {report.stage3_errors.total_error_count}",
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
