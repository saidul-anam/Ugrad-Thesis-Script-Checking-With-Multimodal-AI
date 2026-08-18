"""
Run Full Evaluation Benchmark Across All Student Scripts & Generate Teacher-Ready CSV Comparison Reports.

Outputs generated in outputs/:
1. 1_all_pipeline_evaluations.csv          - Full 25-column predictions matching evaluation.csv format
2. 2_comparative_marks_analysis.csv       - Side-by-side comparison: Teacher vs Gemini vs Gemma Pipeline
3. 3_stage_by_stage_performance.csv       - Diagnostic metrics for Stages 1 to 6
4. 4_hard_cap_and_classification_diagnostics.csv - Confusion matrix (TP, FP, FN, TN), Precision, Recall, Band Agreement
5. 5_summary_teacher_executive_report.csv - QWK, MAE, RMSE, Pearson r, Agreement metrics for supervisor presentation
"""

import os
import sys
import time
import argparse
from typing import List

# Windows console encoding fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure src is discoverable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.schemas import PipelineConfig, PipelineResult
from src.orchestrator import run_script_pipeline, export_evaluation_csv
from src.eval.benchmark_suite import (
    load_ground_truth_csv,
    generate_comparative_marks_csv,
    generate_stage_by_stage_performance_csv,
    generate_ocr_stage_accuracy_csv,
    generate_hard_cap_confusion_csv,
    generate_executive_summary_report_csv
)


def resolve_path(path: str) -> str:
    """Smart multi-root path resolver."""
    candidates = [
        path,
        os.path.abspath(path),
        os.path.join(os.getcwd(), path),
        os.path.join(os.path.dirname(__file__), path),
        os.path.join(os.path.dirname(__file__), "..", path),
        os.path.join(os.path.dirname(__file__), "..", "..", path),
        os.path.join(os.getcwd(), "..", path),
        os.path.join(os.getcwd(), "..", "datasets", os.path.basename(path)),
        os.path.join(os.path.dirname(__file__), "..", "..", "datasets", os.path.basename(path))
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run Full Evaluation & Generate Comparison CSVs")
    parser.add_argument("--dataset_dir", type=str, default="datasets", help="Directory containing student script PDFs")
    parser.add_argument("--extraction_csv", type=str, default="extraction.csv", help="Path to extraction.csv")
    parser.add_argument("--evaluation_csv", type=str, default="evaluation.csv", help="Path to evaluation.csv")
    parser.add_argument("--rubric_path", type=str, default="rubric_v2.txt", help="Path to rubric_v2.txt")
    parser.add_argument("--backend", type=str, default="mock", help="Model backend: mock | transformers | vllm")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory where comparison CSVs will be saved")
    parser.add_argument("--debug", action="store_true", help="Print verbose step execution")

    args = parser.parse_args()

    dataset_dir = resolve_path(args.dataset_dir)
    extraction_csv = resolve_path(args.extraction_csv)
    evaluation_csv = resolve_path(args.evaluation_csv)
    rubric_path = resolve_path(args.rubric_path)
    output_dir = os.path.abspath(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80, flush=True)
    print("STARTING FULL SCRIPT EVALUATION & MULTI-STAGE BENCHMARK COMPARISON", flush=True)
    print("=" * 80, flush=True)
    print(f"Dataset Directory : {dataset_dir}", flush=True)
    print(f"Extraction CSV    : {extraction_csv}", flush=True)
    print(f"Evaluation CSV    : {evaluation_csv}", flush=True)
    print(f"Rubric Path       : {rubric_path}", flush=True)
    print(f"Model Backend     : {args.backend}", flush=True)
    print(f"Output Directory  : {output_dir}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load ground truth references
    extractions, evaluations = load_ground_truth_csv(extraction_csv, evaluation_csv)
    print(f"Loaded {len(extractions)} ground truth extraction records and {len(evaluations)} evaluation records.", flush=True)

    # 2. Discover all PDF scripts in dataset directory
    pdf_files = []
    if os.path.isdir(dataset_dir):
        pdf_files = [
            os.path.join(dataset_dir, f)
            for f in sorted(os.listdir(dataset_dir))
            if f.lower().endswith(".pdf")
        ]
    
    print(f"Discovered {len(pdf_files)} student script PDFs in dataset folder.", flush=True)

    # Read rubric text
    rubric_text = "NCTB HSC English 1st Paper Chief Examiner Evaluation"
    if os.path.exists(rubric_path):
        with open(rubric_path, "r", encoding="utf-8", errors="replace") as f:
            rubric_text = f.read()

    config = PipelineConfig()
    config.model.backend = args.backend
    config.ingestion.pdf_dpi = 100 if args.backend == "mock" else 200
    config.logging.save_per_stage_json = False

    # 3. Run Pipeline across all student scripts
    all_results: List[PipelineResult] = []
    total_pdfs = len(pdf_files)
    start_time = time.time()
    
    # If PDF files exist in dataset directory, run across each
    if pdf_files:
        for idx, pf in enumerate(pdf_files, 1):
            t_script_start = time.time()
            sid = os.path.splitext(os.path.basename(pf))[0]
            print(f"\n[{idx}/{total_pdfs}] Processing Multi-Page Script: {sid} ({pf})...", flush=True)
            
            script_res = run_script_pipeline(
                pdf_path=pf,
                config=config,
                manifest_csv=extraction_csv,
                rubric_text=rubric_text,
                auto_route=True,
                verbose=args.debug
            )
            all_results.extend(script_res)
            dt_script = time.time() - t_script_start
            
            # 1. Immediately save individual script evaluation CSV
            per_script_csv = os.path.join(output_dir, f"eval_{sid}.csv")
            export_evaluation_csv(script_res, per_script_csv)
            
            # 2. Incrementally update running aggregated predictions & comparison CSV
            file1 = os.path.join(output_dir, "1_all_pipeline_evaluations.csv")
            export_evaluation_csv(all_results, file1)
            
            file2 = os.path.join(output_dir, "2_comparative_marks_analysis.csv")
            generate_comparative_marks_csv(all_results, extractions, evaluations, file2)
            
            # 3. Print runtime diagnostics
            elapsed = time.time() - start_time
            avg_per_pdf = elapsed / idx
            est_remaining_sec = avg_per_pdf * (total_pdfs - idx)
            est_rem_str = f"{est_remaining_sec/60:.1f}m" if est_remaining_sec > 60 else f"{est_remaining_sec:.0f}s"
            
            print(f"  -> Graded {len(script_res)} questions in {dt_script:.1f}s.")
            print(f"  -> Saved per-script CSV: {per_script_csv}")
            print(f"  -> Progress: {len(all_results)} total questions graded [{idx}/{total_pdfs} scripts, ~{est_rem_str} remaining].", flush=True)
    else:
        print("No PDF files found directly; evaluating against all tasks in extraction manifest...")
        from ingestion.page_router import load_manifest_from_csv
        from orchestrator import run_pipeline
        from PIL import Image

        manifests = load_manifest_from_csv(extraction_csv)
        dummy_img = [Image.new("RGB", (1000, 1400), color=(255, 255, 255))]
        for sid, manifest in manifests.items():
            for tid, qseg in manifest.questions.items():
                r = run_pipeline(
                    pages=dummy_img,
                    rubric=rubric_text,
                    reference_solution=qseg.question_prompt,
                    config=config,
                    task_type=qseg.question_type,
                    max_mark=qseg.max_mark,
                    task_id=qseg.task_id,
                    student_id=sid,
                    teacher_mark=qseg.teacher_mark,
                    question_no=qseg.question_no,
                    verbose=False
                )
                all_results.append(r)

    print(f"\nTotal Pipeline Evaluations Completed: {len(all_results)} questions in {time.time() - start_time:.1f}s.", flush=True)

    # 4. Generate all comparative CSV reports
    file1 = os.path.join(output_dir, "1_all_pipeline_evaluations.csv")
    export_evaluation_csv(all_results, file1)
    print(f"[Exported 1/5] Full Evaluation Predictions -> {file1}")

    file2 = os.path.join(output_dir, "2_comparative_marks_analysis.csv")
    comp_rows = generate_comparative_marks_csv(all_results, extractions, evaluations, file2)
    print(f"[Exported 2/5] Comparative Marks Analysis (Teacher vs Gemini vs Pipeline) -> {file2}")

    file3 = os.path.join(output_dir, "3_stage_by_stage_performance.csv")
    stage_metrics = generate_stage_by_stage_performance_csv(all_results, extractions, evaluations, file3)
    print(f"[Exported 3/5] Stage-by-Stage Diagnostic Performance -> {file3}")

    file4 = os.path.join(output_dir, "4_hard_cap_and_classification_diagnostics.csv")
    cap_stats = generate_hard_cap_confusion_csv(comp_rows, file4)
    print(f"[Exported 4/6] Hard Cap Confusion Matrix & Classification Diagnostics -> {file4}")

    file5 = os.path.join(output_dir, "5_summary_teacher_executive_report.csv")
    exec_report = generate_executive_summary_report_csv(comp_rows, file5)
    print(f"[Exported 5/6] Executive Statistical Alignment Report -> {file5}")

    file6 = os.path.join(output_dir, "6_ocr_stage_wer_cer_analysis.csv")
    ocr_rows = generate_ocr_stage_accuracy_csv(all_results, extractions, file6)
    print(f"[Exported 6/6] Multi-Stage OCR WER & CER Analysis Report -> {file6}")

    # Print Executive Summary Table to Terminal
    print("\n" + "=" * 85)
    print("EXECUTIVE BENCHMARK SUMMARY (TEACHER REPORT)")
    print("=" * 85)
    for row in exec_report:
        print(f"\nTarget: {row['comparison_target']} (N = {row['sample_size']})")
        print(f"  - MAE                     : {row['MAE (Mean Absolute Error)']}")
        print(f"  - RMSE                    : {row['RMSE (Root Mean Square Error)']}")
        print(f"  - Pearson Correlation (r) : {row['Pearson_r (Correlation)']}")
        print(f"  - Quadratic Weighted Kappa: {row['QWK (Quadratic Weighted Kappa)']}")
        print(f"  - Exact Agreement         : {row['Exact_Agreement (%)']}")
        print(f"  - Adjacent (±1 Mark) Agr. : {row['Adjacent_Agreement_PM1 (%)']}")

    # Print OCR Stage Error Rate Table
    if ocr_rows:
        s1_wers = [r["stage1_cold_ocr_wer"] for r in ocr_rows]
        s3_wers = [r["stage3_reference_primed_wer"] for r in ocr_rows]
        s4_wers = [r["stage4_adjudicated_ocr_wer"] for r in ocr_rows]
        s1_cers = [r["stage1_cold_ocr_cer"] for r in ocr_rows]
        s3_cers = [r["stage3_reference_primed_cer"] for r in ocr_rows]
        s4_cers = [r["stage4_adjudicated_ocr_cer"] for r in ocr_rows]

        avg_s1_wer = sum(s1_wers) / len(s1_wers)
        avg_s3_wer = sum(s3_wers) / len(s3_wers)
        avg_s4_wer = sum(s4_wers) / len(s4_wers)
        avg_s1_cer = sum(s1_cers) / len(s1_cers)
        avg_s3_cer = sum(s3_cers) / len(s3_cers)
        avg_s4_cer = sum(s4_cers) / len(s4_cers)
        delta_wer = ((avg_s1_wer - avg_s4_wer) / avg_s1_wer * 100) if avg_s1_wer > 0 else 0.0

        print("\n" + "=" * 85)
        print("MULTI-STAGE OCR PERFORMANCE & ERROR RATES (GROUND TRUTH COMPARISON)")
        print("=" * 85)
        print(f"Evaluated Tasks (with Ground Truth Extractions): N = {len(ocr_rows)}")
        print(f"  - Stage 1: Extractor A (Cold OCR Read)      : WER = {avg_s1_wer:.4f} ({avg_s1_wer*100:.2f}%), CER = {avg_s1_cer:.4f} ({avg_s1_cer*100:.2f}%)")
        print(f"  - Stage 3: Extractor B (Reference-Primed)   : WER = {avg_s3_wer:.4f} ({avg_s3_wer*100:.2f}%), CER = {avg_s3_cer:.4f} ({avg_s3_cer*100:.2f}%)")
        print(f"  - Stage 4: OCR Supervisor (Adjudicated)    : WER = {avg_s4_wer:.4f} ({avg_s4_wer*100:.2f}%), CER = {avg_s4_cer:.4f} ({avg_s4_cer*100:.2f}%)")
        print(f"  -> Relative WER Reduction (Stage 1 -> Stage 4): {delta_wer:.2f}% improvement")

    print("\n" + "=" * 85)
    print("HARD CAP CONFUSION MATRIX & ACCURACY")
    print("=" * 85)
    print(f"  - True Positives (TP)     : {cap_stats['tp']}")
    print(f"  - False Positives (FP)    : {cap_stats['fp']}")
    print(f"  - False Negatives (FN)    : {cap_stats['fn']}")
    print(f"  - True Negatives (TN)     : {cap_stats['tn']}")
    print(f"  - Precision               : {cap_stats['precision']:.4f}")
    print(f"  - Recall                  : {cap_stats['recall']:.4f}")
    print(f"  - F1 Score                : {cap_stats['f1']:.4f}")
    print(f"  - Exact Band Match (%)    : {cap_stats['exact_band_pct']:.2f}%")
    print(f"  - Adjacent Band Match (±1): {cap_stats['adj_band_pct']:.2f}%")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
