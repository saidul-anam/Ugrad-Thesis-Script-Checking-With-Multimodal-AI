#!/usr/bin/env python
"""
CLI Entrypoint for the Modular Answer-Script OCR & Grading Pipeline.
Supports NCTB HSC English 1st Paper (Rubric v2), multi-page question recognition and segmentation
via extraction.csv manifest or automated VLM detection, and evaluation.csv export.

Usage Examples:
    # Full multi-page student script run (segments Q3, Q7, Q8, Q9, Q10, Q11):
    python run_pipeline.py --input ../datasets/SE_11_Q1_0001.pdf --manifest_csv ../extraction.csv --debug

    # Batch run over all student scripts in a directory:
    python run_pipeline.py --batch_dir ../datasets/ --manifest_csv ../extraction.csv --output_csv outputs/evaluation_predictions.csv

    # Run full 64-combination Stage Ablation Benchmark:
    python run_pipeline.py --input ../datasets/SE_11_Q1_0001.pdf --ablate
"""

import os
import sys
import argparse
import json
import yaml
from typing import Any, Dict, List, Optional, Tuple
import itertools

# Reconfigure stdout/stderr to UTF-8 to handle Unicode on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure src is discoverable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.schemas import PipelineConfig, EnabledStagesConfig, TASK_MAX_MARKS, TASK_QUESTION_NUMBERS
from src.ingestion.pdf_loader import load_images_from_file
from src.ingestion.image_normalizer import normalize_images
from src.ingestion.page_router import PageRouter, load_manifest_from_csv
from src.orchestrator import run_pipeline, run_script_pipeline, export_evaluation_csv
from src.model_client.factory import get_model_client


def resolve_path(path: Optional[str]) -> Optional[str]:
    """Search for path across current working dir, parent dirs, and workspace roots."""
    if not path:
        return None
    candidates = [
        path,
        os.path.abspath(path),
        os.path.join(os.getcwd(), path),
        os.path.join(os.path.dirname(__file__), path),
        os.path.join(os.path.dirname(__file__), "..", path),
        os.path.join(os.path.dirname(__file__), "..", "..", path),
        os.path.join(os.getcwd(), "..", path),
        os.path.join(os.getcwd(), "..", "..", path),
        os.path.join(os.getcwd(), "..", "datasets", os.path.basename(path)),
        os.path.join(os.path.dirname(__file__), "..", "..", "datasets", os.path.basename(path))
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def load_config(config_path: Optional[str]) -> PipelineConfig:
    """Load configuration from YAML or JSON, or fallback to default."""
    resolved_cfg = resolve_path(config_path)
    if resolved_cfg and os.path.exists(resolved_cfg):
        with open(resolved_cfg, "r", encoding="utf-8") as f:
            if resolved_cfg.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f) or {}
        return PipelineConfig.from_dict(data)
    return PipelineConfig()


def read_text_or_literal(path_or_text: Optional[str], default_filename: str = "rubric_v2.txt") -> str:
    """If path_or_text is a valid existing file path or resolves to one, read it; otherwise check default file or return text."""
    resolved = resolve_path(path_or_text)
    if resolved and os.path.isfile(resolved):
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    
    if path_or_text and not path_or_text.endswith((".txt", ".md", ".json")):
        return path_or_text

    found_default = resolve_path(default_filename)
    if found_default and os.path.isfile(found_default):
        with open(found_default, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    
    return "NCTB HSC English 1st Paper Chief Examiner Rubric"


def run_ablation_benchmark(
    pages: List[Any],
    rubric: str,
    reference_solution: str,
    base_config: PipelineConfig,
    student_id: str = "ablation_student",
    task_id: str = "CHART_001",
    task_type: str = "Graph_Chart",
    max_mark: float = 10.0
) -> None:
    """
    Run ablation matrix over critical stage toggle combinations.
    """
    print("\n" + "=" * 90)
    print(f"RUNNING STAGE ABLATION MATRIX BENCHMARK ({task_type}, Max Mark: {max_mark})")
    print("=" * 90)

    test_configs = [
        ("Full Pipeline (All 6 Stages ON)", {}),
        ("Ablation: Stage 2 (RubricAligner) OFF", {"rubric_aligner": False}),
        ("Ablation: Stage 3 (Extractor B) OFF", {"extractor_b": False}),
        ("Ablation: Stage 4 (OCR Supervisor) OFF", {"ocr_supervisor": False}),
        ("Ablation: Stages 3 & 4 both OFF (Cheap OCR)", {"extractor_b": False, "ocr_supervisor": False}),
        ("Ablation: Stage 6 (Compressor) OFF", {"compressor": False}),
        ("Ablation: Minimal (Stage 1 + Stage 5 only)", {
            "rubric_aligner": False, "extractor_b": False, "ocr_supervisor": False, "compressor": False
        })
    ]

    client = get_model_client(base_config.model)

    print(f"{'Config Description':<45} | {'Final Score':<12} | {'Band':<10} | {'Cap Applied':<15}")
    print("-" * 92)

    for desc, overrides in test_configs:
        cfg_dict = base_config.to_dict()
        for k, v in overrides.items():
            cfg_dict["pipeline"]["enabled_stages"][k] = v
        
        cfg = PipelineConfig.from_dict(cfg_dict)
        cfg.logging.save_per_stage_json = False

        res = run_pipeline(
            pages=pages,
            rubric=rubric,
            reference_solution=reference_solution,
            config=cfg,
            task_type=task_type,
            max_mark=max_mark,
            task_id=task_id,
            student_id=student_id,
            model_client=client,
            verbose=False
        )

        cap_str = res.cap_reason if res.cap_applied else "None"
        print(f"{desc:<45} | {res.total_score:<12.2f} | {res.performance_band:<10} | {cap_str:<15}")

    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Modular Answer-Script OCR & Grading Pipeline (Rubric v2)")
    parser.add_argument("--input", type=str, help="Path to student script PDF or image")
    parser.add_argument("--manifest_csv", type=str, help="Path to extraction.csv ground-truth manifest")
    parser.add_argument("--batch_dir", type=str, help="Path to directory of PDFs for batch processing")
    parser.add_argument("--rubric", type=str, help="Path to rubric_v2.txt file or text")
    parser.add_argument("--task_type", type=str, help="Specific task type (e.g. Graph_Chart, Paragraph, Letter_Email, Story, Summary, Theme)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--backend", type=str, default=None, help="Model backend: easyocr, gemini, or huggingface")
    parser.add_argument("--output_csv", type=str, default="outputs/evaluation_results.csv", help="Path for summary CSV")
    parser.add_argument("--auto_route", action="store_true", help="Auto-detect question segments from multi-page PDFs")
    parser.add_argument("--debug", action="store_true", help="Print verbose step-by-step trace")
    parser.add_argument("--ablate", action="store_true", help="Run ablation study across stage toggles")

    args = parser.parse_args()

    config = load_config(args.config)
    if args.backend:
        config.model.backend = args.backend
    rubric = read_text_or_literal(args.rubric, "rubric_v2.txt")
    manifest_csv = resolve_path(args.manifest_csv) or resolve_path("extraction.csv")

    # 1. Ablation Mode
    if args.ablate:
        input_file = resolve_path(args.input) or resolve_path("datasets/SE_11_Q1_0001.pdf")
        if input_file and os.path.exists(input_file):
            pages = load_images_from_file(input_file, dpi=config.ingestion.pdf_dpi)
            pages = normalize_images(pages, max_side=config.ingestion.max_image_side)
        else:
            from PIL import Image
            pages = [Image.new("RGB", (800, 1000), color=(255, 255, 255))]

        run_ablation_benchmark(pages, rubric, "Sources of USA electricity in 1980", config)
        return

    # 2. Batch Mode across a directory of PDFs
    batch_directory = resolve_path(args.batch_dir)
    if batch_directory and os.path.isdir(batch_directory):
        pdf_files = [
            os.path.join(batch_directory, f)
            for f in os.listdir(batch_directory)
            if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg"))
        ]
        print(f"Discovered {len(pdf_files)} script files in {batch_directory}")
        all_results = []
        for pf in sorted(pdf_files):
            sid = os.path.splitext(os.path.basename(pf))[0]
            print(f"\nProcessing student script: {sid} ({pf})...")
            script_res = run_script_pipeline(
                pdf_path=pf,
                config=config,
                manifest_csv=manifest_csv,
                rubric_text=rubric,
                auto_route=args.auto_route,
                verbose=args.debug
            )
            all_results.extend(script_res)

        export_evaluation_csv(all_results, args.output_csv)
        print(f"\nBatch processing complete. Graded {len(all_results)} questions across {len(pdf_files)} scripts.")
        print(f"Results exported to: {args.output_csv}")
        return

    # 3. Single Multi-Page PDF Script Run
    input_file = resolve_path(args.input) or resolve_path("datasets/SE_11_Q1_0001.pdf")
    if not input_file or not os.path.exists(input_file):
        print(f"Error: Input file '{args.input}' not found. Please provide a valid --input PDF path.")
        return

    script_id = os.path.splitext(os.path.basename(input_file))[0]
    print(f"\nLoading and segmenting multi-page student script: {input_file} (Script ID: {script_id})")
    
    results = run_script_pipeline(
        pdf_path=input_file,
        config=config,
        manifest_csv=manifest_csv,
        rubric_text=rubric,
        auto_route=args.auto_route,
        verbose=args.debug or True
    )

    export_evaluation_csv(results, args.output_csv)

    print("\n" + "=" * 75)
    print(f"MULTI-PAGE SCRIPT EVALUATION SUMMARY: {script_id}")
    print("=" * 75)
    print(f"{'Task ID':<15} | {'Task Type':<15} | {'Score':<10} | {'Band':<10} | {'Cap Applied':<15}")
    print("-" * 75)
    total_awarded = 0.0
    total_possible = 0.0
    for r in results:
        total_awarded += r.total_score
        total_possible += r.max_mark
        cap_str = r.cap_reason if r.cap_applied else "None"
        print(f"{r.task_id:<15} | {r.question_type:<15} | {r.total_score:>4.1f}/{r.max_mark:<4.1f} | {r.performance_band:<10} | {cap_str:<15}")
    print("-" * 75)
    print(f"Total Script Marks: {total_awarded:.1f} / {total_possible:.1f}")
    print(f"Detailed CSV Report saved to: {args.output_csv}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
