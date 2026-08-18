"""
Dataset preparation for Phase 2 fine-tuning (LoRA / QLoRA).
Converts logged per-stage JSON files (with human corrections) into SFT instruction tuning datasets.
"""

import os
import json
import glob
from typing import Any, Dict, List, Optional


def build_sft_example(
    instruction: str,
    input_text: str,
    output_json: Dict[str, Any] | str,
    image_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format a single instruction-tuning example for SFTTrainer / HuggingFace.
    """
    if isinstance(output_json, dict):
        target_text = json.dumps(output_json, ensure_ascii=False, indent=2)
    else:
        target_text = str(output_json)

    prompt = f"{instruction}\n\n{input_text}".strip()

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_text}
        ],
        "image_path": image_path
    }


def prepare_ocr_sft_dataset(
    logs_dir: str,
    output_jsonl: str,
    stage_name: str = "stage_1_extractor_a"
) -> int:
    """
    Scrape logged per-stage JSON files from `logs_dir` and export to JSONL for SFTTrainer.
    """
    pattern = os.path.join(logs_dir, "**", f"{stage_name}.json")
    files = glob.glob(pattern, recursive=True)
    count = 0

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                # Find corresponding images if available
                example = build_sft_example(
                    instruction="Transcribe handwritten Bangla and English student answer script accurately.",
                    input_text="",
                    output_json=data
                )
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                continue

    return count


def prepare_examiner_sft_dataset(
    logs_dir: str,
    rubrics_map: Dict[str, str],
    reference_solutions_map: Dict[str, str],
    output_jsonl: str
) -> int:
    """
    Export Stage 5 Examiner training pairs (rubric + answer -> reasoning & marks) for SFT.
    """
    pattern = os.path.join(logs_dir, "**", "stage_5_examiner.json")
    files = glob.glob(pattern, recursive=True)
    count = 0

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as jf:
                    data = json.load(jf)

                # Extract student_id & question_id from folder path
                parts = os.path.normpath(fpath).split(os.sep)
                question_id = parts[-2] if len(parts) >= 2 else "Q1"

                rubric = rubrics_map.get(question_id, "")
                ref_sol = reference_solutions_map.get(question_id, "")

                input_text = f"Operative Rubric:\n{rubric}\n\nReference Solution:\n{ref_sol}"
                example = build_sft_example(
                    instruction="Evaluate the student answer according to the rubric with chain-of-thought.",
                    input_text=input_text,
                    output_json=data
                )
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                continue

    return count
