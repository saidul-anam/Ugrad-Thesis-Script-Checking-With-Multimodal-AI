"""
Ground Truth Teacher Marks Parser and Manager.

Parses manual human markings from gt.txt (or JSON files) and makes them available
to extraction (Stage 0b), dataset exports (raw_tier_dataset.csv), and evaluation (Stage 4).
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional, List, Any
from src.core.schemas import TeacherMarkItem


def canonicalize_question_key(key: str) -> str:
    """Normalize question keys (e.g. 'Q1(A)', '1(a)', 'No-04', 'q2' -> standard format)."""
    k = key.strip()
    # Remove 'Q.', 'Q', 'No-', 'Ans to', etc.
    k = re.sub(r'^(?:ans\s*(?:to)?\s*)?(?:q(?:uestion)?\.?\s*(?:no\.?)?|no\.?)\s*[-:]?\s*', '', k, flags=re.IGNORECASE)
    # Standardize 1(a) -> 1(A)
    m = re.match(r'^([0-9]{1,2})\s*\(([a-zA-Z0-9]+)\)$', k)
    if m:
        return f"{m.group(1)}({m.group(2).upper()})"
    # Strip leading zeroes e.g. 04 -> 4
    if re.match(r'^0+[1-9]$', k):
        k = k.lstrip('0')
    # Strip trailing punctuation
    k = re.sub(r'[.:]+$', '', k).strip()
    return k


def extract_candidate_questions(transcript_text: str) -> List[str]:
    """
    Extract candidate question identifiers mentioned in transcribed student text.
    Handles headers like:
      - Ans to the Question No-01(A)
      - Ans to the question No-01(B)
      - Ans to the question no-2
      - Ans to the Question No-3
      - Ans to the Question No-04
      - Am to the question No -05
      - Ans to the Question No -06
      - Ans to the question no-8
      - Ans to the Question No-10 / No-100
      - Question No - 7
    """
    if not transcript_text:
        return []

    candidates: List[str] = []
    seen = set()

    patterns = [
        r'(?:ans(?:wer)?|am)?\s*(?:to\s*(?:the)?)?\s*(?:q(?:uestion)?\.?\s*(?:no\.?)?|ques\.?\s*no\.?)\s*[-:]?\s*([0-9]{1,2}\s*(?:\([a-zA-Z0-9]+\))?)',
        r'(?:^|\n)\s*([0-9]{1,2}\s*\([a-zA-Z0-9]+\))',
    ]
    for pat in patterns:
        for match in re.finditer(pat, transcript_text, re.IGNORECASE):
            raw_q = match.group(1).strip()
            # Clean leading zeros: 01(A) -> 1(A), 04 -> 4
            clean_q = re.sub(r'^0+([1-9])', r'\1', raw_q)
            if clean_q == "100":
                clean_q = "10"
            canon = canonicalize_question_key(clean_q)
            if canon and canon not in seen:
                seen.add(canon)
                candidates.append(canon)

    return candidates



def parse_ground_truth_file(file_path: str = "gt.txt") -> Dict[str, Dict[str, float]]:
    """
    Parses gt.txt into a dictionary mapping script_id -> {question_id: mark_value}.
    
    Supports formats like:
      SE_11_Q1_0002 marks
      1(A)-5
      1(B)-6
      2-10
      ...
    """
    if not os.path.exists(file_path):
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_gt: Dict[str, Dict[str, float]] = {}
    current_script_id: Optional[str] = None

    for line in lines:
        raw = line.strip()
        if not raw:
            continue

        # Check for script header line (e.g. 'SE_11_Q1_0002 marks' or 'SE_11_Q1_0010')
        script_match = re.search(r'((?:SE|SB)_[0-9A-Za-z_]+)', raw, re.IGNORECASE)
        if script_match and ("marks" in raw.lower() or not any(c in raw for c in "-:=")):
            current_script_id = script_match.group(1).upper()
            if current_script_id not in all_gt:
                all_gt[current_script_id] = {}
            continue

        # Skip header lines like 'question- mark'
        if "question" in raw.lower() and "mark" in raw.lower() and not any(char.isdigit() for char in raw):
            continue

        # Match question-mark pairs: e.g. '1(A)-5', '2-10', '10-3', '9-3.'
        pair_match = re.search(r'([0-9A-Za-z_()]+)\s*[-:=]\s*([0-9]+(?:\.[0-9]+)?)', raw)
        if pair_match and current_script_id:
            raw_q = pair_match.group(1)
            raw_val = pair_match.group(2)
            try:
                val = float(raw_val)
                canon_q = canonicalize_question_key(raw_q)
                all_gt[current_script_id][canon_q] = val
            except ValueError:
                pass

    return all_gt


def get_ground_truth_for_script(
    script_id: str,
    gt_file: str = "gt.txt"
) -> Optional[Dict[str, float]]:
    """Retrieve ground truth marks dictionary for a given script ID if available."""
    all_gt = parse_ground_truth_file(gt_file)
    sid_clean = script_id.strip().upper()
    if sid_clean in all_gt:
        return all_gt[sid_clean]
    
    # Try partial matching without file extensions or prefixes
    for k, v in all_gt.items():
        if k in sid_clean or sid_clean in k:
            return v
            
    return None


def ground_truth_to_teacher_marks(
    gt_dict: Dict[str, float],
    source_desc: str = "Ground Truth Human Examiner Mark (gt.txt)"
) -> List[TeacherMarkItem]:
    """Convert a dictionary of ground truth marks to TeacherMarkItem list."""
    items = []
    for q_no, mark in sorted(gt_dict.items(), key=lambda x: (len(x[0]), x[0])):
        items.append(TeacherMarkItem(
            question_no=q_no,
            mark_value=f"{mark:.1f}" if mark % 1 != 0 else f"{int(mark)}",
            location=source_desc
        ))
    return items


def sync_ground_truth_to_extracted_artifacts(
    extracted_root: str = "outputs/extracted",
    gt_file: str = "gt.txt"
) -> int:
    """
    Synchronize ground truth human marks from gt.txt into pre-extracted script directories.
    Updates:
      - stage0b_teacher_marks.json
      - extraction_result.json (metadata and teacher_marks)
      - raw_tier_records.csv
    Returns count of updated directories.
    """
    import json
    from glob import glob

    all_gt = parse_ground_truth_file(gt_file)
    if not all_gt:
        return 0

    updated_count = 0

    for script_id, gt_marks in all_gt.items():
        matching_dirs = glob(f"{extracted_root}/**/{script_id}", recursive=True)
        if not matching_dirs:
            for p in Path(extracted_root).rglob("*"):
                if p.is_dir() and p.name.upper() == script_id.upper():
                    matching_dirs.append(str(p))

        for s_dir in set(matching_dirs):
            teacher_mark_items = ground_truth_to_teacher_marks(
                gt_marks,
                source_desc=f"Verified Human Ground Truth (gt.txt) for {script_id}"
            )

            # 1. Update stage0b_teacher_marks.json
            s0b_path = os.path.join(s_dir, "stage0b_teacher_marks.json")
            try:
                with open(s0b_path, "w", encoding="utf-8") as f:
                    json.dump([m.model_dump() for m in teacher_mark_items], f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[Ground Truth] Warning: could not write {s0b_path}: {e}")

            # 2. Update extraction_result.json
            ext_json_path = os.path.join(s_dir, "extraction_result.json")
            if os.path.exists(ext_json_path):
                try:
                    with open(ext_json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["teacher_marks"] = [m.model_dump() for m in teacher_mark_items]
                    data.setdefault("metadata", {})
                    data["metadata"]["verified_by_human"] = True
                    data["metadata"]["ground_truth_marks"] = gt_marks
                    with open(ext_json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[Ground Truth] Warning: could not update {ext_json_path}: {e}")

            # 3. Update raw_tier_records.csv
            raw_csv_path = os.path.join(s_dir, "raw_tier_records.csv")
            if os.path.exists(raw_csv_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(raw_csv_path)
                    for idx, row in df.iterrows():
                        trans = str(row.get("transcript_text", ""))
                        cands = extract_candidate_questions(trans)
                        matched = [f"Q{q}:{gt_marks[q]} (Ground Truth)" for q in cands if q in gt_marks]
                        if matched:
                            df.at[idx, "teacher_mark"] = "; ".join(matched)
                            df.at[idx, "question_no"] = cands[0]
                        df.at[idx, "original_marker_id"] = "human_examiner_gt"
                    df.to_csv(raw_csv_path, index=False)
                except Exception as e:
                    print(f"[Ground Truth] Warning: could not update {raw_csv_path}: {e}")

            updated_count += 1
            print(f"[Ground Truth] Synchronized {len(gt_marks)} marks into '{s_dir}'")

    # Refresh any root dataset CSVs in extracted_root or language subfolders
    for root_csv in [
        os.path.join(extracted_root, "raw_tier_dataset.csv"),
        os.path.join(extracted_root, "english", "raw_tier_dataset.csv"),
        os.path.join(extracted_root, "bangla", "raw_tier_dataset.csv")
    ]:
        if os.path.exists(root_csv):
            try:
                import pandas as pd
                root_df = pd.read_csv(root_csv)
                for sid, gt in all_gt.items():
                    mask = root_df["script_id"].astype(str).str.upper() == sid
                    for idx in root_df[mask].index:
                        trans = str(root_df.at[idx, "transcript_text"])
                        cands = extract_candidate_questions(trans)
                        matched = [f"Q{q}:{gt[q]} (Ground Truth)" for q in cands if q in gt]
                        if matched:
                            root_df.at[idx, "teacher_mark"] = "; ".join(matched)
                            root_df.at[idx, "question_no"] = cands[0]
                        root_df.at[idx, "original_marker_id"] = "human_examiner_gt"
                root_df.to_csv(root_csv, index=False)
                print(f"[Ground Truth] Refreshed root dataset CSV -> '{root_csv}'")
            except Exception as e:
                print(f"[Ground Truth] Warning: could not refresh {root_csv}: {e}")

    return updated_count

