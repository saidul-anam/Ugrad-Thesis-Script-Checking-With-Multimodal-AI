"""
Ground Truth Teacher Marks Parser and Manager.

Parses manual human markings from gt.txt (or JSON files) and makes them available
to extraction (Stage 0b), dataset exports (raw_tier_dataset.csv), and evaluation (Stage 4).
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
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


TEACHER_MARKS_DIR = "data/ground_truth/transcripts/teacher_marks"


def resolve_teacher_marks_dir(teacher_marks_dir: Optional[str] = None) -> str:
    """Resolve teacher marks directory relative to current working directory or repository root."""
    if teacher_marks_dir and os.path.isdir(teacher_marks_dir):
        return teacher_marks_dir
    default_rel = TEACHER_MARKS_DIR
    if os.path.isdir(default_rel):
        return default_rel
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / default_rel
    if candidate.is_dir():
        return str(candidate)
    return default_rel


def get_teacher_marks_total(gt_dict: Optional[Dict[str, float]]) -> Optional[float]:
    """
    Retrieve or compute the total teacher mark from a ground truth dict without double counting.
    Prefers explicit 'total_marks' key; otherwise sums distinct scored questions.
    """
    if not gt_dict:
        return None
    for k in ("total_marks", "total", "tot", "total_score"):
        if k in gt_dict:
            return float(gt_dict[k])
    return sum(float(v) for k, v in gt_dict.items() if str(k).lower() not in ("total_marks", "total", "tot", "total_score"))


def parse_teacher_marks_csv(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    Parses a CSV file containing teacher marks (e.g. SE_08_Q1_teacher_marks.csv, SE_11_Q1_teacher_marks.csv).
    Format:
      student_id,Q1,Q2,...,total_marks
    Returns: mapping of student_id -> {canonical_q_no: mark_value, 'total_marks': total_mark_value}
    """
    import csv
    if not os.path.exists(file_path):
        return {}

    marks_by_student: Dict[str, Dict[str, float]] = {}
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("student_id") or "").strip().upper()
            if not sid:
                continue
            student_marks: Dict[str, float] = {}
            for k, v in row.items():
                if not k or k == "student_id" or v is None or v == "":
                    continue
                k_clean = k.strip()
                try:
                    val = float(str(v).strip())
                except ValueError:
                    continue

                if k_clean.lower() in ("total_marks", "total", "tot"):
                    student_marks["total_marks"] = val
                else:
                    canon_k = canonicalize_question_key(k_clean)
                    student_marks[canon_k] = val

            if "total_marks" not in student_marks and student_marks:
                student_marks["total_marks"] = sum(val for qk, val in student_marks.items() if qk != "total_marks")

            marks_by_student[sid] = student_marks

    return marks_by_student


def load_all_teacher_marks(
    teacher_marks_dir: str = TEACHER_MARKS_DIR
) -> Dict[str, Tuple[Dict[str, float], str]]:
    """
    Scans the teacher_marks directory and returns a unified map:
      student_id -> (marks_dict, source_file_path)
    Supports both CSV files (*.csv) and raw text files.
    """
    resolved_dir = resolve_teacher_marks_dir(teacher_marks_dir)
    if not os.path.isdir(resolved_dir):
        return {}

    all_marks: Dict[str, Tuple[Dict[str, float], str]] = {}
    # 1. Process CSV files first (canonical tabular format)
    for entry in sorted(os.listdir(resolved_dir)):
        full_p = os.path.join(resolved_dir, entry)
        if os.path.isfile(full_p) and entry.lower().endswith(".csv"):
            csv_marks = parse_teacher_marks_csv(full_p)
            for sid, m in csv_marks.items():
                all_marks[sid] = (m, full_p)

    # 2. Process text files (merge any missing sub-part questions or add new scripts)
    for entry in sorted(os.listdir(resolved_dir)):
        full_p = os.path.join(resolved_dir, entry)
        if os.path.isfile(full_p) and not entry.lower().endswith(".csv"):
            txt_marks = parse_ground_truth_file(full_p)
            for sid, m in txt_marks.items():
                if sid in all_marks:
                    # Merge sub-parts (e.g. 1(A), 1(B)) if text file contains finer granularity
                    existing_dict, src_file = all_marks[sid]
                    for qk, qv in m.items():
                        if qk not in existing_dict and qk != "total_marks":
                            existing_dict[qk] = qv
                else:
                    if "total_marks" not in m and m:
                        m["total_marks"] = sum(v for k, v in m.items() if k != "total_marks")
                    all_marks[sid] = (m, full_p)

    return all_marks


def get_ground_truth_info_for_script(
    script_id: str,
    gt_file: str = "gt.txt",
    teacher_marks_dir: str = TEACHER_MARKS_DIR,
    lang: Optional[str] = None
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Retrieve ground truth marks dictionary and source file for a given script ID if available.
    Searches:
      1. data/ground_truth/transcripts/teacher_marks/ (CSV & text files)
      2. gt_file (default 'gt.txt')
    """
    sid_clean = os.path.basename(script_id.strip()).upper()
    sid_clean = re.sub(r'\.(?:PDF|PNG|JPG|JPEG|TXT|JSON)$', '', sid_clean, flags=re.IGNORECASE)

    # 1. Search teacher_marks directory
    resolved_dir = resolve_teacher_marks_dir(teacher_marks_dir)
    teacher_marks_map = load_all_teacher_marks(resolved_dir)

    # 1a. Exact match
    if sid_clean in teacher_marks_map:
        return teacher_marks_map[sid_clean]

    # 1b. Match with paper context if present in sid_clean (e.g. SE_11_Q1_0002)
    paper_match = re.search(r'(S[EB]_\d+_[A-Za-z0-9]+)', sid_clean)
    if paper_match:
        pref_prefix = paper_match.group(1)
        for k, (marks, src) in teacher_marks_map.items():
            if pref_prefix in k and (k.endswith(sid_clean) or sid_clean.endswith(k) or sid_clean in k or k in sid_clean):
                return marks, src

    # 1c. Suffix match (e.g. script_id='0002' matching '..._0002')
    for k, (marks, src) in teacher_marks_map.items():
        if k.endswith(f"_{sid_clean}") or sid_clean.endswith(f"_{k}"):
            return marks, src

    # 1d. Substring match
    for k, (marks, src) in teacher_marks_map.items():
        if k in sid_clean or sid_clean in k:
            return marks, src

    # 2. Fallback to gt_file
    if os.path.exists(gt_file):
        all_gt = parse_ground_truth_file(gt_file)
        if sid_clean in all_gt:
            m = all_gt[sid_clean]
            if "total_marks" not in m and m:
                m["total_marks"] = sum(v for k, v in m.items() if k != "total_marks")
            return m, gt_file
        for k, v in all_gt.items():
            if k in sid_clean or sid_clean in k:
                if "total_marks" not in v and v:
                    v["total_marks"] = sum(val for qk, val in v.items() if qk != "total_marks")
                return v, gt_file

    return None, None


def get_ground_truth_for_script(
    script_id: str,
    gt_file: str = "gt.txt",
    teacher_marks_dir: str = TEACHER_MARKS_DIR,
    lang: Optional[str] = None
) -> Optional[Dict[str, float]]:
    """Retrieve ground truth marks dictionary for a given script ID if available."""
    marks, _ = get_ground_truth_info_for_script(script_id, gt_file=gt_file, teacher_marks_dir=teacher_marks_dir, lang=lang)
    return marks


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

            # 1. Update ground_truth_marks.json (keep stage0b_teacher_marks.json strictly for model VLM extractions)
            gt_json_path = os.path.join(s_dir, "ground_truth_marks.json")
            try:
                with open(gt_json_path, "w", encoding="utf-8") as f:
                    json.dump([m.model_dump() for m in teacher_mark_items], f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[Ground Truth] Warning: could not write {gt_json_path}: {e}")

            # 2. Update extraction_result.json metadata (do not overwrite model's visual teacher_marks)
            ext_json_path = os.path.join(s_dir, "extraction_result.json")
            if os.path.exists(ext_json_path):
                try:
                    with open(ext_json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
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

