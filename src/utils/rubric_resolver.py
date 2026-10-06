"""
Rubric and Answer Key Resolver for English Exam Papers.

Centralized registry and resolver for NCTB English exams across class levels:
- SSC Class 10 (e.g. SE_10_Q1)
- HSC Class 11 (e.g. SE_11_Q1)

Provides paper canonicalization, class-level detection, rubric resolution,
objective answer key resolution, and 100-mark schema integrity validation.
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import yaml

from src.core.schemas import ExtractedQuestion


def extract_class_number(paper_id_or_script: str) -> Optional[int]:
    """Extract integer class level from paper ID or path (e.g. SE_3_Q1 -> 3, SE_10_Q1 -> 10)."""
    s = str(paper_id_or_script).replace("\\", "/")
    # Try pattern SE_<num>_Q
    m = re.search(r'SE_0*([0-9]{1,2})_Q', s, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    # Try folder /se_<num>_q<num>/
    m_f = re.search(r'/(?:se_)?0*([0-9]{1,2})_q[0-9]+/', s, re.IGNORECASE)
    if m_f:
        try:
            return int(m_f.group(1))
        except ValueError:
            pass
    # Try <num>_Q
    m_q = re.search(r'(?:^|[_\-/])0*([0-9]{1,2})_Q[0-9]+', s, re.IGNORECASE)
    if m_q:
        try:
            return int(m_q.group(1))
        except ValueError:
            pass
    return None


def canonicalize_paper_id(name_or_path: str) -> str:
    """
    Extract canonical paper ID from script ID, file path, or paper argument across any class (1-12).
    
    Examples:
      - "SE_3_Q1_0001" -> "SE_3_Q1"
      - "se_6_q1" -> "SE_6_Q1"
      - "8_Q1" -> "SE_8_Q1"
      - "SE_10_Q1_0001" -> "SE_10_Q1"
      - "SE_11_Q1_0002" -> "SE_11_Q1"
      - "outputs/extracted/english/se_10_q1/SE_10_Q1_0001" -> "SE_10_Q1"
    """
    s = str(name_or_path).replace("\\", "/")
    # If path contains paper folder e.g. /se_10_q1/ or /se_6_q1/
    match_folder = re.search(r'/(?:english/)?(se_[0-9]+_q[0-9]+)/', s, re.IGNORECASE)
    if match_folder:
        return match_folder.group(1).upper()

    stem = Path(s).stem
    # 1. Match SE_<num>_Q<num>
    m1 = re.search(r'(SE_[0-9A-Za-z]+_Q[0-9A-Za-z]+)', stem, re.IGNORECASE)
    if m1:
        parts = m1.group(1).upper().split('_')
        return f"{parts[0]}_{parts[1]}_{parts[2]}"

    # 2. Match <num>_Q<num>
    m2 = re.search(r'([0-9]{1,4}_Q[0-9]{1,3})', stem, re.IGNORECASE)
    if m2:
        parts = m2.group(1).upper().split('_')
        return f"SE_{parts[0]}_{parts[1]}"

    # Fallback cleanup
    clean = re.sub(r'[^A-Za-z0-9_]', '', stem).upper()
    m_clean = re.match(r'^0*([0-9]{1,2}_Q[0-9A-Z]+)', clean)
    if m_clean:
        return f"SE_{m_clean.group(1)}"
    return clean or "SE_10_Q1"


def detect_class_level(paper_id: str) -> str:
    """
    Detect exam class level from paper ID for any grade from Class 1 to 12.
    'SE_3_Q1'  -> 'Primary_Class_3'
    'SE_6_Q1'  -> 'Junior_Class_6'
    'SE_8_Q1'  -> 'Junior_Class_8'
    'SE_10_Q1' -> 'SSC_Class_10'
    'SE_11_Q1' -> 'HSC_Class_11'
    """
    c = extract_class_number(paper_id)
    if c is not None:
        if 1 <= c <= 5:
            return f"Primary_Class_{c}"
        elif 6 <= c <= 8:
            return f"Junior_Class_{c}"
        elif c in (9, 10):
            return f"SSC_Class_{c}"
        elif c in (11, 12):
            return f"HSC_Class_{c}"
        return f"Class_{c}"

    pid = canonicalize_paper_id(paper_id).upper()
    if "_10_" in pid or pid.startswith("SE_10"):
        return "SSC_Class_10"
    if "_11_" in pid or pid.startswith("SE_11"):
        return "HSC_Class_11"
    return "SSC_Class_10"


def resolve_rubric_path(
    paper_id_or_script: str,
    custom_rubric_path: Optional[str] = None
) -> str:
    """
    Resolve the correct rubric YAML path for a given paper or script across any class (3 to 12).
    Order:
      1. Custom path (if exists)
      2. Exact paper rubric: configs/rubrics/<paper_id>.yaml
      3. Class-specific rubric: configs/rubrics/english_class<N>_*.yaml or english_class<N>.yaml
      4. Tier-level rubric:
         - Primary (3-5): configs/rubrics/english_primary.yaml
         - Junior (6-8): configs/rubrics/english_junior.yaml
         - Secondary (9-10): configs/rubrics/english_class10_ssc.yaml
         - Higher Secondary (11-12): configs/rubrics/english_class11_hsc.yaml
      5. Fallback: configs/rubrics/english_writing.yaml
    """
    if custom_rubric_path and os.path.exists(custom_rubric_path):
        return custom_rubric_path

    paper_id = canonicalize_paper_id(paper_id_or_script)
    class_num = extract_class_number(paper_id)
    rubrics_dir = Path("configs/rubrics")

    # 1. Exact paper rubric
    exact_candidates = [
        rubrics_dir / f"{paper_id}.yaml",
        rubrics_dir / f"{paper_id.lower()}.yaml",
    ]
    for cand in exact_candidates:
        if cand.exists():
            return str(cand)

    # 2. Class-specific rubric (e.g. english_class10_ssc.yaml, english_class8.yaml)
    if class_num is not None:
        class_patterns = [
            f"english_class{class_num}_*.yaml",
            f"english_class{class_num}.yaml",
            f"english_class{class_num}_*.yml",
            f"english_class{class_num}.yml",
        ]
        for pat in class_patterns:
            matches = sorted(rubrics_dir.glob(pat))
            if matches:
                return str(matches[0])

        # 3. Tier-level rubrics
        if class_num <= 5:
            p_prim = rubrics_dir / "english_primary.yaml"
            if p_prim.exists():
                return str(p_prim)
        elif 6 <= class_num <= 8:
            p_jun = rubrics_dir / "english_junior.yaml"
            if p_jun.exists():
                return str(p_jun)
        elif class_num in (9, 10):
            p_sec = rubrics_dir / "english_class10_ssc.yaml"
            if p_sec.exists():
                return str(p_sec)
        elif class_num >= 11:
            p_hsc = rubrics_dir / "english_class11_hsc.yaml"
            if p_hsc.exists():
                return str(p_hsc)

    # General fallback
    p_leg = rubrics_dir / "english_writing.yaml"
    if p_leg.exists():
        return str(p_leg)
    return str(rubrics_dir / "english_class10_ssc.yaml")


def resolve_answer_key_path(
    paper_id_or_script: str,
    answer_keys_dir: str = "configs/answer_keys"
) -> Optional[str]:
    """
    Locate the corresponding objective answer key YAML for a paper.
    """
    paper_id = canonicalize_paper_id(paper_id_or_script)
    root = Path(answer_keys_dir)

    candidates = [
        root / f"{paper_id}.yaml",
        root / f"{paper_id.lower()}.yaml",
        root / f"{paper_id[3:]}.yaml" if paper_id.startswith("SE_") else root / f"SE_{paper_id}.yaml"
    ]

    for c in candidates:
        if c.exists():
            return str(c)
    return None


def resolve_penalty_config_path(
    paper_id_or_script: str,
    penalties_dir: str = "configs/penalties"
) -> str:
    """
    Resolve the tunable penalty YAML configuration for a paper across any class (3 to 12).
    Order:
      1. Paper-specific: configs/penalties/<paper_id>.yaml
      2. Class-specific: configs/penalties/english_class<N>.yaml
      3. Tier-level configs:
         - Primary (3-5): configs/penalties/english_primary.yaml
         - Junior (6-8): configs/penalties/english_junior.yaml
         - Secondary (9-10): configs/penalties/english_class10.yaml
         - Higher Secondary (11-12): configs/penalties/english_class11.yaml
      4. Default: configs/penalties/english_class10.yaml
    """
    paper_id = canonicalize_paper_id(paper_id_or_script)
    class_num = extract_class_number(paper_id)
    root = Path(penalties_dir)

    # 1. Exact paper penalty config
    exact_candidates = [
        root / f"{paper_id}.yaml",
        root / f"{paper_id.lower()}.yaml",
    ]
    for cand in exact_candidates:
        if cand.exists():
            return str(cand)

    # 2. Class-specific penalty config
    if class_num is not None:
        p_class = root / f"english_class{class_num}.yaml"
        if p_class.exists():
            return str(p_class)

        # 3. Tier-level configs
        if class_num <= 5:
            p_prim = root / "english_primary.yaml"
            if p_prim.exists():
                return str(p_prim)
        elif 6 <= class_num <= 8:
            p_jun = root / "english_junior.yaml"
            if p_jun.exists():
                return str(p_jun)
        elif class_num in (9, 10):
            p_sec = root / "english_class10.yaml"
            if p_sec.exists():
                return str(p_sec)
        elif class_num >= 11:
            p_hsc = root / "english_class11.yaml"
            if p_hsc.exists():
                return str(p_hsc)

    return str(root / "english_class10.yaml")


def validate_paper_integrity(
    question_obj: Optional[ExtractedQuestion],
    rubric_data: Optional[Dict[str, Any]] = None,
    expected_total_marks: float = 100.0
) -> Tuple[bool, List[str]]:
    """
    Assert that the question paper and rubric are mathematically consistent
    and cover all expected sub-questions.
    """
    errors: List[str] = []
    if not question_obj:
        return False, ["Question object is None."]

    if not question_obj.sub_questions:
        errors.append(f"Question '{question_obj.question_id}' has 0 sub-questions.")

    sub_marks = sum(float(sq.get("max_marks") or sq.get("marks") or 0.0) for sq in question_obj.sub_questions)
    if abs(sub_marks - expected_total_marks) > 0.01:
        errors.append(f"Sub-question marks sum to {sub_marks:.1f}, expected {expected_total_marks:.1f}.")

    if rubric_data:
        modes = rubric_data.get("scoring_modes", {})
        covered_questions = set()
        for _, m in modes.items():
            for q in m.get("questions", []):
                covered_questions.add(str(q))

        for sq in question_obj.sub_questions:
            q_no = str(sq.get("q_no") or "")
            clean_q = re.sub(r'\(.*?\)', '', q_no).strip()
            if q_no not in covered_questions and clean_q not in covered_questions:
                errors.append(f"Question Q{q_no} is present in paper but missing from rubric scoring modes.")

    is_valid = len(errors) == 0
    return is_valid, errors
