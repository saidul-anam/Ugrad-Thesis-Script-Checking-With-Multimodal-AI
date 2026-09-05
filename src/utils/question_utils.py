"""
Question Matching & Artifact Management Utilities.

Handles matching student exam scripts (e.g. 'SE_11_Q1_0001') to corresponding
extracted questions (e.g. '11_Q1'), and loading/saving question artifacts.
"""

import os
import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.core.schemas import ExtractedQuestion


def extract_question_id(name_or_path: str, lang: Optional[str] = None) -> Optional[str]:
    """
    Extract a canonical question identifier from a script filename, path, or question name.
    
    Examples:
      - "SE_11_Q1_0001.pdf" -> "SE_11_Q1"
      - "SB_11_Q1_0002.pdf" -> "SB_11_Q1"
      - "SE_11_Q1.pdf"      -> "SE_11_Q1"
      - "11_Q1_0001.pdf"    -> "SE_11_Q1" (if lang='english') or "11_Q1"
      - "11_Q1.pdf"         -> "11_Q1"
    """
    stem = Path(name_or_path).stem
    
    # 1. Exact SE / SB prefixed pattern: (SE|SB)_<code/num>_Q<num>
    match_pref = re.search(r'((?:SE|SB)_[0-9A-Za-z]+_Q[0-9A-Za-z]+)', stem, re.IGNORECASE)
    if match_pref:
        parts = match_pref.group(1).upper().split('_')
        return f"{parts[0]}_{parts[1]}_{parts[2]}"

    # 2. Number/Code followed by _Q<num> (e.g. 11_Q1)
    match_std = re.search(r'([0-9]{1,4}_Q[0-9]{1,3})', stem, re.IGNORECASE)
    if match_std:
        parts = match_std.group(1).split('_')
        base_qid = f"{parts[0]}_{parts[1].upper()}"
        if lang:
            prefix = "SE" if lang.lower() == "english" else ("SB" if lang.lower() == "bangla" else "")
            if prefix:
                return f"{prefix}_{base_qid}"
        return base_qid

    # 3. Inverted pattern: Q<number>_<number/code> (e.g., Q1_11)
    match_inv = re.search(r'Q([0-9]{1,3})_([0-9]{1,4})', stem, re.IGNORECASE)
    if match_inv:
        base_qid = f"{match_inv.group(2)}_Q{match_inv.group(1)}"
        if lang:
            prefix = "SE" if lang.lower() == "english" else ("SB" if lang.lower() == "bangla" else "")
            if prefix:
                return f"{prefix}_{base_qid}"
        return base_qid

    return stem


def find_question_artifact(
    question_id: str,
    lang: Optional[str] = None,
    questions_root: str = "outputs/questions"
) -> Optional[Path]:
    """
    Locates an extracted question JSON file based on question_id.
    
    Checks in:
      1. outputs/questions/<lang>/<question_id>.json (and cross-prefix aliases like 11_Q1 <-> SE_11_Q1)
      2. outputs/questions/<question_id>.json
      3. outputs/questions/<lang>/*<question_id>*.json
    """
    q_id_clean = question_id.strip()
    root = Path(questions_root)

    # Generate aliases (e.g. "SE_11_Q1" <-> "11_Q1")
    search_ids = [q_id_clean]
    if q_id_clean.upper().startswith(("SE_", "SB_")):
        search_ids.append(q_id_clean[3:])
    else:
        if lang == "english" or not lang:
            search_ids.append(f"SE_{q_id_clean}")
        if lang == "bangla" or not lang:
            search_ids.append(f"SB_{q_id_clean}")

    candidates: List[Path] = []
    for sid in search_ids:
        if lang:
            candidates.append(root / lang / f"{sid}.json")
        candidates.append(root / f"{sid}.json")

    for cand in candidates:
        if cand.exists():
            return cand

    # Search directory for partial match
    search_dir = (root / lang) if (lang and (root / lang).exists()) else root
    if search_dir.exists():
        for p in search_dir.glob("*.json"):
            for sid in search_ids:
                if sid.lower() in p.stem.lower():
                    return p

    return None


def load_question_for_script(
    script_id_or_path: str,
    lang: str = "english",
    question_override: Optional[str] = None,
    questions_root: str = "outputs/questions"
) -> Optional[ExtractedQuestion]:
    """
    Loads the ExtractedQuestion corresponding to a script ID or question override.
    
    Args:
        script_id_or_path: e.g. "SE_11_Q1_0001" or path to script
        lang: "english" or "bangla"
        question_override: explicit question ID (e.g. "11_Q1") or path to JSON
        questions_root: base directory where extracted questions are saved
    """
    target_q_id = None
    target_path = None

    if question_override:
        # Check if question_override is an existing direct file path
        if os.path.exists(question_override):
            target_path = Path(question_override)
        else:
            target_q_id = question_override
    else:
        target_q_id = extract_question_id(script_id_or_path, lang=lang)

    if not target_path and target_q_id:
        target_path = find_question_artifact(target_q_id, lang=lang, questions_root=questions_root)

    if target_path and target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ExtractedQuestion.model_validate(data)
        except Exception as e:
            print(f"[QuestionUtils] Warning: Failed parsing question artifact {target_path}: {e}")
            return None

    return None


def save_extracted_question(
    question: ExtractedQuestion,
    output_dir: str = "outputs/questions"
) -> Path:
    """
    Saves an ExtractedQuestion as structured JSON and human-readable Markdown.
    """
    lang = question.language or "general"
    dest_dir = Path(output_dir) / lang
    dest_dir.mkdir(parents=True, exist_ok=True)

    json_path = dest_dir / f"{question.question_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(question.model_dump_json(indent=2))

    # Also save Markdown summary for convenient inspection
    md_path = dest_dir / f"{question.question_id}.md"
    md_content = [
        f"# Exam Question: {question.question_id}",
        f"- **Subject / Language**: {question.language.capitalize()}",
        f"- **Total Marks**: {question.total_marks or 'N/A'}",
        f"- **Source File**: `{question.source_file or 'N/A'}`",
        f"- **Extracted At**: {question.extracted_at or datetime.now().isoformat()}",
        "",
        "## Question Text / Prompt",
        "```",
        question.question_text.strip(),
        "```",
        ""
    ]
    if question.sub_questions:
        md_content.append("## Sub-Questions Breakdown")
        for idx, sq in enumerate(question.sub_questions, 1):
            md_content.append(f"### Part {sq.get('part', idx)} (Marks: {sq.get('marks', 'N/A')})")
            md_content.append(sq.get("text", "").strip())
            md_content.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    return json_path
