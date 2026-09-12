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


def parse_sub_questions_from_text(question_text: str, lang: str = "english") -> List[Dict[str, Any]]:
    """
    Parse individual sub-questions and criteria from raw question text or rubric definitions.
    Enriches with canonical question IDs (e.g. '1(A)', '1(B)', '2', ..., '11'), topics, and marks.
    """
    import yaml

    # 1. Try aligning with official rubric YAML if available
    lang_lower = (lang or "english").lower()
    rubric_file = "configs/rubrics/english_writing.yaml" if "english" in lang_lower else "configs/rubrics/bangla_creative_question.yaml"
    rubric_path = Path(rubric_file)

    rubric_sub_qs: List[Dict[str, Any]] = []
    if rubric_path.exists():
        try:
            with open(rubric_path, "r", encoding="utf-8") as rf:
                rdata = yaml.safe_load(rf)
            for c in rdata.get("criteria", []):
                cid = c.get("id", "")
                cname = c.get("name", "")
                m = re.search(r'Q\s*([0-9]{1,2}(?:\([A-Za-z0-9\u0980-\u09FF]\))?)', cname, re.IGNORECASE)
                q_no = m.group(1) if m else cid
                rubric_sub_qs.append({
                    "q_no": q_no,
                    "name": cname,
                    "title": cname,
                    "marks": float(c.get("max_marks", 10.0)),
                    "max_marks": float(c.get("max_marks", 10.0)),
                    "text": c.get("description", "")
                })
        except Exception:
            pass

    if rubric_sub_qs:
        return rubric_sub_qs

    # 2. Fallback: Parse question text using numbered patterns
    parsed: List[Dict[str, Any]] = []
    lines = question_text.split("\n")
    cur_q: Optional[str] = None
    cur_title: str = ""
    cur_lines: List[str] = []

    pattern = re.compile(r'^(?:Part-[A-Z]\s*:.*|(?:([0-9]{1,2})\.\s*|([A-B])\.\s*)(.*))', re.IGNORECASE)
    for line in lines:
        m = pattern.match(line.strip())
        if m and (m.group(1) or m.group(2)):
            if cur_q:
                parsed.append({
                    "q_no": cur_q,
                    "name": cur_title or f"Question {cur_q}",
                    "title": cur_title or f"Question {cur_q}",
                    "text": "\n".join(cur_lines).strip()
                })
            cur_q = m.group(1) or m.group(2)
            cur_title = (m.group(3) or "").strip()
            cur_lines = [cur_title]
        elif cur_q:
            cur_lines.append(line)

    if cur_q:
        parsed.append({
            "q_no": cur_q,
            "name": cur_title or f"Question {cur_q}",
            "title": cur_title or f"Question {cur_q}",
            "text": "\n".join(cur_lines).strip()
        })

    return parsed


def load_question_from_rubric(
    lang: str = "english",
    rubric_path: Optional[str] = None
) -> Optional[ExtractedQuestion]:
    """
    Construct an ExtractedQuestion artifact from rubric YAML configurations
    when pre-extracted question JSON artifacts are not found on disk.
    """
    import yaml

    if not rubric_path:
        lang_lower = (lang or "english").lower()
        if "bangla" in lang_lower:
            rubric_path = "configs/rubrics/bangla_creative_question.yaml"
            default_qid = "SB_11_Q1"
        else:
            rubric_path = "configs/rubrics/english_writing.yaml"
            default_qid = "SE_11_Q1"
    else:
        default_qid = Path(rubric_path).stem

    p = Path(rubric_path)
    if not p.exists():
        return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        sub_qs = parse_sub_questions_from_text("", lang=lang)
        q_text = "\n".join(f"{sq['title']}: {sq['text']}" for sq in sub_qs)

        q_obj = ExtractedQuestion(
            question_id=default_qid,
            language=data.get("subject", lang).lower(),
            title=data.get("question_type", f"{lang.capitalize()} Exam"),
            question_text=q_text,
            total_marks=float(data.get("total_marks", 100.0)),
            sub_questions=sub_qs,
            source_file=str(p),
            extracted_at=datetime.now().isoformat()
        )
        return q_obj
    except Exception as e:
        print(f"[QuestionUtils] Warning: Failed creating ExtractedQuestion from rubric {p}: {e}")
        return None


def load_question_for_script(
    script_id_or_path: str,
    lang: str = "english",
    question_override: Optional[str] = None,
    questions_root: str = "outputs/questions"
) -> Optional[ExtractedQuestion]:
    """
    Loads the ExtractedQuestion corresponding to a script ID or question override.
    Falls back to rubric-defined question context if pre-extracted JSON is absent.
    
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
            q_obj = ExtractedQuestion.model_validate(data)
            if not q_obj.sub_questions:
                q_obj.sub_questions = parse_sub_questions_from_text(q_obj.question_text, lang=q_obj.language)
                try:
                    save_extracted_question(q_obj, output_dir=questions_root)
                except Exception:
                    pass
            return q_obj
        except Exception as e:
            print(f"[QuestionUtils] Warning: Failed parsing question artifact {target_path}: {e}")

    # Fallback to rubric-defined question context
    rubric_q = load_question_from_rubric(lang=lang)
    if rubric_q:
        if target_q_id:
            rubric_q.question_id = target_q_id
        try:
            save_extracted_question(rubric_q, output_dir=questions_root)
        except Exception:
            pass
        return rubric_q

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


def extract_question_vocab(
    question: ExtractedQuestion,
    max_tokens: int = 250
) -> List[str]:
    """
    Extract domain vocabulary terms from an ExtractedQuestion artifact.
    These terms serve as reference vocabulary for Stage 1 handwriting transcription
    to help decipher difficult cursive strokes without autocorrecting student errors.
    Prioritizes key answer entities (MCQ options, cloze box clues, answer key targets)
    over generic reading passage prose.
    """
    if not question or not question.question_text:
        return []

    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
        "shall", "should", "will", "would", "may", "might", "must", "that", "which", "who",
        "whom", "this", "these", "those", "am", "it", "its", "they", "them", "their",
        "theirs", "we", "us", "our", "ours", "you", "your", "yours", "he", "him", "his",
        "she", "her", "hers", "what", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "now", "read", "following",
        "text", "passage", "answer", "questions", "write", "down", "give", "make", "fill",
        "blanks", "suitable", "word", "words", "box", "needed", "grammatical", "changes",
        "necessary", "appropriate", "gap", "sentences", "jumbled", "rearrange", "proper",
        "sequence", "marks", "time", "hours", "subject", "class", "annual", "examination",
        "full", "margin", "indicate", "part", "reading", "writing", "code", "true", "false",
        "correct", "alternatives", "table", "chart", "diagram", "story", "theme", "paragraph"
    }

    priority_tokens: List[str] = []

    # 1. Check if official answer key exists in configs/answer_keys/<qid>.yaml
    try:
        import yaml
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent
        key_path = repo_root / "configs" / "answer_keys" / f"{question.question_id}.yaml"
        if key_path.exists():
            with open(key_path, "r", encoding="utf-8") as f:
                kd = yaml.safe_load(f) or {}
            for part in ("mode_a", "mode_b"):
                for q_id, q_data in kd.get(part, {}).items():
                    if isinstance(q_data, dict):
                        for item in q_data.get("items", []):
                            if isinstance(item, dict):
                                for acc in item.get("accepted", []) + item.get("key_points", []):
                                    priority_tokens.extend(re.findall(r'[A-Za-z\u0980-\u09FF]+', str(acc)))
    except Exception:
        pass

    # 2. Extract clue table cells and MCQ options from the question prompt
    for line in question.question_text.splitlines():
        if "|" in line and not line.strip().startswith("| :"):
            for cell in line.split("|"):
                priority_tokens.extend(re.findall(r'[A-Za-z\u0980-\u09FF]+', cell))
        elif re.search(r'\b(?:i|ii|iii|iv|v)\.\s+', line):
            priority_tokens.extend(re.findall(r'[A-Za-z\u0980-\u09FF]+', line))

    # 3. Sub-questions titles and guidance text
    for sq in question.sub_questions:
        sq_text = str(sq.get("name", "")) + " " + str(sq.get("text", ""))
        priority_tokens.extend(re.findall(r'[A-Za-z\u0980-\u09FF]+', sq_text))

    # 4. General passage tokens
    general_tokens = re.findall(r'[A-Za-z\u0980-\u09FF]+(?:-[A-Za-z\u0980-\u09FF]+)*', question.question_text)

    seen = set()
    vocab: List[str] = []

    for tok in priority_tokens + general_tokens:
        clean_tok = tok.strip()
        low = clean_tok.lower()
        if low in stop_words:
            continue
        if len(clean_tok) < 3:
            continue
        if len(clean_tok) == 3 and not clean_tok.isupper() and low not in {"why", "box", "fly", "net"}:
            continue
        if low in seen:
            continue
        seen.add(low)
        vocab.append(clean_tok)
        if len(vocab) >= max_tokens:
            break

    return vocab
