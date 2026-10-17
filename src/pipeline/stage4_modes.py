"""
Stage 4, rubric-driven scoring modes.

The model JUDGES; the code SCORES. For every question the rubric YAML (`scoring_modes`,
`mode_specifications`) decides the mode:

  Mode A (item-scored)  : model extracts each item answer, code checks it against the answer key
                          and sums `correct * mark_per_item` (rearrangement is position-scored in code).
  Mode B (point-scored) : model awards each item on the rubric's discrete scale, code clamps to the
                          scale and sums.
  Mode C (band-scored)  : model returns the four criterion subscores + a structural audit + evidence,
                          code clamps to the criteria ceilings, computes raw_total, applies the hard
                          caps (with computed verbatim-overlap metrics), snaps to 0.5 and assigns the band.

Nothing here awards marks on a parse failure: the evaluator marks such questions `unscored`.
"""

import difflib
import functools
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import yaml

from src.core.schemas import AlignedAnswerItem
from src.pipeline.token_guard import clean_rubric_answer
from src.utils.ground_truth import canonicalize_question_key
from src.utils.linguistic_sanitizer import get_english_lexicon
from src.pipeline.arbitration.candidate_selector import levenshtein
from src.pipeline.arbitration.symbolic_evidence import align_chars
from src.pipeline.allograph_calibrator import PLAUSIBLE_ALLOGRAPH_PAIRS


# ---------------------------------------------------------------------------
# Rubric specification
# ---------------------------------------------------------------------------
@dataclass
class QuestionSpec:
    q_no: str
    mode: str                       # "A" | "B" | "C"
    task_type: str
    max_mark: float
    mark_per_item: float = 0.0
    n_items: int = 0
    scale: Dict[float, str] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)
    criteria_ceilings: Dict[str, float] = field(default_factory=dict)
    layout_components: Dict[str, List[str]] = field(default_factory=dict)
    bands: Dict[str, str] = field(default_factory=dict)
    hard_caps: Dict[str, str] = field(default_factory=dict)
    general_rules: List[str] = field(default_factory=list)


def rubric_is_mode_based(rubric_data: Dict[str, Any]) -> bool:
    return bool(rubric_data) and "scoring_modes" in rubric_data and "mode_specifications" in rubric_data


def _parse_allocation(text: str) -> Tuple[float, int]:
    """'0.5 x 10 = 5' -> (0.5, 10); '10' -> (0, 0)."""
    m = re.search(r"([0-9.]+)\s*[x×]\s*([0-9]+)", text or "")
    if m:
        return float(m.group(1)), int(m.group(2))
    return 0.0, 0


def load_rubric_specs(rubric_data: Dict[str, Any]) -> Dict[str, QuestionSpec]:
    """Build one QuestionSpec per question from a mode-based rubric. Returns {} for legacy rubrics."""
    if not rubric_is_mode_based(rubric_data):
        return {}
    specs: Dict[str, QuestionSpec] = {}
    modes = rubric_data.get("scoring_modes", {})
    q_to_mode: Dict[str, str] = {}
    for mode_key, m in modes.items():
        letter = mode_key.split("_")[0].upper()[:1]
        for q in m.get("questions", []):
            q_to_mode[canonicalize_question_key(str(q))] = letter
    ms = rubric_data.get("mode_specifications", {})
    mode_a = ms.get("mode_a_item_scored", {}).get("tasks", {})
    mode_b = ms.get("mode_b_point_scored", {}).get("tasks", {})
    mode_c = ms.get("mode_c_band_scored", {})
    mode_c_tasks = mode_c.get("tasks", {})
    hard_caps = mode_c.get("hard_caps", {})
    bands_by_max = mode_c.get("performance_bands_by_max_mark", {})
    # general_evaluation_guidelines is a list of single-key dicts
    general = []
    for g in rubric_data.get("general_evaluation_guidelines", []) or []:
        if isinstance(g, dict):
            for k, v in g.items():
                general.append(f"{k}: {v}")
        else:
            general.append(str(g))

    def _task_entry(tasks: Dict[str, Any], q_no: str) -> Tuple[str, Dict[str, Any]]:
        q_key = q_no.replace("(", "_").replace(")", "")
        for name, entry in tasks.items():
            if name.upper().startswith(f"Q{q_key.upper()}_"):
                return name, entry
        return "", {}

    for qm in rubric_data.get("question_map", []):
        q_no = canonicalize_question_key(str(qm.get("question_no", "")))
        mode = str(qm.get("mode") or q_to_mode.get(q_no, "C")).upper()[:1]
        max_mark = float(qm.get("standard_marks") or 10.0)
        spec = QuestionSpec(q_no=q_no, mode=mode, task_type=str(qm.get("task_type") or ""), max_mark=max_mark,
                            general_rules=general, hard_caps=dict(hard_caps))
        mark_per_item, n_items = _parse_allocation(str(qm.get("breakdown") or ""))
        spec.mark_per_item, spec.n_items = mark_per_item, n_items
        if mode == "A":
            _, entry = _task_entry(mode_a, q_no)
            spec.rules = [str(r) for r in entry.get("rules", [])]
            mpi, n = _parse_allocation(str(entry.get("allocation", "")))
            if mpi:
                spec.mark_per_item, spec.n_items = mpi, n
        elif mode == "B":
            _, entry = _task_entry(mode_b, q_no)
            spec.rules = [str(r) for r in entry.get("rules", [])]
            spec.scale = {float(k): str(v) for k, v in (entry.get("scale") or {}).items()}
            mpi, n = _parse_allocation(str(entry.get("allocation", "")))
            if mpi:
                spec.mark_per_item, spec.n_items = mpi, n
        else:
            _, entry = _task_entry(mode_c_tasks, q_no)
            spec.criteria_ceilings = {k: float(v) for k, v in (entry.get("criteria_ceilings") or {}).items()}
            spec.layout_components = {k: list(v) for k, v in (entry.get("layout_components") or {}).items()}
            if entry.get("max_mark"):
                spec.max_mark = float(entry["max_mark"])
            m_int = int(spec.max_mark)
            raw_b = (
                bands_by_max.get(f"max_{m_int}")
                or bands_by_max.get(m_int)
                or bands_by_max.get(str(m_int))
                or {}
            )
            spec.bands = dict(raw_b)
        specs[q_no] = spec
    return specs


# ---------------------------------------------------------------------------
# Answer keys
# ---------------------------------------------------------------------------
def load_answer_key(question_id: Optional[str], keys_dir: str = "configs/answer_keys") -> Dict[str, Any]:
    if not question_id:
        return {}
    for cand in (f"{question_id}.yaml", f"{question_id}.yml"):
        p = os.path.join(keys_dir, cand)
        if os.path.exists(p):
            return _load_yaml_cached(p, os.path.getmtime(p))
    return {}


@functools.lru_cache(maxsize=32)
def _load_yaml_cached(path: str, mtime: float) -> Dict[str, Any]:
    """Parsed answer key, re-read only when the file changes (the segmenter looks keys up very often)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def key_source_text(answer_key: Dict[str, Any], q_no: str) -> str:
    """Source passage / poem stored for a question under `source_texts:` in the answer key ('' if none)."""
    for k, v in ((answer_key or {}).get("source_texts") or {}).items():
        if canonicalize_question_key(str(k)) == canonicalize_question_key(str(q_no)):
            return str(v or "").strip()
    return ""


def key_for_question(answer_key: Dict[str, Any], q_no: str) -> Tuple[str, Dict[str, Any]]:
    """Return ('mode_a'|'mode_b'|'', entry) for a question."""
    for section in ("mode_a", "mode_b"):
        sec = answer_key.get(section) or {}
        for k, v in sec.items():
            if canonicalize_question_key(str(k)) == q_no:
                return section, v
    return "", {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def snap_half(x: float) -> float:
    return round(round(float(x) * 2) / 2, 1)


def band_for(score: float, bands: Dict[str, str]) -> Optional[str]:
    """bands like {'band_4': '8-10', 'band_3': '6-7: descriptor', ...}."""
    if not bands:
        return None
    for name, rng in bands.items():
        rng_str = str(rng).split("|")[0].split(":")[0].replace("marks", "").strip()
        if "-" in rng_str:
            parts = rng_str.split("-", 1)
            try:
                lo, hi = float(parts[0].strip()), float(parts[1].strip())
                if lo <= score <= hi:
                    return name.replace("_", " ").title()
            except ValueError:
                continue
        else:
            try:
                if float(rng_str) == score:
                    return name.replace("_", " ").title()
            except ValueError:
                continue
    # score between integer bands (e.g. 7.5) -> lower band boundary rule: use floor
    return band_for(float(int(score)), bands) if score != int(score) else None


_WORD = re.compile(r"[a-zঀ-৿']+")


def verbatim_overlap(answer: str, source: str, n: int = 4) -> float:
    """Fraction of the answer's word n-grams that occur verbatim in the source (0..1)."""
    a = _WORD.findall((answer or "").lower())
    s = _WORD.findall((source or "").lower())
    if len(a) < n or len(s) < n:
        return 0.0
    src = {tuple(s[i:i + n]) for i in range(len(s) - n + 1)}
    grams = [tuple(a[i:i + n]) for i in range(len(a) - n + 1)]
    return sum(1 for g in grams if g in src) / len(grams)


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9ঀ-৿ ]", "", str(s or "").lower()).strip()


def _clean_cand_text(cand: str) -> str:
    """Strip student exam boilerplate like 'Ans:', 'Correct Ans:', question labels '(a)', '(i)'."""
    s = str(cand or "").strip()
    s = re.sub(r"^(?:ans(?:wer)?(?:\s+to(?:\s+the)?(?:\s+q(?:uestion)?(?:\s+no\.?)?)?)?|correct\s+ans(?:wer)?)\s*[:.-]*\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\(?[a-j0-9ivx]+\)?[.:\s-]+\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


# Canonical Roman numeral normalization map supporting up to 20 options
_ROMAN = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10",
    "xi": "11", "xii": "12", "xiii": "13", "xiv": "14", "xv": "15",
    "xvi": "16", "xvii": "17", "xviii": "18", "xix": "19", "xx": "20",
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "10": "10"
}


_OPT = r"(?:i{1,3}|iv|v|[a-e]|[1-5])"
_FORMATTED_LABEL_RE = re.compile(rf"(?:^|(?<=\s))(?:\(\s*({_OPT})\s*\)|({_OPT})\s*(?:\)|[.:](?=\s+\S)))", re.I)
_LABELS_ONLY_RE = re.compile(rf"^\s*\(?\s*{_OPT}\s*\)?(?:\s*(?:[,/&]|and|or)?\s*\(?\s*{_OPT}\s*\)?)*\s*[.]?\s*$", re.I)


def selected_option_labels(candidate: str, accepted: List[str] = ()) -> Set[str]:
    """Option labels the student chose: labels written as labels ('(ii)', 'ii)', 'iv.'), or every label of an
    answer made of labels only ('ii, iii'). The text of accepted options is ignored, so 'all of i, ii & iii' or
    the article 'a' in 'He was a patriotic person' are not read as choices."""
    s = str(candidate or "")
    for a in accepted or ():
        a = str(a).strip()
        if len(a) > 3:
            s = re.sub(re.escape(a), " ", s, flags=re.I)
    if _LABELS_ONLY_RE.match(s):
        return {m.lower() for m in re.findall(rf"\b{_OPT}\b", s, flags=re.I)}
    return {(m[0] or m[1]).lower() for m in _FORMATTED_LABEL_RE.findall(s)}


def _matches_accepted(candidate: str, accepted: List[str], task_type: str = "") -> bool:
    c = _norm(candidate)
    if not c:
        return False
    c_clean = _norm(_clean_cand_text(candidate))
    c_alt = _ROMAN.get(c, _ROMAN.get(c_clean, c))
    tt_upper = (task_type or "").upper()
    is_mcq = tt_upper == "MCQ"
    is_matching = "MATCH" in tt_upper or any("+" in a for a in accepted)
    is_tf = "TRUE" in tt_upper or "FALSE" in tt_upper

    # Sentence Matching / Column matching: candidate may contain formula (e.g. a+iv+ii) and/or combined sentence
    if is_matching:
        formulas_accepted = [a for a in accepted if "+" in a]
        if formulas_accepted:
            cand_formulas = re.findall(
                r"([a-e]\s*\+\s*[ivx]+\s*\+\s*[ivx]+|[ivx]+\s*\+\s*[ivx]+)",
                candidate,
                re.IGNORECASE,
            )
            for cf in cand_formulas:
                norm_cf = re.sub(r"[^a-z0-9]", "", cf.lower())
                for fa in formulas_accepted:
                    norm_fa = re.sub(r"[^a-z0-9]", "", fa.lower())
                    if norm_cf == norm_fa or norm_cf.endswith(norm_fa) or norm_fa.endswith(norm_cf):
                        return True
        for a in accepted:
            norm_a = re.sub(r"[^a-z0-9 ]", "", a.lower()).strip()
            if len(norm_a) >= 10:
                if norm_a in c or c in norm_a:
                    return True
                if difflib.SequenceMatcher(None, c, norm_a).ratio() >= 0.65:
                    return True

    # True / False statement handling: if expected answer is True or False with correction
    if is_tf:
        cand_lower = candidate.lower().strip()
        is_cand_false = bool(re.search(r"\b(false|f)\b", cand_lower))
        is_cand_true = bool(re.search(r"\b(true|t)\b", cand_lower)) and not is_cand_false
        has_false_accepted = any(re.search(r"\b(false|f)\b", a.lower()) for a in accepted)
        has_true_accepted = any(re.search(r"\b(true|t)\b", a.lower()) for a in accepted)

        if has_true_accepted and is_cand_true:
            return True
        if has_false_accepted and is_cand_false:
            # Check if student gave the correction
            cand_corr = _norm(re.sub(r"^(?:false|f)[\s.:,-]*(?:correct\s*(?:ans|answer)[\s.:,-]*)?", "", cand_lower))
            cand_corr = re.sub(r"^[()]+|[()]+$", "", cand_corr).strip()
            for a in accepted:
                norm_a = _norm(a)
                a_corr = _norm(re.sub(r"^(?:false|f)[\s.:,-]*(?:correct\s*(?:ans|answer)[\s.:,-]*)?", "", a.lower()))
                a_corr = re.sub(r"^[()]+|[()]+$", "", a_corr).strip()
                if not a_corr or a_corr in ("false", "f"):
                    return True
                if cand_corr and (cand_corr == a_corr or a_corr in cand_corr or cand_corr in a_corr):
                    return True
                if cand_corr and len(cand_corr) >= 8 and len(a_corr) >= 8:
                    if difflib.SequenceMatcher(None, cand_corr, a_corr).ratio() >= 0.70:
                        return True

    for a in accepted:
        an = _norm(a)
        if not an:
            continue
        if c == an or c_clean == an or c_alt == _ROMAN.get(an, an):
            return True
        # OCR whitespace healing: e.g. "com pletely" vs "completely", "phy sically" vs "physically"
        if len(c) >= 6 and len(an) >= 6 and c.replace(" ", "") == an.replace(" ", ""):
            return True
        if len(c_clean) >= 6 and len(an) >= 6 and c_clean.replace(" ", "") == an.replace(" ", ""):
            return True

        # Candidate might contain multiple alternatives separated by commas, slashes, or 'or' (e.g. "classy, cultural")
        cand_parts = [p.strip() for p in re.split(r'[,/;\n]|\bor\b', candidate) if p.strip()]
        if len(cand_parts) > 1:
            for cp in cand_parts:
                cp_norm = _norm(cp)
                cp_clean = _norm(_clean_cand_text(cp))
                if cp_norm == an or cp_clean == an or _ROMAN.get(cp_norm, cp_norm) == _ROMAN.get(an, an):
                    return True

        # Multi-word phrase containment for Table Completion, Cloze, etc.
        # e.g., "was Published completely in six separate Volumes." vs "published completely in six separate volumes"
        # e.g., "the age of ten, afther his grand uncle's death" vs "at the age of ten"
        an_words = an.split()
        if len(an_words) >= 2 and len(an) >= 8:
            if an in c or an in c_clean or c in an or c_clean in an:
                return True
            an_collapsed = an.replace(" ", "")
            c_collapsed = c.replace(" ", "")
            c_clean_collapsed = c_clean.replace(" ", "")
            if an_collapsed in c_collapsed or an_collapsed in c_clean_collapsed or c_collapsed in an_collapsed:
                return True
            # Strip common leading auxiliary verbs/articles like "was", "the", "a", "an", "at", "in"
            an_core = re.sub(r"^(?:was|is|are|were|the|a|an|at|in|by|from)\s+", "", an).strip()
            c_core = re.sub(r"^(?:was|is|are|were|the|a|an|at|in|by|from)\s+", "", c_clean).strip()
            an_core_c = an_core.replace(" ", "")
            c_core_c = c_core.replace(" ", "")
            if len(an_core) >= 6:
                if an_core in c or an_core in c_core or c_core in an_core:
                    return True
                if an_core_c in c_collapsed or an_core_c in c_core_c or c_core_c in an_core_c:
                    return True

        # MCQ only: the student may write the option text (or option letter + text)
        if is_mcq:
            # Case 1: Student wrote option text or option letter + text (e.g. "ii marching", "marching")
            if len(an) >= 3 and re.search(r'\b' + re.escape(an) + r'\b', c):
                return True
            # Case 2: Candidate contains majority of accepted multi-word answer (avoid short slips like "in", "a")
            if len(c) >= 5 and len(an) >= 5 and (c in an) and (len(c) / len(an) >= 0.7):
                return True

        # tolerate a single-character slip in words of length >= 5 (rubric: minor spelling slips),
        # but only if candidate is NOT a different legitimate English word (e.g. "healthy" != "health")
        if len(an) >= 5 and abs(len(an) - len(c)) <= 1 and difflib.SequenceMatcher(None, c, an).ratio() >= 0.85:
            lexicon = get_english_lexicon()
            if c not in lexicon:
                return True
        # Cursive stroke / allograph tolerance for non-words (e.g. uncrossed 't' <-> 'l', 'v' <-> 'r')
        if len(an) >= 4 and levenshtein(c, an) <= 2:
            lexicon = get_english_lexicon()
            if c not in lexicon:
                ops = align_chars(c, an)
                non_match = [o for o in ops if o.op != "match"]
                if non_match:
                    chars_inv = {o.src for o in non_match if o.src} | {o.tgt for o in non_match if o.tgt}
                    if any(chars_inv.issubset(pair) for pair in PLAUSIBLE_ALLOGRAPH_PAIRS):
                        return True
    return False


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
MODE_A_TAG = "MODE A (ITEM-SCORED)"
MODE_B_TAG = "MODE B (POINT-SCORED)"
MODE_C_TAG = "MODE C (BAND-SCORED)"

STAGE4_MODES_SYSTEM_PROMPT = (
    "You are an experienced NCTB board examiner. You do not decide the final mark; you report exactly what "
    "the student wrote for each item, or the criterion-level judgements the rubric asks for, so that the "
    "marks can be computed deterministically. Output valid JSON only."
)


def build_mode_a_prompt(answer: AlignedAnswerItem, spec: QuestionSpec, key_entry: Dict[str, Any], question_prompt_text: str) -> str:
    items = key_entry.get("items") or []
    seq = key_entry.get("correct_sequence")
    if seq:
        sentences = key_entry.get("sentences") or {}
        sent_lines = "\n".join(f"  {k}. {v}" for k, v in sentences.items())
        task = (
            "This is a REARRANGEMENT task. Extract the ORDER of sentence letters the student wrote (or the order of "
            "the sentences they rewrote in full; map each rewritten sentence to its letter). Output the student's "
            "sequence as a list of letters in the order written, including a letter more than once if the student did.\n"
            f"Original sentences:\n{sent_lines}"
        )
        schema = '{"student_sequence": ["c", "h", ...], "notes": "one sentence"}'
    else:
        item_lines = []
        for it in items:
            q = it.get("question") or ""
            item_lines.append(f"  ({it.get('label')}) {q}".rstrip())
        cloze_note = ""
        if "cloze" in (spec.task_type or "").lower():
            cloze_note = (
                "\nThis is a CLOZE (fill-in-the-blanks) task. Students often rewrite the whole passage with the gaps filled "
                "instead of listing (a)...(j). In that case, align the student's text with the gapped passage in the OFFICIAL "
                "QUESTION and report the word the student put at each gap position (a)...(j). Report the word exactly as written."
            )
        task = (
            "For EACH item label below, extract exactly what the student answered (option letter/roman numeral and/or "
            "the words written). If the student left it blank, use \"\". Do not judge correctness; extraction only."
            + cloze_note + "\n"
            "Items:\n" + "\n".join(item_lines)
        )
        schema = '{"items": [{"item_label": "a", "candidate_answer": "the student\'s answer for this item"}], "notes": "one sentence"}'
    clean_ans = clean_rubric_answer(answer.answer_text)
    return f"""{MODE_A_TAG} answer extraction for Question {answer.q_no} ({spec.task_type}, max {spec.max_mark:g} marks).

OFFICIAL QUESTION:
\"\"\"
{question_prompt_text.strip()[:1800]}
\"\"\"

STUDENT'S ANSWER (verified transcription):
\"\"\"
{clean_ans}
\"\"\"

TASK:
{task}

Output ONLY JSON: {schema}"""


def build_mode_b_prompt(answer: AlignedAnswerItem, spec: QuestionSpec, key_entry: Dict[str, Any], question_prompt_text: str) -> str:
    avail = float(key_entry.get("marks_available") or spec.mark_per_item or 2.0)
    scale_lines = "\n".join(f"  {v:g}: {d}" for v, d in sorted(spec.scale.items(), reverse=True)) or f"  {avail:g}: fully correct\n  0: wrong or absent"
    rules = "\n".join(f"  - {r}" for r in spec.rules)
    if key_entry.get("items"):
        ref = "\n".join(
            f"  ({it.get('label')}) {it.get('question', '')}\n      key points: " + "; ".join(it.get("key_points", []))
            for it in key_entry["items"]
        )
        n_items = len(key_entry["items"])
        item_note = "Score each lettered item (a)-(e)."
    else:
        pts = key_entry.get("sequence_points") or []
        ref = "  Expected content in passage order (any 5 earn credit):\n" + "\n".join(f"    - {p}" for p in pts)
        n_items = spec.n_items or 5
        item_note = f"Score each of the {n_items} student boxes/items in order (label them 1..{n_items})."
    clean_ans = clean_rubric_answer(answer.answer_text)
    return f"""{MODE_B_TAG} for Question {answer.q_no} ({spec.task_type}, {n_items} items x {avail:g} marks = {spec.max_mark:g}).

OFFICIAL QUESTION:
\"\"\"
{question_prompt_text.strip()[:1800]}
\"\"\"

REFERENCE ANSWER KEY (from the passage):
{ref}

STUDENT'S ANSWER (verified transcription):
\"\"\"
{clean_ans}
\"\"\"

SCALE PER ITEM (use ONLY these values):
{scale_lines}

RULES:
{rules if rules else '  - Content dominates syntax. Spelling/grammar slips do not reduce marks if meaning is clear.'}
  - Ignore transcription markers such as [struck: ...], [unclear: ...], [illegible].
  - A missing item scores 0.

{item_note}
Output ONLY JSON:
{{"items": [{{"item_label": "a", "candidate_answer": "short quote of what the student wrote", "rationale": "one sentence assessing accuracy against key points", "marks_awarded": {avail:g}}}], "notes": "one sentence"}}"""


def build_mode_c_prompt(answer: AlignedAnswerItem, spec: QuestionSpec, question_prompt_text: str, errors_text: str, source_text: str = "") -> str:
    ceilings = "\n".join(f"  - {k}: max {v:g}" for k, v in spec.criteria_ceilings.items())
    bands_formatted = []
    for k, v in spec.bands.items():
        title = k.replace("_", " ").title()
        v_str = str(v).strip()
        if "marks" in v_str.lower() or "|" in v_str or ":" in v_str:
            bands_formatted.append(f"  - {title}: {v_str}")
        else:
            bands_formatted.append(f"  - {title}: {v_str} marks")
    bands = "\n".join(bands_formatted)
    caps = "\n".join(f"  - {k}: {v}" for k, v in spec.hard_caps.items())
    general_list = list(spec.general_rules[:10])
    general_list.append("OCR Robustness: Do NOT penalise physical paragraph splits, indentations, line breaks, or structural formatting that may be artifacts of OCR extraction. Grade writing quality, logical progression, and organization & cohesion.")
    general = "\n".join(f"  - {g}" for g in general_list)
    layout = ""
    if spec.layout_components:
        layout = "\nLAYOUT COMPONENTS TO CHECK:\n" + "\n".join(f"  - {k}: {', '.join(v)}" for k, v in spec.layout_components.items())
    src = f"\nSOURCE TEXT (for verbatim-copy judgement):\n\"\"\"\n{source_text.strip()[:1200]}\n\"\"\"\n" if source_text else ""
    clean_ans = clean_rubric_answer(answer.answer_text)
    return f"""{MODE_C_TAG} criterion judgement for Question {answer.q_no} ({spec.task_type}, max {spec.max_mark:g} marks).

OFFICIAL QUESTION:
\"\"\"
{question_prompt_text.strip()[:1800]}
\"\"\"
{src}
STUDENT'S ANSWER (verified transcription; handwriting ambiguities already resolved):
\"\"\"
{clean_ans}
\"\"\"

CONFIRMED LINGUISTIC ERRORS (for the language_mechanics criterion only; judge error DENSITY holistically, no per-error arithmetic):
{errors_text}

CRITERIA AND CEILINGS (award each criterion 0 up to its ceiling, in 0.5 steps):
{ceilings}

PERFORMANCE BANDS FOR THIS MAX MARK (the sum of your criteria must fall in the band that matches the overall quality):
{bands}

HARD CAPS (report the facts; the cap itself is applied by the marking software):
{caps}
{layout}
GENERAL GUIDELINES:
{general}

Output ONLY JSON:
{{
  "raw_subscores": {{{", ".join(f'"{k}": <number between 0.0 and {v:g}>' for k, v in spec.criteria_ceilings.items())}}},
  "criteria_reasoning": {{{", ".join(f'"{k}": {{"marks": <number between 0.0 and {v:g}>, "justification": "1 sentence why mark was given against ceiling", "evidence": "exact short quote"}}' for k, v in spec.criteria_ceilings.items())}}},
  "strengths": ["Key element done well"],
  "weaknesses": ["Key weakness or missing point"],
  "student_feedback": {{
    "praise_point": "one clear commendation of what the student did well",
    "actionable_revision_step": "one concrete actionable revision technique"
  }},
  "examiner_reasoning": "2 sentences explaining the final mark allocation across the criteria ceilings",
  "structural_audit": {{
    "verbatim_copy_suspected": false,
    "exceeds_length_limit": false,
    "external_facts_or_personal_opinions": false,
    "attempted": true
  }}
}}"""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class ScoreResult:
    awarded: float
    raw_total: Optional[float] = None
    items: List[Dict[str, Any]] = field(default_factory=list)
    subscores: Dict[str, float] = field(default_factory=dict)
    criteria_reasoning: Dict[str, Any] = field(default_factory=dict)
    student_feedback: str = ""
    examiner_reasoning: str = ""
    cap_applied: bool = False
    cap_reason: str = "None"
    capped_from: Optional[float] = None
    band: Optional[str] = None
    key_source: str = "none"
    notes: List[str] = field(default_factory=list)
    feedback: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


def compute_lcs_alignment(student: List[str], expected: List[str]) -> Tuple[int, Set[int], Set[int]]:
    """
    Computes Longest Common Subsequence of relative sentence order.
    Returns (lcs_length, student_matched_indices, expected_matched_indices).
    """
    m, n = len(student), len(expected)
    if m == 0 or n == 0:
        return 0, set(), set()

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if student[i - 1] == expected[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]

    # Backtrack to identify matched positions in both sequences
    matched_student: Set[int] = set()
    matched_expected: Set[int] = set()
    i, j = m, n
    while i > 0 and j > 0:
        if student[i - 1] == expected[j - 1]:
            matched_student.add(i - 1)
            matched_expected.add(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs_len, matched_student, matched_expected


def score_mode_a(parsed: Dict[str, Any], spec: QuestionSpec, key_entry: Dict[str, Any]) -> ScoreResult:
    res = ScoreResult(awarded=0.0, key_source="answer_key" if key_entry else "none")
    mpi = float(key_entry.get("mark_per_item") or spec.mark_per_item or 1.0)
    seq = key_entry.get("correct_sequence")
    if seq:
        raw_student = [str(x).strip().lower() for x in (parsed.get("student_sequence") or [])]
        from src.pipeline.token_guard import sanitize_rearrangement_sequence
        expected_clean = [str(x).strip().lower() for x in seq]
        repaired_seq, anomalies = sanitize_rearrangement_sequence(raw_student, expected_letters=set(expected_clean))
        student = repaired_seq if repaired_seq and len(repaired_seq) == len(raw_student) else raw_student
        if anomalies:
            res.notes.extend(anomalies)

        # 1. Exact positional slot matching
        pos_correct = 0
        exact_slots: Set[int] = set()
        for pos, letter in enumerate(expected_clean):
            if pos < len(student) and student[pos] == letter:
                pos_correct += 1
                exact_slots.add(pos)

        # 2. Dynamic programming Longest Common Subsequence (LCS) relative order alignment
        lcs_len, matched_student, matched_expected = compute_lcs_alignment(student, expected_clean)

        # Harmonized score: Take the maximum of positional matches and relative order matches
        total_correct = max(pos_correct, lcs_len)

        for pos, letter in enumerate(expected_clean):
            got = student[pos] if pos < len(student) else ""
            is_exact = pos in exact_slots
            is_lcs = pos in matched_expected
            is_ok = is_exact or (lcs_len >= pos_correct and is_lcs)

            if is_exact:
                detail = "correct"
            elif is_ok:
                detail = "correct (relative order)"
            elif not got:
                detail = "not_attempted"
            else:
                detail = "incorrect"

            res.items.append({
                "item_label": str(pos + 1),
                "candidate_answer": got,
                "expected": letter,
                "status": "correct" if is_ok else ("not_attempted" if not got else "incorrect"),
                "status_detail": detail,
                "marks_awarded": mpi if is_ok else 0.0,
                "marks_available": mpi
            })

        res.awarded = min(spec.max_mark, total_correct * mpi)
        res.raw_total = res.awarded
        res.notes.append(
            f"rearrangement scored: {total_correct}/{len(seq)} (slot matches: {pos_correct}, LCS relative order matches: {lcs_len})"
        )
        return res

    raw_key_items = key_entry.get("items") or []
    if isinstance(raw_key_items, dict):
        key_items = {str(k).lower().strip("().: "): v for k, v in raw_key_items.items()}
    else:
        key_items = {str(it.get("label", it.get("item_label"))).lower().strip("().: "): it for it in raw_key_items}
    got_items = {}
    for it in (parsed.get("items") or []):
        lab = str(it.get("item_label") or it.get("item") or it.get("label") or "").strip().lower().strip("().: ")
        if lab:
            got_items[lab] = it
    labels = list(key_items) or list(got_items)
    correct = 0
    for lab in labels:
        it_dict = got_items.get(lab) or {}
        cand = str(it_dict.get("candidate_answer") or it_dict.get("student_answer") or it_dict.get("answer") or "")
        accepted = list((key_items.get(lab) or {}).get("accepted") or [])
        if not cand.strip():
            status = "not_attempted"
        elif accepted:
            # multiple selections earn 0 (MCQ rule): two different option labels chosen
            multi = spec.task_type == "MCQ" and len(selected_option_labels(cand, accepted)) > 1
            status = "correct" if (_matches_accepted(cand, accepted, spec.task_type) and not multi) else "incorrect"
        else:
            status = str((got_items.get(lab) or {}).get("status") or "unscorable")
        ok = status == "correct"
        correct += int(ok)
        res.items.append({"item_label": lab, "candidate_answer": cand, "expected": accepted[:3], "status": status,
                          "marks_awarded": mpi if ok else 0.0, "marks_available": mpi})
    res.awarded = min(spec.max_mark, snap_half(correct * mpi))
    res.raw_total = res.awarded
    if not key_items:
        res.notes.append("no answer key: items could not be verified")
    return res


def score_mode_b(parsed: Dict[str, Any], spec: QuestionSpec, key_entry: Dict[str, Any]) -> ScoreResult:
    res = ScoreResult(awarded=0.0, key_source="answer_key" if key_entry else "none")
    avail = float(key_entry.get("marks_available") or spec.mark_per_item or 2.0)
    allowed = sorted(spec.scale.keys()) if spec.scale else [0.0, 0.5, 1.0, 1.5, 2.0]
    allowed = [v for v in allowed if v <= avail] or [0.0, avail]
    n_items = len(key_entry.get("items") or []) or spec.n_items or 5
    total = 0.0
    for it in (parsed.get("items") or [])[:n_items]:
        try:
            m = float(it.get("marks_awarded", 0.0))
        except (TypeError, ValueError):
            m = 0.0
        m = max(0.0, min(avail, m))
        m = min(allowed, key=lambda v: abs(v - m))          # clamp to the discrete scale
        total += m
        res.items.append({"item_label": str(it.get("item_label", "")).strip("().: "), "candidate_answer": str(it.get("candidate_answer", ""))[:160],
                          "status": "correct" if m >= avail else ("incorrect" if m == 0 else "partially_correct"),
                          "marks_awarded": m, "marks_available": avail, "rationale": str(it.get("rationale", ""))[:200]})
    res.awarded = min(spec.max_mark, snap_half(total))
    res.raw_total = res.awarded
    return res


def score_mode_c(parsed: Dict[str, Any], spec: QuestionSpec, answer_text: str, source_text: str = "") -> ScoreResult:
    res = ScoreResult(awarded=0.0)
    answer_text = clean_rubric_answer(answer_text)
    subs = parsed.get("raw_subscores") or parsed.get("subscores") or {}
    cr = parsed.get("criteria_reasoning") or {}

    # Backward compatibility alias between structure_format_brevity and organization_cohesion
    if "organization_cohesion" in spec.criteria_ceilings and "organization_cohesion" not in subs and "structure_format_brevity" in subs:
        subs["organization_cohesion"] = subs["structure_format_brevity"]
    elif "structure_format_brevity" in spec.criteria_ceilings and "structure_format_brevity" not in subs and "organization_cohesion" in subs:
        subs["structure_format_brevity"] = subs["organization_cohesion"]

    # Tier 1 & Tier 2: Extract numerical subscores per criterion
    extracted_any_subscore = False
    for crit, ceiling in spec.criteria_ceilings.items():
        v = None
        # Tier 1: direct in subs
        if crit in subs:
            try:
                v = float(subs[crit])
                extracted_any_subscore = True
            except (TypeError, ValueError):
                pass
        # Tier 2: check inside criteria_reasoning[crit]
        if v is None and crit in cr and isinstance(cr[crit], dict):
            for k_mark in ("marks", "score", "awarded", "subscore", "mark"):
                if k_mark in cr[crit]:
                    try:
                        v = float(cr[crit][k_mark])
                        extracted_any_subscore = True
                        break
                    except (TypeError, ValueError):
                        pass
        if v is not None:
            v = max(0.0, min(float(ceiling), snap_half(v)))
            res.subscores[crit] = v
        else:
            res.subscores[crit] = 0.0

    # Tier 3: Narrative Band Recovery if no numerical subscores found but student attempted composition
    ans_words = len(_WORD.findall(answer_text.lower()))
    audit = parsed.get("structural_audit") or {}
    is_attempted = audit.get("attempted", True) if answer_text.strip() else False

    if not extracted_any_subscore and is_attempted and ans_words >= 15 and spec.bands:
        all_narrative = (
            str(parsed.get("examiner_reasoning", "")) + " " +
            str(parsed.get("feedback_summary", "")) + " " +
            " ".join(str(cr.get(c, {}).get("justification", "")) for c in spec.criteria_ceilings)
        ).lower()

        band_patterns = [
            ("band_1", r"\bband\s*1\b|\bexcellent\b|\bfull\s+marks\b"),
            ("band_2", r"\bband\s*2\b|\bgood\b|\bhigh\s+marks\b"),
            ("band_3", r"\bband\s*3\b|\bsatisfactory\b|\baverage\b"),
            ("band_4", r"\bband\s*4\b|\blimited\b"),
            ("band_5", r"\bband\s*5\b|\bpoor\b"),
        ]
        detected_band = None
        for b_name, b_re in band_patterns:
            if re.search(b_re, all_narrative):
                matched_key = next((k for k in spec.bands.keys() if b_name in k.lower()), None)
                if matched_key:
                    detected_band = matched_key
                    break

        if detected_band:
            rng_str = str(spec.bands[detected_band])
            if "-" in rng_str:
                lo, hi = map(float, rng_str.split("-", 1))
                target_score = snap_half((lo + hi) / 2.0)
            else:
                target_score = snap_half(float(rng_str))
            reconstruct_reason = f"reconstructed_from_narrative_band: {detected_band} (target={target_score})"
        elif ans_words >= 25:
            # Fallback for attempted substantial composition where narrative didn't name a band
            target_score = snap_half(spec.max_mark * 0.5)
            reconstruct_reason = f"reconstructed_from_attempted_narrative_default (target={target_score})"
        else:
            target_score = 0.0
            reconstruct_reason = None

        if target_score > 0.0:
            tot_ceiling = sum(spec.criteria_ceilings.values()) or spec.max_mark or 1.0
            ratio = min(1.0, target_score / tot_ceiling)
            for crit, ceiling in spec.criteria_ceilings.items():
                res.subscores[crit] = snap_half(ceiling * ratio)
            if reconstruct_reason:
                res.notes.append(reconstruct_reason)

    total = sum(res.subscores.values())
    res.raw_total = snap_half(total)
    audit = parsed.get("structural_audit") or {}
    tt = (spec.task_type or "").lower()
    half = snap_half(spec.max_mark * 0.5)
    cap_value: Optional[float] = None
    reason = "None"
    if audit.get("attempted") is False and not answer_text.strip():
        res.awarded = 0.0
        res.notes.append("not attempted")

    source_words = len(_WORD.findall((source_text or "").lower()))
    if tt == "summary":
        if source_words >= 25:
            overlap = verbatim_overlap(answer_text, source_text)
            too_long = len(_WORD.findall(answer_text.lower())) > (source_words / 3.0 + 5)
            res.notes.append(f"verbatim_overlap={overlap:.2f} too_long={too_long} (src_words={source_words})")
            if overlap > 0.5 or too_long or bool(audit.get("verbatim_copy_suspected")) or bool(audit.get("exceeds_length_limit")):
                cap_value, reason = half, "Summary_Verbatim_Length"
        else:
            res.notes.append(f"source_passage_unresolved_or_short (src_words={source_words}) - length caps bypassed")
    elif tt == "theme":
        if source_words >= 20:
            overlap = verbatim_overlap(answer_text, source_text)
            res.notes.append(f"verbatim_overlap={overlap:.2f}")
            if overlap > 0.5 or bool(audit.get("verbatim_copy_suspected")):
                cap_value, reason = half, "Theme_Verbatim_Copy"
        else:
            res.notes.append(f"source_theme_unresolved_or_short (src_words={source_words}) - verbatim cap bypassed")
    elif tt in ("graph_chart", "graph", "chart") and bool(audit.get("external_facts_or_personal_opinions")):
        cap_val = float(spec.hard_caps.get("Graph_External_Facts", 0.0)) if "Graph_External_Facts" in spec.hard_caps else round(spec.max_mark * 0.6, 1)
        cap_value, reason = min(cap_val if cap_val > 0 else round(spec.max_mark * 0.6, 1), spec.max_mark), "Graph_External_Facts"

    awarded = res.raw_total
    if cap_value is not None and awarded > cap_value:
        res.cap_applied = True
        res.cap_reason = reason
        res.capped_from = awarded
        awarded = cap_value
    elif cap_value is not None:
        res.cap_reason = reason + " (cap not binding)"
    res.awarded = snap_half(awarded)
    res.band = band_for(res.awarded, spec.bands)
    res.criteria_reasoning = parsed.get("criteria_reasoning") or {}
    sf = parsed.get("student_feedback")
    if isinstance(sf, dict):
        praise = str(sf.get("praise_point", "")).strip()
        action = str(sf.get("actionable_revision_step", "")).strip()
        parts = []
        if praise:
            parts.append(f"Commendation: {praise}")
        if action:
            parts.append(f"Actionable Improvement: {action}")
        res.student_feedback = "\n".join(parts)
    elif isinstance(sf, str):
        res.student_feedback = sf.strip()

    res.examiner_reasoning = str(parsed.get("examiner_reasoning") or "").strip()
    res.feedback = res.examiner_reasoning or str(parsed.get("feedback_summary") or "")
    res.strengths = [str(s) for s in (parsed.get("strengths") or parsed.get("positive_aspects") or [])][:5]
    res.weaknesses = [str(s) for s in (parsed.get("weaknesses") or parsed.get("frequent_errors") or [])][:5]
    ev = parsed.get("criterion_evidence") or {}
    if isinstance(ev, dict):
        hallucinations = []
        norm_ans = re.sub(r'\s+', ' ', answer_text.lower())
        for k, v in ev.items():
            s = str(v).strip()
            # If evidence contains quoted excerpts, check against student's answer text
            quote_m = re.findall(r'["\']([^"\']{10,})["\']', s)
            for q in quote_m:
                norm_q = re.sub(r'\s+', ' ', q.lower())
                if norm_q not in norm_ans:
                    hallucinations.append(q[:40])
        if hallucinations:
            res.notes.append(f"evidence_quote_unverified: {hallucinations}")
        res.notes.append("evidence: " + json.dumps({k: str(v)[:80] for k, v in ev.items()}, ensure_ascii=False))
    return res


def source_text_for_question(q_no: str, question_text: str, question_id: Optional[str] = None) -> str:
    """
    Source passage/poem for summary and theme tasks. The answer key's `source_texts` entry wins (it is
    copied verbatim from the paper); otherwise it is located in the question paper text.
    """
    if question_id:
        stored = key_source_text(load_answer_key(question_id), q_no)
        if stored:
            return stored
    if not question_text:
        return ""
    clean = re.sub(r"\(.*?\)", "", q_no).strip()

    # Step 1: Look for multi-question reading headers that include q_no
    # e.g., "Read the text and answer the questions 5, 6 & 7 :"
    # e.g., "Read the following passage carefully and answer questions 1 and 2:"
    group_pattern = re.compile(
        r"(?:Read\s+(?:the\s+)?(?:text|passage)[^\n]*?(?:questions?|question\s+nos?\.)\s*([0-9\s,&andto-]+)[^\n]*?[:\n])\s*(.*?)(?=\n\s*(?:[0-9]{1,2}\.\s|Read\s+the|\*\*Part|\Z))",
        re.DOTALL | re.IGNORECASE,
    )
    for gm in group_pattern.finditer(question_text):
        q_range_text = gm.group(1)
        q_nums = re.findall(r"\b\d+\b", q_range_text)
        if clean in q_nums:
            passage_body = gm.group(2).strip()
            if len(_WORD.findall(passage_body.lower())) >= 25:
                return passage_body

    # Step 2: Check directly under the question number (e.g. "3. Summarize the following poem.\n<poem>")
    pattern = re.compile(
        rf"(?:^|\n)\s*{re.escape(clean)}\.\s+(.*?)(?=\n\s*(?:Read\s+the\s+(?:text|passage)|[0-9]{{1,2}}\.\s|\*\*Part|\Z))",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(question_text)
    if m:
        body = m.group(1).strip()
        lines = [l for l in body.splitlines() if l.strip()]
        quoted = "\n".join(lines[1:]) if len(lines) > 1 else body
        if len(_WORD.findall(quoted.lower())) >= 25:
            return quoted

    # Step 3: If still unresolved, look for the most recent reading passage preceding question q_no
    q_pos_m = re.search(rf"(?:^|\n)\s*{re.escape(clean)}\.\s+", question_text)
    if q_pos_m:
        preceding = question_text[: q_pos_m.start()]
        passages = list(
            re.finditer(
                r"(?:Read\s+(?:the\s+)?(?:text|passage)[^\n]*?[:\n])\s*(.*?)(?=\n\s*(?:[0-9]{1,2}\.\s|Read\s+the|\*\*Part|\Z))",
                preceding,
                re.DOTALL | re.IGNORECASE,
            )
        )
        if passages:
            last_p = passages[-1].group(1).strip()
            if len(_WORD.findall(last_p.lower())) >= 25:
                return last_p

    # Fallback to lines under question if any
    if m:
        return "\n".join(lines[1:]) if len(lines) > 1 else body
    return ""
