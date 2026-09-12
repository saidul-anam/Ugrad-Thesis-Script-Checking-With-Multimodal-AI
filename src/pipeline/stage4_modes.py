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
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.core.schemas import AlignedAnswerItem
from src.utils.ground_truth import canonicalize_question_key


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
            spec.bands = dict(bands_by_max.get(f"max_{int(spec.max_mark)}", {}))
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
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


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
    """bands like {'band_4': '8-10', 'band_3': '6-7', ..., 'band_0': '0'}."""
    if not bands:
        return None
    for name, rng in bands.items():
        rng = str(rng)
        if "-" in rng:
            lo, hi = rng.split("-", 1)
            try:
                if float(lo) <= score <= float(hi):
                    return name.replace("_", " ").title()
            except ValueError:
                continue
        else:
            try:
                if float(rng) == score:
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


_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}


def _matches_accepted(candidate: str, accepted: List[str], task_type: str = "") -> bool:
    c = _norm(candidate)
    if not c:
        return False
    c_alt = _ROMAN.get(c, c)
    is_mcq = (task_type or "").upper() == "MCQ"
    for a in accepted:
        an = _norm(a)
        if not an:
            continue
        if c == an or c_alt == _ROMAN.get(an, an):
            return True
        # MCQ only: the student may write the option text (or option letter + text)
        if is_mcq and len(an) >= 6 and (an in c or c in an):
            return True
        # tolerate a single-character slip in words of length >= 5 (rubric: minor spelling slips)
        if len(an) >= 5 and abs(len(an) - len(c)) <= 1 and difflib.SequenceMatcher(None, c, an).ratio() >= 0.85:
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
    return f"""{MODE_A_TAG} answer extraction for Question {answer.q_no} ({spec.task_type}, max {spec.max_mark:g} marks).

OFFICIAL QUESTION:
\"\"\"
{question_prompt_text.strip()[:1800]}
\"\"\"

STUDENT'S ANSWER (verified transcription; ignore markers like [struck: ...]):
\"\"\"
{answer.answer_text.strip()}
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
    return f"""{MODE_B_TAG} for Question {answer.q_no} ({spec.task_type}, {n_items} items x {avail:g} marks = {spec.max_mark:g}).

OFFICIAL QUESTION:
\"\"\"
{question_prompt_text.strip()[:1800]}
\"\"\"

REFERENCE ANSWER KEY (from the passage):
{ref}

STUDENT'S ANSWER (verified transcription):
\"\"\"
{answer.answer_text.strip()}
\"\"\"

SCALE PER ITEM (use ONLY these values):
{scale_lines}

RULES:
{rules if rules else '  - Content dominates syntax. Spelling/grammar slips do not reduce marks if meaning is clear.'}
  - Ignore transcription markers such as [struck: ...], [unclear: ...], [illegible].
  - A missing item scores 0.

{item_note}
Output ONLY JSON:
{{"items": [{{"item_label": "a", "candidate_answer": "short quote of what the student wrote", "marks_awarded": {avail:g}, "rationale": "one sentence"}}], "notes": "one sentence"}}"""


def build_mode_c_prompt(answer: AlignedAnswerItem, spec: QuestionSpec, question_prompt_text: str, errors_text: str, source_text: str = "") -> str:
    ceilings = "\n".join(f"  - {k}: max {v:g}" for k, v in spec.criteria_ceilings.items())
    bands = "\n".join(f"  - {k.replace('_', ' ').title()}: {v} marks" for k, v in spec.bands.items())
    caps = "\n".join(f"  - {k}: {v}" for k, v in spec.hard_caps.items())
    general = "\n".join(f"  - {g}" for g in spec.general_rules[:10])
    layout = ""
    if spec.layout_components:
        layout = "\nLAYOUT COMPONENTS TO CHECK:\n" + "\n".join(f"  - {k}: {', '.join(v)}" for k, v in spec.layout_components.items())
    src = f"\nSOURCE TEXT (for verbatim-copy judgement):\n\"\"\"\n{source_text.strip()[:1200]}\n\"\"\"\n" if source_text else ""
    return f"""{MODE_C_TAG} criterion judgement for Question {answer.q_no} ({spec.task_type}, max {spec.max_mark:g} marks).

OFFICIAL QUESTION:
\"\"\"
{question_prompt_text.strip()[:1800]}
\"\"\"
{src}
STUDENT'S ANSWER (verified transcription; handwriting ambiguities already resolved):
\"\"\"
{answer.answer_text.strip()}
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
  "raw_subscores": {{{", ".join(f'"{k}": 0.0' for k in spec.criteria_ceilings)}}},
  "criterion_evidence": {{{", ".join(f'"{k}": "exact short quote from the answer"' for k in spec.criteria_ceilings)}}},
  "structural_audit": {{
    "paragraph_subdivisions": false,
    "missing_layout_components": [],
    "verbatim_copy_suspected": false,
    "exceeds_length_limit": false,
    "external_facts_or_personal_opinions": false,
    "attempted": true
  }},
  "frequent_errors": ["..."],
  "positive_aspects": ["..."],
  "feedback_summary": "2 sentences for the student"
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
    cap_applied: bool = False
    cap_reason: str = "None"
    capped_from: Optional[float] = None
    band: Optional[str] = None
    key_source: str = "none"
    notes: List[str] = field(default_factory=list)
    feedback: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


def score_mode_a(parsed: Dict[str, Any], spec: QuestionSpec, key_entry: Dict[str, Any]) -> ScoreResult:
    res = ScoreResult(awarded=0.0, key_source="answer_key" if key_entry else "none")
    mpi = float(key_entry.get("mark_per_item") or spec.mark_per_item or 1.0)
    seq = key_entry.get("correct_sequence")
    if seq:
        student = [str(x).strip().lower() for x in (parsed.get("student_sequence") or [])]
        correct = 0
        for pos, letter in enumerate(seq):
            got = student[pos] if pos < len(student) else ""
            ok = got == str(letter).lower()
            correct += int(ok)
            res.items.append({"item_label": str(pos + 1), "candidate_answer": got, "expected": letter,
                              "status": "correct" if ok else ("not_attempted" if not got else "incorrect"),
                              "marks_awarded": mpi if ok else 0.0, "marks_available": mpi})
        res.awarded = min(spec.max_mark, correct * mpi)
        res.raw_total = res.awarded
        res.notes.append(f"position-scored rearrangement: {correct}/{len(seq)} in correct slot")
        return res

    key_items = {str(it.get("label")).lower(): it for it in (key_entry.get("items") or [])}
    got_items = {str(it.get("item_label", "")).strip().lower().strip("()"): it for it in (parsed.get("items") or [])}
    labels = list(key_items) or list(got_items)
    correct = 0
    for lab in labels:
        cand = str((got_items.get(lab) or {}).get("candidate_answer") or "")
        accepted = list((key_items.get(lab) or {}).get("accepted") or [])
        if not cand.strip():
            status = "not_attempted"
        elif accepted:
            # multiple selections earn 0 (MCQ rule): two different roman numerals / letters
            multi = len(set(re.findall(r"\b(i{1,3}|iv|v|[a-e]|[1-5])\b", cand.lower()))) > 1 and spec.task_type == "MCQ"
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
        res.items.append({"item_label": str(it.get("item_label", "")), "candidate_answer": str(it.get("candidate_answer", ""))[:160],
                          "status": "correct" if m >= avail else ("incorrect" if m == 0 else "partially_correct"),
                          "marks_awarded": m, "marks_available": avail, "rationale": str(it.get("rationale", ""))[:200]})
    res.awarded = min(spec.max_mark, snap_half(total))
    res.raw_total = res.awarded
    return res


def score_mode_c(parsed: Dict[str, Any], spec: QuestionSpec, answer_text: str, source_text: str = "") -> ScoreResult:
    res = ScoreResult(awarded=0.0)
    subs = parsed.get("raw_subscores") or {}
    total = 0.0
    for crit, ceiling in spec.criteria_ceilings.items():
        try:
            v = float(subs.get(crit, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        v = max(0.0, min(float(ceiling), snap_half(v)))
        res.subscores[crit] = v
        total += v
    res.raw_total = snap_half(total)
    audit = parsed.get("structural_audit") or {}
    tt = (spec.task_type or "").lower()
    half = snap_half(spec.max_mark * 0.5)
    cap_value: Optional[float] = None
    reason = "None"
    if audit.get("attempted") is False and not answer_text.strip():
        res.awarded = 0.0
        res.notes.append("not attempted")
    if tt == "paragraph" and bool(audit.get("paragraph_subdivisions")):
        cap_value, reason = half, "Paragraph_Subdivisions"
    elif tt in ("letter_email", "letter", "email"):
        missing = audit.get("missing_layout_components") or []
        if isinstance(missing, list) and len(missing) >= 3:
            cap_value, reason = half, "Letter_Missing_Layout"
    elif tt == "summary":
        overlap = verbatim_overlap(answer_text, source_text) if source_text else 0.0
        too_long = bool(source_text) and len(_WORD.findall(answer_text.lower())) > len(_WORD.findall(source_text.lower())) / 3.0 + 5
        res.notes.append(f"verbatim_overlap={overlap:.2f} too_long={too_long}")
        if overlap > 0.5 or too_long or bool(audit.get("verbatim_copy_suspected")) or bool(audit.get("exceeds_length_limit")):
            cap_value, reason = half, "Summary_Verbatim_Length"
    elif tt == "theme":
        overlap = verbatim_overlap(answer_text, source_text) if source_text else 0.0
        res.notes.append(f"verbatim_overlap={overlap:.2f}")
        if overlap > 0.5 or bool(audit.get("verbatim_copy_suspected")):
            cap_value, reason = half, "Theme_Verbatim_Copy"
    elif tt in ("graph_chart", "graph", "chart") and bool(audit.get("external_facts_or_personal_opinions")):
        cap_value, reason = min(6.0, spec.max_mark), "Graph_External_Facts"

    awarded = res.raw_total
    if cap_value is not None and awarded > cap_value:
        res.cap_applied = True
        res.cap_reason = reason
        res.capped_from = awarded
        awarded = cap_value
    elif cap_value is not None:
        res.cap_reason = reason + " (cap not binding)"
    res.awarded = max(0.0, min(spec.max_mark, snap_half(awarded)))
    res.band = band_for(res.awarded, spec.bands)
    res.feedback = str(parsed.get("feedback_summary") or "")
    res.strengths = [str(s) for s in (parsed.get("positive_aspects") or [])][:5]
    res.weaknesses = [str(s) for s in (parsed.get("frequent_errors") or [])][:5]
    ev = parsed.get("criterion_evidence") or {}
    if isinstance(ev, dict):
        res.notes.append("evidence: " + json.dumps({k: str(v)[:80] for k, v in ev.items()}, ensure_ascii=False))
    return res


def source_text_for_question(q_no: str, question_text: str) -> str:
    """Extract the source passage/poem for summary (Q3) and theme (Q11) tasks from the question paper text."""
    if not question_text:
        return ""
    clean = re.sub(r"\(.*?\)", "", q_no).strip()
    pattern = re.compile(rf"(?:^|\n)\s*{re.escape(clean)}\.\s+(.*?)(?=\n\s*(?:[0-9]{{1,2}}\.\s|\*\*Part|\Z))", re.DOTALL)
    m = pattern.search(question_text)
    if not m:
        return ""
    body = m.group(1)
    # drop the instruction line, keep the quoted text
    lines = [l for l in body.splitlines() if l.strip()]
    return "\n".join(lines[1:]) if len(lines) > 1 else body
