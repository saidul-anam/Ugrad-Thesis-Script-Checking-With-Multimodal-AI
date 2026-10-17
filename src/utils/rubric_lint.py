"""
Consistency checks between a paper's question paper, rubric, answer key and penalty profile.

Stage 4 trusts these four files to agree: the rubric decides each question's mode and marks, the answer key
holds the items Mode A scores against, the question paper decides which questions exist. A mismatch does
not crash anything; it silently mis-scores (a key with 5 items for a 10-item question, a band table with a
gap, criteria ceilings that do not add up to the question's mark). This module finds such mismatches.

Each issue is an ERROR (Stage 4 will mis-score or cannot score) or a WARNING (worth a human look, scoring
still works). Used by scripts/lint_rubrics.py and tests/test_rubric_lint.py.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

from src.pipeline.stage4_modes import QuestionSpec, load_answer_key, key_for_question, key_source_text, load_rubric_specs, \
    rubric_is_mode_based, _norm_task, _parse_allocation
from src.utils.ground_truth import canonicalize_question_key
from src.utils.rubric_resolver import resolve_rubric_path, resolve_penalty_config_path, extract_class_number

ERROR, WARNING = "ERROR", "WARNING"

# hard-cap fields that make a cap measurable by code (otherwise the model's structural audit flag triggers it)
_CAP_TRIGGERS = ("threshold_ratio", "max_length_ratio", "threshold_words", "audit_flag")


@dataclass
class Issue:
    level: str
    paper: str
    question: str
    message: str

    def __str__(self) -> str:
        q = f" Q{self.question}" if self.question else ""
        return f"{self.level:7} {self.paper}{q}: {self.message}"


def _num(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_band_range(text: Any) -> Optional[tuple]:
    """'6.0 - 7.5 | descriptor' -> (6.0, 7.5); '3' -> (3.0, 3.0); unparsable -> None (same reading as band_for)."""
    s = str(text).split("|")[0].split(":")[0].replace("marks", "").strip()
    parts = [p.strip() for p in s.split("-", 1)] if "-" in s else [s, s]
    lo, hi = _num(parts[0]), _num(parts[1])
    return (lo, hi) if lo is not None and hi is not None else None


def check_bands(bands: Dict[str, Any], max_mark: float) -> List[str]:
    """Every half mark from 0 to max_mark must fall in exactly one band."""
    if not bands:
        return [f"no performance bands for a {max_mark:g}-mark question"]
    ranges = {}
    for name, rng in bands.items():
        r = parse_band_range(rng)
        if r is None:
            return [f"band '{name}' has no readable mark range ('{str(rng)[:30]}')"]
        ranges[name] = r
    problems = []
    gaps, overlaps = [], []
    for k in range(int(round(max_mark * 2)) + 1):
        s = k / 2
        hits = [n for n, (lo, hi) in ranges.items() if lo - 1e-9 <= s <= hi + 1e-9]
        if not hits:
            gaps.append(s)
        elif len(hits) > 1:
            overlaps.append(s)
    if gaps:
        problems.append(f"bands leave these marks uncovered: {', '.join(f'{g:g}' for g in gaps)}")
    if overlaps:
        problems.append(f"bands overlap at: {', '.join(f'{o:g}' for o in overlaps)}")
    top = max(hi for _, hi in ranges.values())
    if abs(top - max_mark) > 1e-9:
        problems.append(f"top band ends at {top:g}, question is out of {max_mark:g}")
    return problems


_PRINTED_ALLOC = re.compile(r"([0-9.]+)\s*(?:[x×]|\\times)\s*([0-9]+)\s*=\s*([0-9.]+)")


def printed_marks(question_text: str) -> Dict[str, float]:
    """Marks printed on numbered question lines of the paper text ('10. Answer any five ... 2 x 5 = 10' -> {'10': 10})."""
    out: Dict[str, float] = {}
    for m in re.finditer(r"(?m)^\s*([0-9]{1,2})\s*[.)]\s+(.*)$", question_text or ""):
        alloc = _PRINTED_ALLOC.search(m.group(2))
        if alloc and m.group(1) not in out:
            out[m.group(1)] = float(alloc.group(3))
    return out


def _key_item_count(entry: Dict[str, Any]) -> int:
    return len(entry.get("items") or entry.get("correct_sequence") or entry.get("sequence_points") or [])


def lint_paper(paper_id: str, rubric: Dict[str, Any], question_obj: Any, answer_key: Dict[str, Any],
               rubric_path: str = "", penalty_path: str = "", class_tests: Optional[Dict[str, float]] = None) -> List[Issue]:
    """All issues for one paper. `question_obj` is the parsed question paper (ExtractedQuestion or None)."""
    issues: List[Issue] = []

    def add(level, q, msg):
        issues.append(Issue(level, paper_id, q, msg))

    # ---- files present / resolved to a class-specific file ----
    class_num = extract_class_number(paper_id)
    if rubric_path and class_num is not None and f"class{class_num}" not in os.path.basename(rubric_path) \
            and paper_id.lower() not in os.path.basename(rubric_path).lower():
        add(WARNING, "", f"rubric resolved to a shared fallback ({os.path.basename(rubric_path)}), not a class-{class_num} rubric")
    if penalty_path:
        if not os.path.exists(penalty_path):
            add(ERROR, "", f"penalty profile {penalty_path} does not exist")
        elif class_num is not None and f"class{class_num}" not in os.path.basename(penalty_path):
            add(WARNING, "", f"penalty profile resolved to a shared fallback ({os.path.basename(penalty_path)})")
    if not rubric_is_mode_based(rubric):
        add(ERROR, "", "rubric is not mode-based (needs scoring_modes and mode_specifications); Stage 4 falls back to the generic path")
        return issues
    if question_obj is None or not getattr(question_obj, "sub_questions", None):
        add(ERROR, "", "question paper not found or has no sub-questions")
        return issues

    specs = load_rubric_specs(rubric)
    paper_qs = {canonicalize_question_key(str(sq.get("q_no") or "")): sq for sq in question_obj.sub_questions}

    # ---- totals ----
    paper_total = sum(_num(sq.get("max_marks") or sq.get("marks")) or 0.0 for sq in question_obj.sub_questions)
    map_total = sum(_num(qm.get("standard_marks")) or 0.0 for qm in rubric.get("question_map") or [])
    declared = _num(rubric.get("total_marks"))
    if abs(map_total - paper_total) > 1e-6:
        add(ERROR, "", f"rubric question marks add up to {map_total:g}, the question paper to {paper_total:g}")
    if declared is not None and abs(declared - map_total) > 1e-6:
        add(ERROR, "", f"rubric total_marks says {declared:g} but its questions add up to {map_total:g}")

    # ---- question sets ----
    for q in sorted(set(paper_qs) - set(specs), key=lambda x: (len(x), x)):
        add(ERROR, q, "in the question paper but not in the rubric")
    for q in sorted(set(specs) - set(paper_qs), key=lambda x: (len(x), x)):
        add(ERROR, q, "in the rubric but not in the question paper")

    # ---- every question in exactly one scoring mode, agreeing with question_map ----
    listed: Dict[str, List[str]] = {}
    for mode_key, m in (rubric.get("scoring_modes") or {}).items():
        for q in m.get("questions", []) or []:
            listed.setdefault(canonicalize_question_key(str(q)), []).append(mode_key.split("_")[0].upper()[:1])
    for qm in rubric.get("question_map") or []:
        q = canonicalize_question_key(str(qm.get("question_no", "")))
        modes = listed.get(q, [])
        if len(modes) > 1:
            add(ERROR, q, f"listed under several scoring modes ({', '.join(modes)})")
        elif not modes and not qm.get("mode"):
            add(WARNING, q, "not listed under any scoring mode; defaults to Mode C")
        if qm.get("mode") and modes and str(qm["mode"]).upper()[:1] != modes[0]:
            add(ERROR, q, f"question_map says Mode {qm['mode']} but scoring_modes lists it under Mode {modes[0]}")

    printed = printed_marks(getattr(question_obj, "question_text", "") or "")
    caps = (rubric.get("mode_specifications", {}).get("mode_c_band_scored", {}) or {}).get("hard_caps") or {}
    task_types = {_norm_task(s.task_type) for s in specs.values()}
    class_tests = class_tests or {}

    for q, spec in sorted(specs.items(), key=lambda kv: (len(kv[0]), kv[0])):
        sq = paper_qs.get(q)
        if sq is not None:
            pm = _num(sq.get("max_marks") or sq.get("marks"))
            if pm is not None and abs(pm - spec.max_mark) > 1e-6:
                add(ERROR, q, f"rubric gives {spec.max_mark:g} marks, question paper {pm:g}")
            pt = sq.get("task_type") or sq.get("type")
            if pt and _norm_task(pt) != _norm_task(spec.task_type):
                add(WARNING, q, f"task type differs: rubric '{spec.task_type}', question paper '{pt}'")
        if q in printed and abs(printed[q] - spec.max_mark) > 1e-6:
            add(WARNING, q, f"the printed paper gives {printed[q]:g} marks, the rubric and parsed paper {spec.max_mark:g}")

        section, entry = key_for_question(answer_key or {}, q)
        if entry.get("task_type") and _norm_task(entry["task_type"]) != _norm_task(spec.task_type):
            add(WARNING, q, f"task type differs: rubric '{spec.task_type}', answer key '{entry['task_type']}'")
        if spec.mode in ("A", "B") and spec.mark_per_item and spec.n_items \
                and abs(spec.mark_per_item * spec.n_items - spec.max_mark) > 1e-6 and not entry.get("answer_any"):
            add(ERROR, q, f"allocation {spec.mark_per_item:g} x {spec.n_items} = {spec.mark_per_item * spec.n_items:g}, "
                          f"question is out of {spec.max_mark:g}")

        if spec.mode == "A":
            if section != "mode_a":
                add(ERROR, q, "Mode A (item-scored) but the answer key has no mode_a entry; it cannot be scored"
                    + (f" (found under {section})" if section else ""))
            else:
                n_key = _key_item_count(entry)
                if spec.n_items and n_key != spec.n_items:
                    add(ERROR, q, f"answer key has {n_key} items, the rubric scores {spec.n_items}")
                kmpi = _num(entry.get("mark_per_item"))
                if kmpi is not None and spec.mark_per_item and abs(kmpi - spec.mark_per_item) > 1e-6:
                    add(ERROR, q, f"answer key mark_per_item {kmpi:g}, rubric {spec.mark_per_item:g}")
                for it in entry.get("items") or []:
                    if not it.get("accepted"):
                        add(ERROR, q, f"key item ({it.get('label')}) has no accepted answers")
                if _norm_task(spec.task_type) == "rearrangement" and not entry.get("correct_sequence"):
                    add(ERROR, q, "rearrangement key has no correct_sequence")
        elif spec.mode == "B":
            top = max(spec.scale) if spec.scale else None
            if top is not None and spec.mark_per_item and abs(top - spec.mark_per_item) > 1e-6:
                add(ERROR, q, f"scale tops out at {top:g} but each item is worth {spec.mark_per_item:g}")
            if spec.scale and 0.0 not in spec.scale:
                add(WARNING, q, "scale has no 0 level")
            if section == "mode_a":
                add(WARNING, q, "Mode B (point-scored) but the answer key entry is under mode_a")
            if not entry.get("items") and not entry.get("sequence_points"):
                add(WARNING, q, "Mode B with no reference answers in the key; judged against the passage only")
            elif entry.get("items") and spec.n_items and _key_item_count(entry) < spec.n_items:
                add(WARNING, q, f"key has {_key_item_count(entry)} items, the rubric scores {spec.n_items}")
        else:  # Mode C
            if q in class_tests:
                continue                    # recorded from the class test, not written in this script
            if section:
                add(WARNING, q, f"Mode C (band-scored) question has an answer-key entry under {section}; it is not used")
            if not spec.criteria_ceilings:
                add(ERROR, q, "Mode C question has no criteria_ceilings task entry")
            elif abs(sum(spec.criteria_ceilings.values()) - spec.max_mark) > 1e-6:
                add(ERROR, q, f"criteria ceilings add up to {sum(spec.criteria_ceilings.values()):g}, "
                              f"question is out of {spec.max_mark:g}")
            for p in check_bands(spec.bands, spec.max_mark):
                add(ERROR, q, p)
            tt = _norm_task(spec.task_type)
            for name, cap in caps.items():
                if not isinstance(cap, dict):
                    continue
                targets = cap.get("applies_to") or []
                targets = [targets] if isinstance(targets, str) else targets
                if tt in {_norm_task(t) for t in targets} and "threshold_ratio" in cap:
                    if not key_source_text(answer_key or {}, q):
                        add(WARNING, q, f"cap '{name}' measures copying from the source passage, but the answer key "
                                        f"stores no source_texts entry for this question (located in the paper text instead)")

    # ---- hard caps ----
    for name, cap in caps.items():
        if not isinstance(cap, dict):
            add(ERROR, "", f"hard cap '{name}' is free text; code can only apply structured caps")
            continue
        if cap.get("max_awarded") is None and cap.get("max_ratio") is None:
            add(ERROR, "", f"hard cap '{name}' has neither max_awarded nor max_ratio")
        if not any(k in cap for k in _CAP_TRIGGERS):
            add(WARNING, "", f"hard cap '{name}' has no measurable trigger; it fires on the model's '{name}' audit flag")
        targets = cap.get("applies_to") or []
        targets = [targets] if isinstance(targets, str) else targets
        unknown = [t for t in targets if _norm_task(t) not in task_types]
        if unknown:
            add(WARNING, "", f"hard cap '{name}' applies to task types this paper does not have: {', '.join(unknown)}")
        m = re.search(r"(\d+)_words", name)
        if m and cap.get("threshold_words") is not None and int(m.group(1)) != int(cap["threshold_words"]):
            add(WARNING, "", f"hard cap '{name}' is named for {m.group(1)} words but its threshold_words is {cap['threshold_words']}")

    # ---- answer key metadata ----
    if answer_key:
        kid = answer_key.get("question_id") or answer_key.get("paper_id")
        if kid and str(kid).upper() != paper_id.upper():
            add(ERROR, "", f"answer key is for {kid}")
        if answer_key.get("verified") is not True:
            add(WARNING, "", "answer key is not marked verified: true (not checked by a teacher)")
        for section in ("mode_a", "mode_b"):
            for k in (answer_key.get(section) or {}):
                if canonicalize_question_key(str(k)) not in specs:
                    add(WARNING, str(k), f"answer key {section} entry for a question the rubric does not have")
    elif any(s.mode == "A" for s in specs.values()):
        add(ERROR, "", "no answer key, but the rubric has Mode A questions")
    return issues


def lint_repo_paper(paper_id: str, lang: str = "english") -> List[Issue]:
    """Resolve the paper's files the way the pipeline does, then lint them."""
    from src.utils.question_utils import load_question_for_script
    from src.pipeline.stage4_evaluator import class_test_questions

    rubric_path = resolve_rubric_path(paper_id)
    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric = yaml.safe_load(f) or {}
    question_obj = load_question_for_script(f"{paper_id}_0001", lang=lang)
    return lint_paper(paper_id, rubric, question_obj, load_answer_key(paper_id), rubric_path=rubric_path,
                      penalty_path=resolve_penalty_config_path(paper_id),
                      class_tests=class_test_questions(question_obj) if question_obj else {})
