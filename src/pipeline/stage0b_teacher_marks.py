import json
import re
from typing import Optional, List, Dict, Any, Union, Tuple
from PIL import Image
from pydantic import BaseModel, Field

from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import TeacherMarkItem
from src.prompts.stage0b_marks import STAGE0B_SYSTEM_PROMPT, STAGE0B_BASE_PROMPT, build_stage0b_prompt
from src.utils.ground_truth import canonicalize_question_key


class Stage0bResult(BaseModel):
    """Output of Stage 0b: Extracted red-ink teacher marks."""
    teacher_marks: List[TeacherMarkItem] = Field(default_factory=list)
    raw_response: str = Field("")
    is_valid_json: bool = Field(True)
    needs_manual_review: bool = Field(False)
    total_marks_found: int = Field(0)


def extract_marks(model_output: str) -> Optional[List[Dict[str, Any]]]:
    """
    Mandatory post-processing validator for Stage 0b model output.
    Returns parsed list of mark dicts if schema matches exactly, or None if malformed (routes to review).
    """
    text = model_output.strip()
    
    # Strip markdown codeblocks if model wrapped in ```json ... ```
    if "```" in text:
        text = re.sub(r"^```(?:json)?", "", text, flags=re.MULTILINE)
        text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()

    try:
        marks = json.loads(text)
        if not isinstance(marks, list):
            return None
        for m in marks:
            if not isinstance(m, dict):
                return None
            if not set(m.keys()).issubset({"question_no", "mark_value", "location", "y_position"}):
                return None
            if "mark_value" not in m:
                return None
        return marks
    except (json.JSONDecodeError, AssertionError, Exception):
        return None  # Flag for manual review, don't drop


class Stage0bTeacherMarkExtractor:
    """
    Stage 0b: Teacher Mark Extractor powered by Gemma 4 Multimodal VLM.
    Runs conditionally only on pages where Stage 0 detected has_red_ink == True.
    """

    def __init__(self, engine: BaseVLMEngine):
        self.engine = engine

    def run(
        self,
        image: Image.Image,
        candidate_questions: Optional[List[str]] = None,
        question_max_marks: Optional[Dict[str, float]] = None,
        valid_paper_questions: Optional[List[str]] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 1024,
        thinking_mode: bool = False
    ) -> Stage0bResult:
        # Tier 1: Full-page visual extraction with question constraints
        prompt = build_stage0b_prompt(
            candidate_questions=candidate_questions,
            question_max_marks=question_max_marks,
            valid_paper_questions=valid_paper_questions
        )
        response = self.engine.generate_multimodal(
            prompt=prompt,
            image=image,
            system_prompt=STAGE0B_SYSTEM_PROMPT,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            thinking_mode=thinking_mode
        )

        parsed_marks = extract_marks(response) or []
        found_questions = {canonicalize_question_key(str(m.get("question_no"))) for m in parsed_marks if m.get("question_no")}

        # Tier 2: Targeted Left-Margin Zoom for missing candidate questions
        if candidate_questions:
            missing_cands = [q for q in candidate_questions if q not in found_questions]
            w, h = image.size
            if missing_cands and h > 1000 and w > 500:
                zones = [
                    ("top_left", image.crop((0, 0, int(w * 0.35), int(h * 0.45)))),
                    ("bottom_left", image.crop((0, int(h * 0.50), int(w * 0.35), h)))
                ]
                for zone_name, zone_img in zones:
                    if not missing_cands:
                        break
                    zone_prompt = build_stage0b_prompt(
                        candidate_questions=missing_cands,
                        question_max_marks=question_max_marks,
                        valid_paper_questions=valid_paper_questions
                    )
                    zone_resp = self.engine.generate_multimodal(
                        prompt=zone_prompt,
                        image=zone_img,
                        system_prompt=STAGE0B_SYSTEM_PROMPT,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=512,
                        thinking_mode=thinking_mode
                    )
                    zone_marks = extract_marks(zone_resp) or []
                    for zm in zone_marks:
                        zq = canonicalize_question_key(str(zm.get("question_no")))
                        if zq and zq in missing_cands:
                            zm["location"] = f"left margin ({zone_name})"
                            parsed_marks.append(zm)
                            found_questions.add(zq)
                            missing_cands.remove(zq)

        if not parsed_marks and not response.strip().startswith("["):
            return Stage0bResult(
                teacher_marks=[],
                raw_response=response,
                is_valid_json=False,
                needs_manual_review=True,
                total_marks_found=0
            )

        items = []
        cands_set = set(candidate_questions or [])
        valid_set = set(valid_paper_questions or [])

        for m in parsed_marks:
            raw_q = str(m.get("question_no")) if m.get("question_no") is not None else None
            canon_q = canonicalize_question_key(raw_q) if raw_q else None
            raw_val = str(m.get("mark_value", "")).strip()

            # Fraction conversion: e.g. '1/2' -> '0.5'
            if "1/2" in raw_val:
                raw_val = raw_val.replace("1/2", "0.5")

            # Clean common digit artifacts (e.g. '01' -> '1')
            if raw_val.isdigit() and len(raw_val) > 1 and raw_val.startswith('0'):
                clean_val = str(int(raw_val))
            else:
                clean_val = raw_val

            # Sub-Item Rollup: map loose sub-item letters (e.g. 'C', 'G') to parent candidate question
            if canon_q and len(canon_q) == 1 and canon_q.isalpha():
                letter = canon_q.upper()
                if len(cands_set) == 1:
                    canon_q = next(iter(cands_set))
                elif cands_set:
                    # 1. Direct subpart match e.g. candidate "1(C)" for letter "C"
                    matched_cand = next((c for c in cands_set if c.upper() == letter or c.upper().endswith(f"({letter})")), None)
                    if matched_cand:
                        canon_q = matched_cand
                        # 2. Check if candidates in cands_set are known objective / multi-item questions
                        from src.prompts.stage4_modular import is_objective_question
                        obj_cands = [c for c in cands_set if is_objective_question(c, "", "")]
                        if len(obj_cands) == 1:
                            canon_q = obj_cands[0]
                        else:
                            # Match any candidate question having subparts in cands_set
                            sub_matching = [c for c in cands_set if "(" in c]
                            if len(sub_matching) == 1:
                                canon_q = sub_matching[0]
                            elif obj_cands:
                                canon_q = obj_cands[0]

            # Maximum Marks Ceiling Validation & Disambiguation
            base_q = re.sub(r'\(.*?\)', '', canon_q or '').strip()
            max_allowed = None
            if question_max_marks:
                if canon_q and canon_q in question_max_marks:
                    max_allowed = question_max_marks[canon_q]
                elif base_q and base_q in question_max_marks:
                    max_allowed = question_max_marks[base_q]
                elif base_q:
                    # Match subparts like "4(A)" when canon_q is "4"
                    sub_matches = [v for k, v in question_max_marks.items() if re.sub(r'\(.*?\)', '', k).strip() == base_q]
                    if sub_matches:
                        max_allowed = max(sub_matches)

            if max_allowed is not None:
                try:
                    num_val = float(clean_val)
                    if num_val > max_allowed:
                        # Circled '01' / '1' disambiguation: circled '1' misread as '6' on a <= 5-mark question
                        if num_val == 6.0 and max_allowed <= 5.0:
                            clean_val = "1"
                        else:
                            clean_val = f"{max_allowed:.1f}" if max_allowed % 1 != 0 else f"{int(max_allowed)}"
                except ValueError:
                    pass

            raw_y = str(m.get("y_position", "")).strip().lower() if m.get("y_position") else None
            if raw_y and raw_y not in {"top", "mid", "bottom"}:
                raw_y = "top" if "top" in raw_y or "upper" in raw_y else ("bottom" if "bottom" in raw_y or "lower" in raw_y else "mid")

            unrecognized_q = {"unknown", "null", "none", "not specified", "t specified", "unspecified", "n/a", "undefined", ""}
            if canon_q and canon_q.lower() in unrecognized_q:
                canon_q = None

            items.append(TeacherMarkItem(
                question_no=canon_q,
                mark_value=clean_val,
                location=str(m.get("location", "left margin")),
                y_position=raw_y
            ))

        # Align orphan marks using spatial vertical ordering
        items = align_orphan_marks(
            marks=items,
            candidate_questions=candidate_questions,
            valid_paper_questions=valid_paper_questions
        )

        return Stage0bResult(
            teacher_marks=items,
            raw_response=response,
            is_valid_json=True,
            needs_manual_review=False,
            total_marks_found=len(items)
        )


def align_orphan_marks(
    marks: List[TeacherMarkItem],
    candidate_questions: Optional[List[str]] = None,
    valid_paper_questions: Optional[List[str]] = None
) -> List[TeacherMarkItem]:
    """
    Align orphan teacher marks (missing or unknown question_no) using spatial vertical ordering.
    
    Resolves cases where the examiner wrote a score in the left margin without explicitly
    writing the question label (e.g. wrote '10' next to Question 8).
    """
    if not marks or not candidate_questions:
        return marks

    unrecognized = {"unknown", "null", "none", "not specified", "t specified", "unspecified", "n/a", "undefined", ""}

    # 1. Identify which candidate questions already have an assigned mark
    assigned_cands = set()
    for m in marks:
        if m.question_no and m.question_no.lower() not in unrecognized:
            if m.question_no in candidate_questions:
                assigned_cands.add(m.question_no)

    # 2. Determine remaining unassigned candidate questions in their natural vertical order
    unassigned_cands = [q for q in candidate_questions if q not in assigned_cands]
    if not unassigned_cands:
        return marks

    # 3. Partition marks into assigned and orphan marks
    assigned_marks: List[TeacherMarkItem] = []
    orphan_marks: List[TeacherMarkItem] = []
    for m in marks:
        is_orphan = (not m.question_no) or (m.question_no.lower() in unrecognized)
        if not is_orphan and valid_paper_questions and m.question_no not in valid_paper_questions:
            is_orphan = True

        if is_orphan:
            orphan_marks.append(m)
        else:
            assigned_marks.append(m)

    if not orphan_marks:
        return marks

    # 4. Sort orphan marks vertically
    def y_rank(item: TeacherMarkItem) -> int:
        pos = (item.y_position or "").lower()
        loc = (item.location or "").lower()
        if "top" in pos or "top" in loc or "upper" in loc:
            return 0
        if "bottom" in pos or "bottom" in loc or "lower" in loc:
            return 2
        return 1  # mid

    orphan_marks.sort(key=y_rank)

    # 5. Align orphan marks to unassigned candidates
    aligned_orphans: List[TeacherMarkItem] = []
    for idx, orphan in enumerate(orphan_marks):
        if idx < len(unassigned_cands):
            matched_q = unassigned_cands[idx]
            orphan.question_no = matched_q
            if not orphan.location:
                orphan.location = f"left margin (spatially aligned to {matched_q})"
            else:
                orphan.location += f" (aligned to {matched_q})"
        aligned_orphans.append(orphan)

    return assigned_marks + aligned_orphans


def reconcile_document_teacher_marks(
    marks: List[TeacherMarkItem],
    question_max_marks: Optional[Dict[str, float]] = None
) -> List[TeacherMarkItem]:
    """
    Document-level reconciliation and deduplication of teacher marks across all pages.
    Enforces the single-mark-per-question invariant and resolves duplicate/conflicting entries.
    """
    if not marks:
        return []

    # 1. Deduplicate exact duplicates (same question_no, mark_value, and y_position/location)
    seen_exact = set()
    unique_marks: List[TeacherMarkItem] = []
    for m in marks:
        q_key = (m.question_no or "").strip().lower()
        val_key = (m.mark_value or "").strip().lower()
        exact_sig = (q_key, val_key, (m.y_position or "").strip().lower())
        if q_key and exact_sig in seen_exact:
            continue
        if q_key:
            seen_exact.add(exact_sig)
        unique_marks.append(m)

    # 2. Group by canonical question_no
    grouped: Dict[str, List[TeacherMarkItem]] = {}
    orphans: List[TeacherMarkItem] = []

    for m in unique_marks:
        if not m.question_no:
            orphans.append(m)
            continue
        q = canonicalize_question_key(m.question_no)
        grouped.setdefault(q, []).append(m)

    reconciled: List[TeacherMarkItem] = []

    for q, q_marks in grouped.items():
        if len(q_marks) == 1:
            reconciled.append(q_marks[0])
            continue

        # Multiple marks found for the same question
        values = [m.mark_value for m in q_marks]
        unique_values = set(values)

        if len(unique_values) == 1:
            # Duplicate entries with the exact same score -> keep best location
            best_mark = next((m for m in q_marks if "zoom" in m.location.lower() or "left" in m.location.lower()), q_marks[0])
            reconciled.append(best_mark)
        else:
            # Conflicting mark values for the same question
            # Prioritize targeted zoom reading if available
            zoom_mark = next((m for m in q_marks if "left margin (" in m.location.lower()), None)
            if zoom_mark:
                chosen = zoom_mark
            else:
                # Prefer value within max marks ceiling if known
                max_m = question_max_marks.get(q) if question_max_marks else None
                valid_cands = []
                if max_m is not None:
                    for m in q_marks:
                        try:
                            if float(m.mark_value) <= max_m:
                                valid_cands.append(m)
                        except ValueError:
                            pass
                chosen = valid_cands[-1] if valid_cands else q_marks[-1]

            other_vals = [m.mark_value for m in q_marks if m.mark_value != chosen.mark_value]
            chosen.location += f" (reconciled conflict: kept {chosen.mark_value}, discarded {','.join(other_vals)})"
            reconciled.append(chosen)

    # Append any remaining orphans
    reconciled.extend(orphans)

    # Sort naturally by question number
    def _q_sort_key(item: TeacherMarkItem) -> Tuple[int, str]:
        q = item.question_no or "999"
        num_match = re.search(r'\d+', q)
        num = int(num_match.group(0)) if num_match else 999
        return (num, q)

    reconciled.sort(key=_q_sort_key)
    return reconciled


