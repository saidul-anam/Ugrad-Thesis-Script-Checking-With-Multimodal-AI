import json
import re
from typing import Optional, List, Dict, Any, Union
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
            if not set(m.keys()).issubset({"question_no", "mark_value", "location"}):
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

            # Sub-Item Rollup: map loose sub-item letters (e.g. 'C', 'G') to parent question
            if canon_q and len(canon_q) == 1 and canon_q.isalpha():
                letter = canon_q.upper()
                if "1(B)" in cands_set and letter in ["A", "B", "C", "D", "E"]:
                    canon_q = "1(B)"
                elif "1(A)" in cands_set and letter in ["A", "B", "C", "D", "E"]:
                    canon_q = "1(A)"
                elif "4" in cands_set and letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
                    canon_q = "4"
                elif "5" in cands_set and letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
                    canon_q = "5"

            # Maximum Marks Ceiling Validation & Disambiguation
            if question_max_marks and canon_q in question_max_marks:
                max_allowed = question_max_marks[canon_q]
                try:
                    num_val = float(clean_val)
                    if num_val > max_allowed:
                        # Circled '01' looks like '6'
                        if clean_val == "6" and max_allowed <= 5.0:
                            clean_val = "1"
                        else:
                            clean_val = f"{max_allowed:.1f}" if max_allowed % 1 != 0 else f"{int(max_allowed)}"
                except ValueError:
                    pass

            items.append(TeacherMarkItem(
                question_no=canon_q,
                mark_value=clean_val,
                location=str(m.get("location", "left margin"))
            ))

        return Stage0bResult(
            teacher_marks=items,
            raw_response=response,
            is_valid_json=True,
            needs_manual_review=False,
            total_marks_found=len(items)
        )

