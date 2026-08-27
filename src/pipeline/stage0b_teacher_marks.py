import json
import re
from typing import Optional, List, Dict, Any, Union
from PIL import Image
from pydantic import BaseModel, Field

from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import TeacherMarkItem
from src.prompts.stage0b_marks import STAGE0B_SYSTEM_PROMPT, STAGE0B_BASE_PROMPT


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
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 1024,
        thinking_mode: bool = False
    ) -> Stage0bResult:
        response = self.engine.generate_multimodal(
            prompt=STAGE0B_BASE_PROMPT,
            image=image,
            system_prompt=STAGE0B_SYSTEM_PROMPT,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            thinking_mode=thinking_mode
        )

        parsed_marks = extract_marks(response)

        if parsed_marks is None:
            # Malformed JSON or schema assertion failed -> flag for review, preserve raw response
            return Stage0bResult(
                teacher_marks=[],
                raw_response=response,
                is_valid_json=False,
                needs_manual_review=True,
                total_marks_found=0
            )

        items = []
        for m in parsed_marks:
            items.append(TeacherMarkItem(
                question_no=str(m.get("question_no")) if m.get("question_no") is not None else None,
                mark_value=str(m.get("mark_value", "")),
                location=str(m.get("location", ""))
            ))

        return Stage0bResult(
            teacher_marks=items,
            raw_response=response,
            is_valid_json=True,
            needs_manual_review=False,
            total_marks_found=len(items)
        )
