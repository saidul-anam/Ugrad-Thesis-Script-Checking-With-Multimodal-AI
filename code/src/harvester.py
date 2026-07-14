"""Red-mark harvester — reads ONLY the teacher's red score off the ORIGINAL image.

This is a ground-truth source, so it must see the red ink: it runs on the raw
image, never the erased one. It also reports the full/maximum marks printed on
the page, used as a last-resort denominator when rubric and marks column fail.
"""
from __future__ import annotations

from typing import Optional

from . import config, llm

_SCHEMA = {
    "type": "object",
    "properties": {
        "red_marks_present": {"type": "boolean"},
        "red_score": {"type": "number", "nullable": True},
        "red_score_raw": {"type": "string"},
        "full_marks_on_page": {"type": "number", "nullable": True},
        "notes": {"type": "string"},
    },
    "required": ["red_marks_present", "red_score", "red_score_raw"],
}

_PROMPT = """You are reading a scanned Bangladeshi exam answer script.

A teacher has marked it in RED ballpoint pen. The student's answer and the
printed question are in BLACK/blue-black. Look ONLY at the RED ink.

Report the teacher's awarded SCORE — the red number (it may be a whole number,
a decimal like 2.5, or written as a fraction such as 5/2 meaning 2.5). Ticks,
crosses, circles and underlines are NOT the score; find the numeric red mark.
If a maximum/total mark is printed on the page (e.g. next to the question),
report it as full_marks_on_page.

Return JSON:
- red_marks_present: is there any red teacher ink at all?
- red_score: the numeric score as a number (null if none/illegible)
- red_score_raw: exactly what you see written in red for the score (verbatim)
- full_marks_on_page: printed maximum mark for this question, or null
- notes: anything ambiguous (multiple numbers, overwriting, unclear digit)
"""


def harvest(id_: str, force: bool = False) -> dict:
    img = config.IMAGES_DIR / f"{id_}.jpg"
    parts = [_PROMPT, llm.image_part(img)]
    return llm.generate_json(
        "harvest", id_, "v1", parts, response_schema=_SCHEMA, force=force
    )


def red_score(data: dict) -> Optional[float]:
    v = data.get("red_score")
    return float(v) if isinstance(v, (int, float)) else None
