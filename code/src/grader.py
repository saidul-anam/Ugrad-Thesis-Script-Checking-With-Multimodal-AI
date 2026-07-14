"""Grader — sends Gemini the RED-ERASED image and asks for a structured grade.

INVARIANT: the grader must never see teacher red ink. It runs only on the
erased image produced by redink.erase_red. Two prompt conditions are supported
so results read as "how good under which setup", never a single number:
  - "rubric"   : question + solution + rubric provided
  - "solution" : question + solution only (no rubric)
"""
from __future__ import annotations

from . import config, llm

VARIANTS = ("rubric", "solution")

_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "awarded": {"type": "number"},
                    "max": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["criterion", "awarded", "max"],
            },
        },
        "transcription": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["score", "transcription", "confidence"],
}

_BASE = """You are grading a scanned Bangladeshi exam answer (subject: {subject}).
The teacher's red marks have been digitally ERASED — grade the student's answer
on its own merits, out of {full_marks} marks.

QUESTION:
{question}

MODEL SOLUTION:
{solution}
{rubric_block}
Read the student's handwriting from the image. Then grade.

Return JSON:
- transcription: verbatim what the student actually wrote (as best you can read
  it). Do this FIRST, before scoring, so a misread can be told from a misgrade.
- criteria: per-point breakdown [{{criterion, awarded, max, reason}}]
- score: total awarded (0..{full_marks})
- confidence: 0..1, your confidence in reading + scoring
"""

_RUBRIC_BLOCK = "\nMARKING RUBRIC (mark distribution):\n{rubric}\n"


def _prompt(row: dict, question: str, full_marks, variant: str) -> str:
    rubric_block = ""
    if variant == "rubric" and (row.get("rubric") or "").strip():
        rubric_block = _RUBRIC_BLOCK.format(rubric=row["rubric"].strip())
    return _BASE.format(
        subject=row.get("subject", ""),
        full_marks=full_marks if full_marks is not None else "the stated total",
        question=question or "(see printed question in the image)",
        solution=(row.get("solution") or "(none provided)").strip(),
        rubric_block=rubric_block,
    )


def grade(row: dict, question: str, full_marks, variant: str,
          force: bool = False) -> dict:
    assert variant in VARIANTS, variant
    id_ = row["id"]
    erased = config.ERASED_DIR / f"{id_}.jpg"
    if not erased.exists():
        raise FileNotFoundError(
            f"erased image missing for {id_}; run red erasure first "
            "(never grade the original — it contains teacher red ink)"
        )
    parts = [_prompt(row, question, full_marks, variant), llm.image_part(erased)]
    return llm.generate_json(
        "grade", id_, variant, parts, response_schema=_SCHEMA, force=force
    )
