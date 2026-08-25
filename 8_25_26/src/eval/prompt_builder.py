"""Assemble byte-identical prompts per row.

System prompt = full prompts/rubric_v2.txt (verbatim, from disk). User message is
built from an extraction.csv row per the task spec, with computed metrics passed
as authoritative facts and the Stage 1.5 summary-cap exemption note.

Both graders receive the identical assembled prompt for a given row — the only
difference between an A record and a B record is the model. Placeholder tokens are
substituted by literal replacement (not str.format / Template) so arbitrary
student text containing { } or $ can never corrupt assembly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.config import EvalConfig

# Order matters only for readability; each token is unique.
_METRIC_TOKENS = {
    "question_type": "%%QUESTION_TYPE%%",
    "max_mark": "%%MAX_MARK%%",
    "question": "%%QUESTION%%",
    "word_count": "%%WORD_COUNT%%",
    "source_word_count": "%%SOURCE_WORD_COUNT%%",
    "length_ratio": "%%LENGTH_RATIO%%",
    "exceeds_one_third": "%%EXCEEDS_ONE_THIRD%%",
    "verbatim_overlap_pct": "%%VERBATIM_OVERLAP_PCT%%",
    "paragraph_break_count": "%%PARAGRAPH_BREAK_COUNT%%",
    "letter_components_found": "%%LETTER_COMPONENTS_FOUND%%",
    "letter_components_missing_count": "%%LETTER_COMPONENTS_MISSING_COUNT%%",
    "extracted_text": "%%EXTRACTED_TEXT%%",
}

CAP_NOTE_TEMPLATE = (
    "\n  cap_note: Source text is {n} words. The 1/3 length rule is designed for "
    "prose\n  passages and is NOT applied here. Do not apply the Summary length "
    "cap. The\n  verbatim-copying cap still applies normally."
)


@dataclass
class BuiltPrompt:
    system_prompt: str        # rubric, verbatim (kept for reference)
    user_message: str         # per-row message (kept for reference)
    user_content: str         # UNIFIED delivery: rubric + user, one user turn
    messages: list[dict]      # role-structured message list actually sent
    prompt_bytes: bytes       # canonical JSON of `messages` (role-structured)
    prompt_hash: str          # sha256 of prompt_bytes (encodes role structure)
    rubric_hash: str          # sha256 of the rubric file bytes
    cap_exempt_applied: bool
    cap_note: str
    cap_threshold: int


def load_rubric(cfg: EvalConfig) -> tuple[str, str]:
    """Return (rubric_text, sha256). Loaded verbatim from disk each run."""
    path: Path = cfg.paths.rubric_file
    if not path.exists():
        raise FileNotFoundError(f"rubric file not found: {path}")
    raw = path.read_bytes()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def load_user_template(cfg: EvalConfig) -> str:
    path: Path = cfg.paths.user_template
    if not path.exists():
        raise FileNotFoundError(f"user template not found: {path}")
    return path.read_text(encoding="utf-8")


def _fmt(value: object) -> str:
    """Empty/None -> 'n/a'; everything else verbatim string."""
    if value is None or value == "":
        return "n/a"
    return str(value)


def _cap_decision(row: dict, cfg: EvalConfig) -> tuple[bool, str]:
    """Stage 1.5: exempt short-source Summary rows from the 1/3-length cap.

    Returns (applied, cap_note_text). Threshold 0 => literal rubric (never exempt).
    """
    if row.get("question_type") != "Summary":
        return False, ""
    src = row.get("source_word_count")
    if src in (None, ""):
        return False, ""
    n = int(src)
    if n < cfg.summary_length_cap_exempt_below_words:
        return True, CAP_NOTE_TEMPLATE.format(n=n)
    return False, ""


def build_user_message(
    row: dict, template: str, cfg: EvalConfig
) -> tuple[str, bool, str]:
    """Fill the template for one row. Returns (message, cap_applied, cap_note)."""
    cap_applied, cap_note = _cap_decision(row, cfg)
    msg = template
    for field, token in _METRIC_TOKENS.items():
        msg = msg.replace(token, _fmt(row.get(field)))
    msg = msg.replace("%%CAP_NOTE%%", cap_note)
    return msg, cap_applied, cap_note


def build_prompt(
    row: dict, rubric_text: str, rubric_hash: str, template: str, cfg: EvalConfig
) -> BuiltPrompt:
    """Assemble the prompt for one row with UNIFIED delivery.

    The rubric and per-row message are delivered as a single user turn to BOTH
    graders (Gemma has no system role, so this removes a system-vs-user
    confound). prompt_hash is taken over the role-structured message list, so it
    reflects delivery structure, not just text — a system/user split would hash
    differently and trip the cross-model prompt_hash assertion.
    """
    user_message, cap_applied, cap_note = build_user_message(row, template, cfg)
    user_content = rubric_text + "\n" + user_message
    messages = [{"role": "user", "content": user_content}]
    prompt_bytes = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return BuiltPrompt(
        system_prompt=rubric_text,
        user_message=user_message,
        user_content=user_content,
        messages=messages,
        prompt_bytes=prompt_bytes,
        prompt_hash=hashlib.sha256(prompt_bytes).hexdigest(),
        rubric_hash=rubric_hash,
        cap_exempt_applied=cap_applied,
        cap_note=cap_note,
        cap_threshold=cfg.summary_length_cap_exempt_below_words,
    )
