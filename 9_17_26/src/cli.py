"""Shared CLI helpers.

Every stage takes `--set`. One place decides what that flag means so the stages
cannot drift apart on it.
"""

from __future__ import annotations

from src.config import Config

ALL = "all"


def resolve_set(cfg: Config, requested: str | None) -> str | None:
    """Normalise `--set` into a set id, or None meaning 'every set'.

    Omitted -> config.default_exam_set (the stages are single-set by default, so
    a bare `python scripts/02_extract.py` never silently spends money on a paper
    the user was not thinking about). `--set all` -> None. An unknown id raises
    rather than silently matching nothing.
    """
    if requested == ALL:
        return None
    set_id = requested or cfg.default_exam_set
    cfg.exam_set(set_id)  # raises KeyError with the valid ids listed
    return set_id
