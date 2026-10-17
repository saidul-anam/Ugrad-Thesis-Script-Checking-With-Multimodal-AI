"""
Per-stage wall time, model calls and tokens.

    timer = StageTimer(engine)
    timer.begin("stage1"); ...; timer.begin("stage2"); ...; timer.end()
    timer.summary()  # {"stage1": {"seconds": .., "calls": .., "prompt_tokens": .., "completion_tokens": ..}, ...}

begin() closes the stage that is open, so consecutive stages need one call each. Calls and tokens are the
change in engine.usage_totals over the stage; they are exact when pages run one at a time
(pipeline.parallel_workers: 1), which is the default.
"""

import time
from typing import Any, Dict, Iterable, Optional

_FIELDS = ("seconds", "calls", "prompt_tokens", "completion_tokens")


class StageTimer:
    def __init__(self, engine: Any = None):
        self.engine = engine
        self.stages: Dict[str, Dict[str, float]] = {}
        self._open: Optional[tuple] = None

    def _usage(self) -> Dict[str, int]:
        return dict(getattr(self.engine, "usage_totals", None) or {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})

    def begin(self, name: str) -> None:
        self.end()
        self._open = (name, time.perf_counter(), self._usage())

    def end(self) -> None:
        if not self._open:
            return
        name, t0, u0 = self._open
        u1 = self._usage()
        row = self.stages.setdefault(name, {f: 0 for f in _FIELDS})
        row["seconds"] = round(row["seconds"] + time.perf_counter() - t0, 3)
        for k in ("calls", "prompt_tokens", "completion_tokens"):
            row[k] += u1.get(k, 0) - u0.get(k, 0)
        self._open = None

    def summary(self) -> Dict[str, Dict[str, float]]:
        self.end()
        return {k: dict(v) for k, v in self.stages.items()}


def merge_timings(timings: Iterable[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Sum several summaries stage by stage, and add a "total" row."""
    out: Dict[str, Dict[str, float]] = {}
    for t in timings:
        for stage, row in (t or {}).items():
            if stage == "total":
                continue
            acc = out.setdefault(stage, {f: 0 for f in _FIELDS})
            for f in _FIELDS:
                acc[f] = round(acc[f] + row.get(f, 0), 3)
    out["total"] = {f: round(sum(r[f] for r in out.values()), 3) for f in _FIELDS}
    return out
