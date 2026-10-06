"""
Stage 2b: line reconciliation.

The page-level transcript (Stage 1 + Stage 2) reads with full-page context; a read of a single line
crop sees the strokes at higher resolution but without context. The two make different mistakes
(page reads miss thin cross-out strokes; crop reads garble names and lose words at crop edges).

For every page:
  1. segment the ink into text lines (horizontal projection profile, `LineLocalizer.projection_lines`);
  2. read each line crop independently;
  3. align transcript lines to crop reads (monotonic dynamic programming on text similarity);
  4. for every span where the two readings disagree — different letters, or struck in one and active
     in the other — show the crop with both versions of the line and ask which matches the ink,
     twice with the order swapped. The crop reading is adopted only when both answers choose it.

Nothing here knows about particular words or letters: candidates come from a second visual reading,
and the decision from a visual comparison whose only acceptance rule is order invariance.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image

from src.pipeline.arbitration.localizer import LineLocalizer
from src.prompts.line_reconciliation import (
    FORCED_CHOICE_PROMPT,
    FORCED_CHOICE_SYSTEM_PROMPT,
    LINE_READ_PROMPT,
    LINE_READ_SYSTEM_PROMPT,
)
from src.utils.transcript_markup import MarkupToken, normalize_markup, parse_markup, serialize_markup

_TAGS = re.compile(r"\[struck:|\]|\[unclear:|\[illegible\]|\[truncated\]", re.IGNORECASE)


def _plain(text: str) -> str:
    """Lowercased words only (tags and punctuation removed), for line-level similarity."""
    t = _TAGS.sub(" ", (text or "").lower())
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _line_tokens(line: str) -> List[MarkupToken]:
    return [t for t in parse_markup(line) if t.kind != "nl"]


def _content(tokens: List[MarkupToken]) -> List[MarkupToken]:
    """Tokens that carry a word (punctuation-only tokens are not compared)."""
    return [t for t in tokens if t.kind in ("text", "atom") and _key(t)[0]]


def _key(t: MarkupToken) -> Tuple[str, bool]:
    return re.sub(r"[^\w']", "", t.value.lower()), t.struck


def align_lines(transcript_lines: List[str], reads: List[str]) -> List[Tuple[int, int]]:
    """
    Monotonic one-to-one alignment of transcript lines to crop reads maximising total similarity.
    A pair contributes (similarity - 0.5): two lines are matched only if they are more alike than not.
    """
    n, m = len(transcript_lines), len(reads)
    ta = [_plain(x) for x in transcript_lines]
    rb = [_plain(x) for x in reads]
    sim = [[difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0 for b in rb] for a in ta]
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: List[List[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, step = float("-inf"), None
            if i and score[i - 1][j] > best:
                best, step = score[i - 1][j], "skip_t"
            if j and score[i][j - 1] > best:
                best, step = score[i][j - 1], "skip_r"
            if i and j and score[i - 1][j - 1] + sim[i - 1][j - 1] - 0.5 > best:
                best, step = score[i - 1][j - 1] + sim[i - 1][j - 1] - 0.5, "pair"
            score[i][j], back[i][j] = best, step
    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i or j:
        step = back[i][j]
        if step == "pair":
            if sim[i - 1][j - 1] > 0.5:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == "skip_t":
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def disagreement_spans(line: str, read: str) -> List[Tuple[int, int, List[MarkupToken]]]:
    """(start, end, replacement tokens) over the line's content tokens wherever word or strike differs."""
    a = _content(_line_tokens(line))
    b = _content(_line_tokens(read))
    sm = difflib.SequenceMatcher(None, [_key(t) for t in a], [_key(t) for t in b], autojunk=False)
    return [(a1, a2, b[b1:b2]) for op, a1, a2, b1, b2 in sm.get_opcodes() if op != "equal"]


def replace_span(line: str, start: int, end: int, new: List[MarkupToken]) -> str:
    """Line text with content tokens [start, end) replaced by `new`; all other characters kept."""
    toks = _line_tokens(line)
    content_pos = [k for k, t in enumerate(toks) if t.kind in ("text", "atom") and _key(t)[0]]
    if start < len(content_pos):
        lo = content_pos[start]
    else:
        lo = content_pos[-1] + 1 if content_pos else len(toks)
    hi = content_pos[end - 1] + 1 if end > start else lo
    ins: List[MarkupToken] = []
    for k, t in enumerate(new):
        if k:
            ins.append(MarkupToken("ws", " "))
        ins.append(MarkupToken(t.kind, t.value, t.struck))
    if end == start and ins:
        ins = [MarkupToken("ws", " ")] + ins + [MarkupToken("ws", " ")]
    return re.sub(r"[ \t]{2,}", " ", serialize_markup(toks[:lo] + ins + toks[hi:])).strip()


@dataclass
class ReconcileDecision:
    line_index: int
    current: str
    alternative: str
    votes: List[int]
    accepted: bool


@dataclass
class ReconcileResult:
    transcript: str
    decisions: List[ReconcileDecision] = field(default_factory=list)
    line_reads: List[str] = field(default_factory=list)
    line_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    model_calls: int = 0


class LineReconciler:
    def __init__(self, engine, crop_pad_px: int = 14, crop_min_height_px: int = 96, max_read_tokens: int = 200):
        self.engine = engine
        self.crop_pad_px = crop_pad_px
        self.crop_min_height_px = crop_min_height_px
        self.max_read_tokens = max_read_tokens

    def _generate(self, image: Image.Image, prompt: str, system_prompt: str, max_new_tokens: int) -> str:
        return self.engine.generate_multimodal(
            image=image, prompt=prompt, system_prompt=system_prompt,
            temperature=0.0, top_p=0.1, max_new_tokens=max_new_tokens, thinking_mode=False,
        ) or ""

    def _read_line(self, crop: Image.Image) -> str:
        text = self._generate(crop, LINE_READ_PROMPT, LINE_READ_SYSTEM_PROMPT, self.max_read_tokens).strip()
        return normalize_markup(text.splitlines()[0]) if text else ""

    def _prefers_alternative(self, crop: Image.Image, current: str, alternative: str) -> List[int]:
        """Votes for the alternative (1) or the current line (0) under both presentation orders."""
        votes = []
        for first, second, alt_slot in ((current, alternative, "2"), (alternative, current, "1")):
            r = self._generate(crop, FORCED_CHOICE_PROMPT.format(first=first, second=second), FORCED_CHOICE_SYSTEM_PROMPT, 4)
            m = re.search(r"[12]", r)
            votes.append(1 if (m and m.group(0) == alt_slot) else 0)
        return votes

    def run(self, image: Image.Image, transcript: str, page_no: int = 1) -> ReconcileResult:
        if not transcript or not transcript.strip():
            return ReconcileResult(transcript=transcript)
        loc = LineLocalizer(
            self.engine, [(page_no, image, "")], {page_no: transcript},
            crop_pad_px=self.crop_pad_px, crop_min_height_px=self.crop_min_height_px,
        )
        boxes = loc.projection_lines(page_no)
        crops = [loc.crop(page_no, b) for b in boxes]
        reads = [self._read_line(c) for c in crops]
        calls = len(reads)

        lines = transcript.split("\n")
        decisions: List[ReconcileDecision] = []
        for ti, ri in align_lines(lines, reads):
            line = lines[ti]
            accepted: List[Tuple[int, int, List[MarkupToken]]] = []
            for start, end, new in disagreement_spans(line, reads[ri]):
                alternative = replace_span(line, start, end, new)
                if _plain(alternative) == _plain(line) and \
                        [t.struck for t in _content(_line_tokens(alternative))] == [t.struck for t in _content(_line_tokens(line))]:
                    continue
                votes = self._prefers_alternative(crops[ri], line, alternative)
                calls += 2
                ok = votes == [1, 1]
                decisions.append(ReconcileDecision(ti, line, alternative, votes, ok))
                if ok:
                    accepted.append((start, end, new))
            # apply right to left so earlier content indices stay valid
            for start, end, new in sorted(accepted, key=lambda s: -s[0]):
                line = replace_span(line, start, end, new)
            lines[ti] = line

        return ReconcileResult(
            transcript=normalize_markup("\n".join(lines)),
            decisions=decisions, line_reads=reads, line_boxes=list(boxes), model_calls=calls,
        )
