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
from typing import Any, Dict, List, Optional, Tuple

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


def _tokens_match(w1: str, w2: str) -> bool:
    """
    Checks if two words match for alignment.
    Strict equality for short words (<= 3 chars, e.g. 'in', 'on', 'to') to prevent drift.
    Single edit distance or high similarity (>= 0.79) for words >= 4 chars to handle OCR typos (e.g. 'binth' vs 'birth').
    """
    if w1 == w2:
        return True
    if len(w1) <= 3 or len(w2) <= 3:
        return False
    if abs(len(w1) - len(w2)) > 1:
        return False
    if len(w1) == len(w2) and sum(c1 != c2 for c1, c2 in zip(w1, w2)) <= 1:
        return True
    return difflib.SequenceMatcher(None, w1, w2, autojunk=False).ratio() >= 0.79


def disagreement_spans(line: str, read: str, scope: str = "strikes") -> List[Tuple[int, int, List[MarkupToken]]]:
    """
    (start, end, replacement tokens) over the line's content tokens where the two readings differ.
    scope="strikes": only spans with the same words (allowing minor OCR fuzzy typos) whose struck/active status differs.
    scope="all": also spans whose words differ.
    """
    a = _content(_line_tokens(line))
    b = _content(_line_tokens(read))
    sm = difflib.SequenceMatcher(None, [_key(t) for t in a], [_key(t) for t in b], autojunk=False)
    spans = []
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            continue
        words_a = [_key(t)[0] for t in a[a1:a2]]
        words_b = [_key(t)[0] for t in b[b1:b2]]
        same_words = False
        if len(words_a) == len(words_b) and words_a:
            same_words = all(_tokens_match(wa, wb) for wa, wb in zip(words_a, words_b))

        if same_words:
            # Transfer strike status from b, but retain authentic transcript token text from a (spelling immutability)
            if any(ta.struck != tb.struck for ta, tb in zip(a[a1:a2], b[b1:b2])):
                rep = [MarkupToken(ta.kind, ta.value, struck=tb.struck) for ta, tb in zip(a[a1:a2], b[b1:b2])]
                spans.append((a1, a2, rep))
            elif scope == "all" and [ta.value for ta in a[a1:a2]] != [tb.value for tb in b[b1:b2]]:
                spans.append((a1, a2, b[b1:b2]))
        elif scope == "all":
            spans.append((a1, a2, b[b1:b2]))
    return spans


def has_physical_midline_stroke(
    box: Tuple[int, int, int, int],
    strikethrough_regions: Optional[List[Any]] = None,
    strikethrough_blocks: Optional[List[Any]] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    Checks if any OpenCV Stage 0.5 detected stroke intersects the line box within the text midline.
    Excludes underlines (y_rel > 0.80) and strokes outside the horizontal span of the line box.
    """
    if not strikethrough_regions and not strikethrough_blocks:
        return False

    bx1, by1, bx2, by2 = box
    bh = max(1, by2 - by1)

    # 1. Check StrikethroughRegion list
    if strikethrough_regions:
        for r in strikethrough_regions:
            if getattr(r, "is_underline", False):
                continue
            rx = getattr(r, "x", 0)
            ry = getattr(r, "y", 0)
            rw = getattr(r, "w", 0)
            rh = getattr(r, "h", 0)

            # Horizontal overlap check
            if max(bx1, rx) >= min(bx2, rx + rw):
                continue

            # Vertical midline check within the line box (20% to 80% of text height)
            sy_mid = ry + rh / 2.0
            rel_y = (sy_mid - by1) / bh
            if 0.15 <= rel_y <= 0.80:
                return True

    # 2. Check StrikethroughBlock list (multi-line crossed out blocks)
    if strikethrough_blocks and image_shape:
        img_h, img_w = image_shape
        for b in strikethrough_blocks:
            by_px1 = getattr(b, "y_pct", 0.0) / 100.0 * img_h
            by_px2 = getattr(b, "y2_pct", 0.0) / 100.0 * img_h
            bx_px1 = getattr(b, "x_pct", 0.0) / 100.0 * img_w
            bx_px2 = getattr(b, "x2_pct", 0.0) / 100.0 * img_w

            if max(bx1, bx_px1) < min(bx2, bx_px2) and max(by1, by_px1) < min(by2, by_px2):
                return True

    return False


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


def parse_choice_digit(text: str, alt_slot: str) -> int:
    """Extracts 1 or 2 choice from VLM response; returns 1 if alt_slot chosen, else 0."""
    t = (text or "").strip()
    m = re.search(r"(?:Choice|Answer|Candidate)\s*[:=]?\s*([12])", t, re.IGNORECASE)
    if m:
        return 1 if m.group(1) == alt_slot else 0
    m = re.search(r"\b([12])\b", t)
    if m:
        return 1 if m.group(1) == alt_slot else 0
    return 0


class LineReconciler:
    """
    scope="strikes" (default): only struck/active disagreements are adjudicated. On the class-10 dev
    pages the crop-level judge was right ~70% of the time on strike status (accepted: 11 helped, 4 hurt)
    but at chance on letter identity (accepted: 44 helped, 78 hurt): a higher-resolution crop adds
    information about thin pen strokes, not about letter shapes the same model already read.
    """

    def __init__(self, engine, crop_pad_px: int = 14, crop_min_height_px: int = 96, max_read_tokens: int = 200,
                 scope: str = "strikes"):
        self.engine = engine
        self.scope = scope
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

    def _parse_choice_digit(self, text: str, alt_slot: str) -> int:
        return parse_choice_digit(text, alt_slot)

    def _prefers_alternative(
        self, crop: Image.Image, current: str, alternative: str, disputed: str = ""
    ) -> List[int]:
        """Votes for the alternative (1) or the current line (0) under both presentation orders."""
        votes = []
        disputed_sec = f'Disputed word/phrase: "{disputed}"\n\n' if disputed else ""
        for first, second, alt_slot in ((current, alternative, "2"), (alternative, current, "1")):
            prompt = FORCED_CHOICE_PROMPT.format(
                first=first, second=second, disputed_section=disputed_sec
            )
            r = self._generate(crop, prompt, FORCED_CHOICE_SYSTEM_PROMPT, max_new_tokens=35)
            votes.append(self._parse_choice_digit(r, alt_slot))
        return votes

    def run(
        self,
        image: Image.Image,
        transcript: str,
        page_no: int = 1,
        strikethrough_regions: Optional[List[Any]] = None,
        strikethrough_blocks: Optional[List[Any]] = None,
    ) -> ReconcileResult:
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

        img_shape = (image.height, image.width) if image else None

        lines = transcript.split("\n")
        decisions: List[ReconcileDecision] = []
        for ti, ri in align_lines(lines, reads):
            line = lines[ti]
            accepted: List[Tuple[int, int, List[MarkupToken]]] = []
            box = boxes[ri] if boxes and ri < len(boxes) else (0, 0, 0, 0)
            has_cv_stroke = has_physical_midline_stroke(
                box, strikethrough_regions, strikethrough_blocks, img_shape
            )

            for start, end, new in disagreement_spans(line, reads[ri], self.scope):
                alternative = replace_span(line, start, end, new)
                if _plain(alternative) == _plain(line) and \
                        [t.struck for t in _content(_line_tokens(alternative))] == [t.struck for t in _content(_line_tokens(line))]:
                    continue

                content_tokens = _content(_line_tokens(line))
                disputed_words = " ".join(t.value for t in content_tokens[start:end]) if start < len(content_tokens) else ""

                votes = self._prefers_alternative(crops[ri], line, alternative, disputed=disputed_words)
                calls += 2

                # Unanimous agreement ([1, 1]) -> accepted
                # Split vote ([0, 1] or [1, 0]) -> accepted if and only if OpenCV physically detected a midline stroke
                if votes == [1, 1]:
                    ok = True
                elif (votes == [0, 1] or votes == [1, 0]) and has_cv_stroke:
                    ok = True
                else:
                    ok = False

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
