"""
Transcription accuracy metrics (pure python, no external dependencies).

Provides Character Error Rate (CER) and Word Error Rate (WER) between a reference
(human-corrected verbatim transcript) and a hypothesis (pipeline transcript), plus the
normalization used to make the two comparable (tag stripping, whitespace, Unicode NFC).

Also provides a "silent correction" probe: the set of reference tokens that are NOT in
the lexicon (i.e. genuine student non-words) that the hypothesis failed to reproduce.
This directly measures whether a stage autocorrected student mistakes.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Sequence, Set, Tuple, Optional, Dict, Any


_TAG_STRUCK = re.compile(r"\[struck:\s*([^\]]*)\]", re.IGNORECASE)
_TAG_UNCLEAR = re.compile(r"\[unclear:\s*([^\]]*)\]", re.IGNORECASE)
_TAG_ILLEGIBLE = re.compile(r"\[illegible\]", re.IGNORECASE)
_TAG_TRUNCATED = re.compile(r"\[truncated\]", re.IGNORECASE)
_PAGE_BREAK = re.compile(r"-{2,}\s*page\s*break\s*-{2,}", re.IGNORECASE)
_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*[-:_]{2,}.*$", re.MULTILINE)
_BOX_BORDERS = re.compile(r"[_~=+]{2,}")
_PUNCT = re.compile(r"[^\w\s'ঀ-৿]", re.UNICODE)


def _linearize_envelope_tables(text: str) -> str:
    """
    Detect postal envelope tables/boxes (containing 'from' and ('to' or 'stamp'))
    and linearize them into sequential word blocks: [From block] [To block] [Stamp block].
    Focuses evaluation strictly on handwritten words, removing box layout/border disparity.
    """
    lines = text.split("\n")
    out_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "|" in line or re.match(r"^[\s_\-+=~]{3,}$", line):
            block_lines: List[str] = []
            j = i
            while j < len(lines) and (
                "|" in lines[j]
                or re.match(r"^[\s_\-+=~]{3,}$", lines[j])
                or (block_lines and lines[j].strip() == "" and j + 1 < len(lines) and ("|" in lines[j + 1] or re.match(r"^[\s_\-+=~]{3,}$", lines[j + 1])))
            ):
                block_lines.append(lines[j])
                j += 1

            block_text = "\n".join(block_lines).lower()
            has_from = "from" in block_text
            has_to = "to" in block_text
            has_stamp = "stamp" in block_text

            if (has_from and (has_to or has_stamp)) or (has_to and has_stamp):
                content_rows: List[List[str]] = []
                for bl in block_lines:
                    if re.match(r"^[\s_|\+\-=~]+$", bl):
                        continue
                    stripped = bl.strip()
                    if stripped.startswith("|"):
                        stripped = stripped[1:]
                    if stripped.endswith("|"):
                        stripped = stripped[:-1]
                    cells = [c.strip() for c in stripped.split("|")]
                    if any(cells):
                        content_rows.append(cells)

                from_cells: List[str] = []
                to_cells: List[str] = []
                stamp_cells: List[str] = []
                other_cells: List[str] = []

                for row in content_rows:
                    expanded_row: List[str] = []
                    for c in row:
                        if re.search(r"\bfrom\b.*\s{2,}.*\bto\b", c, re.IGNORECASE):
                            parts = re.split(r"\s{2,}", c, maxsplit=1)
                            expanded_row.extend(parts)
                        elif len(row) > 1 and re.search(r"\S+\s{2,}\S+", c) and not ("stamp" in c.lower()):
                            parts = re.split(r"\s{2,}", c, maxsplit=1)
                            expanded_row.extend(parts)
                        else:
                            expanded_row.append(c)

                    for idx, c in enumerate(expanded_row):
                        c_clean = c.strip()
                        if not c_clean:
                            continue
                        c_lower = c_clean.lower()
                        if "stamp" in c_lower and len(c_clean) < 15:
                            stamp_cells.append(c_clean)
                        elif idx == 0 and len(expanded_row) > 1 and not ("to" in c_lower and "from" not in c_lower):
                            from_cells.append(c_clean)
                        elif idx == 1 and len(expanded_row) > 1 and not ("from" in c_lower and "to" not in c_lower):
                            to_cells.append(c_clean)
                        elif "from" in c_lower:
                            from_cells.append(c_clean)
                        elif "to" in c_lower and not ("p.t.o" in c_lower or "pto" in c_lower):
                            to_cells.append(c_clean)
                        elif len(from_cells) > 0 and len(to_cells) == 0:
                            from_cells.append(c_clean)
                        else:
                            other_cells.append(c_clean)

                linearized: List[str] = []
                if from_cells:
                    linearized.extend(from_cells)
                if to_cells:
                    linearized.extend(to_cells)
                if stamp_cells:
                    linearized.extend(stamp_cells)
                if other_cells:
                    linearized.extend(other_cells)

                out_lines.append("\n".join(linearized))
                i = j
                continue
        out_lines.append(line)
        i += 1
    return "\n".join(out_lines)


def normalize_transcript(
    text: str,
    keep_struck: bool = False,
    strip_punctuation: bool = False,
    lowercase: bool = True,
) -> str:
    """
    Normalize a transcript so reference and hypothesis are comparable.

    - `[struck: x]` -> dropped (or kept as `x` when keep_struck=True): struck text is not the
      student's final answer and its tagging is inconsistent between passes.
    - `[unclear: x]` -> `x` (the model's best reading is still a reading).
    - `[illegible]` -> kept as a single token `[illegible]` so a missing word costs one error.
    - `[truncated]` -> dropped: edge-of-page clipping metadata from reference annotations.
    - envelope tables/boxes are linearized sequentially (From -> To -> Stamp) ignoring borders.
    - page-break markers, markdown table separators, decorative borders, and LaTeX arrows are removed.
    - Unicode NFC, collapse whitespace, optional lowercase / punctuation stripping.
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = _linearize_envelope_tables(t)
    t = _PAGE_BREAK.sub(" ", t)
    t = _MD_TABLE_SEP.sub(" ", t)
    t = _BOX_BORDERS.sub(" ", t)
    t = t.replace("$\\rightarrow$", " ").replace("\\rightarrow", " ").replace("|", " ").replace("_", " ")
    t = _TAG_STRUCK.sub(r"\1" if keep_struck else " ", t)
    t = _TAG_UNCLEAR.sub(r"\1", t)
    t = _TAG_ILLEGIBLE.sub(" [illegible] ", t)
    t = _TAG_TRUNCATED.sub(" ", t)
    t = re.sub(r"(?<!\w)[~=+](?!\w)", " ", t)
    if lowercase:
        t = t.lower()
    if strip_punctuation:
        t = _PUNCT.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two sequences (strings or token lists)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def edit_ops(a: Sequence, b: Sequence) -> Tuple[int, int, int]:
    """Return (substitutions, insertions, deletions) turning a (ref) into b (hyp)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    # backtrace
    i, j = n, m
    subs = ins = dels = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            if a[i - 1] != b[j - 1]:
                subs += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1
    return subs, ins, dels


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = edits / len(reference). Inputs should already be normalized."""
    ref = reference or ""
    hyp = hypothesis or ""
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = token edits / number of reference tokens."""
    ref_t = (reference or "").split()
    hyp_t = (hypothesis or "").split()
    if not ref_t:
        return 0.0 if not hyp_t else 1.0
    return levenshtein(ref_t, hyp_t) / len(ref_t)


@dataclass
class TranscriptionScore:
    cer: float
    wer: float
    ref_chars: int
    ref_words: int
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    # silent-correction probe
    student_nonwords: int = 0
    nonwords_preserved: int = 0
    nonwords_lost: List[str] = field(default_factory=list)

    @property
    def silent_correction_rate(self) -> Optional[float]:
        if self.student_nonwords == 0:
            return None
        return 1.0 - (self.nonwords_preserved / self.student_nonwords)

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "ref_chars": self.ref_chars,
            "ref_words": self.ref_words,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "student_nonwords": self.student_nonwords,
            "nonwords_preserved": self.nonwords_preserved,
            "nonwords_lost": self.nonwords_lost,
        }
        scr = self.silent_correction_rate
        d["silent_correction_rate"] = round(scr, 4) if scr is not None else None
        return d


def _word_tokens(text: str) -> List[str]:
    return [w for w in _PUNCT.sub(" ", text).split() if w and w != "[illegible]"]


def score_transcription(
    reference: str,
    hypothesis: str,
    lexicon: Optional[Set[str]] = None,
) -> TranscriptionScore:
    """
    Score a hypothesis transcript against the reference.

    Both inputs are normalized with `normalize_transcript` (lowercase, tags resolved).
    CER is computed on the normalized strings; WER on whitespace tokens after punctuation
    stripping. If a lexicon is given, the silent-correction probe counts every reference
    token that is not in the lexicon (a student non-word) and checks whether the hypothesis
    contains the same token at least as many times.
    """
    ref_n = normalize_transcript(reference)
    hyp_n = normalize_transcript(hypothesis)
    ref_w = normalize_transcript(reference, strip_punctuation=True)
    hyp_w = normalize_transcript(hypothesis, strip_punctuation=True)

    subs, ins, dels = edit_ops(ref_n, hyp_n)
    score = TranscriptionScore(
        cer=cer(ref_n, hyp_n),
        wer=wer(ref_w, hyp_w),
        ref_chars=len(ref_n),
        ref_words=len(ref_w.split()),
        substitutions=subs,
        insertions=ins,
        deletions=dels,
    )

    if lexicon:
        ref_tokens = _word_tokens(ref_w)
        hyp_tokens = _word_tokens(hyp_w)
        hyp_counts: Dict[str, int] = {}
        for t in hyp_tokens:
            hyp_counts[t] = hyp_counts.get(t, 0) + 1
        seen: Dict[str, int] = {}
        for t in ref_tokens:
            if len(t) < 3 or t.isdigit() or t in lexicon:
                continue
            if not re.search(r"[a-zঀ-৿]", t):
                continue
            score.student_nonwords += 1
            seen[t] = seen.get(t, 0) + 1
            if hyp_counts.get(t, 0) >= seen[t]:
                score.nonwords_preserved += 1
            else:
                score.nonwords_lost.append(t)
    return score
