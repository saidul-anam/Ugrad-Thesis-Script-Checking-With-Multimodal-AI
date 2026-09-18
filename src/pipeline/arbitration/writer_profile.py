"""
Writer-adaptive confusion profile (learned per script; replaces any hand-written confusion table).

For the script being graded we count, from the writer's own ink, which character substitutions
the transcription model produces for this hand. Three sources, all data-driven:

  (a) consensus disagreements: when the same line crop is re-read several times, columns where
      the re-reads disagree reveal which letters are ambiguous in this hand
      (majority char -> minority char);
  (b) anchor near-misses: tokens in the transcript that are not dictionary words but are one
      edit away from one of this writer's own most frequent dictionary words
      ("dhe" next to many "the" => 't>d');
  (c) gated candidates themselves, added after they are scored (weakest source; guarded by
      `min_count` so a single misread cannot self-confirm).

Anchor words are the writer's own high-frequency lexicon tokens, not a fixed list.
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional, Any

from pydantic import BaseModel, Field

from src.pipeline.arbitration.candidate_selector import tokenize, levenshtein
from src.pipeline.arbitration.symbolic_evidence import align_chars, EditOp


_WORD_RE = re.compile(r"^[a-zঀ-৿']+$")


class WriterProfile(BaseModel):
    script_id: str = "unknown"
    pair_counts: Dict[str, int] = Field(default_factory=dict, description="'intended>read' -> count")
    anchors: List[str] = Field(default_factory=list, description="This writer's high-frequency lexicon words")
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    discovered_allographs: Dict[str, str] = Field(default_factory=dict, description="Discovered script-level allographs, e.g. {'terminal_y': 's', 'curvy_s': 's'}")
    adapted_words: Dict[str, str] = Field(default_factory=dict, description="Script-wide adapted words mapping non-words to dictionary words")
    stitched_splits: List[Dict[str, Any]] = Field(default_factory=list, description="List of stitched pen-lift tokens")

    def add_pair(self, key: str, source: str, token: str = "", anchor: str = "", weight: int = 1) -> None:
        if not key:
            return
        self.pair_counts[key] = self.pair_counts.get(key, 0) + weight
        self.evidence.append({"key": key, "source": source, "token": token, "anchor": anchor})

    def count_for_ops(self, ops: List[EditOp]) -> int:
        """Minimum count over the non-match ops (0 if any op has never been observed)."""
        keys = [o.key for o in ops if o.op != "match"]
        if not keys:
            return 0
        return min(self.pair_counts.get(k, 0) for k in keys)

    def is_writer_allograph(self, candidate_token: str, intended_token: str) -> bool:
        """Check if candidate pair matches a discovered allograph rule for this script."""
        c_low = candidate_token.lower()
        i_low = intended_token.lower()
        if c_low in self.adapted_words:
            return True
        for split in self.stitched_splits:
            orig_parts = split.get("original", "").lower().split()
            if c_low in orig_parts:
                return True
        for rule_key, tgt in self.discovered_allographs.items():
            if rule_key == "terminal_y" and tgt == "s":
                if c_low.endswith("y") and i_low.endswith("s") and c_low[:-1] == i_low[:-1]:
                    return True
                if c_low.replace("ey", "ess") == i_low or c_low.replace("y", "s") == i_low:
                    return True
            if rule_key == "curvy_s" and tgt == "s":
                if (c_low.replace("n", "s") == i_low or i_low.replace("n", "s") == c_low):
                    return True
            if rule_key == "cursive_vr" and tgt == "v":
                if (c_low.replace("r", "v") == i_low or c_low.replace("rumore", "remove") == i_low):
                    return True
        return False


def _select_anchors(tokens: List[str], lexicon: Set[str], top_n: int = 40, min_freq: int = 2) -> List[str]:
    counts = Counter(t for t in tokens if len(t) >= 2 and t in lexicon and _WORD_RE.match(t))
    return [w for w, c in counts.most_common(top_n) if c >= min_freq]


def build_writer_profile(
    script_id: str,
    transcript: str,
    lexicon: Set[str],
    question_vocab: Optional[Set[str]] = None,
    top_anchors: int = 40,
) -> WriterProfile:
    """Source (b): count near-miss substitutions between non-words and this writer's own anchor words."""
    prof = WriterProfile(script_id=script_id)
    tokens = tokenize(transcript)
    if not tokens or not lexicon:
        return prof
    q_vocab = {w.lower() for w in (question_vocab or set())}
    anchors = _select_anchors(tokens, lexicon, top_n=top_anchors)
    prof.anchors = anchors
    if not anchors:
        return prof

    anchor_set = set(anchors)
    seen_tokens = Counter(tokens)
    for tok, freq in seen_tokens.items():
        if len(tok) < 2 or tok in lexicon or tok in q_vocab or tok in anchor_set or not _WORD_RE.match(tok):
            continue
        # allow 1 edit, or 2 substitutions for tokens of length >= 4
        best: Optional[Tuple[str, List[EditOp]]] = None
        for a in anchors:
            if abs(len(a) - len(tok)) > 1:
                continue
            dist = levenshtein(tok, a)
            if dist == 0 or dist > 2:
                continue
            ops = align_chars(tok, a)
            non_match = [o for o in ops if o.op != "match"]
            if dist == 2 and (len(a) < 4 or any(o.op != "sub" for o in non_match)):
                continue
            if best is None or dist < levenshtein(tok, best[0]):
                best = (a, non_match)
        if best is None:
            continue
        anchor, non_match = best
        for o in non_match:
            prof.add_pair(o.key, source="transcript", token=tok, anchor=anchor, weight=freq)
    return prof


def add_consensus_disagreements(profile: WriterProfile, columns: List[List[str]], majority: List[str]) -> int:
    """
    Source (a): `columns[i]` are the characters the N re-reads produced at consensus column i,
    `majority[i]` is the winning char ('-' = gap). Every minority char is counted as
    'majority>minority'. Returns the number of pairs added.
    """
    added = 0
    for col, maj in zip(columns, majority):
        if not maj or maj == "-" or not maj.isalpha():
            continue
        for ch in col:
            if ch == maj or ch.isspace() or (ch != "-" and not ch.isalpha()):
                continue
            key = f"{maj}>{ch if ch != '-' else '∅'}"
            profile.add_pair(key, source="consensus")
            added += 1
    return added


def add_candidate_evidence(profile: WriterProfile, candidate_token: str, intended_token: str) -> None:
    """Source (c): a gated candidate contributes its edit path once (after it has been scored)."""
    for o in align_chars(candidate_token.lower(), intended_token.lower()):
        if o.op != "match":
            profile.add_pair(o.key, source="stage3", token=candidate_token, anchor=intended_token)


def writer_prior(profile: WriterProfile, ops: List[EditOp], min_count: int = 2) -> Tuple[float, int]:
    """
    Ambiguity prior in [0,1]: mean over the candidate's edit ops of a saturating function of how
    often this writer's ink has already produced that op (ops seen fewer than `min_count` times
    contribute 0). Returns (prior, min count over ops).
    """
    import math
    keys = [o.key for o in ops if o.op != "match"]
    if not keys:
        return 0.0, 0
    contribs = []
    counts = []
    for k in keys:
        c = profile.pair_counts.get(k, 0)
        counts.append(c)
        contribs.append(0.0 if c < max(1, min_count) else min(1.0, 1.0 - math.exp(-c / 3.0)))
    return sum(contribs) / len(contribs), min(counts)


def save_writer_profile(profile: WriterProfile, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(), f, ensure_ascii=False, indent=2)


def load_writer_profile(path: str) -> WriterProfile:
    with open(path, "r", encoding="utf-8") as f:
        return WriterProfile.model_validate(json.load(f))
