"""
Allograph Calibrator: Self-Supervised Script-Level Handwriting Adaptation.

Across 1,500+ words of an exam script, students exhibit consistent, personal
cursive allographs (glyph formations):
  - Terminal 's' with a descender flourish transcribed as 'y'
    (sourcey -> sources, dreamery -> dreamers, algorithmy -> algorithms, featurey -> features)
  - Medial 's' with an arch transcribed as 'n' (hin -> his)
  - Cursive 'v' with shelf transcribed as 'r' (rumore -> remove, rillage -> village)
  - Asymmetric 'w' transcribed as 'cu' (cuith -> with)

This module statistically detects these script-wide patterns without human intervention
and produces a WriterProfile that normalizes the transcript and shields these words
from false linguistic penalties.
"""

import re
from typing import List, Tuple, Set, Optional, Dict, Any
from collections import Counter, defaultdict

from src.pipeline.arbitration.writer_profile import WriterProfile, _select_anchors
from src.pipeline.arbitration.candidate_selector import tokenize, levenshtein
from src.pipeline.arbitration.symbolic_evidence import align_chars


_ALPHA_WORD = re.compile(r"^[A-Za-z]+$")
_PUNCT_STRIP = re.compile(r"^[^\w]+|[^\w]+$")

# Retained as empty sets for backward-compatibility with external imports
CULTURAL_TERMS: Set[str] = set()
FORBIDDEN_ALLOGRAPH_TARGETS: Set[str] = set()

# Permissible Latin cursive handwriting allograph confusions grounded in stroke topology
PLAUSIBLE_ALLOGRAPH_PAIRS = {
    frozenset({'v', 'r'}),
    frozenset({'u', 'n'}),
    frozenset({'w', 'm'}),
    frozenset({'f', 'p'}),
    frozenset({'l', 't'}),
    frozenset({'c', 'e'}),
    frozenset({'a', 'o'}),
    frozenset({'h', 'b'}),
    frozenset({'s', 'n'}),
    frozenset({'g', 'y'}),
    frozenset({'i', 'l'}),
    frozenset({'e', 'l'}),
}


# Permissible multi-stroke Latin cursive ligature and minim confusions grounded in stroke topology
PLAUSIBLE_LIGATURE_PAIRS = [
    ("curr", "wr"),
    ("cu", "w"),
    ("vv", "w"),
    ("rn", "m"),
    ("cl", "d"),
    ("ney", "ness"),
]


def check_cursive_topology(
    word: str,
    combined_vocab: Set[str]
) -> Optional[Tuple[str, str]]:
    """
    Check if an out-of-vocabulary word maps to a legitimate dictionary word
    via an unambiguous single-character allograph confusion from PLAUSIBLE_ALLOGRAPH_PAIRS
    or a multi-stroke cursive ligature confusion from PLAUSIBLE_LIGATURE_PAIRS.
    Purely algorithmic: zero hardcoded word lists.
    """
    w_low = word.lower().strip()
    if not w_low or len(w_low) < 3 or w_low in combined_vocab:
        return None

    # 1. Check multi-stroke cursive ligature confusions
    for src_seq, dst_seq in PLAUSIBLE_LIGATURE_PAIRS:
        if src_seq in w_low:
            cand = w_low.replace(src_seq, dst_seq, 1)
            if cand in combined_vocab:
                return (f"ligature_{src_seq}_{dst_seq}", cand)

    # 2. Check 1-edit substitutions matching plausible cursive topology pairs
    for i, ch in enumerate(w_low):
        for pair in PLAUSIBLE_ALLOGRAPH_PAIRS:
            if ch in pair:
                alt_ch = next(c for c in pair if c != ch)
                cand = w_low[:i] + alt_ch + w_low[i+1:]
                if cand in combined_vocab:
                    return (f"allograph_{ch}_{alt_ch}", cand)

    return None


class AllographCalibrator:
    """
    Analyzes an entire exam script to discover writer-specific allograph rules
    and map out-of-vocabulary handwritten variants to valid dictionary words.
    """

    def __init__(
        self,
        min_support: int = 3,
        max_edits: int = 2
    ):
        self.min_support = min_support
        self.max_edits = max_edits

    def calibrate(
        self,
        script_id: str,
        transcript: str,
        lexicon: Set[str],
        question_vocab: Optional[Set[str]] = None
    ) -> WriterProfile:
        """
        Scan multi-page transcript, identify systematic allographs, and populate WriterProfile.
        """
        prof = WriterProfile(script_id=script_id)
        if not transcript or not lexicon:
            return prof

        q_vocab = {w.lower() for w in (question_vocab or set())}
        combined_vocab = lexicon | q_vocab

        # 1. Extract raw tokens preserving casing and positions
        raw_words = []
        for line in transcript.split("\n"):
            line_s = line.strip()
            # Ignore headers, tables, and question title lines
            if not line_s or line_s.startswith("|") or line_s.startswith("---") or re.match(r'^(?:ans|answer|dans|q(?:uestion)?|no)\b', line_s, re.IGNORECASE):
                continue
            for raw in line_s.split():
                clean = _PUNCT_STRIP.sub("", raw)
                if clean and _ALPHA_WORD.match(clean):
                    raw_words.append(clean)

        # Populate high-frequency anchors
        prof.anchors = _select_anchors([w.lower() for w in raw_words], lexicon, top_n=40)

        # 2. Identify Out-Of-Vocabulary (OOV) tokens
        word_counts = Counter(raw_words)
        oov_words = [
            w for w in word_counts
            if len(w) >= 3
            and w.lower() not in combined_vocab
        ]

        # 3. Generalized Dynamic Statistical Allograph Discovery
        # For OOV words, find closest dictionary words within Levenshtein <= 2
        # and accumulate character substitution pairs across the script.
        vocab_by_len: Dict[int, List[str]] = defaultdict(list)
        for vw in combined_vocab:
            if len(vw) >= 3:
                vocab_by_len[len(vw)].append(vw)

        # Collect candidate substitutions per word
        word_cand_subs: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for w in oov_words:
            w_low = w.lower()
            w_len = len(w_low)
            cand_pool = vocab_by_len[w_len] + vocab_by_len[w_len - 1] + vocab_by_len[w_len + 1]

            d1_matches = []
            for dw in cand_pool:
                if levenshtein(w_low, dw) == 1:
                    ops = align_chars(w_low, dw)
                    subs = [o for o in ops if o.op == "sub"]
                    non_matches = [o for o in ops if o.op != "match"]
                    if len(subs) == 1 and len(non_matches) == 1:
                        src_c = subs[0].src
                        tgt_c = subs[0].tgt
                        if src_c and tgt_c and src_c.isalpha() and tgt_c.isalpha():
                            d1_matches.append((dw, src_c, tgt_c))

            if d1_matches:
                word_cand_subs[w_low] = d1_matches
            else:
                for dw in cand_pool:
                    if levenshtein(w_low, dw) == 2:
                        ops = align_chars(w_low, dw)
                        subs = [o for o in ops if o.op == "sub"]
                        non_matches = [o for o in ops if o.op != "match"]
                        if len(subs) == 1 and len(non_matches) == 1:
                            src_c = subs[0].src
                            tgt_c = subs[0].tgt
                            if src_c and tgt_c and src_c.isalpha() and tgt_c.isalpha():
                                word_cand_subs[w_low].append((dw, src_c, tgt_c))

        # Count how many distinct words support each (src_c, tgt_c)
        pair_word_support: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for w_low, matches in word_cand_subs.items():
            for dw, src_c, tgt_c in matches:
                pair_word_support[(src_c, tgt_c)].add(w_low)

        # Promote rules with support >= min_support (default 2) that match PLAUSIBLE_ALLOGRAPH_PAIRS
        promoted_rules = {
            pair: words for pair, words in pair_word_support.items()
            if len(words) >= max(self.min_support, 2)
            and frozenset({pair[0], pair[1]}) in PLAUSIBLE_ALLOGRAPH_PAIRS
        }

        for (src_c, tgt_c), supporting_words in promoted_rules.items():
            rule_name = f"allograph_{src_c}_{tgt_c}"
            prof.discovered_allographs[rule_name] = tgt_c
            prof.add_pair(f"{tgt_c}>{src_c}", source="allograph_discovery", weight=len(supporting_words))

            rule_examples = []
            for w_low in supporting_words:
                matching_dws = [dw for dw, sc, tc in word_cand_subs[w_low] if (sc, tc) == (src_c, tgt_c)]
                if matching_dws:
                    best_dw = matching_dws[0]
                    prof.adapted_words[w_low] = best_dw
                    rule_examples.append((w_low, best_dw))

            prof.evidence.append({
                "rule": f"{src_c} <-> {tgt_c}",
                "support_count": len(supporting_words),
                "examples": rule_examples[:6]
            })

        return prof

    def apply_adaptations(
        self,
        transcript: str,
        profile: WriterProfile
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Normalize discovered allographs in the transcript while preserving exact casing.
        """
        if not transcript or not profile.adapted_words:
            return transcript, []

        diffs = []

        def _replace_word(match: re.Match) -> str:
            raw = match.group(0)
            low = raw.lower()
            if low in profile.adapted_words:
                target = profile.adapted_words[low]
                # Preserve capitalization
                adapted = target.capitalize() if raw[:1].isupper() else target
                diffs.append({
                    "original": raw,
                    "adapted": adapted,
                    "reason": f"writer allograph adaptation ({low} -> {target})"
                })
                return adapted
            return raw

        pattern = re.compile(r'\b[A-Za-z]+\b')
        calibrated_transcript = pattern.sub(_replace_word, transcript)
        return calibrated_transcript, diffs
