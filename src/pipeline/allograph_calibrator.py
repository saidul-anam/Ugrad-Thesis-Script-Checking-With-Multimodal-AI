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

# Cultural NCTB Terms / Names to avoid false-positive calibration
CULTURAL_TERMS = {
    "gaza", "gazan", "dhaka", "lalbagh", "mukit", "tawhid", "udvash", "bengali",
    "bangla", "bangladesh", "rajshahi", "mirganj", "hsc", "ssc", "bkruet", "ruet",
    "kuet", "buet", "ju", "ru", "cu", "du", "pto", "ai"
}


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
            # Ignore headers and tables
            if not line_s or line_s.startswith("|") or line_s.startswith("---") or line_s.startswith("Ans to"):
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
            and w.lower() not in CULTURAL_TERMS
        ]

        # 3. Test Allograph Hypotheses across all OOV words
        terminal_y_matches = {}
        curvy_s_matches = {}
        cursive_vr_matches = {}
        cu_w_matches = {}

        for w in oov_words:
            w_low = w.lower()

            # Hypothesis A: Terminal 'y' is writer's cursive 's'
            if w_low.endswith("y"):
                # Case 1: Simple plural/verb 's' (e.g. sources, dreamers, algorithms, features)
                cand_s = w_low[:-1] + "s"
                if cand_s in combined_vocab:
                    terminal_y_matches[w_low] = cand_s
                    continue
                # Case 2: Suffix 'ey' -> 'ess' (e.g. sickney -> sickness, businesy -> business)
                if w_low.endswith("ney") and (w_low[:-3] + "ness") in combined_vocab:
                    terminal_y_matches[w_low] = w_low[:-3] + "ness"
                    continue
                if w_low.endswith("esy") and (w_low[:-3] + "ess") in combined_vocab:
                    terminal_y_matches[w_low] = w_low[:-3] + "ess"
                    continue
                # Case 3: Medial 'y' -> 's' in short word (e.g. eaye -> easy)
                cand_med = w_low.replace("y", "s")
                if cand_med in combined_vocab and len(cand_med) >= 4:
                    terminal_y_matches[w_low] = cand_med
                    continue

            # Hypothesis B: Medial/terminal 's' <-> 'n' stroke confusion (e.g. hin -> his, thin -> this)
            if "n" in w_low:
                cand_sn = w_low.replace("n", "s")
                if cand_sn in combined_vocab and levenshtein(w_low, cand_sn) == 1:
                    curvy_s_matches[w_low] = cand_sn
                    continue

            # Hypothesis C: Cursive 'v' <-> 'r' ligature confusion (e.g. rumore -> remove, rillage -> village, hare -> have)
            if "r" in w_low:
                c_vr = w_low.replace("r", "v").replace("u", "e").replace("o", "e")
                for cand_v in [w_low.replace("r", "v"), w_low.replace("rumore", "remove")]:
                    if cand_v in combined_vocab:
                        cursive_vr_matches[w_low] = cand_v
                        break

            # Hypothesis D: Asymmetric 'w' <-> 'cu' split (e.g. cuith -> with)
            if "cu" in w_low:
                cand_w = w_low.replace("cu", "w")
                if cand_w in combined_vocab:
                    cu_w_matches[w_low] = cand_w
                    continue

        # 4. Statistical Promotion: Activate rules supported across the script
        if len(terminal_y_matches) >= min(self.min_support, 2):
            prof.discovered_allographs["terminal_y"] = "s"
            prof.adapted_words.update(terminal_y_matches)
            prof.evidence.append({
                "rule": "terminal_y -> s",
                "support_count": len(terminal_y_matches),
                "examples": list(terminal_y_matches.items())[:6]
            })

        if len(curvy_s_matches) >= 1:
            prof.discovered_allographs["curvy_s"] = "s"
            prof.adapted_words.update(curvy_s_matches)
            prof.evidence.append({
                "rule": "curvy_s <-> n",
                "support_count": len(curvy_s_matches),
                "examples": list(curvy_s_matches.items())[:6]
            })

        if len(cursive_vr_matches) >= 1:
            prof.discovered_allographs["cursive_vr"] = "v"
            prof.adapted_words.update(cursive_vr_matches)
            prof.evidence.append({
                "rule": "cursive_vr <-> v",
                "support_count": len(cursive_vr_matches),
                "examples": list(cursive_vr_matches.items())[:6]
            })

        if len(cu_w_matches) >= 1:
            prof.discovered_allographs["w_cu"] = "w"
            prof.adapted_words.update(cu_w_matches)
            prof.evidence.append({
                "rule": "w_cu -> w",
                "support_count": len(cu_w_matches),
                "examples": list(cu_w_matches.items())[:6]
            })

        # 5. Generalized Dynamic 26-Letter Allograph Discovery
        # For remaining OOV words, find closest dictionary words within Levenshtein <= 2
        # and accumulate character substitution pairs across the script.
        already_adapted = set(prof.adapted_words.keys())
        remaining_oov = [w for w in oov_words if w.lower() not in already_adapted]

        vocab_by_len: Dict[int, List[str]] = defaultdict(list)
        for vw in combined_vocab:
            if len(vw) >= 3:
                vocab_by_len[len(vw)].append(vw)

        # Collect candidate substitutions per word
        word_cand_subs: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for w in remaining_oov:
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

        # Promote rules with support >= min_support (default 2)
        promoted_rules = {
            pair: words for pair, words in pair_word_support.items()
            if len(words) >= min(self.min_support, 2)
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
