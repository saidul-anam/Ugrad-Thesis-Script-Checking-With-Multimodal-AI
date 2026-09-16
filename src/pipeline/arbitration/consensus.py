"""
Agreement-based confidence from augmented re-reads of a line crop.

Method (after arXiv 2509.09722): the crop is transcribed N times under small geometric and
photometric perturbations (plus one temperature-sampled variant). The N strings are aligned into
a character-level consensus with progressive Needleman-Wunsch alignment; the vote fraction of
the winning character in each column is its confidence, and a word's confidence is the minimum
over its characters. For the disputed token we also measure the share of re-reads that produced
the read token vs the intended token at the aligned position.

The re-read prompt carries no context and no candidate names, so the samples reflect the ink,
not the prior. No model internals (logprobs) are required.
"""

import difflib
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from src.pipeline.arbitration.candidate_selector import tokenize, levenshtein
from src.prompts.stage3b_arbitration import LINE_CROP_SYSTEM_PROMPT, LINE_CROP_VERBATIM_PROMPT


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------
@dataclass
class Augment:
    name: str
    scale: float = 1.0
    rotate_deg: float = 0.0
    gamma: float = 1.0
    contrast: float = 1.0
    pad_shift_px: int = 0
    temperature: float = 0.0


def default_augmentations(n: int, use_sampling: bool = True, temperature: float = 0.7) -> List[Augment]:
    base = [
        Augment("identity"),
        Augment("scale_0.8", scale=0.8),
        Augment("scale_1.2", scale=1.2),
        Augment("rot+1.5_gamma0.8", rotate_deg=1.5, gamma=0.8),
        Augment("rot-1.5_contrast1.2_pad24", rotate_deg=-1.5, contrast=1.2, pad_shift_px=24),
        Augment("pad48_gamma1.2", pad_shift_px=48, gamma=1.2),
        Augment("scale_0.9_rot+1", scale=0.9, rotate_deg=1.0),
        Augment("scale_1.1_contrast0.85", scale=1.1, contrast=0.85),
    ]
    out: List[Augment] = []
    if use_sampling and n >= 2:
        out.append(Augment("sampled_T", temperature=temperature))
    i = 0
    while len(out) < max(1, n):
        out.append(base[i % len(base)])
        i += 1
    return out[:max(1, n)]


def apply_augmentation(img: Image.Image, aug: Augment) -> Image.Image:
    im = img.convert("RGB")
    if aug.scale != 1.0:
        w, h = im.size
        im = im.resize((max(8, int(w * aug.scale)), max(8, int(h * aug.scale))), Image.LANCZOS)
    if aug.rotate_deg:
        im = im.rotate(aug.rotate_deg, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    if aug.gamma != 1.0:
        lut = [int(255 * ((i / 255.0) ** aug.gamma)) for i in range(256)]
        im = im.point(lut * 3)
    if aug.contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(aug.contrast)
    if aug.pad_shift_px:
        p = aug.pad_shift_px
        im = ImageOps.expand(im, border=(p, p // 2, p // 3, p // 2), fill=(255, 255, 255))
    return im


# ---------------------------------------------------------------------------
# Alignment & consensus
# ---------------------------------------------------------------------------
def needleman_wunsch(a: str, b: str, match: int = 1, mismatch: int = -1, gap: int = -1) -> Tuple[str, str]:
    """Global alignment; returns the two gapped strings ('-' marks a gap)."""
    n, m = len(a), len(b)
    S = np.zeros((n + 1, m + 1), dtype=np.int32)
    S[:, 0] = np.arange(n + 1) * gap
    S[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            diag = S[i - 1, j - 1] + (match if ai == b[j - 1] else mismatch)
            up = S[i - 1, j] + gap
            left = S[i, j - 1] + gap
            S[i, j] = max(diag, up, left)
    ra, rb = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and S[i, j] == S[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            ra.append(a[i - 1]); rb.append(b[j - 1]); i -= 1; j -= 1
        elif i > 0 and S[i, j] == S[i - 1, j] + gap:
            ra.append(a[i - 1]); rb.append("-"); i -= 1
        else:
            ra.append("-"); rb.append(b[j - 1]); j -= 1
    return "".join(reversed(ra)), "".join(reversed(rb))


@dataclass
class ConsensusResult:
    consensus: str                       # gap-free consensus string
    majority: List[str]                  # winning char per column (may include '-')
    columns: List[List[str]]             # chars each sample contributed per column
    column_votes: List[float]            # vote fraction of the winner per column
    n: int
    consensus_votes: List[float] = field(default_factory=list)  # votes for the gap-free consensus chars


def _seed_index(samples: List[str]) -> int:
    if len(samples) <= 2:
        return 0
    best_i, best_score = 0, -1.0
    for i, s in enumerate(samples):
        score = sum(difflib.SequenceMatcher(None, s, t).ratio() for j, t in enumerate(samples) if j != i)
        if score > best_score:
            best_i, best_score = i, score
    return best_i


def progressive_consensus(samples: List[str]) -> ConsensusResult:
    """Center-star progressive alignment with per-column majority voting."""
    samples = [s.strip() for s in samples if s is not None]
    samples = [s for s in samples if s]
    n = len(samples)
    if n == 0:
        return ConsensusResult("", [], [], [], 0)
    seed_i = _seed_index(samples)
    columns: List[List[str]] = [[c] for c in samples[seed_i]]
    n_done = 1
    for idx, s in enumerate(samples):
        if idx == seed_i:
            continue
        # profile is gap-free by construction (columns with a gap majority are dropped below)
        profile = "".join(_majority(col) for col in columns)
        pa, pb = needleman_wunsch(profile, s)
        new_cols: List[List[str]] = []
        ci = 0
        for ca, cb in zip(pa, pb):
            if ca == "-":
                new_cols.append(["-"] * n_done + [cb])
            else:
                new_cols.append(columns[ci] + [cb])
                ci += 1
        n_done += 1
        columns = [col for col in new_cols if _majority(col) != "-"]
    majority = [_majority(col) for col in columns]
    votes = [col.count(maj) / len(col) for col, maj in zip(columns, majority)]
    consensus_chars, consensus_votes = [], []
    for maj, v in zip(majority, votes):
        if maj != "-":
            consensus_chars.append(maj)
            consensus_votes.append(v)
    return ConsensusResult("".join(consensus_chars), majority, columns, votes, n, consensus_votes)


def _majority(col: List[str]) -> str:
    counts: Dict[str, int] = {}
    for c in col:
        counts[c] = counts.get(c, 0) + 1
    # tie-break: prefer non-gap, then lexical
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0] == "-", kv[0]))[0][0]


def word_confidences(res: ConsensusResult) -> List[Tuple[str, float]]:
    """(word, min char vote) for each whitespace-delimited word of the consensus."""
    out: List[Tuple[str, float]] = []
    cur, cur_votes = [], []
    for ch, v in zip(res.consensus, res.consensus_votes):
        if ch.isspace():
            if cur:
                out.append(("".join(cur), min(cur_votes)))
            cur, cur_votes = [], []
        else:
            cur.append(ch)
            cur_votes.append(v)
    if cur:
        out.append(("".join(cur), min(cur_votes)))
    return out


def aligned_token(sample_tokens: List[str], ref_tokens: List[str], target_idx: int) -> Optional[str]:
    """Token of `sample_tokens` aligned to ref_tokens[target_idx] (None if deleted/unaligned)."""
    if target_idx < 0 or target_idx >= len(ref_tokens):
        return None
    sm = difflib.SequenceMatcher(a=ref_tokens, b=sample_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if i1 <= target_idx < i2:
            if tag == "equal":
                return sample_tokens[j1 + (target_idx - i1)]
            if tag == "replace":
                if (i2 - i1) == (j2 - j1):
                    return sample_tokens[j1 + (target_idx - i1)]
                # unequal replace: pick the sample token closest by edit distance to the ref token
                ref = ref_tokens[target_idx]
                cands = sample_tokens[j1:j2]
                return min(cands, key=lambda t: levenshtein(t, ref)) if cands else None
            return None
    return None


def _find_target_index(
    ref_tokens: List[str],
    candidate: str,
    intended: str,
    context_sentence: Optional[str] = None,
) -> int:
    cand, inten = candidate.lower(), intended.lower()
    for i, t in enumerate(ref_tokens):
        if t == cand:
            return i
    for i, t in enumerate(ref_tokens):
        if t == inten:
            return i
    best_i, best_d = -1, 10 ** 9
    for i, t in enumerate(ref_tokens):
        d = min(levenshtein(t, cand), levenshtein(t, inten))
        if d < best_d:
            best_i, best_d = i, d
    if best_d <= 2:
        return best_i

    # Context-anchored fallback:
    # If the candidate/intended token is not found within 2 edits (e.g. OCR read 'renny', dictionary guessed 'runny',
    # but the paper has 'verry'), use surrounding context words from context_sentence to find the target position in ref_tokens.
    if context_sentence:
        ctx_tokens = tokenize(context_sentence)
        cand_in_ctx = -1
        for i, t in enumerate(ctx_tokens):
            if t == cand or t == inten or min(levenshtein(t, cand), levenshtein(t, inten)) <= 1:
                cand_in_ctx = i
                break
        if cand_in_ctx >= 0:
            prev_tok = ctx_tokens[cand_in_ctx - 1] if cand_in_ctx > 0 else None
            next_tok = ctx_tokens[cand_in_ctx + 1] if cand_in_ctx + 1 < len(ctx_tokens) else None
            if prev_tok and next_tok:
                for j in range(len(ref_tokens) - 2):
                    if ref_tokens[j] == prev_tok and ref_tokens[j + 2] == next_tok:
                        return j + 1
            if prev_tok:
                for j in range(len(ref_tokens) - 1):
                    if ref_tokens[j] == prev_tok:
                        return j + 1
            if next_tok:
                for j in range(1, len(ref_tokens)):
                    if ref_tokens[j] == next_tok:
                        return j - 1

    return -1


def agreement(
    samples: List[str],
    reference_line: str,
    candidate: str,
    intended: str,
    context_sentence: Optional[str] = None,
) -> Tuple[float, float, Optional[float], List[Optional[str]]]:
    """
    Share of re-reads whose aligned token equals the candidate / the intended token, plus the
    token-level confidence of the disputed word in the consensus (None if not locatable).
    """
    ref_tokens = tokenize(reference_line)
    t_idx = _find_target_index(ref_tokens, candidate, intended, context_sentence)
    aligned: List[Optional[str]] = []
    for s in samples:
        aligned.append(aligned_token(tokenize(s), ref_tokens, t_idx) if t_idx >= 0 else None)
    n = len(samples) or 1
    agr_c = sum(1 for t in aligned if t == candidate.lower()) / n
    agr_i = sum(1 for t in aligned if t == intended.lower()) / n

    word_conf: Optional[float] = None
    res = progressive_consensus(samples)
    wc = word_confidences(res)
    if wc:
        # the disputed word in the consensus: closest to candidate/intended
        best = min(wc, key=lambda p: min(levenshtein(p[0].lower(), candidate.lower()), levenshtein(p[0].lower(), intended.lower())))
        if min(levenshtein(best[0].lower(), candidate.lower()), levenshtein(best[0].lower(), intended.lower())) <= 2:
            word_conf = best[1]
    return agr_c, agr_i, word_conf, aligned


def consensus_signal(agr_candidate: float, agr_intended: float, word_conf: Optional[float]) -> float:
    """1 = re-reads favour the intended word / disagree a lot; 0 = re-reads firmly reproduce the read token."""
    base = min(1.0, max(0.0, 0.5 + 0.5 * (agr_intended - agr_candidate)))
    if word_conf is None:
        return base
    if base >= 0.8:
        return min(1.0, max(base, 0.8 + 0.2 * word_conf))
    if base <= 0.2:
        return max(0.0, min(base, base * (1.2 - 0.2 * word_conf)))
    return 0.7 * base + 0.3 * (1.0 - word_conf)


# ---------------------------------------------------------------------------
# Engine-facing transcriber
# ---------------------------------------------------------------------------
import threading


class ConsensusTranscriber:
    def __init__(
        self,
        engine,
        n_samples: int = 5,
        use_sampling: bool = True,
        temperature: float = 0.7,
        seed: int = 0,
        parallel_workers: int = 1,
    ):
        self.engine = engine
        self.augs = default_augmentations(n_samples, use_sampling, temperature)
        self.model_calls = 0
        self.token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.parallel_workers = max(1, int(parallel_workers or 1))
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def _accumulate_usage(self) -> None:
        with self._lock:
            self.model_calls += 1
            u = self.engine.get_last_usage() or {}
            for k in self.token_usage:
                self.token_usage[k] += int(u.get(k, 0) or 0)

    def read_once(self, crop: Image.Image, aug: Optional[Augment] = None) -> str:
        img = apply_augmentation(crop, aug) if aug else crop
        out = self.engine.generate_multimodal(
            image=img,
            prompt=LINE_CROP_VERBATIM_PROMPT,
            system_prompt=LINE_CROP_SYSTEM_PROMPT,
            temperature=(aug.temperature if aug else 0.0),
            top_p=(0.95 if aug and aug.temperature > 0 else 0.1),
            max_new_tokens=96,
            thinking_mode=False,
        )
        self._accumulate_usage()
        return _clean_line(out)

    def run(
        self,
        crop: Image.Image,
        reference_line: str,
        candidate: str,
        intended: str,
        context_sentence: Optional[str] = None,
    ) -> Dict[str, Any]:
        samples: List[str] = [""] * len(self.augs)
        if self.parallel_workers > 1 and len(self.augs) > 1:
            from concurrent.futures import ThreadPoolExecutor

            def _read_idx(pair):
                idx, aug = pair
                try:
                    return idx, self.read_once(crop, aug)
                except Exception:
                    return idx, ""

            with ThreadPoolExecutor(max_workers=min(len(self.augs), self.parallel_workers)) as executor:
                for idx, res in executor.map(_read_idx, enumerate(self.augs)):
                    samples[idx] = res
        else:
            for i, aug in enumerate(self.augs):
                try:
                    samples[i] = self.read_once(crop, aug)
                except Exception as ex:  # a failed re-read is simply a missing sample
                    samples[i] = ""
        samples_ok = [s for s in samples if s]
        if not samples_ok:
            return {"samples": samples, "agreement_candidate": None, "agreement_intended": None,
                    "word_confidence": None, "signal": None, "consensus": None,
                    "consensus_token": None, "consensus_token_agreement": 0.0}
        agr_c, agr_i, word_conf, aligned = agreement(samples_ok, reference_line, candidate, intended, context_sentence)
        res = progressive_consensus(samples_ok)

        # Extract dominant visual consensus token across aligned tokens
        valid_aligned = [t.lower() for t in aligned if t]
        cons_tok = None
        cons_tok_agr = 0.0
        if valid_aligned:
            from collections import Counter
            top_tok, top_cnt = Counter(valid_aligned).most_common(1)[0]
            cons_tok = top_tok
            cons_tok_agr = round(top_cnt / len(samples_ok), 3)

        return {
            "samples": samples,
            "aligned_tokens": aligned,
            "agreement_candidate": agr_c,
            "agreement_intended": agr_i,
            "word_confidence": word_conf,
            "signal": consensus_signal(agr_c, agr_i, word_conf),
            "consensus": res,
            "consensus_token": cons_tok,
            "consensus_token_agreement": cons_tok_agr,
        }


def _clean_line(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
    # keep the first non-empty line only
    for line in t.splitlines():
        if line.strip():
            return line.strip().strip('"').strip()
    return t
