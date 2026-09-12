"""
Symbolic (model-free) evidence for one candidate pair.

Only two generic, writer-independent things are computed here:

1. `align_chars` — the character edit path between the read token and the intended token.
   It is used to *describe* the discrepancy (e.g. ['sub:f>d']) so the writer profile can count it.
   It carries no judgement about which letters are confusable; that is learned per writer.

2. `phonetic_plausibility` — how similar the two tokens sound. A genuine learner misspelling is
   usually phonologically motivated (it sounds like the target word); a perceptual misread often is
   not. This is a general linguistic property, not a rule about any word or letter. Its influence is
   decided by the fitted fusion weight. Non-Latin scripts return a neutral 0.5.
"""

from dataclasses import dataclass
from typing import List, Tuple

from src.pipeline.arbitration.candidate_selector import levenshtein


@dataclass(frozen=True)
class EditOp:
    op: str      # match | sub | ins | del
    src: str     # char in the read (candidate) token, '' for ins
    tgt: str     # char in the intended token, '' for del

    @property
    def key(self) -> str:
        """Confusion-pair key in the form 'intended>read' (∅ marks an absent char)."""
        if self.op == "match":
            return ""
        return f"{self.tgt or '∅'}>{self.src or '∅'}"

    def __str__(self) -> str:
        return f"{self.op}:{self.tgt or '∅'}>{self.src or '∅'}" if self.op != "match" else f"match:{self.src}"


def align_chars(candidate: str, intended: str) -> List[EditOp]:
    """Levenshtein backtrace between candidate (read) and intended tokens."""
    a, b = candidate or "", intended or ""
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
    ops: List[EditOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            ops.append(EditOp("match" if a[i - 1] == b[j - 1] else "sub", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(EditOp("ins", a[i - 1], ""))      # extra char in the read token
            i -= 1
        else:
            ops.append(EditOp("del", "", b[j - 1]))      # char of intended missing in read
            j -= 1
    ops.reverse()
    return ops


def edit_op_keys(ops: List[EditOp]) -> List[str]:
    return [o.key for o in ops if o.op != "match"]


# ---------------------------------------------------------------------------
# Phonetic coding
# ---------------------------------------------------------------------------
_VOWELS = set("aeiou")


def _metaphone_fallback(word: str) -> str:
    """Compact implementation of the original Metaphone algorithm (English)."""
    w = "".join(ch for ch in word.lower() if ch.isalpha())
    if not w:
        return ""
    # initial exceptions
    if w[:2] in ("kn", "gn", "pn", "ae", "wr"):
        w = w[1:]
    elif w[0] == "x":
        w = "s" + w[1:]
    elif w[:2] == "wh":
        w = "w" + w[2:]

    out = []
    n = len(w)
    i = 0
    while i < n:
        c = w[i]
        prev = w[i - 1] if i > 0 else ""
        nxt = w[i + 1] if i + 1 < n else ""
        nxt2 = w[i + 2] if i + 2 < n else ""

        if c == prev and c != "c":
            i += 1
            continue
        if c in _VOWELS:
            if i == 0:
                out.append(c.upper())
        elif c == "b":
            if not (prev == "m" and i == n - 1):
                out.append("B")
        elif c == "c":
            if nxt == "i" and nxt2 == "a":
                out.append("X")
            elif nxt == "h":
                out.append("K" if prev == "s" else "X")
                i += 1
            elif nxt in ("i", "e", "y"):
                if prev != "s":
                    out.append("S")
            else:
                out.append("K")
        elif c == "d":
            if nxt == "g" and nxt2 in ("e", "y", "i"):
                out.append("J")
                i += 2
            else:
                out.append("T")
        elif c == "g":
            if nxt == "h" and (i + 2 < n and w[i + 2] not in _VOWELS):
                pass
            elif nxt == "n" and (i + 2 == n or (nxt2 == "e" and w[i + 3:i + 4] == "d")):
                pass
            elif nxt in ("i", "e", "y") and prev != "g":
                out.append("J")
            elif nxt == "h":
                out.append("F")
                i += 1
            else:
                out.append("K")
        elif c == "h":
            if prev in _VOWELS and nxt not in _VOWELS:
                pass
            elif prev in ("c", "s", "p", "t", "g"):
                pass
            else:
                out.append("H")
        elif c == "k":
            if prev != "c":
                out.append("K")
        elif c == "p":
            if nxt == "h":
                out.append("F")
                i += 1
            else:
                out.append("P")
        elif c == "q":
            out.append("K")
        elif c == "s":
            if nxt == "h":
                out.append("X")
                i += 1
            elif nxt == "i" and nxt2 in ("o", "a"):
                out.append("X")
            else:
                out.append("S")
        elif c == "t":
            if nxt == "i" and nxt2 in ("o", "a"):
                out.append("X")
            elif nxt == "h":
                out.append("0")
                i += 1
            elif not (nxt == "c" and nxt2 == "h"):
                out.append("T")
        elif c == "v":
            out.append("F")
        elif c in ("w", "y"):
            if nxt in _VOWELS:
                out.append(c.upper())
        elif c == "x":
            out.append("KS")
        elif c == "z":
            out.append("S")
        else:  # f j l m n r
            out.append(c.upper())
        i += 1
    return "".join(out)


def metaphone(word: str) -> str:
    """Metaphone code of an English word (jellyfish if installed, else the built-in fallback)."""
    try:
        import jellyfish  # type: ignore
        return jellyfish.metaphone(word or "").replace(" ", "").upper()
    except Exception:
        return _metaphone_fallback(word or "")


def _is_latin(word: str) -> bool:
    return all(ord(ch) < 0x0250 for ch in word if ch.isalpha())


def phonetic_plausibility(candidate: str, intended: str) -> float:
    """
    Similarity in [0,1] of the phonetic codes of the read and intended tokens.
    1.0 = they sound the same (typical of a genuine learner misspelling);
    0.0 = phonetically unrelated (typical of a perceptual misread).
    Non-Latin tokens (e.g. Bangla) return 0.5 (neutral).
    """
    c = (candidate or "").strip().lower()
    t = (intended or "").strip().lower()
    if not c or not t:
        return 0.5
    if not (_is_latin(c) and _is_latin(t)):
        return 0.5
    if c == t:
        return 1.0
    pc, pt = metaphone(c), metaphone(t)
    if not pc or not pt:
        return 0.5
    if pc == pt:
        return 1.0
    dist = levenshtein(pc, pt)
    return max(0.0, 1.0 - dist / max(len(pc), len(pt)))


def symbolic_evidence(candidate: str, intended: str) -> Tuple[float, List[str]]:
    """Return (phonetic_plausibility, edit_op_keys)."""
    ops = align_chars(candidate.lower(), intended.lower())
    return phonetic_plausibility(candidate, intended), [str(o) for o in ops if o.op != "match"]
