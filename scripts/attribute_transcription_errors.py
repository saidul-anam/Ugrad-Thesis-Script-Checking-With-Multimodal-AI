#!/usr/bin/env python3
"""
Word-level error attribution of pipeline transcripts against human ground truth.

Where scripts/evaluate_transcription.py gives one CER/WER number per stage, this script says *what
kind* of mistakes make up that number. Struck text is kept with a struck flag (not dropped), so
strike mistakes are counted separately from reading mistakes:

  false_strike    word is active in the ground truth but [struck:] in the transcript
  missed_strike   word is [struck:] in the ground truth but active in the transcript
  misread         wrong letters (active in both)
  autocorrection  student non-word (not in the lexicon) transcribed as a dictionary word
  dropped_word    active ground-truth word missing from the transcript
  extra_word      active transcript word not in the ground truth
  (differences inside struck text on both sides are listed but marked "ignored")

Usage:
  python scripts/attribute_transcription_errors.py \
      --gt-dir data/ground_truth/transcripts/english_se_10_q1 \
      --extracted-dir outputs/extracted/english/se_10_q1_v2 [--examples 20] [--per-page]
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.linguistic_sanitizer import get_english_lexicon
from src.utils.transcript_markup import struck_word_spans


def words(text: str):
    t = re.sub(r"\[unclear:\s*([^\]|]*)(?:\|[^\]]*)?\]", r"\1", text or "")
    t = re.sub(r"\[(?:truncated|illegible)\]", " ", t)
    out = []
    for tok, struck in struck_word_spans(t):
        for w in re.sub(r"[^\w']", " ", tok.lower()).split():
            w = w.strip("'")
            if w:
                out.append((w, struck))
    return out


def align(a, b):
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1][0] != b[j - 1][0]))
    i, j, ops = n, m, []
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1][0] != b[j - 1][0]):
            ops.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif j and d[i][j] == d[i][j - 1] + 1:
            ops.append((None, b[j - 1])); j -= 1
        else:
            ops.append((a[i - 1], None)); i -= 1
    return ops[::-1]


def classify(g, h, lexicon):
    if g and h and g[0] == h[0]:
        if g[1] == h[1]:
            return None
        return "missed_strike" if g[1] else "false_strike"
    if g and h:
        if g[1] and h[1]:
            return "struck_misread (ignored)"
        if g[1]:
            return "missed_strike"
        if h[1]:
            return "false_strike"
        return "autocorrection" if (g[0] not in lexicon and h[0] in lexicon and len(g[0]) >= 3) else "misread"
    if g:
        return "dropped_struck (ignored)" if g[1] else "dropped_word"
    return "extra_struck (ignored)" if h[1] else "extra_word"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--extracted-dir", required=True)
    ap.add_argument("--examples", type=int, default=0, help="print up to N examples per category")
    ap.add_argument("--per-page", action="store_true")
    ap.add_argument("--include-drafts", action="store_true")
    args = ap.parse_args()

    lexicon = get_english_lexicon()
    stages = {
        "stage1": lambda d: d["stage1_transcription"]["raw_transcript"],
        "final": lambda d: d["stage2_verification"]["verified_transcript"],
    }
    totals = {s: collections.Counter() for s in stages}
    examples = {s: collections.defaultdict(list) for s in stages}
    active_words = 0
    for sd in sorted(p for p in Path(args.gt_dir).iterdir() if p.is_dir()):
        for txt in sorted(sd.glob("page_*.txt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0])):
            n = int(re.findall(r"\d+", txt.stem)[0])
            meta = sd / f"page_{n}.meta.json"
            if meta.exists() and not args.include_drafts:
                if str(json.load(open(meta)).get("status", "")).upper().startswith("DRAFT"):
                    continue
            ck = Path(args.extracted_dir) / sd.name / "checkpoints" / f"page_{n}.json"
            if not ck.exists():
                continue
            data = json.load(open(ck, encoding="utf-8"))
            gt = words(txt.read_text(encoding="utf-8"))
            active_words += sum(1 for _, s in gt if not s)
            row = []
            for stage, get in stages.items():
                c = collections.Counter()
                for g, h in align(gt, words(get(data))):
                    k = classify(g, h, lexicon)
                    if k:
                        c[k] += 1
                        examples[stage][k].append(f"{sd.name}p{n}: {g[0] if g else '-'} -> {h[0] if h else '-'}")
                totals[stage] += c
                row.append(c)
            if args.per_page:
                print(f"{sd.name} p{n}: " + " | ".join(f"{s}: {dict(c)}" for s, c in zip(stages, row)))

    print(f"\nactive ground-truth words: {active_words}")
    cats = sorted(set(totals["stage1"]) | set(totals["final"]))
    print(f"{'category':28s} {'stage1':>8s} {'final':>8s}")
    for k in cats:
        print(f"{k:28s} {totals['stage1'][k]:8d} {totals['final'][k]:8d}")
    if args.examples:
        for k in cats:
            print(f"\n## {k}\n  final: {examples['final'][k][:args.examples]}")


if __name__ == "__main__":
    main()
