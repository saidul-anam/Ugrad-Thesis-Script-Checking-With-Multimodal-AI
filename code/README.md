# Exam-script grading benchmark

Benchmarks Gemini (`gemini-2.5-flash`) grading against teachers' manual marks on
548 Bangladeshi exam answer scripts (Biology, English, Bangla, ICT, General
Knowledge), and measures where and why the two diverge.

See `CLAUDE.md` for invariants and `docs/PLAN.md` for the full plan + status.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install google-genai opencv-python-headless numpy pillow pandas pytest

# Offline stages only — no API key needed (erases red ink, builds ground-truth
# scaffolding, resolves full marks, routes question text):
python3 -m src.pipeline --no-llm

# Full run (needs a Vertex AI key):
export GEMINI_API_KEY=...        # or API_KEY, or put it in .env
python3 -m src.pipeline          # add --subject Biology --limit 20 to sample

python3 -m pytest tests/ -q
```

## How it works

| Stage | Module | Notes |
|------|--------|-------|
| Red-ink erasure | `src/redink.py` | HSV red mask + inpaint. The grader only ever sees the erased image. |
| Red-mark harvest | `src/harvester.py` | Gemini reads **only** the red score, on the **original** image → authoritative ground truth. |
| Cross-check | `src/ground_truth.py` | Red mark vs every reading of the `marks` `a/b` token → GOLD / FLAG. FLAG share = gold-label error bar. |
| Full marks | `src/fullmarks.py` | rubric total → proper `a/b` denominator → question-stated → max mark on page. |
| Question router | `src/normalize_bn.py` | EN present → use EN; genuine Bengali → NFC; Bijoy-garbled → defer to image. |
| Grader | `src/grader.py` | Erased image + question + solution (+rubric). Two conditions: `rubric`, `solution`. Structured JSON. |
| Analysis | `src/analyze.py` | Per-subject exact / ±0.5 / normalized-MAE; ranked disagreement table. |

Raw LLM responses are cached under `data/cache/` keyed by `id`, so reruns and
reprompts cost nothing.

## Key data facts

- `marks` is empty or `a/b`; `5/2` appears in 170 rows and is **corrupt** (means
  different scores in different rows) — hence the red mark, not the column, is ground truth.
- Some scripts are **blank** (student wrote nothing); red fraction ≈ 0.
- `questionBN` is often legacy-Bijoy garble; `questionEN` is the reliable source when present.
