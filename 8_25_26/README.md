# 8_25_26 — Extraction + Two-Model Evaluation Pipeline

Verbatim transcription of handwritten Class XI English exam scripts, followed by
independent grading with two models, for a divergence study.

## Stages
1. **PDF → images** (`scripts/01_convert.py`, `src/pdf_to_images.py`) — 300 DPI PNGs.
2. **Extraction** (`scripts/02_extract.py`, `src/extract.py`) — verbatim JSON
   transcripts via a vision model; fidelity-preserving (no spell-check/normalise).
3. **Extraction CSV** (`scripts/04_build_csv.py`, `src/build_csv.py`) — answer-grain
   `data/extraction.csv` with computed metrics.
4. **Evaluation** (`scripts/eval/01_evaluate.py`, `src/eval/`) — two graders,
   k runs each, identical prompts; per-criterion scores, reasoning, and verbatim
   evidence spans.
5. **Eval CSVs** (`scripts/eval/02_emit_csv.py`, `04_emit_avg.py`) — per-run and
   per-(task×model) averaged views.
6. **Validation** (`scripts/eval/03_validate.py`) — assertions + within-model
   noise floor + evidence-not-found rate + disagreement report.

## Config
`config.yaml` holds all model configuration (provider, model IDs, thinking level,
k, concurrency, cost rates, paths). API keys are read from environment / a local
`.env` (never committed).

## Current dataset
20 of 24 scripts extracted and evaluated (k=3, two graders).
- `data/transcripts/` — per-script verbatim transcripts (canonical).
- `data/extraction.csv` — answer-grain extraction (120 rows).
- `output/evaluation.csv` — per-run evaluations (720 rows).
- `output/evaluation_avg.csv` — averaged per (task × model) (240 rows).
- `output/evaluations/` — one JSON record per (task × model × run).
- `logs/evaluation_report.md` — validation + noise/evidence/disagreement report.

## Run
```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python scripts/01_convert.py --limit N
python scripts/02_extract.py --limit N
python scripts/04_build_csv.py
python scripts/eval/01_evaluate.py --limit N --k 3
python scripts/eval/02_emit_csv.py && python scripts/eval/04_emit_avg.py
python scripts/eval/03_validate.py
```

> Note: `data/images/` (300-DPI renders, ~0.8 GB) is regenerable from the PDFs
> and is not tracked.
