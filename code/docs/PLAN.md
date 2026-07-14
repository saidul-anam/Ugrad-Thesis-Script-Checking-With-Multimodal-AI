# Plan — Exam-script grading benchmark

Goal: benchmark automated (Gemini) grading against the manual marks, and measure
where and why it diverges. Track progress with the checkboxes below.

> **Status (2026-07-14):** all code is built and the offline stages (CV + all
> deterministic logic) are run and validated on the full 548-row set. The two
> Gemini stages — red-mark harvest and grading — are implemented, cached, and
> ready but **not yet executed**: they need a Vertex AI API key
> (`export GEMINI_API_KEY=...` then `python3 -m src.pipeline`). 14 unit tests pass.

---

## 1. Ground truth (two sources that check each other)

- [x] Extract the red mark on each image as the authoritative score — Gemini call
      reads **only** the red score → `src/harvester.py` (runs on the ORIGINAL image).
      *Built; runs once the API key is set.*
- [x] Parse the `marks` column as the cross-check → `src/ground_truth.py`.
- [x] Compare the two per row:
  - [x] Agree → high-confidence **gold label**.
  - [x] Disagree → flag for a quick human look.
- [x] Expect disagreements to cluster on the half-mark rows (corrupt `5/2`-style tokens).
      Cross-check tests every plausible reading of `a/b` ({a, a/b}) against the red mark.
      Result: a clean gold set **plus a measured error bar** (FLAG share).

## 2. Full marks (denominator)  → `src/fullmarks.py`

- [x] Parse full marks from the **rubric text** (sum of per-clause `X নম্বর` allocations,
      Bengali digits). Covers 306 rows.
- [x] Fallback chain: marks-column denominator **only when `a<=b`** (a proper "out of";
      the corrupt improper `5/2` is refused) → marks stated in the question → the max mark
      the harvester reads off the page. 418/548 resolved offline; the rest fill in at harvest.

## 3. Question-text normalizer (route per row)  → `src/normalize_bn.py`

- [x] `questionEN` present → English-typed-in-Bijoy case → **use `questionEN`** (247 rows).
- [x] Genuinely Bengali → **Unicode (NFC) normalization only** (294 rows).
- [x] Bengali-only **and** Bijoy-garbled → **flagged `BN_GARBLED`** so the grader reads the
      printed question off the image (7 rows). *Plan modification:* rather than a blind
      reverse-Bijoy decode (which risks silently mangling real Bengali, and is moot because
      the grader sees the image anyway), garbled rows defer to image OCR — honoring the
      "never blind-convert" invariant.
- [x] Validated on a sample: the 7 garbled rows are genuinely garbled; BN_UNICODE rows read
      as clean Bengali.

## 4. Grading (defeats contamination)  → `src/grader.py`

- [x] Erase the red ink from the image → `src/redink.py` (HSV red mask + inpaint).
      **Validated by hand on ~20 images**, then run on all 548.
- [x] Send Gemini the **erased image** + normalized question + solution + rubric.
- [x] Structured JSON back: score · per-criterion breakdown · transcription · confidence.
- [x] Cache every raw response keyed by `id` (+stage+variant) → `data/cache/`.
- [x] Transcription field returned first, so a **misread** is separable from a **misgrade**.

## 5. Analysis (both outputs)  → `src/analyze.py`

- [x] Per-subject metrics: exact agreement · within ±0.5 · MAE normalized by full marks.
- [x] Ranked disagreement table (Gemini reasoning + transcription next to the gold mark)
      → `outputs/disagreements_<variant>.csv`.

## Build in from the start

- [x] Grading runs under **two prompt conditions** (`rubric` vs `solution`-only) →
      `src/grader.py::VARIANTS`.
- [x] **Gold-label error rate** from the cross-check is always reported →
      `analyze.gold_label_error_rate`.

---

## Build order

1. [x] Red-erasure + red-mark extraction — **validated on ~20 images by hand**, then all 548.
2. [x] Bengali normalizer.
3. [x] Grader.  *(built + cached; executes on API key)*
4. [x] Analysis.  *(code complete; produces numbers once grades exist)*

## Remaining to produce the actual benchmark numbers

- [ ] Set `GEMINI_API_KEY` and run `python3 -m src.pipeline` (harvest + grade all rows).
- [ ] Eyeball a handful of harvested red scores against the images (harvester accuracy).
- [ ] Review the FLAG rows (the gold-label error bar) by hand.
