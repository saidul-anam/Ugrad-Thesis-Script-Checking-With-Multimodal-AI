# Implementation Notes — Extraction-First Rebuild (2026-09-12)

This file records **what was implemented**, **why**, and **how to use it**. It is the user-facing
instruction sheet for the measurement harness (Phase 0) and the new Stage 3b evidence gate (Phase 1).
Plan of record: `~/.claude/plans/the-main-issue-we-adaptive-wozniak.md`.

---

## 1. Problem being solved

Stage 3 flagged perceptual misreads of one writer's letter shapes as student spelling errors:
`powerdul`→powerful, `electricidty`→electricity, `illustrodes`→illustrates, `thouths`→thoughts.
On `SE_11_Q1_0002` the student writes every `t` with a looped ascender that reads as `d`
(`the`→"dhe", `that`→"dhad", `chart`→"chard"). The same confusion also produced errors that Stage 3
labelled *grammar* (`want do`→want to, `Do the`→to the, `he fed thad`→he felt that, `stanled`, `his heard`),
which the old gate never even looked at.

Why the old Stage 3b could not fix it:
- it listed the four target words as examples inside the prompt (label leakage, a per-word rule);
- it sent the whole page and asked the VLM to "inspect the ink" — with no localization the model
  answers from its language prior;
- it only gated errors whose type contained "spell";
- it always used the **first page** of a multi-page answer;
- Stage 1's `[unclear: …]` tags never fired (0 tags in 19 pages), so verbalized uncertainty is useless.

**User constraint honoured: no hand-written letter rules.** Nothing in the new gate says which letters
look alike. The writer's confusions are learned from their own script, ambiguity is measured from the
ink, and the fusion weights are fitted on human labels.

---

## 2. What was implemented

### Phase 0 — Measurement harness (transcription accuracy)
| File | Purpose |
| --- | --- |
| `src/utils/text_metrics.py` | `levenshtein`, `cer`, `wer`, `normalize_transcript` (tags, page breaks, NFC), `score_transcription` incl. the **silent-correction probe** (student non-words in the reference that the hypothesis "fixed") |
| `scripts/make_transcription_gt.py` | builds `data/ground_truth/transcripts/<lang>/<script>/page_<n>.txt` pre-filled with the Stage 2 transcript + page PNG + meta; you correct the text |
| `scripts/evaluate_transcription.py` | scores Stage 1 and Stage 1+2 per page → `outputs/benchmarks/transcription_*.json/.md` (CER/WER macro+micro, silent-correction rate) |
| `tests/test_text_metrics.py` | unit tests |

### Phase 1 — Stage 3b evidence gate (`src/pipeline/arbitration/`)
| Module | Signal / role | Hand-written knowledge? |
| --- | --- | --- |
| `candidate_selector.py` | gates **any** Stage 3 error (spelling, grammar, syntax) whose read/intended token pairs differ by ≤2 character edits (`want do`→`want to` yields the pair `do/to`) | none |
| `symbolic_evidence.py` | character edit path (used only to *name* the discrepancy, e.g. `sub:f>d`) and `phonetic_plausibility` (Metaphone similarity: genuine learner errors sound like the target, misreads often don't) | generic phonetics only, weight is fitted |
| `writer_profile.py` | **learned per script**: counts which substitutions this writer's ink produces, from (a) disagreements between augmented re-reads, (b) non-words one edit from the writer's *own* most frequent dictionary words (`dhe`≈`the` ⇒ `t>d`), (c) gated candidates (guarded by `min_count`). Persisted as `writer_profile.json` | none |
| `localizer.py` | finds the physical line: VLM bounding box (validated by re-reading the crop and fuzzy-matching the context) with an OpenCV projection-profile fallback (Otsu + opening + valley splitting at the page's own line pitch). Also `attribute_error_to_page` (fixes the first-page-only bug) | none |
| `consensus.py` | re-reads the line crop N times under scale / rotation / gamma / contrast / padding perturbations + one sampled read, aligns them with progressive Needleman-Wunsch, votes per character; measures agreement for the read vs the intended token | none |
| `forced_choice.py` | one VLM call on the crop: "does the ink read A or B (or NEITHER)?", options in seeded random order, no example words | none |
| `fusion.py` | logistic fusion → `ambiguity_score` → `GENUINE_ERROR` / `HANDWRITING_AMBIGUITY` / `UNCERTAIN`; `fit_weights` (numpy logistic regression) and `choose_thresholds` from labels | weights fitted from labels (provisional defaults until then) |
| `gate.py` | `EvidenceArbitrationGate.arbitrate(errors, q_no)` orchestrates the above per question; `finalize()` writes `stage3b_arbitration.json` + `writer_profile.json` + `arbitration_crops/` | — |

Integration and supporting changes:
- `src/prompts/stage3b_arbitration.py` (new prompts, no example words), `src/prompts/stage3_arbitration.py`
  (legacy prompt de-leaked), `src/prompts/stage2_verification.py` (Direction B de-leaked).
- `src/core/schemas.py`: `ArbitrationCandidate`, `ArbitrationEvidence`, `ArbitrationRecord`, `Stage3bArbitrationResult`.
- `src/core/config.py` + `configs/pipeline_config.yaml`: `arbitration:` block (`mode: evidence|legacy|off`, thresholds, weights…).
- `src/pipeline/stage3_error_analyzer.py`: `arbitrate_visual_errors(..., gate=, q_no=)`; old behaviour kept as `_arbitrate_legacy`.
- `src/pipeline/orchestrator.py`: builds the gate once per script; gates every question's errors (any type);
  normalizes cleared tokens **inside the error's own context** in both the answer text and the per-page
  transcripts (so exports and Stage 4 see the same text; a cleared `do`→`to` no longer rewrites every other `do`);
  fallback branch handled; `metadata["arbitration"]` summary; per-candidate token usage accounted.
- `src/utils/export_utils.py`: `extraction_summary.md` gains "Stage 3b Handwriting Ambiguity Arbitration"
  with the benefit-of-doubt table and a **"Flagged for Human Review (UNCERTAIN)"** table.
- `src/engine/mock_engine.py`: mock branches for the crop / bbox / forced-choice prompts.
- `scripts/label_arbitration_candidates.py`: labelling sheet (`data/labels/arbitration_labels.csv`) stratified
  per script + contact-sheet PNGs of the crops.
- `scripts/evaluate_arbitration.py`: precision/recall/F1, ECE, per-signal ablation, **leave-one-script-out**,
  `--fit` prints (or `--write-config` writes) fitted weights/thresholds.
- Tests: `tests/test_arbitration_*.py` (6 files) + assertions in `tests/test_pipeline_mock.py`.

Later fixes after the first real run (same day):
- A writer prior below `writer_profile_min_count` is now **neutral (None)**, not 0: absence of evidence
  was being read as evidence of a genuine error.
- A candidate whose line crop cannot be located is always `UNCERTAIN` (never penalised, never cleared).
- **Lexicon scan** (`arbitration.lexicon_scan`): Stage 3 is an LLM and does not flag the same tokens on
  every run (the second run of 0002 missed `powerdul`, `illustrodes`, `electricidty`, `thouths`). Every
  alphabetic token of a subjective answer that is not a dictionary word, not question vocabulary and not
  a likely proper noun is now gated too, with the closest dictionary word as the intended reading. If the
  ink says it is genuine it is appended as a spelling error; if ambiguous it is normalized. This is a
  dictionary lookup, not a letter rule.
- Writer-profile counts from consensus disagreements ignore punctuation/digit columns.
- Default weights changed to phonetic 0.6 / forced_choice 2.4 (still provisional until fitted).

### Phase 1b — Stage 4 rubric-driven scoring (audit fixes)
Audit findings that prompted this: the English rubric YAML was never applied (both modular and monolithic
paths only read `subject`/`penalties`/`criteria`, none of which the English rubric has); objective
questions (30 marks) were graded without an answer key; malformed model output silently became 60 % /
50 % marks; marks were not snapped to 0.5; questions the segmenter missed vanished from the MAE; the
benchmark was per script only.

| File | Purpose |
| --- | --- |
| `src/pipeline/stage4_modes.py` | rubric spec loader (`load_rubric_specs`), answer-key loader, prompt builders and **code-side scorers** for Mode A (item vs key, rearrangement position-scored), Mode B (discrete scale clamp), Mode C (criteria ceilings → raw total → hard caps with computed verbatim overlap → 0.5 snap → band) |
| `configs/answer_keys/SE_11_Q1.yaml` | answer key for the English paper: MCQ, both cloze tests, rearrangement sequence, short-answer key points, flow-chart sequence. **`verified: false` — check it once** |
| `src/pipeline/stage4_evaluator.py` | `evaluate_modular` dispatches to the modes when the rubric is mode-based (else the old generic prompt); one strict JSON retry, then `scoring_status: unscored`; questions expected by rubric/GT but absent from segmentation become `scoring_status: missing`; `mae_vs_human` (scored only) and `mae_including_missing` |
| `src/core/schemas.py` | `QuestionEvaluationItem`: task_mode, task_type, scoring_status, items, subscores, raw_total, cap_*, performance_band, key_source; `Stage4EvaluationResult`: mae_including_missing, missing_questions, unscored_questions, rubric_driven |
| `src/utils/export_utils.py` | evaluation report shows mode, status, raw, cap, band per question, plus item-level and criterion-level tables |
| `scripts/benchmark_evaluation.py` | cross-script benchmark vs `gt.txt`: MAE (scored / incl. missing), per mode, per question, Pearson, quadratic weighted kappa, leave-one-script-out median baseline, bootstrap CI, `--compare` for A/B ablations |
| `tests/test_stage4_modes.py` | 9 unit tests incl. mock-engine end-to-end |
| `src/pipeline/answer_segmenter.py` | first real run exposed two segmentation faults: (1) headers like `Ans to the question no - 2` / `No - 01(B)` were not split (space after "No"), losing 1(B) and 2 on script 0010; (2) topic-keyword matching ran *before* the explicit numeric header and matched stop words, so the theme answer of 0002 (`Theme: The poem … with …`) was filed under Q3 and Q11 kept only its header. Explicit headers now take precedence; keyword matching picks the best sub-question and ignores stop words |
| `scripts/resegment_extraction.py` | re-run the segmenter on existing extraction artifacts without re-running the VLM stages (use after any segmenter fix, then re-evaluate) |

### Deliberately NOT implemented
- No confusion table / letter rules of any kind (user constraint).
- Bangla runs, exemplar-anchored Mode C prompts, self-consistency sampling, more MAE ground truth
  (deferred; see plan Phase 2 / out of scope).

---

## 3. How to use it (user instructions)

### 3.1 Run extraction with the gate (default `mode: evidence`)
```bash
cd /mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI
python scripts/extract_scripts.py --lang english --script-name SE_11_Q1_0002 --local-only --non-interactive
# add --force-extract to redo Stages 1/2 as well (otherwise page checkpoints are reused; Stage 3 + 3b always run)
```
Outputs per script in `outputs/extracted/english/<script>/`:
- `stage3b_arbitration.json` — every candidate with all raw signals, score, verdict, crop path
- `writer_profile.json` — the learned confusion counts for this writer
- `arbitration_crops/*.png` — the line crops that were judged
- `extraction_summary.md` — includes the benefit-of-doubt and human-review tables

Ablation: set `arbitration.mode: legacy` (old single-call gate) or `off` in `configs/pipeline_config.yaml`
and re-run; diff `stage3_errors.json`.

### 3.2 Label ~100 crops and fit the weights (turns the gate from provisional into calibrated)
```bash
python scripts/label_arbitration_candidates.py --lang english --per-script 20
#   -> data/labels/arbitration_labels.csv  (fill human_label: G genuine / A ambiguity / U cannot tell)
#   -> data/labels/contact_sheets/<script>.png  (look at the crops)
python scripts/evaluate_arbitration.py --labels data/labels/arbitration_labels.csv            # metrics + ablation + leave-one-script-out
python scripts/evaluate_arbitration.py --labels data/labels/arbitration_labels.csv --fit      # prints YAML weights/thresholds
python scripts/evaluate_arbitration.py --labels data/labels/arbitration_labels.csv --fit --write-config   # writes them into the config
```
Label candidates from **all** scripts, not just 0002, so the fitted weights generalize across writers.

### 3.3 Transcription accuracy (CER / WER / silent-correction rate)
```bash
python scripts/make_transcription_gt.py --lang english --max-pages 20
#   correct data/ground_truth/transcripts/english/<script>/page_<n>.txt to the exact handwriting
#   (keep student mistakes! use [illegible] / [struck: …]); then set "status": "CORRECTED" in page_<n>.meta.json
python scripts/evaluate_transcription.py --lang english
#   -> outputs/benchmarks/transcription_english_<timestamp>.md  (Stage 1 vs Stage 1+2)
```
Use `--extracted-dir <other run> --tag <name>` to compare ablations (e.g. Stage 2 off, inpainting off).

### 3.4 Grade and benchmark (Stage 4, rubric-driven)
```bash
# 0. check the answer key once and set verified: true
#    configs/answer_keys/SE_11_Q1.yaml
# 1. grade every extracted English script (modular = rubric-driven modes A/B/C)
python scripts/evaluate_scripts.py --lang english --non-interactive --force-evaluate --eval-mode modular
#    -> outputs/evaluated/english/<script>/stage4_evaluation.json + evaluation_report.md
# 2. benchmark against gt.txt across scripts
python scripts/benchmark_evaluation.py --lang english --tag rubric_v1
#    -> outputs/benchmarks/evaluation_english_rubric_v1_<stamp>.md
# 3. ablations: grade into another dir and compare
python scripts/evaluate_scripts.py --lang english --non-interactive --force-evaluate --eval-mode monolithic --output-dir outputs/evaluated_monolithic/english
python scripts/benchmark_evaluation.py --lang english --compare outputs/evaluated_monolithic/english --tag modular_vs_monolithic
```
Ground truth: add every verified script to `gt.txt` in the existing `SCRIPT_ID` / `q-mark` format; the
benchmark picks up whatever is there. Ten scripts × twelve questions is the target for a credible MAE.

### 3.5 Tests
```bash
python -m pytest tests -q
```

---

## 4. Reading the verdicts
| Verdict | Effect on grading | Effect on transcript |
| --- | --- | --- |
| `GENUINE_ERROR` | deduction kept | unchanged |
| `HANDWRITING_AMBIGUITY` | deduction dropped (benefit of the doubt) | token normalized to the intended word inside that sentence |
| `UNCERTAIN` | deduction dropped, listed under "Flagged for Human Review" | unchanged |

A candidate for which no crop could be located is never cleared outright unless the writer profile
already supports the confusion; it becomes `UNCERTAIN`.

---

## 5. Results of the first real runs (2026-09-13, Gemma 4 31B 4-bit, RTX 5090)

### Stage 3b gate on SE_11_Q1_0002 (second run, after the fixes)
- 24 candidates gated, 178 crop-level calls, ~8 min for Stage 3 + 3b with Stages 1/2 resumed from checkpoints.
- Cleared as handwriting ambiguity: `illustrodes`→illustrates (score 0.96), `do`→to (0.90), `cand`→and (0.94), `almighly`→almighty (first run).
- Kept as genuine: `familyes`, `fullfill`, `accroding`, `stanled`, `become`→became and the other agreement errors.
- `powerdul`, `electricidty` were caught only by the lexicon scan (Stage 3 did not flag them this run) and ended
  `UNCERTAIN` because the line could not be localized on the bleed-through page 13 (no deduction, listed for review).
  The projection fallback now searches more lines on such pages; not yet re-run.
- Known weakness: the intended word for lexicon-scan candidates is the nearest dictionary word without context
  (`cand` → "and" instead of "sand"). Labels will show how often this matters.
- Stage 3 itself is non-deterministic across runs (53 → 31 → 30 errors on the same transcript); the lexicon scan
  is what makes the gate's coverage stable.

### Stage 4 rubric-driven grading, 2 scripts with ground truth (22 questions)
| Metric | rubric_v1 (before segmenter fix) | rubric_v2 |
| --- | --- | --- |
| MAE, scored questions | 1.43 (n=20) | **1.02** (n=22) |
| MAE incl. missing/unscored as 0 | 2.07 | 1.02 |
| Baseline: leave-one-script-out median | 1.72 | 1.60 |
| Pearson r | 0.70 | 0.86 |
| Quadratic weighted kappa (0.5-mark bins) | 0.64 | 0.84 |
| Within 1 mark | 60 % | 73 % |
| Missing segments | 4 (2 real, 2 segmenter) | 2 (both genuinely unattempted) |
| Hard caps applied | 0 | 1 (Q3 summary verbatim/length) |

Per mode (v2): A_ITEM MAE 1.31 (n=8), B_POINT 1.00 (n=4), C_BAND 0.80 (n=10). Script totals: 0002 AI 63.0 vs
human 70.0; 0010 AI 42.5 vs human 46.0 (same questions). The AI is systematically slightly stricter on the
objective items (bias −1.0 to −2.0 on Q2, Q5, Q6) and more generous on the story (Q9, +1.25).

Caveats: two scripts is far too few for any claim; the bootstrap CI is meaningless at n=2; the answer key is
unverified; Q5 on 0010 scores 0 because the student rewrote the cloze passage and the extraction did not
align the gaps (prompt now carries cloze guidance, not yet re-run); Q6 is position-scored strictly as the rubric
says while the examiner evidently gave partial credit.

### What to do next (user)
1. Verify `configs/answer_keys/SE_11_Q1.yaml` (10 minutes) and set `verified: true`.
2. Add marks for the other 8 English scripts to `gt.txt`, run extraction + evaluation + benchmark on all 10.
3. Label ~100 crops (`scripts/label_arbitration_candidates.py`) and fit the gate weights.
4. Optionally correct ~15 transcript pages for the CER table.
