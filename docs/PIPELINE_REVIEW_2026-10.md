# Pipeline review, October 2026

A teammate asked for five things once class-10 extraction and the class 6/7/8/10 evaluations were done:

1. audit `answer_segmenter.py`;
2. cut extraction runtime and bloat;
3. improve extraction accuracy;
4. standardise the rubrics per class;
5. make evaluation more reliable.

This document records what was found, what changed and why, the numbers, and how to reproduce them.

**Ground rules:**
- Teacher marks are treated as the truth only for objective questions (MCQ, cloze, tables, rearrangement).
- Writing questions are never tuned toward teachers.
- No per-word rules or hand-set thresholds are added.
- Every calibration choice was made on the dev split (`configs/splits/eval_split.csv`, by script, seeded).
- The held-out class-10 transcription pages (scripts 0029–0033) are scored once, at the end.

Work order: correctness bugs, then runtime, then accuracy, then rubric and evaluation calibration.

> Status (2026-10-17): every code change below is committed. The GPU measurements are still running:
> - the page benchmark and its noise floor;
> - the script-level before/after;
> - the band-stitching comparison on real band texts;
> - the two prompt ablations.
>
> Their numbers will be filled in under [Pending GPU results](#pending-gpu-results).

---

## Results at a glance

| Area | Before | After | How measured |
|---|---|---|---|
| Questions marked missing although the teacher gave marks (segmenter, 153 transcripts) | 22 | 9 | `scripts/audit_segmentation.py` |
| Missing questions overall / page-list jumps / empty page lists | 44 / 47 / 9 | 34 / 24 / 0 | same |
| MCQ MAE vs teacher, SE_10 Q1 (marks) | 3.18 | 0.73 | stored items re-scored offline |
| MCQ MAE vs teacher, SE_06 / SE_07 / SE_08 | 0.86 / 0.73 / 0.57 | 0.62 / 0.63 / 0.57 | same |
| Rearrangement MAE vs teacher (dev), SE_06 / SE_07 / SE_10 | 1.57 / 1.33 / 0.52 | 0.57 / 1.06 / 0.17 | same, per-paper convention |
| Stored class 6–8 summaries capped by a rule their rubric does not have | 61 | 0 | re-applying caps offline |
| Evaluation repeatable (same transcript, same segments) | no (LLM boundary layers) | yes (deterministic by default) | `segmentation_llm_recovery: false` |
| Rubric/key/paper consistency errors (5 papers) | 8 (class 11) | 0 | `scripts/lint_rubrics.py` |
| Unit tests | 310 pass, 8 fail | 348 pass | `pytest tests` |
| Extraction time per script | 20.5 min (Oct 13 run) | pending | per-stage timers |
| Dev-page CER | 4.30% (stored outputs) | pending | `scripts/evaluate_transcription.py` |

---

## 1. Segmenter audit (`src/pipeline/answer_segmenter.py`)

### What was wrong

Run on the stored transcripts with no model (`scripts/audit_segmentation.py`), the segmenter left 44 questions without an answer. For 22 of these the teacher had given marks. The causes:

- **An explicit header always won**, even when it named a question the student had already finished, or had jumped back to. Students re-write numbers, cross them out, or write "2. No Ans to the Qs No 9".
- **Switches based on content were thrown away.** A new answer that started at a page top without a number was kept in the previous question.
- **Gaps in the header grammar:**
  - markup inside headers;
  - `no:` written inline;
  - several numbers on one line;
  - table rows read as headers;
  - `4 (a)` when only `4` exists on the paper;
  - a hard-coded guess that `No. Z` means 7.
- **`reconcile_delayed_headers` could drop text** and gave it to the previous question in paper order rather than reading order.
- **Page lists drifted**, because pages were found again by matching 40-character substrings.
- **The deterministic recovery only ran when a model engine was passed.** Evaluation re-segmented with model-based layers, so two evaluations of the same transcript could give different answers. Totals moved by up to 29 marks.
- **Stage 3 errors** were matched to questions loosely, and on the raw-transcript path every error was attached to every question.

### What changed (commits d6727bf, 7995936, a5ceb8d, 8d00e22)

- **Parts carry their own page number**, so page lists cannot drift.
- **`classify_header()` records how each question was decided.** An explicit header that names a finished question, or jumps back, is checked against evidence from the paper:
  - question and passage terms, each weighted 1/k over the k questions that share it;
  - answer-key items;
  - the layout fingerprint.

  It is remapped only when an unanswered question fits *strictly better than every other question*. Mid-script switches without a header use the same test.
- **Evidence rules:**
  - Answer-key matches count only when a line *is* the item answer, which stops essays about a passage looking like the cloze question.
  - SE_10's key declares which reading passage serves which questions.
  - Crossed-out text is never evidence.
- **Header grammar fixed generically:**
  - markup is stripped;
  - `no:` is accepted inline;
  - when a line has several numbers, the last one wins;
  - table rows are never headers;
  - `4 (a)` reads as 4;
  - `(6) (a)` is split;
  - the `No. Z → 7` guess is removed.
- **Normalisation rewrites header lines only**, so answer text ("Ans: Due to …") reaches Stage 3/4 unchanged.
- **Delayed-header text is never dropped**: it goes to the question answered just before, in reading order. Inner-header recovery always runs. Gap filling needs content evidence and never empties a question.
- **The model-based boundary layers are off by default** (`pipeline.segmentation_llm_recovery: false`). Re-segmenting the same transcript now always gives the same answers. Turning the layers on is an ablation.
- **Stage 3 errors are attributed in this order:** page first, then context, then text of 4+ characters, then question number. Errors that match nothing are listed, not dropped.
- **Every run writes `segmentation.json`** next to the outputs: the parts with page and source, each remap and recovery with its evidence, missing questions and unattributed errors.
- **Stage 4 tells a blank answer from a question with no segment:** `blank_answer` vs `missing_no_segment`. The report says which.

### Numbers

`scripts/audit_segmentation.py`, 153 transcripts, no model:

| | before | after |
|---|---|---|
| missing questions | 44 | 34 |
| missing, teacher gave marks | 22 | 9 |
| header-only answers | 8 | 6 |
| page-list jumps | 47 | 24 |
| empty page lists | 9 | 0 |

The remaining 9 are listed per script in `outputs/benchmarks/segmentation_audit_after10_*.json`. Most are answers the student wrote with no number, in the middle of another answer.

---

## 2. Runtime

### Where the time went (Oct 13 run, before any change)

20.5 min per script: 71 s per page and about 442 model calls per script.

| Stage | Share | Why |
|---|---|---|
| Stage 2 (autocorrection audit) | ~31% | runs on every page: all 555 class-10 pages had at least one trigger |
| Stage 3 (error extraction) | ~26% | 5 of 11 segments are objective questions whose errors are never used |
| Stage 2b (line reconciliation) | ~15% | 95% of line re-reads find nothing |
| Stage 1 (transcription) | ~14% | 3 bands per page |
| Stage 3b | ~7% | |

The engine ran one request at a time and called `gc.collect + empty_cache + ipc_collect` after every call. Nothing was timed per stage.

### What changed

- **Timers first** (f186cbe):
  - every page checkpoint stores `token_usage.timing` per stage (seconds, calls, tokens);
  - `extraction_result.json` stores `metadata.stage_timing` for the run;
  - `scripts/report_stage_timing.py` sums them over folders.

  The Stage 1 token log now sums all bands; it reported only the last band.
- **No per-call CUDA cache clearing** (ed230aa). It now clears once per script, and before an out-of-memory retry.
- **Stage 3/3b is skipped for item-scored (Mode A) answers.** Their errors are never used: objective items get no language deductions. The mode comes from the rubric. In the stored runs, 18 Stage 3b normalisations fell on such answers, mostly rearrangement text where only the letter order counts.
- **Stage 3b with one consensus sample** reuses the localizer's read of the same crop (same prompt, temperature 0) instead of reading it again.
- **The evidence gate no longer re-runs red-ink detection on pre-cleaned PDFs.**
- **Removed** `scripts/audit_pipeline_bloat.py`: it read the deleted `gt.txt` and guessed call counts.

### Checked and not adopted

- **Stage 2b gating** (re-read a line only when the page transcript tags it `[struck:]` or OpenCV sees a stroke). Offline on the 555 stored pages, it would re-read 48% of lines but lose 62 of the 285 strike corrections Stage 2b adopted (22%). Not worth it. A better route, *not done* and offered as a proposal, is to send a page's line crops in one batched call.
- **Skipping the PDF re-render on resume:** 0.2 s per page, negligible.
- **Writing page checkpoints once:** negligible.

### Measured effect

Pending: see [Pending GPU results](#pending-gpu-results).

---

## 3. Extraction accuracy

### Band stitching (dca6324, b2e7003)

A page is read in three overlapping horizontal bands, and the band texts are joined. The old joiner compared whole lines from the end of one band with whole lines from the start of the next. The line at a band edge is usually cut: read partly, or marked `[truncated]`. When that happened the joiner found no match and **appended the lower band whole**, so the overlap lines appeared twice. 123 of the 555 stored class-10 pages contain a `[truncated]` marker. Student 0031's repeated lines are this bug.

The new joiner (`src/pipeline/band_tiler.py`):

1. **Search window:** it looks for the longest run of identical words between the last half of the upper band and the first half of the lower band. The tiler overlaps bands by a quarter to a third of their height, so the shared text always lies there.
2. **Seam:** it joins the bands at that run. Text before the run comes from the upper band and text after it from the lower band, so a cut line on either side falls away.
3. **Acceptance:** the run must contain a complete line of both bands and more than one word. A line inside the overlap is read in full by both bands, while a lone "Ans:" can match by chance.
4. **No match:** if no run qualifies, the bands are joined unchanged. Only edge lines that hold nothing but `[truncated]` are dropped.

Of 7 synthetic cases (`tests/test_band_stitching.py`), the old joiner fails 5.

Each band's raw text is now kept in the page checkpoint (`stage1_transcription.band_transcripts`). The old and new joiners can therefore be compared on exactly the same model output, with no generation noise.

### Ablation switches added

| Switch | Default | What it tests |
|---|---|---|
| `pipeline.prompt_question_vocab` | true | Stage 1/2 prompts list up to 250 words from the question paper. Does that push the model toward dictionary words (silent autocorrection)? |
| `pipeline.stage2_compact_output` | false | Stage 2 replies without the per-patch `confidence` and the page `verification_notes`: no decision reads them, and they cost about 10% of Stage 2's output tokens. `context_anchor` stays because it picks which copy of a repeated word gets patched. |

Both have flags in `scripts/bench_pages.py` (`--no-prompt-vocab`, `--stage2-compact`).

`stage2_conditional` is now described honestly in the YAML: every handwritten page triggers Stage 2, so it skips nothing in practice.

---

## 4. Rubric standardisation

### Rubric linter (e100a9b)

`python scripts/lint_rubrics.py` resolves, for each paper, the rubric, answer key, question paper and penalty profile that Stage 4 would use, and checks that they agree.

**Errors** (Stage 4 would mis-score):
- question sets or marks differ between paper and rubric;
- allocations that do not add up;
- criteria ceilings that do not add up to the question's mark;
- a band table that leaves a half mark uncovered or overlaps;
- a Mode A question without a key, or with the wrong number of key items;
- an unknown rearrangement convention.

**Warnings** (worth a human look):
- the key is not verified by a teacher;
- a Mode B question has no reference answers;
- the printed paper shows marks that differ from the parsed paper;
- task types are named differently in different files;
- a cap's name contradicts its threshold.

`tests/test_rubric_lint.py` runs it on all five papers.

**Errors found and fixed (class 11):**
- **Band tables had gaps at every half mark.** Rewritten so every half mark falls in exactly one band; every mark gets the same band as before.
- **Q1(A) allocation.** The rubric said 0.5 × 10, but the paper has 5 items worth 5 marks; the key already scored 1 × 5. The rubric now says 1 × 5.

### One rubric per paper, and refusal on mismatch

- **`english_writing.yaml` removed.** It was a copy of the class-11 rubric, and Stage 4 used it for *every* English paper when no `--rubric` was given (for example from `scripts/run_pipeline.py`). The stored evaluations were not affected, because `scripts/evaluate_scripts.py` resolves the rubric per paper.
- **Also removed:** `english_junior.yaml` (a copy of class 8, unused) and `english_primary.yaml` (legacy format, unusable by the rubric-driven scorer).
- **`rubric_mismatches()`:** Stage 4 and `evaluate_scripts.py` now **refuse** a rubric whose questions or marks differ from the question paper, instead of scoring one paper with another paper's rubric.
- **Reports name the rubric and answer key**, and say whether a teacher has verified the key.

### The guide's hand-set gates are off by default (19096d6)

Section 4 of `docs/NCTB_Standard_Evaluation_Rubric_Guide.md` adds three gates, derived from one script (SE_10_0001, teacher 7.5/15, model 14.5/15):

- **Error-density gating:** 3+ spelling errors cap language at 50%.
- **Story baseline:** a linear story is Band 3.
- **Verbatim-overlap summary cap:** ≥50% overlap caps the summary at 3.0/10. Classes 6/7 use 70% → 2.5/5 and class 8 uses 60% → 5/10.

These were calibrated toward a teacher's mark on writing questions, which this project does not do. As decided, they moved into a `calibration_gates:` block in each rubric. They apply only with `pipeline.stage4_calibration_gates: true` (`evaluate_scripts.py --calibration-gates`), as an ablation row.

**Caps that stay active:**
- the minimum-length caps and the off-topic cap, which state task requirements;
- class 11's own summary, theme and graph caps, which come from the HSC rubric, not the guide.

This also removes a contradiction: the class 6/7 rubrics said "without punitive verbatim caps" but had one.

### Grade-level expectations (756d032)

Each class rubric now states the guide's grade expectation (section 5) as a qualitative guideline. Class 6 and class 7 already had different band wordings from the teammate, but nothing about compound sentences and connectives at class 7, or turn-taking at class 8. No thresholds were added.

### Answer keys (800f5bc, 176bbf5)

- **Rearrangement keys** for SE_06 and SE_10 now include the sentences by letter, copied from the paper. The model can then map sentences a student rewrote in full back to letters.
- **Rearrangement convention per paper.** These are objective questions, so the teachers are the reference. The convention was chosen on the dev split and stored as `scoring:` in the key:

  | Paper | Positional MAE | Relative-order MAE | Chosen |
  |---|---|---|---|
  | SE_06 (14 dev scripts) | 0.57 | 1.57 | positional |
  | SE_07 (18 dev scripts*) | 1.06 | 1.33 | positional |
  | SE_10 (23 dev scripts) | 0.17 | 0.52 | positional |
  | SE_08 | — | — | waits for the key check below |

  \*4 SE_07 scripts were read as `a b c d e f g h` while the teacher gave 8: the students rewrote the sentences, and without a sentence map the model listed them in writing order. Both conventions get these wrong, so they are excluded. A sentence map for SE_07 is drafted for the teacher (see below).

  Under the order most SE_08 students wrote, relative order fits better (0.81 vs 1.29). That is decided once the key is confirmed. Keys without `scoring:` keep the old rule: the better of the two.
- **Draft reference answers** for the six Mode B questions that have none (SE_07 Q2/Q10, SE_08 Q2/Q10, SE_10 Q8/Q9) are in `configs/answer_keys/drafts/mode_b_drafts.yaml`. The pipeline does not read this file.
  - Passage questions are drafted from the paper.
  - Textbook questions are drafted only where the drafter was sure, and marked `textbook`.
  - The rest are marked TODO.

### Review list for the teammate / a teacher

1. **SE_08 Q9 rearrangement:** the key says `e b d h a c g f`. 20 of 35 students wrote `e b d h a f c g` and the teacher gave 8. Which is right?
2. **SE_07 Q10:** the printed paper says `2 x 5 = 10`. The parsed paper, the rubric and the 90-mark total use 5. Which is right?
3. **SE_07 Q8:** the rubric calls it Sentence_Making (Mode A); the key calls it Sentence_Matching. The paper text lists full questions as the "cues", so it may have been typed from the answers.
4. **SE_07 Q1c/Q1d and SE_10 Q1f:** the keys accept two options. Is that intended?
5. **SE_06 Q5:** check the wording of the accepted answers.
6. **SE_07 Q9:** confirm the sentence-to-letter map in the drafts file.
7. **Check the drafted Mode B answers**, then move them into the keys (`verified: true` covers them).
8. **SE_10 key:** add `verified: true` once checked. The key also has no `question_id`; it uses `paper_id`.
9. **Writing questions (for the expert):** in SE_10 the teacher never gave more than 8.5/15 on Q10 (story) or Q11 (dialogue). Model means are 8.5 and 9.6 against teacher means of 6.8. Is the teacher's scale compressed, or is the model lenient? This is not tuned either way.

---

## 5. Evaluation reliability (Stage 4)

Most of the error against teachers on objective questions came from bugs, not from model judgement. The fixes:

| Fix | Commit | Effect |
|---|---|---|
| **MCQ "multiple selection" counted any stray letter.** The article "a", or "i, ii & iii" inside the option text, made a correct answer score 0. Now only option labels count. | 64ec15b | 106 of 114 flagged items were single choices; SE_10 Q1 MAE 3.18 → 0.73 |
| **The last question of a paper lost its prompt text** (it was graded against "Question 11: Dialogue Writing") | 1882203 | SE_06/07 Q13, SE_10 Q11, SE_11 Q11 |
| **Hard caps come from the rubric.** The hard-coded "50% if over 1/3 of the source" rule is gone. | 34a8f0f | it capped 61 class 6–8 summaries whose rubrics have no such cap |
| **No invented marks.** Narrative band recovery, the 50% defaults and the monolithic 5.0 fallback are removed; unusable output is `unscored` and listed for review. | 34a8f0f | |
| **Penalties by mode:** additive language deductions only in Mode B. Mode C already judges language in `language_mechanics`, so deducting again counted it twice. | 34a8f0f | |
| **Half marks round up** (Python sent 7.25 to 7.0 but 7.75 to 8.0) | d226c18 | |
| **"Answer any N" scores the first N attempted items** (SE_06 Q10 summed all 8 answers) | d226c18 | 4 dev scripts: first-5 MAE 1.50 vs best-5 2.00, too few to be conclusive |
| **Mode B without a key** gets "no reference answers, judge against the passage" instead of a stray flow-chart instruction | d226c18 | |
| **SE_08 class test** is left out of the paper maximum (scored out of 90) | 30d8d8f | |
| **Feedback cannot contradict the score:** the model does not state a total; the report shows criteria total → cap → awarded | 2ad8217 | |
| **Provenance:** each evaluation stores the git commit, config hash and the hashes of the rubric, key, penalty profile and scoring code | 30d8d8f | two evaluations can be told apart |
| **Rearrangement convention per paper** | 176bbf5 | see section 4 |
| **One rubric per paper, refusal on mismatch** | e100a9b | see section 4 |

### Writing questions: agreement with teachers, descriptive only

These numbers describe the stored evaluations; they are not targets. Stored summary marks were produced under the old hard-coded caps, so the summary rows will change on re-evaluation.

| Paper | Q | Task (max) | n | Teacher mean (max) | Model mean (max) | MAE | r |
|---|---|---|---|---|---|---|---|
| SE_06 | 7 | Summary (5) | 21 | 2.95 (5) | 2.43 (3.5) | 0.90 | −0.22 |
| SE_06 | 11 | Story (10) | 21 | 5.48 (9) | 5.62 (7.5) | 1.10 | 0.79 |
| SE_06 | 12 | Paragraph (10) | 21 | 5.67 (9) | 5.52 (8.5) | 1.14 | 0.66 |
| SE_06 | 13 | Dialogue (10) | 19 | 6.11 (9) | 6.11 (8.5) | 0.74 | 0.88 |
| SE_07 | 7 | Summary (5) | 34 | 3.76 (4) | 2.41 (3) | 1.35 | 0.17 |
| SE_07 | 11 | Story (10) | 33 | 7.62 (9) | 6.55 (9.5) | 1.29 | 0.71 |
| SE_07 | 12 | Paragraph (10) | 34 | 7.56 (9) | 6.18 (10) | 1.62 | 0.62 |
| SE_07 | 13 | Dialogue (10) | 34 | 7.75 (9) | 6.71 (9.5) | 1.22 | 0.52 |
| SE_08 | 7 | Summary (10) | 35 | 7.09 (9) | 4.76 (8) | 2.61 | −0.03 |
| SE_08 | 11 | Story (7) | 35 | 5.23 (7) | 4.63 (6.5) | 1.03 | 0.47 |
| SE_08 | 12 | Paragraph (7) | 35 | 4.74 (6) | 4.36 (7) | 1.01 | 0.51 |
| SE_08 | 13 | Dialogue (6) | 33 | 4.79 (6) | 3.70 (6) | 1.18 | 0.42 |
| SE_10 | 5 | Summary (10) | 32 | 6.27 (8) | 6.77 (9.5) | 0.94 | 0.66 |
| SE_10 | 10 | Story (15) | 32 | 6.83 (8.5) | 8.52 (12.5) | 2.22 | 0.60 |
| SE_10 | 11 | Dialogue (15) | 33 | 6.83 (8.5) | 9.64 (14.5) | 2.86 | 0.65 |

Stories, paragraphs and dialogues rank students much as the teachers do (r 0.42–0.88). Summaries do not, which makes them the question type to show the expert first.

---

## Switches and ablation rows

All of these live in `configs/pipeline_config.yaml` under `pipeline:`. The default is the thesis setting.

| Switch | Default | Turn on/off to test |
|---|---|---|
| `segmentation_llm_recovery` | false | model-based boundary layers on top of the deterministic segmenter |
| `prompt_question_vocab` | true | question-paper word list in the Stage 1/2 prompts |
| `stage2_compact_output` | false | shorter Stage 2 replies |
| `stage4_calibration_gates` | false | the guide's hand-set gates (error density, story baseline, verbatim summary cap) |
| `stage4_thinking_mode` | false | thinking mode in Stage 4 (`evaluate_scripts.py --thinking`) |

---

## How to reproduce

```bash
PY=./script_checking/bin/python

# unit tests
$PY -m pytest tests -q

# segmenter audit (CPU, no model); compare with the stored baseline JSON
$PY scripts/audit_segmentation.py

# rubric / key / paper consistency
$PY scripts/lint_rubrics.py

# Stage 4 agreement with teacher marks, per paper / mode / question
$PY scripts/benchmark_evaluation.py

# page benchmark (GPU): re-run Stages 0-2b on the 56 dev ground-truth pages only, with timing and CER
$PY scripts/bench_pages.py --tag after
$PY scripts/bench_pages.py --tag no_vocab --no-prompt-vocab
$PY scripts/bench_pages.py --tag s2_compact --stage2-compact

# per-stage time summed over extraction folders
$PY scripts/report_stage_timing.py outputs/extracted/english/se_10_q1
```

Run the page benchmark twice with the same settings: the difference between the two runs is the noise floor. A CER change smaller than that is not claimed.

---

## Pending GPU results

The queue (each job starts only when no process is using the GPU):

1. `baseline_a`: page benchmark on the code after the runtime changes, before the new stitcher.
2. `baseline_b`: page benchmark on the code before the runtime changes. A vs B gives the CER noise floor and the cache-clearing effect.
3. `after`: page benchmark with the new stitcher, which also saves the band texts. It is compared with A/B, and offline with the old stitcher on the same band texts.
4. `sl_before` / `sl_after`: script-level stages (segmentation, Stage 3, 3b) on 5 dev scripts, before and after the runtime changes. It also checks that Stage 3 errors of the non-Mode-A answers and the 3b verdicts are unchanged.
5. `no_vocab`, `s2_compact`: the two prompt ablations, compared with `after`.

After that:
- the full re-evaluation of the 123 scripts (dev and test splits reported separately);
- a repeatability check (10 scripts evaluated twice must give identical totals);
- the one-time score of the held-out class-10 test pages.

---

## Proposals not done (need a decision)

- **Language bug in `orchestrator.py`.** Stage 4 reads `extraction.metadata["paper"]` as the language. Digitized imports store the paper ID there ("SE_08_Q1"), so without an explicit `question_input` the wrong question paper can be loaded. `evaluate_scripts.py` always passes it, so stored runs are unaffected. A small fix is ready to propose.
- **Batch Stage 2b's line reads** into one call per page instead of one call per line.
- **Before pushing:** the commits on `gemma` are authored with the git account currently set on this PC. Re-author them before pushing if that account is going to change.
