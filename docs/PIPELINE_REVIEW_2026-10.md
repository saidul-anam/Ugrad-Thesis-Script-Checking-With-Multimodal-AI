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
| Unit tests | 310 pass, 8 fail | 351 pass | `pytest tests` |
| Page stages (0–2b), seconds per page, 56 dev pages | 48.2 | **41.3** | `scripts/bench_pages.py` timers |
| Post-reading stages (segmentation, Stage 3, 3b), 5 dev scripts | 2220 s | **1865 s** | script-level timers |
| Dev-page CER, Stage 1 / Stage 1+2 (56 GT pages, same benchmark) | 6.07% / 4.27% | **5.20% / 4.04%** | `scripts/bench_pages.py` |

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

### What changed (commits 0eb4d7d, c4dd120, 1817378, f756183)

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

**Follow-up (2026-10-17): headers with an unreadable number.**
- **The gap:** 15 of about 760 answer headers in the imported text have no readable number ("Qno: Ques: Ans:", "Long:Ques: Ans:", "Ans to the Q/A-g"). The section splitter needs a digit, so their text was glued to the answer before. In SE_08_Q1_0017, Q9's letter sequence was filed under Q8.
- **The change:** such a line now starts a new section. The section's question is still decided by the existing rules: a keyword or layout match, or a content switch needing evidence ≥ `MIN_REMAP_EVIDENCE` that is uniquely best. No threshold was added.
- **Effect** (audit, all 123 imported transcripts and 30 class-10 pipeline transcripts): one assignment changed, the 0017 fix above, and nothing else moved.
- **Still missing:** 0017's Q10 and Q12. Their text carries too little paper evidence: Q10 scores 1.1, below 2.
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

- **Timers first** (470710a):
  - every page checkpoint stores `token_usage.timing` per stage (seconds, calls, tokens);
  - `extraction_result.json` stores `metadata.stage_timing` for the run;
  - `scripts/report_stage_timing.py` sums them over folders.

  The Stage 1 token log now sums all bands; it reported only the last band.
- **No per-call CUDA cache clearing** (884d960). It now clears once per script, and before an out-of-memory retry.
- **Stage 3/3b is skipped for item-scored (Mode A) answers.** Their errors are never used: objective items get no language deductions. The mode comes from the rubric. In the stored runs, 18 Stage 3b normalisations fell on such answers, mostly rearrangement text where only the letter order counts.
- **Stage 3b with one consensus sample** reuses the localizer's read of the same crop (same prompt, temperature 0) instead of reading it again.
- **The evidence gate no longer re-runs red-ink detection on pre-cleaned PDFs.**
- **Removed** `scripts/audit_pipeline_bloat.py`: it read the deleted `gt.txt` and guessed call counts.

### Checked and not adopted

- **Stage 2b gating** (re-read a line only when the page transcript tags it `[struck:]` or OpenCV sees a stroke). Offline on the 555 stored pages, it would re-read 48% of lines but lose 62 of the 285 strike corrections Stage 2b adopted (22%). Not worth it. A better route, *not done* and offered as a proposal, is to send a page's line crops in one batched call.
- **Skipping the PDF re-render on resume:** 0.2 s per page, negligible.
- **Writing page checkpoints once:** negligible.

### Measured effect

Page benchmark on the 56 dev GT pages (Stages 0–2b re-run; decoding is deterministic):

| Run | Code | Stage 1 CER | Stage 1+2 CER | s/page S1 / S2 / S2b / total |
|---|---|---|---|---|
| `baseline_b` | before the runtime changes | 6.07% | 4.27% | 12.3 / 22.7 / 13.1 / 48.2 |
| `baseline_a` | runtime changes only | 6.07% | 4.27% | 12.3 / 22.7 / 13.2 / 48.3 |
| `final` | all changes (hybrid joiner, compact Stage 2) | **5.20%** | **4.04%** | 11.9 / 17.6 / 11.6 / **41.3** |

- **The runtime changes do not change the output:** A and B agree on every page. So the noise floor for identical code is zero, and dropping the per-call cache clearing gave no measurable speed-up (48.3 vs 48.2 s/page).
- **`final` vs `baseline_b`:**
  - Stage 1 improves by 0.86 ± 0.29 points per page (16 better, 5 worse). This is the joiner.
  - Stage 1+2 improves by 0.23 ± 0.24 points (21 better, 14 worse), within the noise.
  - The page stages are 14% faster, mostly Stage 2's shorter reply.
- **Post-reading stages:** on 5 dev scripts they take 1865 s instead of 2220 s (−16%). Stage 3 calls fall from 60 to 35 (Mode A answers skipped), and the Stage 3 errors of the other answers are identical on 29 of 29 questions.
- **Sanity check:** `final`'s Stage 1 CER equals the offline re-join of the `after` band texts (5.20%), as it must with deterministic decoding.

---

## 3. Extraction accuracy

### Band stitching (2be532d, 97262cd)

A page is read in three overlapping horizontal bands, and the band texts are joined. The old joiner compared whole lines from the end of one band with whole lines from the start of the next. The line at a band edge is usually cut: read partly, or marked `[truncated]`. When that happened the joiner found no match and **appended the lower band whole**, so the overlap lines appeared twice. 123 of the 555 stored class-10 pages contain a `[truncated]` marker. Student 0031's repeated lines are this bug.

The new joiner (`src/pipeline/band_tiler.py`):

1. **Search window:** it looks for the longest run of identical words between the last half of the upper band and the first half of the lower band. The tiler overlaps bands by a quarter to a third of their height, so the shared text always lies there.
2. **Seam:** it joins the bands at that run. Text before the run comes from the upper band and text after it from the lower band, so a cut line on either side falls away.
3. **Acceptance:** the run must contain a complete line of both bands and more than one word. A line inside the overlap is read in full by both bands, while a lone "Ans:" can match by chance.
4. **No match:** if no run qualifies, edge lines that hold nothing but `[truncated]` are dropped and the old line-similarity joiner joins the pair, with its original ratios. This fallback was added after the first real-page test (below).

Of 7 synthetic cases (`tests/test_band_stitching.py`), the old joiner fails 5.

Each band's raw text is now kept in the page checkpoint (`stage1_transcription.band_transcripts`). The old and new joiners can therefore be compared on exactly the same model output, with no generation noise.

**Measured on real band texts (56 dev GT pages, `after` run, Stage 1 output only):**

| Joiner | Stage 1 CER (macro) | Inserted / deleted chars |
|---|---|---|
| Old: line similarity only | 6.07% | 867 / 363 |
| Exact word run only, plain join when none is found (97262cd) | 9.00% | much higher: repeated lines |
| Exact word run, line similarity when none is found (current) | **5.20%** | 805 / 207 |

- The exact run alone made things worse. When the two bands misread the overlap differently ("normal troees … trees" / "norormal troees … troes"), no complete line is shared word for word, so it joined the bands whole and the overlap appeared twice. An exact seam was found on 86 of the 109 band joins.
- The old joiner failed the other way on some pages: it cut lines that were not repeats.
- The current joiner uses each method where it works. Against the old joiner it is better on 16 pages, worse on 5 and the same on 35.
- `test_overlap_read_differently_by_the_two_bands_is_not_repeated` holds the real case.
- The fallback was chosen after looking at these dev pages, so the 5.20% is a dev number. The check that settles it runs Stage 1 on 99 dev pages that have an Opus 5.5 reference but no hand-checked GT (`silver_trunc`).

### Stage 1 clean-up could hang a page (7c0d060)

The first class-6 scan run stopped on SE_06_Q1_0006 page 8 for 3 hours. The model's text for that page held a struck word inside a struck phrase, like `[struck: [struck: the] cat]`. The Stage 1 clean-up (`sanitize_and_normalize_stage1_output`, step 7c) flattens nested struck tags in a loop:
- the loop runs while the text contains `[struck: [struck:`;
- each pass only flattens the form `[struck: [struck: x]]`;
- with text after the inner tag, the pass changes nothing, and the loop never ends.

The loop now stops when a pass changes nothing. Output is unchanged on every page where the old loop finished. `test_nested_struck_tag_that_cannot_be_flattened_does_not_hang` holds the case.

The scan queue now also has a watchdog for our own jobs. If a job's log has not changed for 30 minutes, the queue logs a stack trace, ends the job, sets the script aside and goes on.

### Ablation switches added

| Switch | Default | What it tests |
|---|---|---|
| `pipeline.prompt_question_vocab` | true | Stage 1/2 prompts list up to 250 words from the question paper. Does that push the model toward dictionary words (silent autocorrection)? |
| `pipeline.stage2_compact_output` | **true** (since the measurement below) | Stage 2 replies without the per-patch `confidence` and the page `verification_notes`: no decision reads them, and they cost about 10% of Stage 2's output tokens. `context_anchor` stays because it picks which copy of a repeated word gets patched. |

Both have flags in `scripts/bench_pages.py`: `--no-prompt-vocab`, and `--stage2-full` for the old, longer Stage 2 reply. Before the default changed, that flag was `--stage2-compact`.

**Measured (56 dev GT pages, compared with the `after` run; same Stage 1 text, deterministic decoding):**

| Run | Stage 1 CER | Stage 1+2 CER | Silent corrections | s/page S1 / S2 / S2b / total |
|---|---|---|---|---|
| `after` (word list on, full Stage 2 reply) | 9.00% | 7.84% | 26.2% | 12.0 / 23.7 / 11.6 / 47.4 |
| `s2_compact` | 9.00% | 7.81% | 24.8% | 12.0 / 19.7 / 11.8 / 43.6 |
| `no_vocab` | 9.00% (4.76% re-joined with the current joiner, vs 5.20%) | 6.88% | 27.2% | 10.2 / 21.6 / 11.7 / 43.6 |

All three runs used the exact-run-only joiner that was in the code at the time, hence the high Stage 1 CER.

- **Compact Stage 2 reply: adopted.** Stage 1+2 CER is unchanged (per-page mean difference −0.03 ± 0.44 points) and Stage 2 is 17% faster.
- **Word list: not decided yet.** Removing it lowered Stage 1+2 CER by 0.96 ± 1.05 points per page (22 pages better, 22 worse), which is within the noise, and it did not reduce silent corrections. A larger check (99 more dev pages against the Opus reference, with the final code) is queued before the decision.

`stage2_conditional` is now described honestly in the YAML: every handwritten page triggers Stage 2, so it skips nothing in practice.

---

## 4. Rubric standardisation

### Rubric linter (65147a5)

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

### The guide's hand-set gates are off by default (48ee074)

Section 4 of `docs/NCTB_Standard_Evaluation_Rubric_Guide.md` adds three gates, derived from one script (SE_10_0001, teacher 7.5/15, model 14.5/15):

- **Error-density gating:** 3+ spelling errors cap language at 50%.
- **Story baseline:** a linear story is Band 3.
- **Verbatim-overlap summary cap:** ≥50% overlap caps the summary at 3.0/10. Classes 6/7 use 70% → 2.5/5 and class 8 uses 60% → 5/10.

These were calibrated toward a teacher's mark on writing questions, which this project does not do. As decided, they moved into a `calibration_gates:` block in each rubric. They apply only with `pipeline.stage4_calibration_gates: true` (`evaluate_scripts.py --calibration-gates`), as an ablation row.

**Caps that stay active:**
- the minimum-length caps and the off-topic cap, which state task requirements;
- class 11's own summary, theme and graph caps, which come from the HSC rubric, not the guide.

This also removes a contradiction: the class 6/7 rubrics said "without punitive verbatim caps" but had one.

### Grade-level expectations (1e1917b)

Each class rubric now states the guide's grade expectation (section 5) as a qualitative guideline. Class 6 and class 7 already had different band wordings from the teammate, but nothing about compound sentences and connectives at class 7, or turn-taking at class 8. No thresholds were added.

### Answer keys (fe87a74, 8b9fb13)

- **Rearrangement keys** for SE_06 and SE_10 now include the sentences by letter, copied from the paper. The model can then map sentences a student rewrote in full back to letters.
- **Rearrangement convention per paper.** These are objective questions, so the teachers are the reference. The convention was chosen on the dev split and stored as `scoring:` in the key:

  | Paper | Positional MAE | Relative-order MAE | Chosen |
  |---|---|---|---|
  | SE_06 (14 dev scripts) | 0.57 | 1.57 | positional |
  | SE_07 (18 dev scripts*) | 1.06 | 1.33 | positional |
  | SE_10 (23 dev scripts) | 0.17 | 0.52 | positional |
  | SE_08 | — | — | waits for the key check below |

  \*4 SE_07 scripts were read as `a b c d e f g h` while the teacher gave 8: the students rewrote the sentences, and without a sentence map the model listed them in writing order. Both conventions get these wrong, so they are excluded. A sentence map for SE_07 was drafted for the teacher (see below).

  **Update (2026-10-17): SE_07 sentence map added to the key.** The full re-evaluation showed the cost of the missing map. Three students who rewrote the story in the solved order (0002, 0014, 0027) got 0 from the model and 8/8 from the teacher, and 0009 got 0 vs 6. Q9 MAE on the 24 dev scripts rose to 2.46, from 2.00 in the stored run.
  - The paper text prints the story in the solved order without letters. So the map gives each sentence the letter at its position in `correct_sequence`, which is exactly the order those students were given full marks for.
  - Teacher marks are the reference for objective questions, so the map went into the key without waiting for the review.
  - The sentences are listed a..h, so their order gives nothing away.
  - SE_07 Stage 4 is re-run with the map (`eval_se_07_map`).

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
6. **SE_07 Q9:** the sentence-to-letter map is now in the key, confirmed by teacher marks (see above). Still check it against the printed paper. No scan of the SE_07 question paper is on the PC.
7. **Check the drafted Mode B answers**, then move them into the keys (`verified: true` covers them).
8. **SE_10 key:** add `verified: true` once checked. The key also has no `question_id`; it uses `paper_id`.
9. **Writing questions (for the expert):** in SE_10 the teacher never gave more than 8.5/15 on Q10 (story) or Q11 (dialogue). Model means are 8.5 and 9.6 against teacher means of 6.8. Is the teacher's scale compressed, or is the model lenient? This is not tuned either way.
10. **SE_06_Q1_0004 is a different paper.** The scan answers grammar questions: parts of speech, articles, verb forms, transformation, a paragraph and a dialogue. That looks like English 2nd Paper. Its teacher-marks row matches that scan (Q1 = 2, as written in red on page 1), but it is filed under SE_06_Q1. It has no Gemini transcript and is left out of every class-6 comparison. Is there a 1st Paper scan for this student?
11. **SE_08_Q1_0007** is a normal 1st Paper scan with teacher marks, but it has no Gemini transcript. Its pipeline grade can only be compared with the teacher.

---

## 5. Evaluation reliability (Stage 4)

Most of the error against teachers on objective questions came from bugs, not from model judgement. The fixes:

| Fix | Commit | Effect |
|---|---|---|
| **MCQ "multiple selection" counted any stray letter.** The article "a", or "i, ii & iii" inside the option text, made a correct answer score 0. Now only option labels count. | a180afd | 106 of 114 flagged items were single choices; SE_10 Q1 MAE 3.18 → 0.73 |
| **The last question of a paper lost its prompt text** (it was graded against "Question 11: Dialogue Writing") | f874190 | SE_06/07 Q13, SE_10 Q11, SE_11 Q11 |
| **Hard caps come from the rubric.** The hard-coded "50% if over 1/3 of the source" rule is gone. | 092a73e | it capped 61 class 6–8 summaries whose rubrics have no such cap |
| **No invented marks.** Narrative band recovery, the 50% defaults and the monolithic 5.0 fallback are removed; unusable output is `unscored` and listed for review. | 092a73e | |
| **Penalties by mode:** additive language deductions only in Mode B. Mode C already judges language in `language_mechanics`, so deducting again counted it twice. | 092a73e | |
| **Half marks round up** (Python sent 7.25 to 7.0 but 7.75 to 8.0) | 8c67c12 | |
| **"Answer any N" scores the first N attempted items** (SE_06 Q10 summed all 8 answers) | 8c67c12 | 4 dev scripts: first-5 MAE 1.50 vs best-5 2.00, too few to be conclusive |
| **Mode B without a key** gets "no reference answers, judge against the passage" instead of a stray flow-chart instruction | 8c67c12 | |
| **SE_08 class test** is left out of the paper maximum (scored out of 90) | 554a8bc | |
| **Feedback cannot contradict the score:** the model does not state a total; the report shows criteria total → cap → awarded | 4f29629 | |
| **Provenance:** each evaluation stores the git commit, config hash and the hashes of the rubric, key, penalty profile and scoring code | 554a8bc | two evaluations can be told apart |
| **Rearrangement convention per paper** | 8b9fb13 | see section 4 |
| **One rubric per paper, refusal on mismatch** | 65147a5 | see section 4 |

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
| `stage2_compact_output` | true | shorter Stage 2 replies (adopted: same CER, faster) |
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

Done: `baseline_a`/`b`, `after`, `sl_before`/`sl_after`, `no_vocab`, `s2_compact`, `final` (numbers above). The class 6/7/8 Stage 4 re-evaluation on the imported text is also done.

Still running (2026-10-17 evening, in this order):
1. **Class 6/7/8 read from the scans** (uploaded 2026-10-17), then graded on that text:
   - a seeded sample of 8 scripts per class first, then the rest;
   - Stage 3 and 3b are skipped, because every other Stage 4 run here graded text without Stage 3 errors, so only the transcription differs.
   - The first attempt hung on SE_06_Q1_0006 (see "Stage 1 clean-up could hang a page"). It was restarted at 23:49 from 7c0d060, which also includes the segmenter fix df86f90. 0001 and 0004 had been read and graded before the hang, with e7c2ff9.
2. **`final_no_vocab`, `silver_final`, `silver_final_no_vocab`:** the word-list decision, and the joiner on 99 more dev pages scored against the Opus reference.
3. **Class 10 Stage 4 re-evaluation:** the first run was interrupted. Then the repeatability check, SE_07 with the Q9 sentence map, and class 10 graded on the Oct-13 pipeline text (the cost of transcription errors).
4. **The one-time score of the held-out class-10 test pages.**

## Proposals not done (need a decision)

- **Language bug in `orchestrator.py`.** Stage 4 reads `extraction.metadata["paper"]` as the language. Digitized imports store the paper ID there ("SE_08_Q1"), so without an explicit `question_input` the wrong question paper can be loaded. `evaluate_scripts.py` always passes it, so stored runs are unaffected. A small fix is ready to propose.
- **Batch Stage 2b's line reads** into one call per page instead of one call per line.

---

## Commit IDs before re-authoring

On 2026-10-17 the unpushed commits on `gemma` were re-authored from the PC's previous git account to Md. Shadman Shafie. Content, messages and dates are unchanged; only the IDs changed. Outputs written before then (the evaluation provenance, benchmark reports) record the old ID; this table maps it to the new one. The IDs in this document are the new ones.

| Old | New | Date | Commit |
|---|---|---|---|
| 9f7475b | e62eb5b | 2026-10-07 | Docs: test pages must be extracted before scoring; stale test checkpoints removed from se_10_q1_v2 |
| 4f4b7a8 | fb2063e | 2026-10-07 | Docs: held-out class-10 test result (20 pages), uncertainty and test-set size estimate |
| 5d6ac53 | 99d4c73 | 2026-10-16 | Snapshot: teammate's class 6-8 rubrics/keys/penalties, schema-driven segmenter, Stage 4 and report changes |
| 2840afa | bd7e4c3 | 2026-10-16 | Phase 0: teacher-mark benchmark, seeded eval dev/test split, segmentation audit harness |
| d6727bf | 0eb4d7d | 2026-10-16 | Segmenter: page-tracked parts, content-checked headers, deterministic gap filling |
| 7995936 | c4dd120 | 2026-10-16 | Segmenter evidence: weighted paper terms, unique-best rule, answer-key source passages |
| a5ceb8d | 1817378 | 2026-10-16 | Segmenter evidence: item-form key matches, declared passages, struck text ignored |
| 8d00e22 | f756183 | 2026-10-17 | Segmenter wiring: LLM recovery switch (default off), segmentation.json, missing vs blank status |
| 64ec15b | a180afd | 2026-10-17 | Stage 4 MCQ: count only option labels as selections |
| 1882203 | f874190 | 2026-10-17 | Stage 4: last question keeps its prompt text |
| 34a8f0f | 092a73e | 2026-10-17 | Stage 4: rubric-driven caps, no invented marks, penalties by mode |
| d226c18 | 8c67c12 | 2026-10-17 | Stage 4: half marks round up, "answer any N" scores the first N attempted, no-key Mode B prompt |
| 30d8d8f | 554a8bc | 2026-10-17 | Stage 4: provenance stamp, class-test marks out of the total, evaluate_scripts fixes |
| 2ad8217 | 4f29629 | 2026-10-17 | Report: show each writing answer's computed mark and the cap that set it |
| f186cbe | 470710a | 2026-10-17 | Runtime instrumentation: per-stage time, model calls and tokens |
| 0fc218b | 957a740 | 2026-10-17 | scripts/bench_pages.py: re-run Stages 0-2b on ground-truth pages only, with timing and CER |
| ed230aa | 884d960 | 2026-10-17 | Runtime: no per-call CUDA cache clearing, skip Stage 3/3b for item-scored answers, reuse 3b crop read |
| dca6324 | 2be532d | 2026-10-17 | Stage 1: keep each band's raw text in the page result so stitching can be compared offline |
| b2e7003 | 97262cd | 2026-10-17 | Band stitching: join bands at the longest shared word run instead of line-by-line similarity |
| bdfc45f | 09ce9af | 2026-10-17 | Config switch for the question-word list in the Stage 1/2 prompts (ablation), honest stage2_conditional comment |
| fdda19a | 5d270d9 | 2026-10-17 | Stage 2 ablation switch: reply without the unread confidence / verification_notes fields |
| e100a9b | 65147a5 | 2026-10-17 | Rubric linter; each English paper uses its own rubric and Stage 4 refuses one that does not fit the paper |
| 19096d6 | 48ee074 | 2026-10-17 | Rubrics: the guide's hand-set calibration gates move to a calibration_gates block, off by default |
| 756d032 | 1e1917b | 2026-10-17 | Rubrics: each class states the guide's grade-level expectation (classes 6, 7, 8, 10) |
| 800f5bc | fe87a74 | 2026-10-17 | Answer keys: rearrangement sentence maps (SE_06, SE_10); draft Mode B reference answers for review (not read by the pipeline) |
| 176bbf5 | 8b9fb13 | 2026-10-17 | Rearrangement: per-key scoring convention; SE_06, SE_07, SE_10 use positional scoring |
| 440aeca | f7a7160 | 2026-10-17 | docs/PIPELINE_REVIEW_2026-10.md: findings, changes, numbers and reproduction commands (GPU results pending) |
