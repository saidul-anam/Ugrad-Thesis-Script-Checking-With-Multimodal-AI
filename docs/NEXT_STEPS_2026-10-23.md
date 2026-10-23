# Pipeline review, Oct 2026: status and next steps (2026-10-23)

This replaces `docs/NEXT_STEPS_2026-10-18.md` as the place to restart from. The full findings, every code change
and how to reproduce the numbers are in `docs/PIPELINE_REVIEW_2026-10.md`.

**Nothing of this project is running.** The GPU runner was stopped after class 7 on 2026-10-22, because another
user needed the PC. The Opus transcription finished on 2026-10-23.

---

## 1. Where things stand

### Code

| What | State |
|---|---|
| Branch `gemma` | at `b697c19`. Two local commits are not pushed: `df9c151` (folder restructure, see `archive/README.md`) and `b697c19` (`scripts/opus_transcribe.py`). Push only when you decide to. |
| Branch `review-followups` | 7 commits, **not merged yet** (language fix, SE_08 Q9 positional key, SE_06_Q1_0004 excluded, batched Stage 2b reads, vLLM serving, docs). It is merged only after the runner has finished (step 1 below), so that class 8 is read with the same code as classes 6 and 7. |
| Unit tests | 351 pass on `gemma`; 357 with the branch. |

### GPU runner (`outputs/review_2026_10/run_remaining.sh`)

| Part | State |
|---|---|
| Class 6 scans: read + graded | **done**: 21 scripts (SE_06_Q1_0004 left out, it is a 2nd Paper scan) |
| Class 7 scans: read + graded | **done**: 34 scripts |
| Class 8 scans | **8 sample scripts done**, 28 still to read and grade |
| Word-list / joiner benchmarks (`final_no_vocab`, `silver_final`, `silver_final_no_vocab`) | not started |
| Class-10 Stage 4 (`eval_se_10`, `eval_repeat`, `eval_se_07_map`, `eval_pipeline_text`) | not started |

It stopped at 2026-10-22 15:55 through the STOP file, after `grade_rest_se_07`. Progress log:
`outputs/benchmarks/pages/queue.log`.

### Opus 5.5 transcription of classes 6 and 7: done

- **Finished:** 2026-10-23 08:55, with `scripts/opus_transcribe.py`. It covers 55 scripts and 799 pages (about 63,000 words). There were no failed calls and no empty pages.
- **Output**, in the same layout as class 10:
  - `data/ground_truth/transcripts/SE_06_Q1_extraction_opus_5.5/` (21 scripts) and `SE_07_Q1_extraction_opus_5.5/` (34 scripts): one `<script_id>.txt` per script, pages separated by `--- PAGE N ---`;
  - `pages/<script_id>/page_N.jpg / .txt / .json`: the image sent, the page text, and Opus's notes on anything uncertain.
- **Markup** is the same as on the hand-checked GT pages: `[struck: x]`, `[illegible]`, `[unclear: x]`. Teacher red ink is left out.
- **Left out:** SE_06_Q1_0004 (a 2nd Paper scan) and SE_07_Q1_0000 (no teacher marks, no text).
- **It is a silver reference, not ground truth.** Nobody has checked it by hand yet (step 3a).

### Results so far: grades on the pipeline's own text vs the Gemini text vs the teacher

Average gap in script totals, and average error per question by question type (marks):

| Class | Scripts | Total gap: pipeline–Gemini | pipeline–teacher | Gemini–teacher | Objective (A): pipeline / Gemini vs teacher | Short answer (B) | Writing (C) |
|---|---|---|---|---|---|---|---|
| 6 | 21 | 4.31 | 5.69 | 4.29 | 1.10 / 0.87 | 1.24 / 1.25 | 1.08 / 0.88 |
| 7 | 34 | 3.88 | 7.68 | 7.85 | 0.97 / 1.11 | 1.38 / 1.32 | 1.13 / 1.04 |
| 8 (sample only) | 8 | 5.06 | 8.50 | 7.69 | 0.73 / 0.95 | 1.94 / 0.75 | 1.25 / 1.20 |

- **Saved tables:** `outputs/review_2026_10/scan_compare_se_06_full.txt` and `scan_compare_se_07_full.txt`. Re-make them with `scan_compare.py` (section 4).
- **Per question, the grades on the two texts differ by only 0.3–0.8 marks on average,** and much of that comes from questions that lost their section. So ordinary reading mistakes cost few marks.
- **What costs marks is question splitting.** Teacher-marked answers that got no section, and so scored 0:

  | Class | Pipeline text | Gemini text |
  |---|---|---|
  | 6 | 12 questions, 55 teacher marks | 2 questions, 9 marks |
  | 7 | 6 questions, 20 teacher marks | 2 questions, 5 marks |

  The causes are header forms the header patterns don't know: "( 5 Anso )", "5to. Q/A", "/ Ans.to.the.qus.no9", "Ans. to the quies no: 8", "12. Rickshaw puller", a circled "⑥", "Answerting to the question No 5". Then come misread question digits (3→9, 9→Q, 1→7), and answers that start a new page with no header at all.
- **Writing and short answers:** the model gives about 1 mark less than the teacher per question on class 7, on both texts. You said teachers often mis-mark descriptive questions, so this can't be judged against teachers. Expert marks are needed (step 5).
- **One out-of-memory error on the GPU:** SE_07_Q1_0028 page 3. Stage 2 was skipped for that page and the Stage 1 text kept. The log said "517 multi-line strike blocks" for a 72-word page with nothing crossed out, which looks like a counting bug (section 3).

---

## 2. Next steps, in order

### Step 1: finish the GPU runner (when the GPU is free)

```bash
cd /mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI
nvidia-smi                                                        # is anyone else on the GPU?
tmux new -d -s scans outputs/review_2026_10/run_remaining.sh      # start, in your own terminal
tail -5 outputs/benchmarks/pages/queue.log                        # progress: "start …" / "end … rc=0"
```

- **Waiting for the GPU:** the runner starts a job only when 26 GB of GPU memory is free. It never touches other people's jobs, so starting it while someone else is on the GPU is safe: it just waits.
- **Classes 6 and 7 again:** their jobs run first but skip every finished script, so they should take only minutes.
- **Then, in order:**
  - class 8: reading about 5 h, grading about 1.5 h;
  - the word-list and joiner benchmarks: about 3 h;
  - class-10 Stage 4: about 5 h.
- **Stopping and restarting:** `touch outputs/review_2026_10/STOP` stops it after the running job. `tmux kill-session -t scans` stops it at once. A restart skips finished work.
- **After class 8 is graded:**
  ```bash
  PYTHONPATH=. ./script_checking/bin/python outputs/review_2026_10/scratchpad_backup/scan_compare.py 06 07 08
  ```
- **Word list:** compare `final_no_vocab` with `final`, and `silver_final_no_vocab` with `silver_final`. The summaries come from `outputs/review_2026_10/scratchpad_backup/bench_summary.py`. Then decide whether to keep the word list (section 3, question 6).

### Step 2: merge the follow-ups branch and run its checks (after step 1)

Answer questions 1 and 2 in section 3 first: the last part starts a vLLM server.

```bash
git log --oneline gemma..review-followups                         # review the 7 commits first
git checkout gemma && git merge review-followups                  # a plain merge (no --ff-only); no conflicts
tmux new -d -s followups outputs/review_2026_10/run_followups.sh  # in your own terminal, about 6 h
```

`run_followups.sh` does three things:
1. It re-grades SE_08 on both texts with the positional Q9 key.
2. It checks batched Stage 2b reads on the 56 dev pages.
3. It starts the vLLM server (Google's QAT w4a16 checkpoint, port 8020) and benchmarks it with batch 1 and batch 16.

If vLLM is not OK to use, stop it before part 3 with `touch outputs/review_2026_10/STOP`, or ask Claude to take that part out.

### Step 3: use the Opus text (CPU only, can run any time)

- **a. Hand-check about 10 Opus pages per class (you).**
  - Pick pages with many crossed-out words and with tables, and correct them against the scan.
  - This tells us Opus's own error rate on younger handwriting. On class 10 it added only about 0.3 CER points.
- **b. Reading accuracy on classes 6 and 7.** Measure CER of the pipeline text against the Opus text, and of the Gemini text against the Opus text.
  - The Opus pages first need the GT page layout. `outputs/review_2026_10/scratchpad_backup/make_silver_trunc_gt.py` built the class-10 silver pages and can be adapted.
  - Then score with `scripts/evaluate_transcription.py --gt-dir <that folder> --extracted-dir outputs/extracted/english`.
  - Report the numbers as "vs Opus reference", separate from hand-checked GT.
- **c. Question splitting on the Opus text.** Run the segmenter on the Opus text with `scripts/audit_segmentation.py` (check its options with `--help`), and list which missing sections remain.
  - Misses that disappear on the Opus text come from reading: misread header digits.
  - Misses that stay come from header forms. They show what the segmenter fix must handle.

### Step 4: grade the Opus text (GPU, about 40 min for class 6 and 75 min for class 7)

This gives a clean-text grading reference for classes 6 and 7, to replace the less accurate Gemini text.

```bash
# import into a NEW folder: data_without_pipeline/ holds the Gemini text and must not be overwritten
./script_checking/bin/python scripts/setup_data_without_pipeline.py \
    --input-dir data/ground_truth/transcripts/SE_06_Q1_extraction_opus_5.5 --output-dir outputs/opus_inputs --paper SE_06_Q1
# check the folder it created and that only the 21 script files were imported (not pages/); then grade from it:
PYTHONPATH=. ./script_checking/bin/python scripts/evaluate_scripts.py --extraction-dir outputs/opus_inputs/english/se_06_q1 \
    --paper SE_06_Q1 --output-dir outputs/evaluated_opus/english/se_06_q1 --non-interactive
```

Do the same for SE_07_Q1. Then compare three sets of grades:
- **pipeline text vs Opus text:** what reading and question splitting cost;
- **Opus text vs teacher, objective questions:** grading error;
- **Opus text vs teacher, writing:** this needs expert marks to judge (step 5).

`scan_compare.py` compares with `outputs/evaluated_review`. Point its `GEM` path at `outputs/evaluated_opus/english` for this.

### Step 5: reference marks for writing questions

Teacher marks are the reference for objective questions only, because teachers often mis-mark paragraphs and stories.

For writing, the plan in `docs/EVALUATION_IMPROVEMENT_GUIDE.md` is expert marks:
- on a random sample of writing answers;
- blind, without seeing the teacher's or the model's mark;
- if possible two experts on part of the sample, to know how much experts disagree with each other.

Claude can prepare the sample and a marking sheet once you confirm the expert and the number of answers (section 3, question 5).

### Step 6: segmenter fix (needs your OK before any code is written)

**Proposal:** let content decide which question an answer belongs to, and treat the header as one signal among others. Content means:
- the answer keys (MCQ option text, accepted answers);
- the source passages;
- the kind of answer (a letter list, a true/false list, a story or paragraph).

The header forms vary too much to cover with more patterns. Measure the fix on:
- the 153 Gemini texts (`scripts/audit_segmentation.py`);
- the class 6/7/8 pipeline text;
- the Opus text.

Then re-grade the affected scripts.

### Step 7: final evaluation and write-up

1. Score the grading benchmark on the dev split, then on the test split **once**:
   ```bash
   ./script_checking/bin/python scripts/benchmark_evaluation.py --eval-dir outputs/evaluated_review/english --compare outputs/evaluated/english --split dev
   ```
   Then run the same command with `--split test`. Use SE_07 from `outputs/evaluated_review_map`, and leave SE_10 0012 and 0013 out of the stored-vs-new comparison.
2. **Score the held-out class-10 test pages once,** at the very end: scripts 0029–0033, 20 GT pages. 0032 and 0033 need full runs. Never tune on them.
3. **Write-up:**
   - add the class 6/7/8 results, the Opus-reference results and the follow-up checks to `docs/PIPELINE_REVIEW_2026-10.md`;
   - write the short reply to the teammate;
   - commit (no Claude trailers);
   - push only when you decide to.

---

## 3. Open questions for you

1. **vLLM install:** is `/home/ubuntu/venvs/vllm` (vLLM 0.30.0) yours, and may we use it? If not, a new environment goes on `/mnt/models`, because the root disk is 98% full.
2. **QAT checkpoint:** may we read `/mnt/models/google/gemma-4-31B-it-qat-w4a16-ct`? It may belong to the bangla-ai-tutor user.
3. **SE_08 Q9 scoring:** scored by position or by relative order?
   - The branch uses position, following the key's own note and SE_06/07/10.
   - By relative order, the 20 students who wrote `e b d h a f c g` would get 7/8 instead of 5/8.
4. **Segmenter fix:** may Claude write it (step 6)? The diff comes to you first.
5. **Expert marking for writing:** is it still planned, and roughly how many answers can the expert mark?
6. **Word list:** keep it or drop it, after the step 1 benchmarks.

**Smaller items (each needs your OK before any code change):**
- Find out why Stage 2 counted "517 multi-line strike blocks" on SE_07_Q1_0028 page 3 and ran out of memory.
- Move the two 15-line re-export files `src/pipeline/strikethrough_collision_resolver.py` and `src/pipeline/edge_truncation_detector.py` into `archive/`. The real modules are in `src/utils/`. One test imports the second file, so its import line changes too.

---

## 4. Useful commands

```bash
tmux ls                                                           # what is running ("scans", "followups")
tail -5 outputs/benchmarks/pages/queue.log                        # runner progress
nvidia-smi --query-compute-apps=pid,used_memory --format=csv      # who is on the GPU
PYTHONPATH=. ./script_checking/bin/python outputs/review_2026_10/scratchpad_backup/scan_compare.py 06 07 08
./script_checking/bin/python -m pytest -q                         # unit tests
```

**Running Claude jobs from tmux.**
- Jobs started from a Claude session die when that session ends. Start long jobs yourself in tmux.
- If Claude starts one, it uses `tmux new -d`, which survives the session.

**Opus transcription again** (for example for class 8):
```bash
tmux new -d -s opus -c "$PWD" "./script_checking/bin/python -u scripts/opus_transcribe.py --papers SE_08_Q1"
```
- **Restarting:** it skips finished pages.
- **Usage limits:** it waits for each usage-limit reset, and pauses at 90% of the weekly limit (`--weekly-cap`).
- **Cost:** each page uses about 0.5% of the 5-hour window and about 0.03% of the weekly window. That is about 200 pages per 5 hours.
- **Shared allowance:** the job shares your Claude subscription. While it runs at the limit, interactive Claude use is blocked too.
