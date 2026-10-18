# Pipeline review, Oct 2026: status and next steps (2026-10-18)

The review was stopped on 2026-10-18 at 16:30, at the user's request, to continue in a later session.

- **The full findings,** every change and the reproduction commands are in `docs/PIPELINE_REVIEW_2026-10.md`.
- **This file says** where the work stands, what is left, and how to restart it.

## Where things stand

**The code work for all five tasks is done.** The tasks were:
1. segmenter audit;
2. runtime;
3. extraction accuracy;
4. rubric standardisation;
5. evaluation reliability.

It is committed on the `gemma` branch: 33 commits since 2026-10-16, ending at `2e1effd`. 351 unit tests pass.

**The last 7 commits are not pushed** (`ee8b648` … `2e1effd`). Push only when the user asks.

**What is left is GPU work:**
- the class 6/7/8 scans;
- the class-10 checks;
- the one-time test score.

It was interrupted because the GPU was shared (see "Lessons"). Nothing is running now.

## Results so far

### Code changes (measured)

| Area | Before | After |
|---|---|---|
| Questions marked missing although the teacher gave marks (segmenter, 153 transcripts) | 22 | 9 |
| MCQ MAE vs teacher, class 10 Q1 (marks) | 3.18 | 0.73 |
| Class 6–8 summaries capped by a rule their rubric does not have | 61 | 0 |
| Same transcript graded twice gives the same marks | no | yes |
| Rubric/key/paper consistency errors | 8 | 0 |
| Unit tests | 310 pass, 8 fail | 351 pass |
| Page stages, seconds per page (56 dev pages) | 48.2 | 41.3 |
| Dev-page CER, Stage 1 / Stage 1+2 (56 GT pages) | 6.07% / 4.27% | 5.20% / 4.04% |

On CER:
- The Stage 1 gain (−0.86 ± 0.29 points) is real.
- The Stage 1+2 gain (−0.23 ± 0.24) is within noise.

### Classes 6/7/8 re-graded with the new Stage 4, on the Gemini text (dev split, new vs stored)

Teacher marks are the reference for objective (Mode A) questions only. Agreement on writing (Mode C) is descriptive.

| Paper | Total MAE, stored → new | Mode A MAE | Mode C MAE |
|---|---|---|---|
| SE_06 | 4.83 → 4.37 | 1.056 → 0.750 | 1.025 → 0.897 |
| SE_07 | 8.46 → 9.17 | 1.012 → 1.098 | 1.400 → 1.121 |
| SE_08 | 8.81 → 7.06 | 0.862 → 0.869 | 1.495 → 0.914 |

- **Mode C feedback that states a different band or total than the mark:** 144 of 316 before, 0 of 252 now.
- **SE_07 got worse because of a key gap.** Q9's key had no sentence map, so students who rewrote the story got 0. The map is now in the key (`e7c2ff9`), but SE_07 has not been re-graded with it yet (`eval_se_07_map` below).
- **Report file:** `outputs/review_2026_10/bench_eval_dev/evaluation_english_20261018_163521.md`.

### Class 6/7/8 read from the scans (started; about 5% done)

**Read** (pipeline text in `outputs/extracted/english/se_06_q1/`):
- SE_06_Q1_0001, 0004 and 0006 are complete.
- 0009 has pages 1–6 saved; a restart resumes from page 7.

**Graded** (`outputs/evaluated_pipeline/english/se_06_q1/`): 0001 and 0004.

**SE_06_Q1_0001:**

| Graded on | Total |
|---|---|
| Pipeline's own text | 36.5 |
| Gemini text | 33.0 |
| Teacher | 30.0 |

On Mode A, the error against the teacher is 1.83 marks per question on the pipeline text and 2.50 on the Gemini text. One script is too few to conclude anything.

**SE_06_Q1_0004 is a different paper.** The scan answers grammar questions, which looks like English 2nd Paper. Its teacher-marks row matches that scan, but it is filed under SE_06_Q1. It is left out of the comparisons.

**Speed:** reading takes about 46 s per page and grading about 2.5 min per script.

**How the scans were run:**
- Stage 3 (error list) and 3b are skipped, because every other Stage 4 run here graded text without Stage 3 errors. Only the transcription differs from the Gemini-text grades.
- The word list stays on.
- The sample is seeded (20261017), 8 scripts per class:

| Class | Sample scripts |
|---|---|
| 6 | 0001 0004 0006 0009 0010 0011 0015 0016 |
| 7 | 0003 0006 0012 0018 0019 0023 0026 0031 |
| 8 | 0005 0011 0012 0017 0019 0021 0025 0029 |

## Lessons from this session

1. **A Stage 1 clean-up loop could hang a page forever.** It happened on `[struck: [struck: the] cat]`, and SE_06_Q1_0006 page 8 sat for 3 hours. It is fixed in `7c0d060` and tested. The runner below also ends any job whose log is silent for 30 minutes.
2. **Jobs started from a Claude session die when that session ends**, even with `setsid nohup`. This happened at 17:56, 23:59 and 12:08. Start the runner yourself in tmux (below), not through Claude.
3. **The GPU is shared.** Another user's vLLM server (Gemma 4 31B QAT w4a16, port 8010, project `bangla-ai-tutor`) takes about 31 of the 32.6 GB. Our pipeline needs about 23 GB, so the two cannot run together. The runner waits for free memory and never touches other processes.
   - The user decided (2026-10-18) to wait for the GPU rather than send work to that server.

## How to restart

Everything needed is on disk:
- **Runner:** `outputs/review_2026_10/run_remaining.sh`. It runs from the repo root with the checked-out code.
- **Helper scripts and inputs:** `outputs/review_2026_10/scratchpad_backup/`.
- `outputs/` is gitignored, so these files are on this PC only.

```bash
cd /mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI
tmux new -d -s scans outputs/review_2026_10/run_remaining.sh     # start (in your own terminal)
tail -5 outputs/benchmarks/pages/queue.log                       # progress: "start …" / "end … rc=0"
tmux ls                                                          # "scans: 1 windows" = running
touch outputs/review_2026_10/STOP                                # stop after the running job
tmux kill-session -t scans                                       # stop now (also: pkill -f extract_noS3.py / evaluate_scripts.py)
```

The runner skips scripts that are already read or graded, so it can be stopped and started at any time. Job logs go to `outputs/benchmarks/pages/<job>.log`.

**Check results as they come in:**
```bash
./script_checking/bin/python outputs/review_2026_10/scratchpad_backup/scan_compare.py 06 07 08
```
It compares the pipeline-text grades with the Gemini-text grades and with the teacher: totals, and per-mode MAE and bias.

## Next tasks, in order

GPU times assume the GPU is free.

1. **Class 6/7/8 scans:** steps 1–2 of the runner.
   - Samples first, about 5 h. Then the rest, about 16 h.
   - Report each class's sample as soon as it is graded, with `scan_compare.py`.
   - Put the results in the review doc.
   - If the pipeline text has headers with no readable question number, check that `df86f90` handled them.
2. **Word-list decision and the joiner check:** step 3 of the runner, about 3 h.
   - **Word list:** compare `final_no_vocab` with `final` on CER. The `no_vocab` result so far is −0.96 ± 1.05 points, which is undecided.
   - **Joiner:** compare `silver_final` with `silver_final_no_vocab` on 99 dev pages scored against the Opus reference.
   - Then run the offline stitch comparison on the `silver_final` band texts with `scratchpad_backup/stitch_hybrid.py`.
   - Summaries: `scratchpad_backup/bench_summary.py`.
3. **Class-10 Stage 4:** step 4 of the runner, about 5 h.
   - **Re-grade class 10:** `eval_se_10`.
   - **Repeatability:** `eval_repeat`, 10 scripts graded twice; the totals must be identical.
   - **SE_07 with the Q9 map:** `eval_se_07_map`.
   - **Transcription cost:** `eval_pipeline_text`, the class-10 dev scripts graded on the Oct-13 pipeline text.
   - **Then:**
     ```bash
     ./script_checking/bin/python scripts/benchmark_evaluation.py --eval-dir outputs/evaluated_review/english --compare outputs/evaluated/english --split dev
     ```
     Then the same with `--split test`, once only.
   - Use SE_07 from `outputs/evaluated_review_map`.
   - Leave SE_10 0012/0013 out of the stored-vs-new comparison.
4. **Re-grade SE_08_Q1_0017 on the Gemini text with HEAD.** The segmenter fix `df86f90` files its Q9 answer correctly, and the stored class-8 re-grade predates it.
5. **Score the held-out class-10 test pages once,** at the very end. These are scripts 0029–0033 (20 GT pages); 0032 and 0033 need full runs. Never tune on them.
6. **Write-up:**
   - update `docs/PIPELINE_REVIEW_2026-10.md` with the results above;
   - write the short reply to the teammate;
   - commit without Claude trailers.

## Decisions needed (user / teammate)

The full list is in the review doc, "Review list for the teammate / a teacher".

1. **SE_08 Q9 key:** the key says `e b d h a c g f`. 20 of 35 students wrote `e b d h a f c g` and got 8. Which is right?
2. **SE_07 Q10:** the printed paper says 2 × 5 = 10; the rubric and total use 5. Which is right?
3. **SE_07 Q9 sentence map:** confirmed by teacher marks. Check it against the printed paper; no SE_07 question PDF is on the PC.
4. **SE_06_Q1_0004:** is there a 1st Paper scan for this student?
5. **SE_08_Q1_0007** has no Gemini transcript; it can only be compared with the teacher.
6. **Not done, need a go-ahead:**
   - the language bug in `orchestrator.py` (`metadata["paper"]`);
   - batching Stage 2b's line reads;
   - serving the model through vLLM with batching. This needs a CER re-baseline and GPU memory.

## Exact code versions used (for reproduction)

| Run | Commit |
|---|---|
| Interim class 6/7/8 Stage 4 re-grade (`evaluated_review`) | `440aeca` (= `f7a7160`) |
| Benchmark `final` | `17000c0` |
| Scans read so far | `e7c2ff9`: 0001, 0004 and 0006 p1–7. `7c0d060`: 0006 p8–13 and 0009 p1–6. |

The commits in between change no Stage 1/2 output except that a hanging page now finishes. The old session's git worktrees lived in its scratchpad; if that is gone, run `git worktree prune`.
