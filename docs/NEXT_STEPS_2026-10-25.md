# Pipeline review, Oct 2026: status and next steps (2026-10-25)

This replaces `docs/NEXT_STEPS_2026-10-23.md` as the place to restart from. That file still has the results of
classes 6 and 7 and the full commands for steps 1, 2 and 4. The findings and code changes of the review are in
`docs/PIPELINE_REVIEW_2026-10.md`.

**Nothing of this project is running on the lab PC.** There is no GPU access for the next few days, so only CPU
work is done until you say the GPU is back.

---

## 1. Where things stand

| What | State |
|---|---|
| Branch `gemma` | at `c977e1e`. Three local commits are not pushed (`df9c151`, `b697c19`, `c977e1e`). Push only when you decide to. |
| Branch `review-followups` | 9 commits, not merged yet; the last two (`3192802`, `1fbd7c9`) are the team-check change of section 3. It is merged after the GPU runner (step 1), so class 8 is read with the same code as classes 6 and 7. |
| GPU runner | stopped after class 7 (2026-10-22). Class 8: 8 sample scripts done, 28 to go. |
| Opus in the cloud | runs in a claude.ai/code session on the private repo. It transcribes class 8, then grades classes 6/7/8/10 **with the code unchanged**. It asks you before grading. |
| Untracked file | `scripts/eval_opus_benchmark.py` (Oct 22, not from the Claude sessions). It computes CER against the Opus text. Left untouched. |

---

## 2. New results (CPU, 2026-10-25)

Scripts and outputs:
- scripts: `outputs/review_2026_10/scratchpad_backup/` (`dup_scan.py`, `cer_vs_opus.py`, `audit_opus.py`, `tc_same_text.py`);
- outputs: `outputs/review_2026_10/cpu_2026_10_25/`.

How to re-run them is in section 5. All numbers use dev scripts only: class-10 test scripts 0029–0033, SE_06_Q1_0004 and SE_07_Q1_0000 are left out.

### 2a. Questions answered twice

Sometimes a student answers a question twice and crosses one copy out with one or two long diagonal or vertical
strokes. The question splitter then joins both copies into one answer, so the grader sees both.

**Real cases in the Opus text (classes 6, 7 and the class-10 dev scripts): 9.**

| Class | Cases |
|---|---|
| 6 | SE_06_Q1_0001 Q4, 0016 Q2 |
| 7 | SE_07_Q1_0006 Q9, 0014 Q3, 0014 Q9, 0025 Q5, 0025 Q8, 0025 Q9 |
| 10 | SE_10_Q1_0019 Q7 |

- **Opus missed most of them.** It tagged the crossed-out copy as `[struck: …]` in 3 cases, partly in 1, and missed it in 5. It does catch a line drawn through words. Gemini and the pipeline (Gemma) miss these strokes too.
- **A simple flag finds them:** a question whose text comes in 2 or more separate runs, each starting with its own question header.
  - It finds 8 of the 9 on the Opus text and 6 of the 9 on the pipeline text.
  - Its other flags are mostly misread question numbers (3 on the Opus text, 7 on the pipeline text). The same check catches those too, which is useful.
- **Class 8:** the Gemini text gives 12 flags in 7 scripts. They are not checked yet. SE_08_Q1_0004 is a false alarm: its numbered items "1. ans:" are read as question headers. The real check waits for the class-8 Opus text from the cloud.
- **Class-10 Opus files drop crossed-out words entirely**, even on the pasted hand-checked pages ("make sure [struck: po] Portia" became "make sure Portia"). Classes 6 and 7 keep the `[struck: …]` tags. This doesn't change CER, because CER drops struck text on both sides.

**Decision (2026-10-25):** the thesis team checks these questions; the teachers aren't involved. The report shows both marks until the team decides. No fixed rule such as "take the higher mark".

### 2b. Reading accuracy on classes 6 and 7, vs the Opus reference (silver)

CER = character error rate; WER = word error rate.

| Class | Text | Pages | CER, whole script (micro) | WER |
|---|---|---|---|---|
| 6 | pipeline, Stage 1 | 295 | 8.84% | 14.30% |
| 6 | pipeline, Stage 1+2 | 295 | **7.75%** | 12.29% |
| 6 | Gemini | 294 | 10.56% | 14.83% |
| 7 | pipeline, Stage 1 | 503 | 8.78% | 12.68% |
| 7 | pipeline, Stage 1+2 | 503 | 7.97% | 11.45% |
| 7 | Gemini | 503 | **6.25%** | 8.79% |
| 10 (Opus-only dev pages) | pipeline, Stage 1 | 437 | 6.73% (macro per page 7.23%) | 9.47% |
| 10 (Opus-only dev pages) | pipeline, Stage 1+2 | 437 | 5.84% (macro per page 6.45%) | 7.81% |

- **Scoring:** classes 6 and 7 are scored on the whole script, with all pages joined. The Gemini import of class 6 has shifted page numbers: its page-level CER is 21%, but its whole-script CER is 10.6%.
- **The class-10 numbers equal the earlier silver result (7.23% / 6.45%)**, so the scorer matches `src/utils/text_metrics.py`.
- **Stage 2 helps on every class:** about 1 CER point.
- **Gemini is better than the pipeline on class 7, and worse on class 6.**
- **Caution:** Opus and Gemini are both large models and may tidy the same student mistakes in the same way, which would flatter Gemini here. The hand check (step 3a) settles this.
- **Report these as "vs Opus reference",** separate from the hand-checked GT.

### 2c. Question splitting on the Opus text

These are teacher-marked questions that got no section, and so scored 0. The table counts questions, from the same audit as `scripts/audit_segmentation.py`.

| Class | Pipeline text | Gemini text | Opus text | Opus text, markup unwrapped in headers |
|---|---|---|---|---|
| 6 | 12 | 2 | 8 | 3 |
| 7 | 6 | 2 | 11 | 10 |
| 10 (dev) | 3 (+1 header only) | (is the Opus text) | 0 | 0 |

**Why the Opus text loses sections that the Gemini text keeps:**
- **Markup inside header lines:**
  - "Answering to the [unclear: Quostion] No 2";
  - "Q-No.[unclear: 02]";
  - "Q NO.[struck: illegible]5";
  - a circled "⑤".

  The header patterns allow a `[struck: …]` only at the start of a line. Unwrapping the markup in the first 60 characters of each line fixes 6 of the 19 Opus misses. This was a what-if run in the scratch copy; no code has changed.
- **Header forms the patterns don't know**, which the pipeline text often misses too:
  - a header in the middle of a line, after answer text: "courage. Answers to the Q No: 6", "Growing important Answers to the Q No: 4";
  - the student's shorthand: "A: to the question No.5", "An to the Q. NO. 2" (SE_07_Q1_0013 alone loses 5 questions);
  - typos and out-of-order answers: "Answe to the QNo:9" after "Answer to the QNo:3" (SE_07_Q1_0025);
  - no header at all: SE_06_Q1_0020 Q2 starts directly with "ⓐ) Ans:".

**Reading problems:** the pipeline loses 7 class-6 sections that the Opus text keeps. These come from misread header digits.

**What this means for the segmenter fix (step 6):**
- unwrap markup in header lines;
- find headers in the middle of a line;
- for the rest, let the content of an answer decide its question (out of order, typos, no header).

---

## 3. Code change: grade each copy, team check when the marks differ (approved 2026-10-25)

Committed on `review-followups` as `3192802` and `1fbd7c9` (not on `gemma`, for the reasons below). 365 tests pass, 9 of them new. The one failing test (`test_setup_data_without_pipeline`) also fails without the change: the branch's folder has no ground-truth mark files. The diff is also saved in `outputs/review_2026_10/cpu_2026_10_25/team_check_proposal.diff`.

| File | Change |
|---|---|
| `src/pipeline/answer_segmenter.py` | Finds the questions answered twice, with the flag of section 2a. It keeps each copy's text and pages in the new `attempts` / `attempt_pages` fields, and writes `repeated_questions` into `segmentation.json`. A copy with no answer words once struck text is removed does not count: it is a header alone, or the transcription already caught the crossing out (`1fbd7c9`). **The joined answer text is unchanged:** checked on 1,980 real questions, 0 differ, 22 have copies. |
| `src/pipeline/stage4_evaluator.py` | Grades each copy on its own. Same mark: the question is scored as usual. Different marks, or a copy that can't be scored: the status is `team_check`, and all copy marks are kept. `final_score` uses the lower mark and the new `final_score_max` the higher one, so the total is a range. `team_check` questions are left out of the MAE. |
| `src/core/schemas.py` | The new fields: `attempts`, `attempt_pages`, `attempt_marks`, `attempt_writing`, `team_check_questions`, `final_score_max`. |
| `src/utils/export_utils.py` | The report shows "👥 Answered twice: team check", both marks ("1.0 or 3.0"), each copy with its pages, and the total as a range ("41.0–43.0"). |
| `scripts/team_check.py` (new) | Collects every `team_check` question into one CSV (`outputs/team_check/english.csv`): pages and mark of each copy, and the first line of each copy. The team fills in `chosen_copy`. Running it again keeps the decisions. |
| `scripts/benchmark_evaluation.py` | New `--team-check <csv>`: a decided question counts with the chosen copy's mark; an undecided one stays out of every metric. The summary counts questions answered twice and those still open. |
| `tests/test_answered_twice.py` (new) | 8 tests. |

**Why it is on `review-followups` and not on `gemma`:**
- It is built on top of `review-followups`: that branch also changes `scripts/benchmark_evaluation.py`, so the diff does not apply to `gemma`.
- The GPU runner reads and grades class 8 with the code on `gemma`. Classes 6 and 7 were done with that code, so `gemma` stays unchanged until class 8 is done.

It is merged into `gemma` together with the other follow-ups (step 2 below). Push only when you say so.

---

## 4. Next steps, in order

### Now (CPU only)

- **A. Team-check code:** done (section 3); it is merged with the follow-ups.
- **B. Cloud results (Opus in claude.ai/code).**
  - **What the cloud did:**
    - transcribed class 8;
    - graded classes 6/7/8/10 with the code unchanged, so a question answered twice got one mark for both copies read together;
    - a follow-up task grades each copy of those questions separately. Struck words are ignored as usual, and a copy that is entirely `[struck]` gets an empty mark with the note "fully struck".
  - **Its output:** `results/answered_twice.csv` (one row per copy: mark, original_mark, verdict, note) and `results/answered_twice_summary.md` (each script's total as a range).
  - **The user copies the whole cloud repo folder,** with its README and grading instructions, to `/mnt/models/script_checking/opus_cloud_results/`. The user is not reachable after that.
  - **Then, on the PC (CPU only):**
    1. Read the folder's README, and copy `results/` into the repo following its mapping. Don't overwrite anything in `data_without_pipeline/` or `data/ground_truth/`; new folders only.
    2. Run the repeat flag on the class-8 Opus text (section 5). Compare it with the cloud's `answered_twice.csv`, and check each class-8 case on the scan pages (`data/raw_pdfs/english/SE_08_Q1/`): a real repeat, or a misread question number?
    3. Build the team list, one row per question: the pages and mark of each copy, and an empty `chosen_copy` column. Use the same columns as `scripts/team_check.py` (on `review-followups`), and save it under `outputs/team_check/`. The team decides which copy counts; Claude doesn't.
    4. Compare the Opus grades with the teacher (objective questions, the reference there) and with the grades on the pipeline text and the Gemini text. Leave class-10 test scripts 0029–0033 out.
    5. Write the results into this file or a new `docs/` file, and commit locally (no Claude trailers, no push).
  - **The PC does no `claude -p` re-grading:** the cloud does it.
- **C. Step 3a, hand-check about 10 Opus pages per class (you).** Pick pages with crossed-out words, and pages where the Gemini text and the pipeline text disagree. This also shows whether Opus and Gemini tidy the same mistakes (section 2b).
- **D. Segmenter fix (step 6): draft it in a scratch copy, don't commit** (the user agreed, 2026-10-25).
  - **What it covers** (section 2c):
    - unwrap markup in header lines;
    - find headers in the middle of a line;
    - let the content of an answer decide its question when the header is missing, out of order or misread.
  - **Measure it** with `audit_opus.py` (section 5) on the Opus, Gemini and pipeline texts, and check that no other question changes (like `tc_same_text.py --compare`).
  - **The diff and the numbers go into `docs/` for the user's OK,** with no per-word lists or hand-set thresholds (the user's rule).
- **While the user is away:** no push, no merge into `gemma`, no GPU work, and never touch other people's processes.
- **E. Expert marks for writing (step 5):** Claude can prepare the sample and the marking sheet once question 5 in section 6 is answered.

### When the GPU is back (you tell Claude)

The full commands are in `docs/NEXT_STEPS_2026-10-23.md`.

1. **Step 1: finish the GPU runner.** You start it in tmux: `tmux new -d -s scans outputs/review_2026_10/run_remaining.sh`.
   - class 8: reading about 5 h, grading about 1.5 h;
   - the word-list and joiner benchmarks: about 3 h;
   - class-10 Stage 4: about 5 h.
2. **Step 2: merge `review-followups`, then `run_followups.sh`** (about 6 h). Questions 1 and 2 in section 6 come first, because it starts a vLLM server.
3. **Questions answered twice:** re-grade them in classes 6, 7 and 10 (class 8 is re-graded by `run_followups.sh`). Then run `scripts/team_check.py` and fill in the team CSV.
4. **Step 4: grade the Opus text with the pipeline's Stage 4** (classes 6/7, about 2 h). This is grading on clean text, to compare with the pipeline-text grades. The cloud run is different: it grades with Opus.
5. **Step 6: the segmenter fix**, then re-grade the affected scripts.
6. **Step 7: final evaluation.**
   - Run the dev split, then the test split **once**.
   - Score the class-10 test pages (0029–0033) once, at the very end.
   - Write up the results.

---

## 5. Re-running the CPU checks

From the repo root:

```bash
B=outputs/review_2026_10/scratchpad_backup
O=outputs/review_2026_10/cpu_2026_10_25
# questions answered twice (FLAG lines; --details lists every run of every flagged question)
./script_checking/bin/python $B/dup_scan.py SE_07_Q1 data/ground_truth/transcripts/SE_07_Q1_extraction_opus_5.5 --details
./script_checking/bin/python $B/dup_scan.py SE_10_Q1 data/ground_truth/transcripts/SE_10_Q1_extraction_opus_5.5 \
    --exclude SE_10_Q1 SE_10_Q1_0029 SE_10_Q1_0030 SE_10_Q1_0031 SE_10_Q1_0032 SE_10_Q1_0033
./script_checking/bin/python $B/dup_scan.py SE_08_Q1 <class-8 Opus folder from the cloud>
# CER of pipeline and Gemini text vs the Opus text (a few seconds)
./script_checking/bin/python $B/cer_vs_opus.py $O/cer
# question-splitting audit on Opus, pipeline and Gemini text; --unwrap = the what-if of section 2c
./script_checking/bin/python $B/audit_opus.py $O/split
./script_checking/bin/python $B/audit_opus.py $O/split_unwrap --unwrap
```

`dup_scan.py` reads Opus `.txt` folders and `<script>/extraction_result.json` folders. None of these scripts write into `data/` or `src/`.

---

## 6. Open questions for you

1. **vLLM install:** is `/home/ubuntu/venvs/vllm` yours, and may we use it?
2. **QAT checkpoint:** may we read `/mnt/models/google/gemma-4-31B-it-qat-w4a16-ct`?
3. **SE_08 Q9:** score by position (the branch does this) or by relative order?
4. **Segmenter fix (step 6):** may Claude write it? This includes unwrapping markup in header lines and finding headers in the middle of a line (section 2c).
5. **Expert marking for writing:** is it still planned, and roughly how many answers?
6. **Word list:** keep it or drop it, after the step 1 benchmarks.
