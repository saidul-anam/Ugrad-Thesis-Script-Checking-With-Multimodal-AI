# Opus cloud run: results (2026-10-25)

This is step B of section 4 in `docs/NEXT_STEPS_2026-10-25.md`. A Claude Code cloud session transcribed class 8 with
Opus 5.5, graded classes 6, 7, 8 and 10 on the Opus text with Opus 5.5, then graded each copy of the questions
answered twice. This file says where the results are, what the PC checks found, and what needs your decision.

All numbers here use **dev scripts only**, unless a line says otherwise:
- grading: the dev split of `configs/splits/eval_split.csv`, without the class-10 transcription test scripts
  0029–0033 (they are in the grading dev split, but stay unseen) and SE_07_Q1_0000 (no teacher marks);
- the team list covers every script, because it records decisions and tunes nothing.

---

## 1. What came back and where it is

The cloud repo arrived as `Grading-using-claude--claude-practical-albattani-gq6f0t.zip` in the repo root
(262,842,003 bytes, 5,967 entries, every file's checksum OK). It is extracted to
`/mnt/models/script_checking/opus_cloud_results/`. The zip stays in the repo root, untracked.

Copied into the repo following the mapping in the folder's `README.md` (all new folders; nothing overwritten):

| From `opus_cloud_results/` | To the repo | What |
|---|---|---|
| `results/transcripts/SE_08_Q1_extraction_opus_5.5/` | `data/ground_truth/transcripts/SE_08_Q1_extraction_opus_5.5/` | class-8 Opus text: 36 scripts, 579 pages, same layout as classes 6 and 7 |
| `results/evaluated_opus/english/` | `outputs/evaluated_opus/english/` | Opus grades: classes 6 (21), 7 (34), 8 (36), 10 (33) |
| `results/report/` | `outputs/benchmarks/opus/` | the cloud's benchmark tables (`benchmark_dev.txt`, `benchmark_test.txt`, `benchmark_all.txt`) |

Left in `opus_cloud_results/results/` only (the README says to keep them with the results):
- `ledger.jsonl`: one line per Opus call with its cost;
- `claude_cache/`: every Opus reply with its prompt;
- `answered_twice.csv`, `answered_twice_summary.md`: the follow-up task (section 2);
- `opus_inputs/`: the Opus text of all four classes, already imported for Stage 4. Step 4 (Gemma grading the Opus
  text) can use it instead of importing again.

**Cost at list price:** $76.00 in the ledger ($33.72 for 579 pages, $42.27 for 1,537 grading calls), plus $0.63 for
the answered-twice follow-up (not in the ledger).

**Code used:** branch `review-followups` at `4792d0d`, so the SE_08 Q9 answer is scored by position. It did not
include the team-check change (`3192802`, `1fbd7c9`): each answered-twice question got one mark for both copies, and
the follow-up graded the copies separately.

---

## 2. Questions answered twice

### 2a. What the cloud found and graded

The cloud graded each copy with the unchanged Stage 4. A copy whose words are all `[struck: …]` was not graded.

| Script | Question (max) | Earlier mark (both copies) | Copy marks | Team decision needed? |
|---|---|---|---|---|
| SE_06_Q1_0001 | Q4 (5) | 3 | c1 1 / c2 3 | yes |
| SE_06_Q1_0016 | Q2 (5) | 1.5 | c1 fully struck / c2 1.5 | no: the struck copy does not count |
| SE_07_Q1_0006 | Q9 (8) | 2 | c1 fully struck / c2 2 | no: the struck copy does not count |
| SE_07_Q1_0014 | Q3 (5) | 1 | c1 1 / c2 1 | no: same mark |
| SE_07_Q1_0014 | Q9 (8) | 8 | c1 0 / c2 8 | yes |
| SE_07_Q1_0025 | Q5 (5) | 4 | c1 1 / c2 0 / c3 4 | yes (copy 2 is not Q5, see 2c) |
| SE_07_Q1_0025 | Q8 (5) | 2 | c1 0 / c2 2 | yes |
| SE_07_Q1_0025 | Q9 (8) | 4 | c1 3 / c2 4 | yes |
| SE_08_Q1_0020 | Q4 (5) | 3 | c1 0 / c2 3 | yes |
| SE_08_Q1_0027 | Q7 (10) | 6.5 | c1 5 / c2 5.5 | yes |
| SE_10_Q1_0019 | Q7 (8) | 8 | c1 3 / c2 8 | yes |

### 2b. The team list

**`outputs/team_check/opus_cloud.csv`: 8 questions.** It uses the columns of `scripts/team_check.py` (on
`review-followups`): the pages and mark of each copy, the first line of each copy, an empty `chosen_copy`, and a
`note` with what the cloud and the scan check saw. The team writes the number of the copy that counts in
`chosen_copy`. Claude does not choose.

The rule for "decision needed" is the one in the team-check code: a copy with no answer words once struck text is
removed does not count, and copies with the same mark are scored as usual.

### 2c. Class 8: the repeat flag on the Opus text, and the scan check

The flag of section 2a in `NEXT_STEPS_2026-10-25.md` (a question with two or more separate runs that each start with
a header) on the class-8 Opus text: **6 questions in 2 scripts**. Both are misread question numbers, not repeats.
The cloud's list agrees. I checked every class-8 case on the scans (`data/raw_pdfs/english/SE_08_Q1/`):

| Case | What the scan shows | Effect on the grades |
|---|---|---|
| SE_08_Q1_0004 Q1–Q5 (flagged) | Page 9 is Q8 ("Ans to the question no - 8"). Its items "1. ans:" … "5. ans:" were taken as headers for Q1–Q5. | Q8 graded 0 (teacher 5); the Q8 items were added to Q1–Q5. |
| SE_08_Q1_0007 Q3 (flagged) | Page 8 header "8No Q:A:" (it follows "7 No", and the items are matchings). Opus read "3No". | Q8 graded 0 (teacher 1); the Q8 items were graded under Q3. |
| SE_08_Q1_0020 Q4 (not flagged) | **Real repeat.** Page 7: the table crossed out by one long black diagonal stroke (plus a red teacher stroke). Page 8: the rewrite. | Team list. |
| SE_08_Q1_0027 Q7 (not flagged) | **Real repeat.** Page 7 is labelled "Rough" and crossed by one diagonal stroke; its header is struck too. Page 8 is the final summary. | Team list. |
| SE_08_Q1_0036 Q12 | The student wrote "no. 12" above a dialogue. The dialogue is Q13. Not a reading error. | The start of the Q13 dialogue is graded under Q12. |
| SE_08_Q1_0014 Q2 | Header "no 9" read as "no 2". The rearrangement below is Q9. Its first story paragraph is crossed by a long stroke, and the Opus text tags it `[struck: …]`. | The segmenter already moved it to Q9 by content. |
| SE_08_Q1_0023 Q1 | Header "no. 11" read as "no.1)". The story below is Q11. | The segmenter already moved it to Q11 by content. |
| SE_08_Q1_0008 Q8 | Page 15, the top of the second answer sheet: a paragraph on deforestation with no header, after Q12. Page 16: the real Q8 header. | The paragraph is graded with Q8 (3, teacher 5). |

The other class-8 candidates in the cloud's file (0003, 0013, 0018, 0030, 0031, 0035) came from the Gemini text.
The flag doesn't fire on them in the Opus text.

**Why the flag misses the two real class-8 repeats:** both copies sit on consecutive pages. Each copy has its own
header, but the parts form one run of the same question, and the flag only counts separate runs. The team-check code
on `review-followups` (`_repeated_attempts`) uses the same run rule, so it misses them too.

A header repeated inside one run is not always a repeat. I listed every such case in the dev Opus text
(`consec_headers.py`, section 6). There are 7:

| Case | What it is |
|---|---|
| SE_08_Q1_0020 Q4, SE_08_Q1_0027 Q7 | real repeats (above) |
| SE_06_Q1_0011 Q2 | a continuation page with the header written again: items a–c on page 2, d–e on page 3 |
| SE_08_Q1_0036 Q12 | the student's wrong number above the Q13 dialogue (above) |
| SE_08_Q1_0008 Q8 | a headerless paragraph filed under Q8 by content (above) |
| SE_07_Q1_0018 Q13 | pages 12–13 continue the Q12 paragraph ("…present at the function"). The segmenter switched to Q13 by content at the page start (evidence 2.87 vs 0.87) |
| SE_10_Q1_0020 Q11 | the student swapped the numbers: "10" above the Q11 dialogue, "11" above the Q10 story |

So "the same header again inside one run" would find both real repeats among 7 cases. The other 5 are a
continuation page and 4 splitting errors. Word overlap between the two parts doesn't separate them (59% on the
continuation page, 22% and 56% on the two repeats). See question 2 in section 5.

### 2d. Classes 6, 7 and 10 cases the cloud could not see

The cloud had no class 6, 7 or 10 page images. I checked its two "unsure" cases on the scans:
- **SE_07_Q1_0014 Q9, copy 1 (page 6):** the top of the page is the end of a first Q9 story ("…Austin car. I waved
  my torch…"), not crossed out. Page 5 ends with Q8, so the start of this copy is not on the scanned pages. It is a
  real, partial first attempt.
- **SE_07_Q1_0025 Q5, copy 2 (page 15):** "5. Ans: The teacher describe his role in the poem…" is item 5 of the poem
  questions (Q10), read as a Q5 header. It is not a copy of Q5.

Both notes are in the team list.

---

## 3. Opus grades vs teacher and vs Gemma (dev split)

Script: `grade_compare.py` (section 6). Output: `outputs/review_2026_10/cpu_2026_10_25/opus_grades/compare.md`.

**What is compared:**
- **Opus:** Opus 5.5 grading the Opus text (`outputs/evaluated_opus`).
- **Gemma, pipeline text:** Gemma grading the pipeline's own text (`outputs/evaluated_pipeline`).
- **Gemma, Gemini text:** Gemma grading the imported Gemini text (`outputs/evaluated_review`).

Each Gemma source is compared with Opus on the same questions: both scored, with a teacher mark. Each difference
comes with a 95% interval from a paired bootstrap over scripts. **Teacher marks are the reference for Mode A
(objective items) only.** The B and C rows only describe how far each grader is from the teacher.

**What differs between the runs:**
- **Model:** Opus 5.5 vs Gemma.
- **Text:** the Opus text vs the pipeline or Gemini text.
- **Code:** `review-followups` vs older `gemma` code.

So a difference mixes all three. Step 4 (Gemma grading the Opus text) separates the model from the text.

### Mode A, the objective items

| Class | n | Opus MAE / bias | Gemma, pipeline text | Opus − Gemma (95% CI) | n | Opus MAE / bias | Gemma, Gemini text | Opus − Gemma (95% CI) |
|---|---|---|---|---|---|---|---|---|
| 6 (15 scripts) | 85 | 0.71 / +0.26 | 0.78 / +0.09 | −0.07 [−0.26, +0.10] | 86 | 0.72 / +0.23 | 0.73 / +0.10 | −0.01 [−0.14, +0.13] |
| 7 (24 scripts) | 157 | 0.66 / −0.03 | 0.89 / −0.27 | **−0.22 [−0.39, −0.07]** | 158 | 0.66 / −0.03 | 1.11 / −0.47 | **−0.46 [−0.67, −0.26]** |
| 8 (24 scripts, Q9 left out) | – | – | only 4 dev scripts graded | – | 143 | 0.50 / −0.25 | 0.68 / −0.43 | **−0.17 [−0.29, −0.08]** |
| 10 (18 scripts) | – | 0.39 / +0.03 (196 questions, all modes) | no Gemma grades with this code | – | | | | |

- **Opus is closer to the teacher on objective items in classes 7 and 8** (the interval excludes 0), and level on
  class 6.
- **Gemma marks objective items lower than the teacher** on classes 7 and 8 (bias −0.27 to −0.47); Opus barely
  does (−0.03 on class 7).
- **Class 8 Q9 is left out** of this table, because the two runs used different answer keys (section 4).

### Writing and point-scored questions (descriptive)

| Class | Mode | Opus MAE / bias | Gemma, Gemini text MAE / bias | Opus vs Gemma MAE |
|---|---|---|---|---|
| 6 | B (points) | 1.37 / −0.59 | 1.41 / +0.29 | 1.07 |
| 6 | C (bands) | 0.77 / −0.09 | 0.90 / +0.11 | 0.57 |
| 7 | B | 1.38 / −1.18 | 1.41 / −0.94 | 0.72 |
| 7 | C | 1.29 / −1.20 | 1.12 / −0.98 | 0.66 |
| 8 | B | 1.85 / +0.20 | 1.74 / +0.56 | 1.10 |
| 8 | C | 0.95 / −0.52 | 0.92 / −0.41 | 0.62 |

- **Opus and Gemma are about as far from the teacher on writing.** Both mark class-7 writing about 1 mark below the
  teacher.
- **The two models differ from each other** by 0.6–1.1 marks per question. That is about as much as each differs
  from the teacher.
- **The expert marks (step 5) are what can judge writing.** Teachers often mis-mark writing, so matching them there
  is not the goal.

### Whole scripts (dev; a missing section counts as 0)

| Class | Scripts | Opus total MAE / bias | Gemma, Gemini text | Gemma, pipeline text | Missing sections the teacher marked (Opus / Gemini / pipeline) |
|---|---|---|---|---|---|
| 6 | 15 | 4.83 / −2.37 | 4.37 / +0.90 | 4.47 / −0.27 | 6 / 2 / 7 |
| 7 | 24 | 8.81 / −8.60 | 9.17 / −9.17 | 8.62 / −8.62 | 10 / 2 / 4 |
| 8 | 25 (Gemini: 24) | 7.86 / −7.22 | 7.06 / −5.02 | (4 scripts) | 4 / 2 / – |
| 10 | 18 | 4.67 / +1.39 | – | – | 0 / – / – |

- **Missing sections cost Opus more than Gemma:** 6 in class 6 and 10 in class 7, against 2 for the Gemini text.
  These are the question-splitting misses on the Opus text in `NEXT_STEPS_2026-10-25.md` section 2c: markup inside
  header lines and header forms the patterns don't know. Step D (the segmenter draft) works on them.
- **Class 8's total is pulled down by Q9** (section 4).

---

## 4. SE_08 Q9: the answer key looks wrong

Q9 asks the student to put eight story sentences in order. The key in `configs/answer_keys/SE_08_Q1.yaml` is
`e b d h a c g f`, scored by position. Its comment says the sequence was confirmed on 2026-10-22, and that the
teacher gave 8 to nearly every script "whatever the order".

The class-8 data, all 25 dev scripts (script: `q9_key_check.py`; output: `opus_grades/q9_key_check_dev.txt`):
- **20 of 25 students wrote `e b d h a f c g`**, and the teacher gave all 20 full marks (8).
- **The teacher's marks follow `e b d h a f c g` by position:**

  | Scoring | MAE vs teacher | Bias | Equal to the teacher |
  |---|---|---|---|
  | by position, key `ebdhacgf` (what the cloud used) | 3.04 | −2.96 | 0 of 25 |
  | by relative order, key `ebdhacgf` | 1.28 | −0.88 | 0 of 25 |
  | **by position, `ebdhafcg`** | **0.44** | −0.44 | **22 of 25** |
  | by relative order, `ebdhafcg` | 0.32 | −0.08 | 20 of 25 |

- **The 5 dev scripts with another order:**
  - two get exactly the position count against `ebdhafcg` from the teacher: 0007 `ebacgfdh` gets 3, 0026 `ebdhacfh` gets 5;
  - two are reading gaps in the Opus text: 0017 has only 5 letters read, and 0034 has one letter unread;
  - 0008 `ebdhacfg` got 8 from the teacher, where position would give 6.
- **The test-split scripts point the same way** (looked at once, not used for anything else):
  - 0006 `ebdhacfg` gets 6;
  - 0010 `ehbdafcg` gets 5;
  - 0012 `ebhdafca` gets 5.

  All three equal the position count against `ebdhafcg`. Over all 35 scripts it is 32 of 35.
- **The story reads naturally in that order too:** he finds the purse (a), thinks a passer-by dropped it (f), picks
  it up and carries it home (c), and shows it to his wife (g).

So the teacher marked by position against `e b d h a f c g`. "Whatever the order" holds only for the students who
wrote that order. With the current key, Opus loses 3 marks on almost every class-8 script, and that is most of the
class-8 total bias.

Objective questions are where the teacher is the reference. **I have not changed the key.** It is your decision, as
question 1 in section 5.

---

## 5. Questions for you

1. **SE_08 Q9 key:** change `correct_sequence` to `e b d h a f c g`, still scored by position? This would change
   `configs/answer_keys/SE_08_Q1.yaml` on `review-followups`. The class-8 Q9 grades are then recomputed in code from
   the items Stage 4 already parsed, with no new model calls. Class-8 grades from the old key need the same.
2. **Repeats on consecutive pages:** the team-check code (`_repeated_attempts`) misses a rewrite on the next page,
   as in 0020 Q4 and 0027 Q7. Counting every header repeated inside one run would find both real repeats among 7
   cases (section 2c). Should the team list include these cases, with the false alarms? Or should the team-check
   code wait for the segmenter fix, which removes 4 of the 5 false alarms (they are splitting errors)?
3. **Step 4 output folder:** `docs/NEXT_STEPS_2026-10-23.md` step 4 writes Gemma's grades of the Opus text to
   `outputs/evaluated_opus/english/`. That folder now holds the cloud's Opus grades. Step 4 should write to a new
   folder, for example `outputs/evaluated_opus_text_gemma/english/`. Its input can be
   `/mnt/models/script_checking/opus_cloud_results/results/opus_inputs/english/` (already imported).

---

## 6. Re-running the checks

From the repo root. The scripts are in `outputs/review_2026_10/scratchpad_backup/`. None of them write into `data/`
or `src/`.

```bash
B=outputs/review_2026_10/scratchpad_backup
O=outputs/review_2026_10/cpu_2026_10_25
# repeat flag on the class-8 Opus text (section 2c)
./script_checking/bin/python $B/dup_scan.py SE_08_Q1 data/ground_truth/transcripts/SE_08_Q1_extraction_opus_5.5 --details
# headers repeated inside one run of parts (section 2c), for each class
./script_checking/bin/python $B/consec_headers.py SE_08_Q1 data/ground_truth/transcripts/SE_08_Q1_extraction_opus_5.5
# team list from the cloud's answered_twice.csv (section 2b); the scan notes were added afterwards by hand
./script_checking/bin/python $B/team_list_from_cloud.py \
    /mnt/models/script_checking/opus_cloud_results/results/answered_twice.csv outputs/team_check/opus_cloud.csv
# grade comparison (section 3) and the Q9 key check (section 4; --all includes the test split)
./script_checking/bin/python $B/grade_compare.py $O/opus_grades/compare.md
./script_checking/bin/python $B/q9_key_check.py
```

Outputs:
- `outputs/review_2026_10/cpu_2026_10_25/repeat_scan/opus_08.txt`, `consec_opus.txt`;
- `outputs/review_2026_10/cpu_2026_10_25/opus_grades/compare.md`, `q9_key_check_dev.txt`, `q9_key_check_all.txt`;
- `outputs/team_check/opus_cloud.csv`.

Note: re-running `team_list_from_cloud.py` overwrites the scan notes in `opus_cloud.csv`. The notes are also in
section 2.
