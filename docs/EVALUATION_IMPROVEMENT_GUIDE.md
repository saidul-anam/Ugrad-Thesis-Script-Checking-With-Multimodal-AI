# Evaluation Improvement Guide: grading that an expert would accept

Written 2026-09-15. This guide explains, step by step, how to improve Stage 4 (marking the answers) and how to
prove the result in the thesis. It builds on the extraction guide (`docs/EXTRACTION_ACCURACY_GUIDE.md`): marking
can only be as good as the text it reads.

---

## 0. The new idea

Classroom teachers mark the **objective** questions reliably, because each item is simply right or wrong.
They are **not reliable on the writing questions**. Reducing that gap is one of the purposes of AI-assisted
evaluation. So the reference ("the correct mark") depends on the question type:

| Group | Questions | Marks | Reference mark | Expert involved? |
| --- | --- | --- | --- | --- |
| Objective | 1(A) multiple choice, 4 fill in blanks from a box, 5 fill in blanks without a box, 6 rearrange sentences | 30 | The teacher's mark | No |
| Borderline | 1(B) short answers from the passage, 2 flow chart | 20 | Decided by an expert check of the teachers (Step 5.2) | Yes, as a check |
| Writing | 3 summary, 7 paragraph, 8 chart, 9 story, 10 letter, 11 theme | 50 | The expert's mark | Yes, marks the answers |

**Why 1(B) and 2 are borderline.** The answers come from the passage and a key exists, but the student writes
sentences or notes and the marker gives part marks (0 to 2 per item). That needs some judgement, so an expert
checks whether the teachers' part marks can be trusted.

**What success means now.** Being close to the teachers is no longer the goal on writing questions. The thesis
shows two things:

1. **The AI is closer to the expert than the teachers are.**
2. **The AI works as a second marker.** When the AI disagrees with a teacher, it points at the teacher marks an
   expert would reject, so a reviewer checks a few answers instead of all of them.

---

## Current status (checked on 2026-09-15)

| Item | Finding |
| --- | --- |
| English scripts available | 24 PDFs; 4 extracted (`0002`, `0010`, `0011`, `0015`) |
| Typed teacher marks in `gt.txt` | 4 scripts |
| Expert marks | None yet |
| Earlier result vs teachers | MAE 1.02 on 2 scripts, 22 questions. Measured against teachers, so it is no longer the right target for writing questions |
| Question splitting (segmentation) | Script `0015`: only 4 of 12 answers found, because headers say "Ans to the **que** no: 3" and one says "**due** no: 7". Script `0010`: saved split is older than a segmenter fix and is missing 1(B) |
| Typed marks vs red margin marks (read by Stage 0b) | Disagree on `0002` Q2 (typed 10, margin 6); `0011` Q3 (6 vs 3), Q4 (5 vs 3), Q5 (3 vs 5); `0015` Q2 (3 vs 2), Q7 (5 vs 8), Q10 (3 vs 2) |
| Margin reading errors | `0010`: margin read as "4(A) = 6", impossible for a 5-mark question |
| Objective scoring bug | The spelling tolerance accepts a different real word: "healthy" for "health", "right" for "rights" |
| Rearrangement (Q6) | AI below the teacher on all 3 graded scripts (9 vs 10, 1 vs 3, 1 vs 3). On `0002` the 1-mark gap is a misread: "j" was read as "o" |
| Which questions count as objective | Two parts of the code disagree. The older helper counts Q2 as objective but not 1(B), and also matches keywords such as "table". The rubric scoring treats 1(B) and 2 as part-mark questions |
| Evidence quotes in writing marks | Requested from the model but never checked |

---

## Rules that apply to every step

1. **Split the scripts once, at the start.** Put most scripts in a *development* set (used to improve things)
   and the rest in a *test* set (used once, at the end). Split by script, never by question, so a student's
   handwriting and style never appear on both sides. Suggestion for 24 scripts: 16 development, 8 test, chosen at
   random. Write the list down and never move a script between sets.
2. **Decide limits before looking at results.** The tolerance for "acceptable" marks and the rule in Step 5.2 are
   agreed first. Choosing them after seeing the numbers makes any result look good.
3. **The expert always marks blind.** No teacher mark, no AI mark, no student identity.
4. **Keep the three kinds of marks separate.** Typed teacher marks, margin marks read by the machine, and expert
   marks live in separate lists. Never overwrite one with another.
5. **Log every change.** One line per experiment in `docs/evaluation_log.md`:

| Date | Scripts | Change | Objective exact-match with teacher | AI vs expert MAE | Teacher vs expert MAE | Catch rate | False-alarm rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-xx | dev | baseline | | | | | |


---

## Step 1 — Make sure every answer is filed under the right question

### Why
Grading happens question by question. If a header is missed, one answer is merged into another and both get
the wrong mark. On script `0015`, headers written as "que no" were not recognised, so the summary "answer" holds
475 words from several questions. Every mark computed from that is meaningless, and no grading improvement can
fix it.

### What you do
**1.1 Extract the remaining scripts.** Grading needs more than 4 scripts.
```bash
python scripts/extract_scripts.py --lang english --non-interactive
```
Already-extracted scripts are skipped.

**1.2 Collect every header line from all transcripts.** Look at how students actually write headers:
"Ans to the Question No - 01 (B)", "Ans:to the Q. No. 1", "Ans to the que no:3", and misreads such as "Ansito"
or "due no". The header recogniser must accept all of them. A robust approach is: a line that starts like
"Ans" and contains a number matching a question on the paper is a header, checked against the order of the
questions on the paper.

**1.3 Re-split all scripts after any change to the recogniser.** This does not re-run the model:
```bash
python scripts/resegment_extraction.py --lang english
```

**1.4 Check the split for every script.** Red flags:

| Red flag | Example |
| --- | --- |
| A question has a teacher margin mark but no answer | `0015`: Q4, Q5, Q6 marked by the teacher but not found |
| One answer is far longer than usual | `0015` Q3 summary with 475 words |
| An objective answer has no items | Q4 found, but no (a) to (j) inside |
| Two answers in the wrong order | Q9 text found under Q3 |

### Effort and done-criteria
- Your time: about 1 hour to review header lines. Code: half a day.
- Done when: on every script, the questions found match the questions the teacher marked.
- Tools: re-split script **exists**. Header collection report and split check report **to be built**. Header fix **to be built**.

---

## Step 2 — Clean the teacher marks

### Why
For objective questions the teacher's mark is the reference, so a typing slip corrupts the correct answer
itself. For writing questions the teacher's mark is what you compare the expert against, so a slip distorts the
"teacher gap". This check needs no expert: reading a red number is easy.

### What you do
**2.1 Put typed and margin marks side by side** for every script and question.

**2.2 Look at every disagreement on the page yourself** and correct `gt.txt`. Current cases to check:

| Script | Question | Typed | Margin (machine-read) |
| --- | --- | --- | --- |
| 0002 | 2 | 10 | 6 |
| 0011 | 3 | 6 | 3 |
| 0011 | 4 | 5 | 3 |
| 0011 | 5 | 3 | 5 |
| 0015 | 2 | 3 | 2 |
| 0015 | 7 | 5 | 8 |
| 0015 | 10 | 3 | 2 |

The machine can misread margins too (`0010` "4(A) = 6"), so the page decides, not the machine.

**2.3 Run simple sanity checks.**
- A mark above the question's maximum.
- A mark that the item values cannot produce, such as 4.25 on Q4 where each blank is worth 0.5.
- A question with a mark but no answer, or an answer but no mark.

**2.4 Type the marks for all scripts**, then propagate them:
```bash
python scripts/sync_ground_truth.py
```

### Effort and done-criteria
- Your time: about 2 minutes per script.
- Done when: no typed-vs-margin disagreement is left unexplained.
- Tools: sync **exists**. Side-by-side comparison and sanity-check report **to be built** (small).

---

## Step 3 — Make objective scoring match the teachers

### Why
These 30 marks should be almost pure arithmetic. When the AI disagrees with a trusted teacher, there is a
concrete, fixable cause.

### What you do
**3.1 Fix the spelling tolerance.** Forgiving a slip is fair when the student clearly meant the right word:
"atain" for "attain". It is wrong when the student wrote a *different real word*: "healthy" instead of "health",
"right" instead of "rights". The grammatical form earns the mark on these questions. Change: forgive a slip only
when the student's word is **not itself a dictionary word**. This is a dictionary lookup, not a letter rule.

**3.2 Find out which items the teacher accepted.** `gt.txt` holds only a total per question. Teachers tick or
cross each blank in red, so for every question where the AI total differs from the teacher total, look at the
ticks on the page and write down the teacher's decision per item. This turns a vague "3 vs 5" into exact item
differences.

**3.3 Complete the answer key from those decisions.** The key in `configs/answer_keys/SE_11_Q1.yaml` is a draft.
- An answer the teacher accepted but the key rejects is added, with a note of which script it came from.
- An answer the key accepts but teachers reject is removed.
- Q5 matters most: the rubric says "generous on vocabulary, strict on grammatical fit", so its list must come
  from what teachers really accept.
- Mark the key `verified: true` when every objective item has been checked this way on the development scripts.

**3.4 Confirm the rearrangement scoring method.** In NCTB board marking, examiners evaluate the rearrangement table
slot-by-slot (position-by-position):
1. 1 mark is awarded for each slot whose letter matches the correct sequence (positions 1 to 10).
2. There is no cascading domino penalty: an error or omitted item in an earlier slot does not break or zero out
   subsequent correct slots.
3. Confirm the key sequence itself: a script the teacher gave full marks (e.g. `0002` Q6 = 10) confirms the accepted order.

**3.5 Use one list of objective questions everywhere.** The rubric's list decides which questions are objective,
borderline or writing. The older helper that switches off language deductions must read the same list instead of
its own numbers and keywords.

**3.6 Explain every remaining disagreement.** For each objective question where AI and teacher still differ, write
one cause:

| Cause | Example | Fixed in |
| --- | --- | --- |
| Reading mistake | `0002` Q6: "j" read as "o" | Extraction guide |
| Question-splitting mistake | Q4 answers merged into Q3 | Step 1 |
| Answer key gap | Teacher accepts a word the key lacks | Step 3.3 |
| Scoring method | Rearrangement convention | Step 3.4 |
| Typing slip in marks | Q4 and Q5 swapped | Step 2 |
| Teacher slip | Adding up the ticks wrongly | Note it; keep the teacher mark as the reference only if the ticks support it |

### How you measure
```bash
python scripts/evaluate_scripts.py --lang english --non-interactive --force-evaluate --eval-mode modular
python scripts/benchmark_evaluation.py --lang english --tag objective_v1
```
Read the rows for Q1(A), Q4, Q5 and Q6: exact-match rate with the teacher and mean absolute error.

### Effort and done-criteria
- Your time: about 5 minutes per script for the tick check. Code: half a day.
- Done when: on the development scripts, every objective disagreement has a cause, and no cause is
  "unexplained".
- Tools: evaluation and benchmark **exist**. Tolerance fix, method comparison and disagreement table **to be built**.

---

## Step 4 — Prepare the expert's work

### Why
The expert's time is the scarcest resource in the project. A careful setup makes every hour count, and makes the
expert's marks defensible in the thesis.

### What you do
**4.1 Choose and describe the expert.** Ideally someone with board-examiner or head-examiner experience in HSC
English. Record their background; the thesis needs it to justify "expert".

**4.2 Hold one standardisation session, about an hour.** Before marking anything, the expert:
- reads the rubric and confirms or changes the criteria ceilings, hard caps and bands for each writing question.
  Some parts of the rubric came from syllabus documents rather than real marking;
- confirms the expected key points for 1(B) (a to e) and the flow-chart content and order for Q2;
- decides the **tolerance**: how far a mark may be from the expert's own and still be acceptable, per question
  type. Example only: "within 1 mark on a 10-mark question, and in the same band". The expert decides the actual line;
- agrees the **decision rule for borderline questions** in Step 5.2, before any marking.

Any rubric change made here is applied before the AI is evaluated, and never changed after seeing test results.

**4.3 Build blind marking packets.** One packet per question, e.g. "all Q7 answers". Examiners mark more
consistently question by question than script by script.

Each answer in the packet contains:
- the question text;
- the student's handwritten answer as page images: this is the real answer;
- the machine transcript, clearly labelled "machine reading, may contain errors", as a reading aid only;
- an answer code instead of the student's name or script number;
- no teacher mark and no AI mark.

Answers are shuffled across scripts.

**4.4 Prepare the marking sheet.**

| Question type | The expert records |
| --- | --- |
| Writing | Score for each of the four criteria; yes/no for paragraph split, missing letter layout parts, copying from the source, outside facts or personal opinions in the chart answer; total; band; one-line comment |
| Borderline | Mark per item on the rubric scale (0, 0.5, 1, 1.5, 2 for 1(B); 0, 1, 2 for flow-chart boxes); one-line comment when not full marks |

The yes/no answers let you check later whether the AI applies the hard caps correctly.

### Effort and done-criteria
- Expert time: about 1 hour for the session.
- Done when: rubric confirmed, tolerance and decision rule written down, packets and sheet ready.
- Tools: packet builder and expert-marks sheet format **to be built**.

---

## Step 5 — Expert marking

### 5.1 Writing questions (3, 7, 8, 9, 10, 11)
The expert marks every attempted writing answer in the chosen scripts, blind, question by question.

Rough workload (estimates; measure the real time in the first batch):

| Batch | Answers | Estimated expert time |
| --- | --- | --- |
| First batch: 10 scripts | up to 60 writing answers | 3 to 5 hours |
| All 24 scripts | up to 144 writing answers | 7 to 12 hours |

Start with the first batch, taken from both the development and test sets, then continue.

### 5.2 Borderline questions (1(B) and 2): check the teachers first
The expert does not mark every borderline answer straight away. The first job is to find out whether teachers'
part marks can be trusted.

**Stage A — random check.** The expert blind-marks every item of 1(B) and every flow-chart box for a **random**
sample of scripts, for example 8 scripts: 40 short answers and 40 boxes.

**Stage B — decision, using the rule agreed in Step 4.2.** Compare the teacher's total with the expert's total
on each sampled answer.

| Result on the random sample | What happens next |
| --- | --- |
| Teachers are within tolerance often enough, e.g. at least 9 answers in 10 | Teacher marks become the reference for 1(B) and 2. The expert only resolves future cases where the AI and the teacher disagree by more than the tolerance |
| Teachers are not | 1(B) and 2 are treated like writing questions: the expert marks all of them |

**Important.** Only the random sample decides. Answers chosen because the AI and the teacher disagree are useful
to study, but they overstate teacher errors, so they must never be used to judge the teachers.

### 5.3 Re-marking for consistency
One or two weeks later, mix about 10 to 15 % of the answers back into a new packet without telling the expert. The
agreement between the expert's two markings is the realistic ceiling: no marker, human or AI, can be expected to
agree with the expert more closely than the expert agrees with themselves. If a second expert can mark about 20
answers, that is even stronger evidence.

### 5.4 Reasons for unacceptable teacher marks
Only after blind marking is finished, show the expert the teacher marks. For each teacher mark outside the
tolerance, the expert writes one line: which criterion the teacher ignored or over-rewarded. Group the reasons
afterwards; do not invent categories in advance. These groups describe the teacher gap, which is a thesis finding
in itself.

### Done-criteria
- Expert marks for writing answers on the chosen scripts.
- Borderline decision made and recorded.
- Re-marks collected.
- Reasons for every unacceptable teacher mark.

---

## Step 6 — Measure the baseline: three comparisons

### Why
Before improving the AI, record where it stands. Every later step is judged against these numbers, on the
development scripts only.

### What you measure

| Comparison | Questions | Metrics |
| --- | --- | --- |
| AI vs teacher | Objective | Exact-match rate, mean absolute error |
| Teacher vs expert | Writing (and borderline if Stage 5.2 said so) | Mean absolute error, within-tolerance rate, bias (teacher higher or lower), band agreement, weighted kappa |
| AI vs expert | Writing and borderline | The same metrics |
| Expert vs expert (re-marks) | Re-marked answers | The same metrics; this is the ceiling |
| AI as second marker | Writing and borderline | See below |

**How the second-marker numbers work.** The AI "flags" a teacher mark when the AI and the teacher differ by more
than the tolerance. The expert says which teacher marks are unacceptable.

| Metric | Meaning |
| --- | --- |
| Catch rate | Of the teacher marks the expert rejects, the share the AI flagged |
| False-alarm rate | Of the teacher marks the expert accepts, the share the AI flagged |
| Review share | Share of all answers flagged, i.e. how much a reviewer must re-check |

Report everything per question and per criterion, not only as one overall number.

### Effort and done-criteria
- Code: about a day. The current benchmark compares only with `gt.txt`; it must also read expert marks, compute
  the three comparisons and the second-marker numbers.
- Done when: the baseline line is in the log.
- Tools: benchmark **exists**, expert comparison **to be built**.

---

## Step 7 — Find out where the AI's marking mistakes come from

### Why
An AI mark can be wrong for three different reasons: the text was misread, the answer was filed under the wrong
question, or the judgement was wrong. Each has a different fix.

### What you do
**7.1 Transcript swap test.** For scripts with hand-corrected pages (extraction guide, Step 1), grade the same
answers twice: once with the corrected text, once with the machine text. Compare both with the expert. The
difference is caused by extraction alone.
```bash
python scripts/benchmark_evaluation.py --lang english --eval-dir outputs/evaluated/english --compare outputs/evaluated_corrected_text/english --tag transcript_swap
```
The comparison exists; building the copy with corrected text is **to be built**.

**7.2 Count splitting mistakes separately** (Step 1). They are not marking errors.

**7.3 Compare per criterion.** Find the criterion where the AI is furthest from the expert. For example, if
*language mechanics* is off while *content* is close, look at the error list the grader receives from Stage 3
and Stage 3b.

**7.4 Check the language evidence.** Words marked *unclear* or cleared as handwriting ambiguity must not count
as student errors. If the AI's language scores are much lower than the expert's, compare them with the number of
confirmed errors per answer.

### Done-criteria
- A table of AI mistakes by cause: reading, splitting, judgement, per criterion. It tells you which step to work
  on next.

---

## Step 8 — Teach the grader with expert-marked examples

### Why
Real examiners are trained by studying answers the head examiner has already marked, across the bands. Giving
the AI the same examples pulls its marks toward the expert's standard in both directions, for example less
generosity on the story question and a fairer view of the flow chart.

### What you do
**8.1 Pick examples from development scripts only.** For each writing question, choose a few expert-marked
answers: one low, one middle, one high band. Include the expert's criterion scores and one-line comment.

**8.2 Put them in the grader's prompt for that question**, introduced as answers marked by a senior examiner.

**8.3 Never leak.**
- An example must never come from the script being graded.
- An example must never come from the test set.
- On development scripts, measure with *leave-one-script-out*: when grading a script, its own answers are removed
  from the example pool.

**8.4 Find the smallest set that works.** Try 0, 1 per band and 2 per band. Longer prompts cost time on the
4-bit model and can crowd out the student's answer.

**8.5 Borderline questions.** If Step 5.2 put the expert in charge of 1(B) and 2, add examples of short
answers worth 0, 1 and 2 marks in the same way.

**8.6 Turn repeated expert comments into rubric wording.** If the expert's comments keep repeating one point,
for example "copies the question instead of answering it", clarify that criterion's description in the rubric.
Do not add rules for individual answers.

### How you measure
AI vs expert mean absolute error and within-tolerance rate, before and after, on development scripts. Log each try.

### Effort and done-criteria
- Code: half a day. GPU: one grading run per try.
- Done when: the best example setting is chosen on development scripts.
- Tools: **to be built**.

---

## Step 9 — Let the grader say "not sure", and check its evidence

### Why
Some answers are hard for any marker. A grader that admits doubt and sends those answers to a person is more
useful, and more honest, than one that is always confident. This is the same idea as the "unclear" words in the
extraction guide.

### What you do
**9.1 Grade each writing answer three times** with small variations: slightly different sampling, or the rubric
criteria listed in a different order. Keep the middle total and the middle score for each criterion.

**9.2 Flag the answer if** the three totals are further apart than the expert's tolerance, or if they fall in
different bands.

**9.3 Check the evidence quotes.** For each criterion the grader must quote the student's answer. The code checks
that every quote really appears in the answer. In an earlier run, the "evidence" for one theme answer was only
the question header, and that answer scored zero. A missing or invented quote triggers one regrade; if it still
fails, the answer is flagged.

**9.4 Flag the answer if the AI and the teacher differ by more than the tolerance.** This is the second-marker flag.

**9.5 Send flagged answers to a reviewer.** During the project that is the expert; in real use it would be a
senior teacher.

**9.6 Choose how strict the flag is.** On development scripts, vary the flag setting and plot two numbers:

| Number | Meaning |
| --- | --- |
| Coverage | Share of answers not flagged |
| Accuracy on unflagged answers | How close those AI marks are to the expert |

A stricter flag gives higher accuracy and lower coverage. Pick the setting you can defend, then confirm it on the
test set. That plot is a thesis figure.

### Effort and done-criteria
- Code: about a day. GPU: three grading runs per writing answer.
- Done when: every writing answer has a status (accepted or flagged), and the setting is chosen on development
  scripts.
- Tools: **to be built**.

---

## Step 10 — Check for unfair patterns

### Why
AI graders have known habits that can look like good accuracy on average while being unfair to some students.

### What you check
| Check | Question it answers |
| --- | --- |
| Length | Does the AI give extra marks to longer answers, compared with the expert? |
| Handwriting quality | Do scripts with more unclear words get lower AI marks than the expert gives? That would mean the AI punishes messy handwriting |
| Hard caps | When the AI applied a cap, did the expert's yes/no answers agree? And when the expert said a cap applies, did the AI apply it? |
| Question type | Is the AI consistently stricter or more generous on one question type? |

### Done-criteria
- Each check reported on development scripts, with a short explanation of any pattern found and what was changed.

---

## Step 11 — Final test and reporting

### What you do
1. **Freeze everything**: prompts, examples, answer key, rubric, flag setting.
2. **Run once on the test scripts.** No changes after this run. If a bug forces a change, say so in the thesis.
3. **Report these tables:**

| Table | Content |
| --- | --- |
| Objective marking | AI vs teacher exact-match and error, per question |
| Writing and borderline marking | Teacher vs expert, AI vs expert, expert vs expert |
| Second marker | Catch rate, false-alarm rate, review share |
| Doubt flag | Coverage and accuracy on accepted answers |
| Error sources | Reading, splitting, judgement, per criterion |
| Ablations | With and without expert examples; with and without the doubt flag; question-by-question vs whole-script grading |
| Fairness checks | Length, handwriting quality, caps, question type |

4. **Give uncertainty ranges.** With 8 test scripts the ranges will be wide. Say so plainly; it is more convincing
   than hiding it.

---

## Summary

| Step | What | Who | Your or expert time | Tools |
| --- | --- | --- | --- | --- |
| 1 | File every answer under the right question | You + code | ~1 h | Partly built |
| 2 | Clean the teacher marks | You | ~2 min per script | Partly built |
| 3 | Make objective scoring match teachers | You + code | ~5 min per script | Partly built |
| 4 | Prepare expert work: rubric, tolerance, blind packets | Expert + code | ~1 h expert | To build |
| 5 | Expert marks writing; checks borderline; re-marks | Expert | 4 to 12 h, in batches | To build |
| 6 | Baseline: three comparisons, second marker | Code | 0 | To build |
| 7 | Where AI mistakes come from | Code | 0 | Partly built |
| 8 | Expert-marked examples in the grader | Code + GPU | 0 | To build |
| 9 | Doubt flag and evidence check | Code + GPU | 0 | To build |
| 10 | Fairness checks | Code | 0 | To build |
| 11 | Final test and report | You + code | ~1 day | Partly built |

**Start now with Steps 1 to 3.** They need no expert and fix the largest current errors. At the same time,
arrange the expert, because Steps 4 and 5 depend on their schedule.
