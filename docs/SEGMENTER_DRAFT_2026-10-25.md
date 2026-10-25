# Segmenter fix (step 6), draft for your OK (2026-10-25)

This is step D of `docs/NEXT_STEPS_2026-10-25.md`, section 4. **It is a draft: no code in `src/` or `scripts/` has
changed on any branch.** The code is in a scratch copy, and the whole diff is in
`docs/SEGMENTER_DRAFT_2026-10-25.patch`. It is applied only after you say yes (section 9).

The segmenter splits a script's text into questions: it decides which question each piece of text answers. When it
gets this wrong, a teacher-marked answer gets no text (and scores 0), or text is graded under the wrong question.

---

## 1. In short

The draft adds three things. The two learned models are trained on the dev scripts only, and nothing is set by hand:

| Part | What it does |
|---|---|
| **D1** | Reads a header that transcription markup hides: "Answering to the [unclear: Quostion] No 2", "Q NO.[struck: illegible]5". |
| **D2** | A **learned header reader** for headers the patterns don't know: the student's shorthand ("A: to the question No.5"), misspellings ("Ansuler No.12"), Bengali ("9 নং"), and headers in the middle of a line after answer text ("courage. Answers to the Q No: 6"). |
| **D3** | A **learned content model**: which question a piece of text answers, learned from the other students' answers to the same paper. It decides when a written header is wrong (a misread or miswritten number, an item numbered like a question), and it stops structural evidence from moving text away from the question it clearly answers. |

**Results** (Opus, Gemini and pipeline texts of classes 6, 7, 8 and 10; class-10 transcription test pages
0029–0033 left out):

- **Teacher-marked questions that get no text:**
  - dev split: Opus 20 → 7, pipeline 15 → 5, Gemini 6 → 5;
  - test split: Opus 3 → 1, pipeline 12 → 7, Gemini 0 → 0.
- **Do the three texts of a script file the same words under the same question?** On dev, Opus vs pipeline goes
  from 90.1% to 92.0%. The most a perfect splitter could reach is 93.7%, because the readings differ, so the draft
  closes more than half of the gap. This holds for every pair of texts, on dev and on test, and the number rises at
  every step (section 4b).
- **Changes:** 266 questions change in 100 scripts (counting each text of a script separately).
  - D3's changes are all listed and judged in section 5: 27 of the 30 script texts are right; the other 3 are
    partly wrong.
  - The D2 header reads are judged in section 5 too: 97 of 99 are real headers.
- **Tests:** all 366 tests pass on the draft with both models on.
- **Questions answered twice:** the team-check code's flags go from 46 to 23. 22 of the 23 that go were false
  alarms; 1 was a real repeat (section 6).

---

## 2. Where everything is

| What | Where |
|---|---|
| The diff (against `review-followups` at `1fbd7c9`; it applies cleanly there) | `docs/SEGMENTER_DRAFT_2026-10-25.patch` (865 lines added, 17 removed) |
| Scratch copy with the draft | `outputs/review_2026_10/segfix_draft/draft/` (git-ignored) |
| Scratch copy of the code before the draft (`review-followups` `1fbd7c9`) | `outputs/review_2026_10/segfix_draft/base/` |
| The trained models (not in the patch; the trainers rebuild them) | `draft/configs/segmenter/header_model.json` (184 KB), `content_model.json` (238 KB) |
| Every run and comparison | `outputs/review_2026_10/segfix_draft/runs/` |
| The measuring tools | `outputs/review_2026_10/scratchpad_backup/seg_*.py` |

The patch changes one file and adds four:

| File | Change |
|---|---|
| `src/pipeline/answer_segmenter.py` | D1 in `normalize_header_text`; D2 in `_read_learned_headers`, called by `_assign_sections`; D3 in `resolve_header_collision`, `_content_overrules_header`, `_claimed_questions`, `QuestionContext.content_keeps`, and in the content switch and the gap filler. |
| `src/pipeline/header_model.py` (new) | The learned header reader. |
| `src/pipeline/content_model.py` (new) | The learned content model. |
| `scripts/train_header_model.py` (new) | Trains the header reader on the dev scripts and writes `configs/segmenter/header_model.json`. |
| `scripts/train_content_model.py` (new) | Trains the content model on the dev scripts and writes `configs/segmenter/content_model.json`. |

**Without the two model files the segmenter still works:** it then runs D1 and the D3a rules only (section 3). A
paper the content model was not trained on gets no content decisions either. This was checked: the draft with the
content model off gives exactly the D3a run (0 changes).

---

## 3. What each part does, in plain words

### D1: markup in header lines

A header line is read once as written and once with its markup removed. The markup is `[unclear: …]`, `[struck: …]`,
`struck[: …]` and `~~…~~`. The line without markup is used only when the header patterns read it as a header and
don't read the original. Other lines keep their markup.

### D2: the learned header reader

**What it learns from.** Every line with a number that is a question on its paper is an example. The current header
patterns give the label: header, answer text, or text followed by a header on the same line.
- Data: 82 dev scripts × 3 texts, 5,536 lines (2,861 text, 2,673 header, 2 mid-line).
- The model never sees the test scripts.

**Two parts:**
1. **The shape of the line.** This is a naive Bayes model over its character 2- to 4-grams, with the number replaced
   by a placeholder, plus the line's length and where the number sits. It learns that "A: to the qustion NO. 10" looks
   like a header, and that "(v) 6" or "1. a+v=" looks like an answer item.
2. **Where the line is.** The number can be the first question, the one being answered, the next one, one further on,
   or an earlier one. The line can start the page, follow a blank line, or follow text. A logistic regression combines
   the two parts into the probability that the line is a header. It is fitted on scores of scripts the shape model
   did not see (5 folds by script).

**When it reads a header.** It reads one when that probability is above 1/2, which is the plain "more likely than not"
decision, not a tuned cut-off. The line is then rewritten to `Ans to the Q No. N`. Answer text before the header stays
on its own line, so mid-line headers are split.

**The fitted weights.** These are log-odds, so a positive weight makes a header more likely:

| Feature | Weight |
|---|---|
| bias | +0.31 |
| shape score (per nat) | +0.016 |
| number = first question | +1.43 |
| number = the question being answered | +0.06 |
| number = the next question | +0.59 |
| number = a question further on | −0.45 |
| number = an earlier question | −1.32 |
| line starts the page | +0.41 |
| line follows a blank line | +0.37 |
| line follows text | −0.47 |
| header after text on the same line | −4.55 |

**Two rules around it.** These are structural, not thresholds:
- It only adds headers for questions that have no header the patterns read anywhere in the script. A second header
  for a question is more often an item numbered like it.
- A read that names the question already open adds nothing and is skipped (items "2ⓐ", "2ⓑ").

### D3a: the old header-collision rule, made more careful (no model)

These are plain code changes, and they apply with or without the models:
- **A header that repeats the question being answered, over the next answer, gives way to that answer's question.**
  For example, "Answer to the question no. 12" stands over a dialogue: it is the dialogue question (SE_08_Q1_0036).
  This is checked only after 8 or more words of the answer: a repeat right after the student's name is not one.
- **A header that names a question already answered can be an item inside the current answer.** For example,
  "Ans to the Q No. 1 (a+v) Deforestation…" inside question 8 (SE_08_Q1_0004), or "Q No 1. In the poem…" inside
  question 8 (pipeline SE_10_Q1_0004). The current question is then a candidate too.

### D3: the learned content model

**What it learns from.**
- **Word model:** every question of a paper gets a model of the words in other students' answers to it: words,
  numbers, and single symbols such as `+`, `|` and `:`, which carry layout like matching pairs, tables and dialogue
  turns. It is trained on 2,638 dev answers filed under a written header, with the header's own words left out.
- **Calibration:** 2,677 dev header sections, scored by word models that did not see their script (5 folds by
  script).

**What is fitted.** All values come from maximum likelihood and expectation-maximisation. None is set by hand.

| Number | Value | Meaning |
|---|---|---|
| β, γ | 2.95, 0.30 | Turn the word model's scores into honest probabilities. Naive Bayes is far too sure of itself on long texts. |
| λ | 0.48% | Share of answers whose words say nothing about their question (they look like no other student's answer). |
| ε, header names the next question | 0.42% | How often a written header names the wrong question, in each situation… |
| ε, a question further on | 1.06% | |
| ε, an earlier question not yet answered | 1.39% | |
| ε, a question already answered | 50.9% | …so a header naming an already-answered question is wrong half the time (items numbered like questions, misread digits). |
| ε, the question being answered | 7.8% | |

**How the segmenter uses it.** Every decision takes the most probable question. There is no cut-off.
1. **A written header can be overruled.** The winner is the question with the highest probability, given how often a
   header in its situation is wrong and the words under it.
   - Candidates: the header's question; questions not answered yet that no other header of the script claims (one
     question, one header); and the current question, when the header names one already answered.
   - With these numbers, **a header naming the next question is never overruled**: a wrong "next" header (0.42%) is
     rarer than content that says nothing (0.48%).
   - Headers in the other situations can be overruled. For a header naming an answered question, the words must be
     about 10–13 times more likely under the other question; for one naming a question further on, about
     900–1,200 times.
   - The words written on the header line after its number count as content ("(3) (e) Ans: Robin teaches us…").
2. **Structural evidence no longer moves text away from where it clearly belongs.** The content switch and the gap
   filler move text by the paper's terms and layout. They now leave it alone when the content model finds the
   question it is filed under the single most probable one. Example: the rickshaw-puller paragraph stays in the
   paragraph question instead of going to the summary question.

---

## 4. Numbers

Each step is measured with the models trained without the script's own text: dev scripts in 5 folds by script, test
scripts with the model of all dev scripts. Left out: class-10 transcription test pages 0029–0033, SE_06_Q1_0004 (a
2nd Paper scan) and SE_07_Q1_0000 (no text). The steps:
- **base** is `review-followups` (`1fbd7c9`);
- **D1** to **D3** add the parts in order;
- **D3a** is D2 plus the D3a rules (no content model);
- **D3** is the whole draft.

### 4a. Teacher-marked questions with no text, per paper and text

The number in brackets counts teacher-marked questions that get 8 words or fewer ("header only").

**Dev split**

| Paper | Text | Scripts | base | D1 | D2 | D3a | D3 |
|---|---|---|---|---|---|---|---|
| SE_06_Q1 | gemini | 15 | 2 | 2 | 2 | 2 | 2 |
| SE_06_Q1 | opus | 15 | 6 | 4 | 2 | 2 | 2 |
| SE_06_Q1 | pipe | 15 | 7 | 7 | 2 | 2 | 3 |
| SE_07_Q1 | gemini | 24 | 2 | 2 | 1 | 1 | 1 |
| SE_07_Q1 | opus | 24 | 10 | 10 | 5 (1) | 5 (1) | 4 (1) |
| SE_07_Q1 | pipe | 24 | 4 (1) | 4 (1) | 1 (2) | 1 (2) | 1 (2) |
| SE_08_Q1 | gemini | 24 | 2 (4) | 2 (4) | 2 (4) | 2 (2) | 2 (2) |
| SE_08_Q1 | opus | 25 | 4 (1) | 4 (1) | 1 (1) | 1 | 1 |
| SE_08_Q1 | pipe | 4 | 3 | 3 | 2 | 2 | 1 |
| SE_10_Q1 | gemini | 18 | 0 | 0 | 0 | 0 | 0 |
| SE_10_Q1 | opus | 18 | 0 | 0 | 0 | 0 | 0 |
| SE_10_Q1 | pipe | 18 | 1 (1) | 1 (1) | 0 (1) | 0 | 0 |
| **all** | **gemini** | | **6** (4) | **6** (4) | **5** (4) | **5** (2) | **5** (2) |
| **all** | **opus** | | **20** (1) | **18** (1) | **8** (2) | **8** (1) | **7** (1) |
| **all** | **pipe** | | **15** (2) | **15** (2) | **5** (3) | **5** (2) | **5** (2) |

**Test split**

| Paper | Text | Scripts | base | D1 | D2 | D3a | D3 |
|---|---|---|---|---|---|---|---|
| SE_06_Q1 | gemini | 6 | 0 | 0 | 0 | 0 | 0 |
| SE_06_Q1 | opus | 6 | 2 | 1 | 1 | 1 | 1 |
| SE_06_Q1 | pipe | 6 | 5 | 5 | 4 | 4 | 3 |
| SE_07_Q1 | gemini | 10 | 0 | 0 | 0 | 0 | 0 |
| SE_07_Q1 | opus | 10 | 1 | 1 | 0 | 0 | 0 |
| SE_07_Q1 | pipe | 10 | 2 | 2 | 1 | 1 | 1 |
| SE_08_Q1 | gemini | 11 | 0 (1) | 0 (1) | 0 | 0 | 0 |
| SE_08_Q1 | opus | 11 | 0 | 0 | 0 | 0 | 0 |
| SE_08_Q1 | pipe | 4 | 3 | 3 | 1 | 1 | 1 |
| SE_10_Q1 | gemini | 10 | 0 | 0 | 0 | 0 | 0 |
| SE_10_Q1 | opus | 10 | 0 | 0 | 0 | 0 | 0 |
| SE_10_Q1 | pipe | 10 | 2 | 2 | 2 | 2 | 2 |
| **all** | **gemini** | | **0** (1) | **0** (1) | **0** | **0** | **0** |
| **all** | **opus** | | **3** | **2** | **1** | **1** | **1** |
| **all** | **pipe** | | **12** | **12** | **8** | **8** | **7** |

**Two things to know about these counts:**
- **They count only missing answers, not text filed under the wrong question.** Most of D3's work is the second kind,
  so 4b and section 5 show it better.
- **The one rise (pipeline class 6 dev, 2 → 3) is SE_06_Q1_0003 Q7.** The pipeline text lost that question's header.
  Before D3, Q7 "had text", but it was the poem answers (section 5).

### 4b. Do the three texts agree on which words belong to which question?

For every script, question and pair of texts, the words in the question's section in each text are compared: the
shared words count against all words, summed over all questions. Reading differences lower this the same way at
every step, so the changes come from the splitting. The **ceiling** puts each script's words in one bag: it is the
agreement a perfect splitter would reach with these readings.

| Split | Texts | base | D1 | D2 | D3a | D3 | ceiling |
|---|---|---|---|---|---|---|---|
| dev | opus/gemini | 92.55% | 92.72% | 93.73% | 93.97% | 94.07% | 95.32% |
| dev | opus/pipe | 90.09% | 90.13% | 91.14% | 91.52% | 92.01% | 93.69% |
| dev | gemini/pipe | 89.17% | 89.17% | 90.59% | 90.93% | 91.25% | 93.18% |
| test | opus/gemini | 93.27% | 93.27% | 94.09% | 94.22% | 94.51% | 95.74% |
| test | opus/pipe | 89.62% | 90.02% | 91.40% | 91.40% | 91.84% | 93.46% |
| test | gemini/pipe | 89.13% | 89.13% | 90.71% | 90.87% | 91.25% | 93.19% |

**How much of the gap to the ceiling the draft closes:**
- dev opus/pipe: 3.60 points before, 1.68 after;
- test opus/pipe: 3.84 points before, 1.62 after.

### 4c. How many questions change at each step

| Step | Questions changed | In scripts (texts counted separately) |
|---|---|---|
| D1 vs base | 27 | 15 |
| D2 vs D1 | 147 | 61 |
| D3a vs D2 | 34 | 14 |
| D3 vs D3a | 74 | 30 |
| **D3 vs base** | **266** | **100** |

**By text** (base → D3), these are the scripts to re-grade once the fix is in:

| Text | Dev | Test |
|---|---|---|
| pipeline | 28 | 10 |
| Opus | 23 | 10 |
| Gemini | 22 | 7 |

The 38 pipeline scripts are:
- SE_06_Q1: 0003, 0005, 0007, 0010, 0013, 0015, 0016, 0020, 0021;
- SE_07_Q1: 0001, 0003, 0005, 0008, 0009, 0011, 0012, 0013, 0014, 0018, 0020, 0021, 0022, 0023, 0024, 0025, 0026, 0029, 0031, 0034;
- SE_08_Q1: 0005, 0012, 0017, 0025, 0029;
- SE_10_Q1: 0004, 0014, 0019, 0026.

The full list of changed questions is in `runs/d3_vs_base.txt`.

---

## 5. Every change, judged

### D1 and D2

- **D1:** each change is a header line that only markup hid, so the header was always there.
- **D2:** all 99 header reads (models trained without the script, all texts, dev and test) are listed in
  `runs/d2_header_reads.txt`, with the line, the question before it and the odds.
  - **97 are real headers.** Examples: "A: to the question No.5", "Ansuler No.12", "9 নং", "4. 4no. Question's
    Answer", and the mid-line "Part-A (Reading) Ans. to ques. No.1".
  - **2 are not:** "5) Quiet = start (Antonym)" in the Gemini and Opus text of SE_07_Q1_0022, an item of the synonym
    question numbered "5)". The one line ends up in Q5 instead of Q4; the real Q5 still gets its own text (D3).
- **D3a:** the 34 changed questions were checked one by one, and all are right.

### D3 (D3 vs D3a: 74 questions in 30 script texts)

✓ = right; ✗ = wrong; ~ = partly right.

**A header overruled by the content below it:**

| Script | Texts | Split | What happens | |
|---|---|---|---|---|
| SE_07_Q1_0031 | Gemini, pipe | dev | "Ans to the Q. No. 7" over MCQ answers ("a) iii. rural area") → Q1. "Ans to the Q.no.2" over the short answers stays Q2 (D3a had moved it to Q1). | ✓ |
| SE_07_Q1_0012 | Opus, pipe | dev | The header "১" (Opus) or "3" (pipe) over the short answers "a. Joynul and Nazma live in…" → Q2. This removes a false "answered twice". | ✓ |
| SE_06_Q1_0006 | Opus | dev | "Ans to the Q No. 4" over the story "A friend in need is a friend indeed" → Q11 (story). Q4 keeps its own text. | ✓ |
| SE_06_Q1_0016 | Opus, pipe | test | A second "Ans to the Q No. 6" over Victory Day short answers → Q2. Pipe also: "Ans to the QS.NO 9" over cloze words → Q3, which was missing (teacher 1); the real Q9 table on p7 → Q9 (was Q8). | ✓ (pipe ~, see the ✗ below) |
| SE_08_Q1_0007 | Opus | dev | A second "Ans to the Q No. 3" over matching pairs "(a + iii)…" → Q8 (matching). The tree-plantation paragraph on p17–18 → Q12 (was Q8). | ✓ |
| SE_08_Q1_0025 | pipe | dev | The first of two "Ans to the Ques no- 9" over the deforestation sentences → Q8, which was missing (teacher 5). | ✓ |
| SE_07_Q1_0019 | Gemini | dev | "9. ans:- Nature is the source of joy…" (the item "g." read as 9) → Q10 (poem). | ✓ |
| SE_07_Q1_0005 | pipe | test | On a garbled pipeline page after the dialogue, "Ans to the Question No - 09" → Q13, not a second Q9 copy. "Ans to the Q No. ,9" over synonyms → Q4 (as before). | ✓ |
| SE_07_Q1_0018 | pipe | test | "Ans. to the Ques. No.8" over cloze words → Q3 (as before, now by the content decision). | ✓ |
| SE_07_Q1_0022 | Gemini, Opus | dev | "Ans to the Q No. 5" over the Apollo table → Q5. D3a had sent it to Q7, because the false D2 read "5) Quiet" had already opened Q5. | ✓ |
| SE_07_Q1_0025 | Opus | dev | "Answer to the QNo:3" over the cloze → Q3, which was missing (teacher 3). But the rest of page 10 goes with it: a second copy of Q9, under the unread header "Answe to the QNo:9" (the Q9 table and 3 lines of the story). It was in Q9 before. | ~ |

**Structural evidence no longer moves text away from where its content clearly belongs:**

| Script | Texts | Split | What happens | |
|---|---|---|---|---|
| SE_06_Q1_0003 | Gemini, Opus | dev | "He wants to improve himself…" (the summary of Raju) stays Q7 (was Q11, story). | ✓ |
| SE_06_Q1_0003 | pipe | dev | The pipeline text lost the Q7 header. The poem answers on p7–8 → Q10 (they were Q7's text). Q7 now has no text: its summary is in Q11, as before. | ~ |
| SE_06_Q1_0021 | Gemini, Opus, pipe | dev | The rickshaw-puller paragraph stays Q12 (was moved to Q7, summary; teacher Q7 0). | ✓ |
| SE_07_Q1_0018 | Gemini, Opus, pipe | test | The prize-giving paragraph on p12–13 stays Q12 (was Q13, dialogue). | ✓ |
| SE_08_Q1_0008 | Gemini, Opus | dev | "Nowadays deforestation is very dangerous…" inside the tree-plantation paragraph stays Q12 (was Q8). | ✓ |
| SE_06_Q1_0022 | Opus | test | "ⓓ Subject he did poorly in: ICT, Science…", a table item, stays Q5 (was Q7). | ✓ |
| SE_08_Q1_0029 | Gemini, Opus, pipe | test | The woodcutter story on p2–4 stays Q11 (was Q7, summary). Opus and pipe also: "(d) Ans: Hygiene is…" stays Q2 (was moved into Q1; teacher Q1 0). | ✓ |
| SE_07_Q1_0022 | Gemini, Opus, pipe | dev | "(c) Ans: Mr Joynul Ali is a carpenter…" stays Q2 (was moved into Q1; Q1 has no teacher mark). | ✓ |

**Still wrong in D3:**
- **Pipe SE_06_Q1_0016 p8:** the rickshaw-puller paragraph (it belongs to Q12) moves from Q8 to Q9. It was wrong
  before too.
- **The two ~ cases above.**

**How the last version was chosen.** An earlier D3 version let the content model block a structural move whenever it
found the current question more probable than the target. That was wrong 3 times, where the content model's own top
choice was a third question (it had no real opinion):
- Opus SE_06_Q1_0020 Q4, under a circled ④;
- pipeline SE_10_Q1_0011 Q3, under a misread "13";
- and, separately, Gemini SE_06_Q1_0020, where the only answer was on the header line itself.

The final version blocks a move only when the current question is the content model's top choice, and it reads the
words on a header line. Those 3 went back to the earlier result, and nothing else changed. In every case where
the gap filler was rightly blocked (7), the content model gave the current question 0.99 or more; in the 2 wrong
ones it gave it 0.18.

---

## 6. Questions answered twice (the team check)

`docs/OPUS_CLOUD_RESULTS_2026-10-25.md` section 5, question 2, asks about repeats on consecutive pages. The check that
counts a header repeated inside one run of parts (`consec_headers.py`), on the Opus text of all four classes, gives:

| | base | draft |
|---|---|---|
| Real rewrites found | SE_08_Q1_0020 Q4, SE_08_Q1_0027 Q7 | the same, **and SE_07_Q1_0013 Q10** (pages 6 and 7 both start "ⓒA; The poet feels about life…"; base could not read its "A: to the qustion NO. 10" headers) |
| False alarms | 5: SE_06_Q1_0011 Q2, SE_07_Q1_0018 Q13, SE_08_Q1_0008 Q8, SE_08_Q1_0036 Q12, SE_10_Q1_0020 Q11 | 3: SE_06_Q1_0011 Q2, SE_10_Q1_0020 Q11, and SE_07_Q1_0025 Q10 (the overruled "No. 5" header over a poem answer is counted as a header) |

The draft run used the stored models, trained on all dev scripts, so this is in-sample for dev scripts.
SE_07_Q1_0013 Q10 is a rewrite in the Opus text and has not been checked on the scan.

**The team-check code itself** (`review-followups`) flags a question when its copies are in separate runs, each with
its own header. Run over all texts and all four classes (`seg_repeats.py`):
- **Flags:** 46 → 23. The draft adds no new flag.
- **Of the 23 that go:**
  - 22 were false alarms:
    - the items "1. ans:" of SE_08_Q1_0004, read as the headers of Q1–Q5, in the Gemini and Opus text;
    - "Ans to the Q. No. 7" over MCQ answers (SE_07_Q1_0031);
    - an item "Q No 1." inside Q8 (pipeline SE_10_Q1_0004);
    - the rest are the false copies of section 5 (SE_06_Q1_0006, SE_07_Q1_0012, …). The full lists are
      `runs/repeats_base.txt` and `runs/repeats_d3.txt`.
  - 1 was real: Opus SE_07_Q1_0025 Q9. Its first copy has the unread header "Answe to the QNo:9" (section 7,
    limit 1).
- **On the Opus text:**
  - base flags 7 of the 9 real cases of `NEXT_STEPS_2026-10-25.md` section 2a, plus 9 other flags: 7 false alarms,
    which the draft removes, and SE_10_Q1_0005 Q7 and SE_10_Q1_0014 Q10 (not checked);
  - the draft flags 6 of the 9, plus those same 2.
- **Neither finds the class-8 rewrites on the next page** (SE_08_Q1_0020 Q4, 0027 Q7). The team-check code looks for
  separate runs only, so that question for you is still open.

---

## 7. Known limits

1. **A second copy whose header the patterns can't read stays unread** when the question has a readable header
   elsewhere. This is the D2 rule that avoids items numbered like questions. Example: Opus SE_07_Q1_0025, where page
   10's copy of Q9 ends up in Q3.
2. **Circled numbers as headers** ("③", "④" alone on a line, Opus SE_06_Q1_0020) are not read by either model. The
   old gap filler still finds them from the paper's terms and layout, and D3 leaves that alone.
3. **An item numbered like the next question** ("5) Quiet = start") can be read as its header. Only that one line
   is misfiled, because the student's own header for the question still opens the real answer.
4. **When a pipeline page loses a header and the answer looks like no other**, the text stays where it is (pipeline
   SE_06_Q1_0003 Q7).
5. **The content model knows only the papers it was trained on** (classes 6, 7, 8, 10). For a new paper, the
   segmenter runs D1, D2 and the D3a rules, and no content decisions.
6. **The models must be retrained when the texts change**, for example when the GPU runner adds the class-8 pipeline
   text: 8 class-8 pipeline scripts exist now, 36 after step 1. Retraining takes a few minutes on the CPU.

### A note on the test split

- The learned models never saw a test script.
- But the test-split segmentation tables were looked at while the draft was built.
- One of the three cases that led to the last change (pipeline SE_10_Q1_0011) is a test script; the other two are
  dev.
- No grades of the test split have been computed, so the grading test of step 7 is still unseen.
- In the thesis, the segmentation numbers of the test split should be reported as "looked at during development".

---

## 8. Re-running the measurements

From `outputs/review_2026_10/segfix_draft/draft/` (`PY` is the repo's Python, `B` the tools folder):

```bash
PY=../../../../script_checking/bin/python
B=../../scratchpad_backup
# train both models on the dev scripts (writes configs/segmenter/*.json)
$PY scripts/train_header_model.py
$PY scripts/train_content_model.py
# segment every text, models trained without the script's own text (about 1.5 min)
$PY $B/seg_measure.py run ../runs/d3.json --header-cv --content-cv
# the same with the content model off: gives exactly ../runs/d3a.json
$PY $B/seg_measure.py run ../runs/d3_nocontent.json --header-cv --no-content
# compare two runs: the summary table and every changed question
$PY $B/seg_measure.py compare ../runs/d3a.json ../runs/d3.json
# the tables of section 4a and 4b
$PY $B/seg_stages.py base=../runs/base.json D1=../runs/d1.json D2=../runs/d2.json D3a=../runs/d3a.json D3=../runs/d3.json
$PY $B/seg_agree.py base=../runs/base.json D1=../runs/d1.json D2=../runs/d2.json D3a=../runs/d3a.json D3=../runs/d3.json
# what moved, line by line, and the decisions behind it
$PY $B/seg_linediff.py ../runs/d3a.json ../runs/d3.json "opus SE_07_Q1_0025"
$PY $B/seg_events.py "opus SE_07_Q1_0025" "pipe SE_06_Q1_0016"
$PY $B/seg_trace.py opus SE_07_Q1_0025 --header-cv --content-cv
# repeats on consecutive pages (section 6) and the answered-twice flags
$PY $B/consec_headers.py SE_07_Q1 data/ground_truth/transcripts/SE_07_Q1_extraction_opus_5.5
$PY $B/seg_repeats.py ../runs/repeats_d3.txt
# the tests
$PY -m pytest -q
```

For the base run, run the same `seg_measure.py run ../runs/base.json` from `../base/`. The D1, D2 and D3a runs in
`runs/` come from earlier versions of the draft. They are kept as files; only D3a can be re-made from the final draft
(with `--no-content`).

---

## 9. How to apply it, after your OK

The fix belongs after step 1 (class 8 is read with the old code, like classes 6 and 7) and after `review-followups`
is merged into `gemma` (step 2). Then:

```bash
git show gemma:docs/SEGMENTER_DRAFT_2026-10-25.patch | git apply
./script_checking/bin/python scripts/train_header_model.py     # with the class-8 pipeline text in by then
./script_checking/bin/python scripts/train_content_model.py
./script_checking/bin/python -m pytest -q
```

1. Commit the code and the two model files, `configs/segmenter/*.json` (422 KB together), which the segmenter needs.
2. Measure again (section 8), because class 8 then has more pipeline text.
3. Re-grade only the questions that changed (section 4c, and `seg_measure.py compare` for the new list).
4. The cloud's Opus grades were made with the old splitting too: 33 Opus script texts change. For a fair Opus vs
   teacher comparison, those questions need grading again.

---

## 10. Questions for you

1. **May Claude apply the patch** at step 6, as in section 9, and commit it with the two model files?
2. **The three known wrong cases** (section 5): leave them, or should circled-number headers and the "second copy
   under a readable header" case get a follow-up? Both would need a labelled example set, because the current header
   patterns, which give the training labels, can't read them.
3. **Test split:** fine to report the test-split segmentation numbers as "looked at during development" (section 7)?
