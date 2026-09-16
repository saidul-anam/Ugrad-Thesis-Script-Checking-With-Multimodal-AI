# Extraction Accuracy Guide: from "pretty accurate" to measurably trustworthy


---


## Current status (measured on 2026-09-14)

| Item | Value |
| --- | --- |
| Scripts extracted | 4 (`SE_11_Q1_0002`, `0010`, `0011`, `0015`) |
| Pages / words | 46 pages / 4,481 words |
| Human-corrected pages (ground truth) | **0**, so there is no accuracy number yet |
| Words the model marked `[unclear]` or `[illegible]` | **0**, so every mistake it makes is silent |
| LaTeX arrows corrupted by Stage 2 parsing | 19 (scripts 0010, 0011, 0015) |
| Stage 2 changes applied without being declared | 50 of 62 changed spans |
| Pages where Stage 2 deleted `[struck: ...]` tags | 4 |

---

## The improvement loop (use it for every step) 

Every change you make follows the same loop:

1. Run the pipeline on the scripts.
2. Score the output against the **same** corrected pages from Step 1.
3. Keep the change only if **no number gets worse**: CER, silent-correction rate, accepted-word accuracy.
4. Write one line in a results log, for example `docs/accuracy_log.md`:

| Date | Change | CER Stage 1 | CER Stage 1+2 | Silent-correction rate | Coverage | Accepted-word accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-xx | baseline | | | | n/a | n/a |

This log becomes a results table in the thesis almost for free.

---

## Step 1 — Build a small ground-truth set and measure the baseline

> **Order changed from my earlier list:** measure *before* fixing anything. If you fix first, you can never
> show how much the fixes helped.

### Why
You cannot improve what you have not measured, and the thesis needs a number. The corrected pages are
also the test set for every later step, so this is the most valuable hour you can spend.

### Which pages
- **10 pages**, from **all 4 writers** (2–3 pages each).
- Prefer continuous writing (Q3, Q7–Q11), because that is where language marks depend on exact spelling.
- Include at least **2 hard pages** on purpose, e.g. `SE_11_Q1_0002` page 13 (reverse-side bleed-through)
  and `SE_11_Q1_0015` page 1 (overwritten words, red ink across the text). If you only pick neat pages,
  the number will be too optimistic.

### What you do
**1.1 Create the drafts** (the machine's text, pre-filled, next to a copy of the page image):

```bash
# automatic selection of continuous-writing pages across all scripts
python scripts/make_transcription_gt.py --lang english --max-pages 10

# or choose pages yourself
python scripts/make_transcription_gt.py --lang english --pages SE_11_Q1_0002:12,13 SE_11_Q1_0010:3,4 SE_11_Q1_0011:4,5 SE_11_Q1_0015:1,2
```

This creates, for each page, in `data/ground_truth/transcripts/english/<script>/`:
`page_<n>.png` (image), `page_<n>.txt` (draft to correct), `page_<n>.meta.json` (status).

**1.2 Correct each draft so it matches the ink exactly.** Open the `.png` and `.txt` side by side.

| Situation | What to write |
| --- | --- |
| Student misspelled a word (`familyes`, `he see`) | Keep it exactly. **Never fix the student.** |
| Word you cannot read at all | `[illegible]` |
| Word you can read but are not sure | `[unclear: your best reading]` |
| Word crossed out by the student | `[struck: the crossed-out text]` |
| Letters overwritten by the student | The letters the ink finally shows |
| Red teacher ink, ticks, margin marks | Ignore completely |
| Line breaks | Keep them as on the page |

> **Warning: anchoring.** The draft is the machine's text, and people tend to accept what they read.
> A draft you did not really check scores 0 errors, because it is just the machine compared with itself
> (a dry run confirmed this). To avoid it:
> read each line **on the image first**, then compare with the text, and stop at every word that is not a
> normal dictionary word.

**1.3 Mark the page as done.** In `page_<n>.meta.json` change `"status"` to `"CORRECTED"`.
Pages still marked `DRAFT` are skipped by the benchmark.

**1.4 Run the benchmark:**

```bash
python scripts/evaluate_transcription.py --lang english --tag baseline
```

It writes `outputs/benchmarks/transcription_english_baseline_<time>.md` with:

| Metric | Meaning | Good direction |
| --- | --- | --- |
| CER (character error rate) | Wrong, missing or extra characters divided by characters in your corrected text | Lower |
| WER (word error rate) | The same at word level | Lower |
| Silent-correction rate | Share of the student's real misspellings that the machine "fixed" | Lower, ideally 0 |
| Stage 1 vs Stage 1+2 | Whether Stage 2 helps or hurts | Stage 1+2 should be lower |

**1.5 Sort the errors into kinds.** Print the word differences for one page:

```python
import difflib, json
script, page = "SE_11_Q1_0015", 1
ref = open(f"data/ground_truth/transcripts/english/{script}/page_{page}.txt").read().split()
hyp = json.load(open(f"outputs/extracted/english/{script}/checkpoints/page_{page}.json"))["stage2_verification"]["verified_transcript"].split()
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
    if tag != "equal":
        print(f"{tag:8s} truth: {' '.join(ref[i1:i2])!r:30}  machine: {' '.join(hyp[j1:j2])!r}")
```

Put every difference in one bucket and count them:

| Bucket | Example | Fixed by |
| --- | --- | --- |
| A. Software bug | arrow command corrupted, struck tag deleted | Step 2 |
| B. Stage 2 made it worse | Stage 1 was right, Stage 2 changed it | Step 2 |
| C. Genuinely ambiguous ink | *ask* / *ash* | Step 3 + Step 4 |
| D. Plain misread | clear ink, wrong word | Step 3 (and Step 6 if frequent) |
| E. Layout | table cells, line breaks, flow-chart boxes | Usually ignore; note it |

The counts tell you where the effort pays off. If bucket A+B is large, Step 2 alone gives a big gain.

**1.6 (Recommended) Measure the human ceiling.** Ask a friend to correct **2** of the same pages
independently, from the images only. Compare the two humans:

```python
from src.utils.text_metrics import score_transcription
a = open("friend/SE_11_Q1_0010/page_3.txt").read()
b = open("data/ground_truth/transcripts/english/SE_11_Q1_0010/page_3.txt").read()
print(score_transcription(a, b).as_dict())
```

The CER between two humans is the ceiling. If the machine reaches it, the remaining "errors" are
disagreements a human would also make.

### Effort and done-criteria
- Your time: about 5–8 minutes per page, so **1–1.5 hours** for 10 pages.
- Done when: 10 pages are `CORRECTED`, the baseline report exists, and the first log line is filled in.
- Tools: **already built.**

---

## Step 2 — Fix the two software bugs in Stage 2

### Why
These are not handwriting problems; they are deterministic bugs. Fixing them costs no GPU research and
removes errors from bucket A and B.

### 2a. The backslash bug
**What happens.** For flow charts the model writes arrows as the LaTeX command `$\rightarrow$`. Stage 2
returns its answer as JSON. In JSON a backslash followed by `r` means "carriage return", so the parser
turns `\rightarrow` into an invisible line-break character followed by `ightarrow`:

```text
Stage 1:  2. Becoming $\rightarrow$ 3. Being
Stage 2:  2. Becoming $<carriage return>ightarrow$ 3. Being
```

The same would happen to `\times` (becomes a tab), `\frac` (form feed) and `\bullet` (backspace).

**The fix.** In `src/pipeline/stage2_verifier.py`, after parsing, compare with the Stage 1 text: any
carriage-return, tab, form-feed or backspace character that Stage 1 did not contain is turned back into the
backslash sequence it came from. A one-off repair script fixes the 19 existing cases in the saved
checkpoints without using the GPU.

### 2b. Stage 2 applies changes it never declared
**What happens.** Stage 2 returns two things: a complete rewritten page, and a list of the changes it
claims to have made. The code keeps the **complete rewritten page**. The safety check
(`is_spurious_reversion`) only looks at the **listed** changes. So any change the model did not list goes
in without any check. In the current outputs, **50 of 62** changed spans were never listed, and 4 pages
lost their `[struck: ...]` tags this way.

Some unlisted changes are right (the student really wrote "poemishows" as one word). Others are wrong.
Right now nobody can tell which.

**The fix.**
1. Start from the Stage 1 text, not from Stage 2's rewritten page.
2. Apply each **listed** change only inside its own context snippet, after the existing safety check.
3. Never delete `[struck: ...]` tags.
4. Record every **unlisted** difference as a *Stage 2 disagreement* instead of applying it. This is not
   thrown away: it becomes a free "unsure" signal for Step 3.

### 2c. Re-run and compare
Add an option to re-run **only Stage 2** from the saved Stage 1 checkpoints. If you re-run Stage 1 too,
Stage 1 can change slightly between runs and you will not know which change caused the difference.

```bash
python scripts/evaluate_transcription.py --lang english --tag stage2_fixed
```

Compare with the baseline report and add a log line.

### Effort and done-criteria
- Code: about 1–2 hours (I can implement it). GPU: one Stage 2 pass over the pages.
- Done when: bucket A is empty on the 10 pages, and Stage 1+2 CER is not higher than Stage 1 CER.
- Tools: **must be built** (parser repair, declared-only apply, Stage-2-only re-run, checkpoint repair).

---

## Step 3 — Teach the system to say "I am not sure" (abstention)

### Why
Across 4,481 words the model wrote **zero** `[unclear]` tags. It guesses every word with full confidence,
so every error is silent. You cannot reach "100 % of accepted words" without a way to set doubtful words aside.

### The idea, in plain words
Two people read the same word. If both say "ask", it is almost certainly "ask". If one says "ask" and the
other says "ash", nobody should decide alone. The system does the same: it reads each word more than once
in slightly different ways, and **disagreement means "unsure"**. Research on VLM OCR shows that agreement
across perturbed re-reads is a much more reliable confidence signal than asking the model how confident it is
(arXiv 2509.09722). This is measured from the ink, not a hand-written rule.

### Design: three layers, from cheap to expensive

**Layer 1 — Find disputed words (cheap).**
- Read every page a **second time** with Stage 1 on a slightly altered image (for example scaled to 90 %, or
  higher contrast). That is one extra model call per page.
- Line up the two transcripts word by word. Identical words are **agreed**; different words are **disputed**.
- Add Stage 2's unlisted disagreements from Step 2b as disputed words; they cost nothing extra.

**Layer 2 — Resolve disputed words (more expensive, but only for the few disputed words).**
- Use the machinery that already exists for Stage 3b: find the line on the page (`localizer.py`), cut it out,
  read the crop 5 times under small changes and vote (`consensus.py`), and ask the model to choose between the
  two readings on the crop (`forced_choice.py`).
- A clear winner makes the word **resolved**.
- Cost: about 7 small calls per disputed word.

**Layer 3 — Abstain.**
- If Layer 2 has no clear winner, write `[unclear: best reading | other reading]` into the transcript and put
  the word on the review list (Step 4).

Every word ends with a status: **agreed**, **resolved**, or **unclear**.

### How you prove it works (with the Step 1 pages)
Split the corrected words by status and measure each group:

| Status | Expected accuracy | What it means if it is lower |
| --- | --- | --- |
| Agreed | Close to 100 % | Both reads share the same mistake. Make the second read more different (another perturbation, or a second model). |
| Resolved | High | Tighten how clear the Layer 2 winner must be. |
| Unclear | Low (that is the point) | Nothing; these go to a human. |

### Things to check
- Stage 3 must never penalise a word inside an `[unclear: ...]` tag. The rubric already says transcription
  markers are not penalised; confirm the code follows it.
- Question headers must be read correctly: a misread header moves a whole answer to the wrong question.

### Effort and done-criteria
- Code: about a day. GPU: roughly doubles Stage 1 time, plus Layer 2 calls for disputed words.
- Done when: every word has a status, and on the 10 pages the agreed words are (near) 100 % correct.
- Tools: Layer 2 **exists**. Layer 1, word statuses and `[unclear]` writing **must be built**.

---

## Step 4 — The human review list

### Why
Unclear words need a final decision. A teacher looking at a cropped word makes that decision in seconds,
much faster than re-reading a page.

### What it looks like
For each script, a list with one row per unclear word: the crop image, reading A, reading B, and an empty
column where the reviewer writes the final word. The existing `extraction_summary.md` already has a
"Flagged for Human Review" table for Stage 3b, and crops are saved in `arbitration_crops/`, so the format is
familiar.

### Workflow
1. The pipeline writes `review_queue.csv` and a contact sheet of crops per script.
2. The reviewer fills in the final word for each row.
3. A script applies the decisions to the transcript.
4. Re-run from Stage 3 onward (`scripts/resegment_extraction.py`, then `scripts/evaluate_scripts.py`).

### The trade-off dial
A stricter Layer 2 sends more words to review and makes the accepted words more accurate. A looser one does
the opposite. Plot both numbers from the 10 pages as the setting changes, and pick the point where accepted-word
accuracy reaches 100 % (or the human ceiling) with a review list you can live with. That plot is a thesis figure.

Every teacher decision is also new labelled data: for the Stage 3b weights and for Step 6.

### Effort and done-criteria
- Code: half a day (`scripts/label_arbitration_candidates.py` is a good template).
- Reviewer time: measure it per script and report it.
- Done when: a script can go from PDF to final transcript with only the review list touched by a human.
- Tools: **must be built** (export queue, apply decisions).

---

## Step 5 — Aim for 100 % where marks depend on it

### Why
Not every character matters for grading. A misread comma changes no marks. A misread cloze answer, MCQ
option or penalised spelling does. 100 % on this small set of words is realistic.

### Which words
| Word type | Why it matters | How many per script |
| --- | --- | --- |
| MCQ option letters (`ii` vs `iii`) | One mark each | 5 |
| Cloze answers | 0.5–1 mark each | 20 |
| Rearrangement letters | 1 mark each | 10 |
| Words Stage 3 penalises | Language marks | varies; already checked by Stage 3b |
| Question headers | A wrong header moves a whole answer | about 12 |

### How
- Send every one of these words through Layer 2 (crop re-reads and forced choice), **even if Layer 1 agreed**.
  There are only about 50 per script, so the cost is small.
- Anything not clearly resolved goes to the review list.

### The "transcript swap" test
This is the clearest way to show that extraction is accurate enough for grading:

1. Make a copy of the extraction in which the 10 corrected pages replace the machine's pages.
2. Grade both versions with Stage 4 into two folders.
3. Compare them:
   ```bash
   python scripts/benchmark_evaluation.py --lang english --eval-dir outputs/evaluated/english --compare outputs/evaluated_human_transcript/english --tag transcript_swap
   ```
4. If the marks are the same, the remaining extraction errors do not change any grade. That is
   "100 % for grading purposes", and it is a claim the thesis can make.

### Effort and done-criteria
- Code: half a day (a mark-relevant word list per question type, plus the swap-copy script).
- Done when: mark-relevant word accuracy is 100 % on the 10 pages, and the swap test shows no mark change.
- Tools: benchmark comparison **exists**; the word list and swap-copy script **must be built**.

---

## Step 6 (optional, later) — Fine-tune the model

### When
Only after Steps 1–5, and only if the review list is still too long.

### How
- Training data: line crops paired with their exact corrected text, from the Step 1 pages and the teacher
  decisions from Step 4. You need several hundred lines.
- Method: LoRA fine-tuning of Gemma 4 on those crops.
- **Test on writers the model never saw in training.** If you train and test on the same student's
  handwriting, the accuracy is inflated and the thesis reviewers will notice.
- The corrected text must stay verbatim, with student mistakes. Otherwise the model learns to autocorrect.

### Effect
Fewer disputed words, so a shorter review list. It does not remove the ceiling from Step 0.

---

## Summary

| Step | What | Who | Your time | Status |
| --- | --- | --- | --- | --- |
| 0 | Activate environment | You | 2 min | Ready |
| 1 | Correct 10 pages, measure baseline | You | 1–1.5 h | Tools built |
| 2 | Fix the two Stage 2 bugs, re-measure | Code | 0 | To build |
| 3 | Agreed / resolved / unclear for every word | Code | 0 | Layer 2 built; rest to build |
| 4 | Human review list | Code + reviewer | a few min per script | To build |
| 5 | 100 % on mark-relevant words; transcript swap test | Code | 0 | Partly built |
| 6 | Fine-tuning (optional) | Code + GPU | 0 | Later |

**Start with Step 1 today.** Everything after it is judged against those 10 pages.
