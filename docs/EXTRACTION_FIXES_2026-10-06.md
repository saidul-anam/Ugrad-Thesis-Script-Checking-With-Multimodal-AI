# Extraction accuracy fixes — 2026-10-06

Response to `docs/PIPELINE_PROBLEMS_AUDIT.md` (P1–P10). Every change below is either removing a
rule that rewrites the transcript from text patterns alone, or replacing it with a decision that
comes from the image. No word lists, no per-letter rules, no hand-tuned thresholds.

> **Data split.** Class-10 scripts 0001–0028 (56 ground-truth pages) were used to find these
> problems, so from now on they are the **dev set**: numbers on them are development numbers.
> Scripts 0029–0033 were never scored. Their ground-truth drafts are in
> `data/ground_truth/transcripts/english_se_10_q1_test/` (20 pages, status `DRAFT`). Once they are
> proofread they are the **held-out test set**. Score it once, at the end.
> The honest held-out number for the *old* pipeline is the first class-10 run
> (`outputs/benchmarks/transcription_english_20261006_001828.md`: Stage 1+2 CER 13.2% macro).

---

## 0. Result (fresh end-to-end run, 56 dev pages)

Stage 1 → Stage 2 → Stage 2b were re-run from scratch on every dev ground-truth page with the new
code and the cleaned Stage 1 prompt (`outputs/extracted/english/se_10_q1_v2/`, benchmark
`outputs/benchmarks/transcription_english_se10_dev_v2_fixed_*.md`).

| | CER macro | CER micro | WER macro | silent autocorrection |
|---|---|---|---|---|
| Old pipeline, Stage 1 | 6.00% | 5.45% | 8.94% | 29.5% |
| Old pipeline, Stage 1+2 (as shipped) | 7.24% | 6.98% | 10.44% | 30.1% |
| New Stage 1 (leaked examples removed) | 6.07% | 5.45% | 8.95% | 28.1% |
| New Stage 1+2 | 4.91% | 4.36% | 7.22% | 26.8% |
| **New Stage 1+2+2b (final)** | **4.52%** | **4.01%** | **6.77%** | **26.5%** |

Word-level errors on the same 4,933 words:

| Error type | Old shipped | New final |
|---|---|---|
| false strike | 276 | **20** |
| missed strike | 71 | 58 |
| misread | 155 | 128 |
| autocorrection | 29 | 34 |
| dropped word | 37 | 25 |
| extra word | 39 | 48 |

Removing the leaked prompt examples did not cost Stage 1 anything (6.00% → 6.07%, within run-to-run
noise). **These are dev numbers.** The pipeline was changed while looking at these pages, so the
number for the thesis is the test set (section 7), once its ground truth is proofread.

## 1. Where the errors actually come from (measured, not guessed)

Word-level attribution of every difference between pipeline output and ground truth on the 56 dev
pages (4,933 active words). Struck text is kept with a "struck" flag so strike mistakes are counted
separately from reading mistakes.

| Error type | Stage 1 | Stage 1+2 as shipped |
|---|---|---|
| **false strike** (active words marked `[struck:]`) | 32 | **276** |
| missed strike (crossed-out words left active) | 98 | 71 |
| misread (wrong letters) | 171 | 155 |
| autocorrection (student non-word → dictionary word) | 30 | 29 |
| dropped word | 44 | 37 |
| extra word | 43 | 39 |

Stage 2 lowers misreads and missed strikes. It also adds about 240 false strikes, and that one
category explains most of the regression.

## 2. What caused the false strikes

The saved Stage 2 patches were replayed onto Stage 1 offline, with and without the step that runs
after Stage 2 ("Stage 2.5 strikethrough grounding", `ground_and_reconcile_strikethroughs`):

| Variant (56 dev pages) | CER macro | CER micro | WER macro |
|---|---|---|---|
| Stage 1 | 6.00% | 5.45% | 8.94% |
| Stage 1+2 as shipped | 7.24% | 6.98% | 10.44% |
| Stage 1+2, **without** Stage 2.5 | **5.04%** | **4.54%** | **7.47%** |

**The Stage 2 VLM itself helps; the rule layer after it is what hurts.** The rules:

* *Draft-block snapping.* This treats any line that **ends with `]`** as the end of a crossed-out
  draft, and a normal struck word at the end of a line counts. It then wraps every line above it
  up to the previous blank line. This produced 0010 p13 (11 lines), 0013 p18 and 0023 p8.
* *Tag inversion* and *t-bar unwrapping.* These decide whether a strike stroke covers a word from a
  **guessed** position: `line_index / n_lines` for the vertical position and character offset
  within the line for the horizontal position. They never measure where the text is on the page.
* The script-level rewriters measured the same way:
  * Stutter→strike regexes (`resolve_strikethrough_collisions`): +0.04 CER points.
  * Lexicon pen-lift stitching (`stitch_pen_lift_splits`): +0.09 CER points and +0.6 WER points.
    It turns "for a long time" into "fora longtime", "are a" into "area", and "can not" into
    "cannot".

## 2b. Stage 2's own patch filters

Stage 2 was re-run on 36 dev pages and the **raw** patch proposals were kept. Each applied patch was
labelled automatically by whether it lowered or raised the page CER against ground truth.

| Variant (36 dev pages, same Stage 1 input) | CER macro | WER macro |
|---|---|---|
| Stage 1 | 6.13% | 9.20% |
| Stage 2 with hand-written filters (old) | 4.67% | 7.02% |
| Stage 2 without filters (new default) | **4.28%** | **6.51%** |

Of 138 applied patches, 77 helped, 42 hurt and 19 were neutral. The filters kept 54/77 (70%) of
the helpful patches and 29/42 (69%) of the harmful ones, which is no better than chance. They
were removed from the default path. The harmful patches still need a check, and that check has to
come from the image (section 4).

## 3. Changes made

| # | Change | Files | Audit items |
|---|---|---|---|
| 1 | **Markup normaliser.** Brackets are matched with a stack. Nested `[struck:]` is flattened, an unclosed tag closes at the end of its line, and a stray `]` is dropped. Output has one balanced tag per struck run per line. Purely structural. | `src/utils/transcript_markup.py` (new), `tests/test_transcript_markup.py` | P7 |
| 2 | Stage 1 and Stage 2 output pass through the normaliser, so every downstream consumer sees well-formed tags. | `stage1_transcriber.py`, `stage2_verifier.py` | P7 |
| 3 | Stage 2.5 grounding, stutter→strike regexes and pen-lift stitching are **off by default**. `pipeline.legacy_text_rules: true` re-enables them for the ablation table only. | `orchestrator.py`, `core/config.py` | P1, P4, P6, P8 |
| 4 | Removed the 15% length-disparity fallback. Patches are applied one at a time onto the Stage 1 base, so the page cannot drift as a whole. The fallback only threw away valid patches on short pages. | `stage2_verifier.py` | P9 |
| 5 | Stage 2 hand-written patch filters are **off by default** (function-word lists, vowel-doubling list, 50% similarity floor, 3-strike quota, keyword checks on the patch's free-text reason, regex mining of `verification_notes`). Every structurally valid patch (target exists, differs) is applied and the tags are normalised. `legacy_filters=True` / `legacy_text_rules: true` restores them for ablation. See section 2b. | `stage2_verifier.py` | P6, P10 |
| 6 | Removed label leakage from the Stage 1 prompt. It contained examples copied from class-10 ground truth (`[struck: po] Portia`, `he[struck: e] ordered`, `[struck: had] he had`), which have been replaced by neutral placeholders. | `prompts/stage1_verbatim.py` | P2 (leakage) |

### Audit remediations deliberately *not* implemented

| Audit proposal | Why not |
|---|---|
| P3: "force a patch if swapping n↔r gives a dictionary word" | A per-letter confusion rule. It would also "correct" real misspellings, which is the opposite of verbatim transcription. |
| P1: veto strikes spanning "more than 3 lines" | A hand-picked threshold. The actual bug was the `]`-ending heuristic, which is now removed. |
| P8: "never unwrap OOV word followed by its correction" | A word-pattern rule. The unwrapping rule it patches is now removed. |
| P9: `abs_diff > 80` | Swaps one hand threshold for another. The fallback was removed instead (change 4). |
| P10 / P2: put `neve cuted`, `had`, `inste` into prompts | Test-set words in the prompt (label leakage). |
| P5: negative logit bias on dictionary tokens | Not supported by the engine, and it would make the model invent misspellings. |
| P6: "page-level CER safety gate" | CER needs ground truth, which doesn't exist at run time. |

## 4. Stage 2b — line reconciliation (strike status only)

Page-level reading (Stage 1/2) has context and good word accuracy. Reading a single line crop at
higher resolution is worse overall (it garbles names and loses context), but it is better at
seeing thin cross-out strokes. `src/pipeline/line_reconciler.py`:

1. Segment the page into text lines from the ink itself (horizontal projection profile,
   `arbitration/localizer.py`).
2. Read each line crop independently (`src/prompts/line_reconciliation.py`).
3. Align the transcript lines to the crop reads (monotonic dynamic programming; a pair counts only
   if the two lines are more alike than not).
4. Wherever the two readings have the **same words but disagree on which are crossed out**, show
   the crop with both versions of the line and ask which one matches the ink. Ask twice with the
   order swapped, and adopt the crop reading only if both answers choose it.

**Why strikes only (measured, dev pages).** The first version adjudicated every disagreement:

| Decision type | Accepted: helped / hurt | Rejected: would have helped / hurt |
|---|---|---|
| Strike status only | 11 / 4 | 3 / 7 |
| Letters / words | 44 / 78 | 35 / 31 |

On strike status the judge is right about 70% of the time. On letters it is at chance, and it
accepted invented non-words such as `kmog` and `thimb`. A higher-resolution crop adds information
about pen strokes, not about letter shapes the same model has already read. Applied to all
disagreements, the version made things worse (CER 4.88% → 5.11%). Restricted to strike status:

| Variant (56 dev pages) | CER macro | WER macro | false strikes | missed strikes |
|---|---|---|---|---|
| Stage 2, new default | 4.50% | 6.69% | 51 | 70 |
| + Stage 2b (strikes only) | **4.21%** | **6.33%** | 36 | 63 |

27 changes were accepted: 19 helped and 8 hurt. Cost: one short VLM call per text line plus two per
disagreement, about 15–20 s per page. It is on by default (`pipeline.line_reconciliation: true`).
Each page's decisions are saved next to its checkpoint as `page_<n>.reconcile.json`.

Not solved by this: letter-level misreads (≈130 words on the dev pages) and silent
autocorrection of student misspellings (≈28% of student non-words). The second reading did keep
misspellings better (silent-correction rate 28% → 21% when letters were adjudicated too), but the
judge could not tell which reading was right. That would need a judge that is independent of the
reader (a different model, or labelled crops to fit one), which is left as future work.

## 5. Results log

| Date | Variant | Set | CER macro | WER macro | Notes |
|---|---|---|---|---|---|
| 2026-10-06 | Stage 1 (old prompt with leaked examples) | dev 56 pp | 6.00% | 8.94% | checkpoint |
| 2026-10-06 | Stage 1+2 shipped | dev 56 pp | 7.24% | 10.44% | checkpoint |
| 2026-10-06 | Stage 1+2 without text rules (replay) | dev 56 pp | 5.04% | 7.47% | offline replay |
| 2026-10-06 | Stage 2 old filters, fresh run | dev 36 pp | 4.67% | 7.02% | same Stage 1 |
| 2026-10-06 | Stage 2 no filters, fresh run | dev 36 pp | 4.28% | 6.51% | same Stage 1 |
| 2026-10-06 | Stage 2 old filters | dev 56 pp | 4.87% | 7.12% | same Stage 1 (old checkpoint) |
| 2026-10-06 | Stage 2 new default | dev 56 pp | 4.50% | 6.69% | same Stage 1 (old checkpoint) |
| 2026-10-06 | + Stage 2b all disagreements | dev 32 pp | 5.11% (from 4.88%) | 9.48% (from 7.74%) | rejected design |
| 2026-10-06 | + Stage 2b strikes only | dev 56 pp | 4.21% | 6.33% | adopted (old Stage 1) |
| 2026-10-06 | **Fresh run, final pipeline** | dev 56 pp | **4.52%** | **6.77%** | new Stage 1 prompt; Stage 1 alone 6.07% |

## 6. How to reproduce

```bash
source ./script_checking/bin/activate
# score any extraction directory against dev ground truth
python scripts/evaluate_transcription.py --lang english \
  --gt-dir data/ground_truth/transcripts/english_se_10_q1 \
  --extracted-dir outputs/extracted/english/se_10_q1
# what kind of errors make up the CER (false/missed strike, misread, autocorrection, dropped/extra)
python scripts/attribute_transcription_errors.py \
  --gt-dir data/ground_truth/transcripts/english_se_10_q1 \
  --extracted-dir outputs/extracted/english/se_10_q1_v2 --examples 10
# ablation: old text rules back on
#   set pipeline.legacy_text_rules: true in configs/pipeline_config.yaml, re-extract into another --output-dir
```

## 7. What is left to do (needs a person or a long GPU run)

1. **Proofread the test set.** Open each `page_<n>.png` / `page_<n>.txt` pair in
   `data/ground_truth/transcripts/english_se_10_q1_test/SE_10_Q1_00{29..33}/` (20 pages, Q10 and
   Q11). Correct the `.txt` to the exact ink: keep misspellings, use `[struck: ...]` for crossed-out
   text and one line per handwritten line. Then change `"status"` in the `.meta.json` to `"CORRECTED"`.
   The drafts come from the *old* pipeline's output; check names and crossed-out words especially
   carefully so the drafts do not bias the result.
2. **Score the test set once** (the new pipeline's test-page outputs already exist):
   ```bash
   python scripts/evaluate_transcription.py --lang english \
     --gt-dir data/ground_truth/transcripts/english_se_10_q1_test \
     --extracted-dir outputs/extracted/english/se_10_q1_v2 --tag se10_test_final
   ```
   For the old pipeline on the same pages (comparison row), use `--extracted-dir outputs/extracted/english/se_10_q1`.
3. **Re-extract all 33 scripts** so Stage 3 and Stage 4 use the new transcripts. The existing
   `outputs/extracted/english/se_10_q1/` was produced by the old pipeline, and so was the Stage 4
   evaluation run on it. This takes about 10–12 h on the GPU (Stage 2b adds about 15–20 s per page):
   ```bash
   python scripts/extract_scripts.py --lang english --pdf-dir data/raw_pdfs/english/se_10_q1 \
     --output-dir outputs/extracted/english_v2 --quant 4bit --force-extract -y
   ```
4. **Ablation rows for the thesis**: `pipeline.legacy_text_rules: true` (old rules back) and
   `pipeline.line_reconciliation: false` (no Stage 2b), each extracted into its own `--output-dir`.

Not addressed here: Stage 3's text-only error analysis (audit P6 part 2). It affects grading, not
transcription CER, and belongs with the Stage 3/4 work.
