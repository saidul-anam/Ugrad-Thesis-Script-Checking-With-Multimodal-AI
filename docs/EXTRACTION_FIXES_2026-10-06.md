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

## 4. In progress: visual reconciliation of disagreements

(Updated when the experiments finish; see section 5.)

Page-level reading (Stage 1/2) has context and good word accuracy. Reading a single line crop at
higher resolution is worse overall, but much better at noticing thin cross-out strokes. The two
make different mistakes. The planned mechanism:

1. Segment the page into text lines from the ink itself (horizontal projection profile, already in
   `arbitration/localizer.py`).
2. Read each line crop separately.
3. Align the page transcript lines to the crop reads (monotonic dynamic programming on text
   similarity).
4. For every span where the two readings disagree (different letters, or one of them struck), show
   the VLM the crop and both versions of the line, and ask which one matches the ink. Ask twice
   with the order swapped. Accept the change only if both answers pick it.

The decision comes from the pixels. No word, letter or threshold is hand-set; the only
acceptance rule is that the model's choice does not depend on the order the options were shown in.

## 5. Results log

| Date | Variant | Set | CER macro | WER macro | Notes |
|---|---|---|---|---|---|
| 2026-10-06 | Stage 1 (old prompt with leaked examples) | dev 56 pp | 6.00% | 8.94% | checkpoint |
| 2026-10-06 | Stage 1+2 shipped | dev 56 pp | 7.24% | 10.44% | checkpoint |
| 2026-10-06 | Stage 1+2 without text rules (replay) | dev 56 pp | 5.04% | 7.47% | offline replay |
| 2026-10-06 | Stage 2 old filters, fresh run | dev 36 pp | 4.67% | 7.02% | same Stage 1 |
| 2026-10-06 | Stage 2 no filters, fresh run | dev 36 pp | 4.28% | 6.51% | same Stage 1 |

## 6. How to reproduce

```bash
source ./script_checking/bin/activate
# score any extraction directory against dev ground truth
python scripts/evaluate_transcription.py --lang english \
  --gt-dir data/ground_truth/transcripts/english_se_10_q1 \
  --extracted-dir outputs/extracted/english/se_10_q1
# ablation: old text rules back on
#   set pipeline.legacy_text_rules: true in configs/pipeline_config.yaml, re-extract into another --output-dir
```
