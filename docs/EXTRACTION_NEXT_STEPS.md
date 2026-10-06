# Extraction accuracy — suggested next steps

Suggestions only; nothing here is implemented yet. What was already done, with numbers, is in
`docs/EXTRACTION_FIXES_2026-10-06.md`.

**Starting point** (fresh run on 56 class-10 dev pages, scripts 0001–0028): CER 4.52%, WER 6.77%,
26.5% of student misspellings silently corrected.

| Error type (4,933 words) | Count | Share |
|---|---|---|
| Misread letters | 128 | ~40% |
| Missed strikes (crossed-out words left as active text) | 58 | ~18% |
| Extra words | 48 | ~15% |
| Autocorrected misspellings | 34 | ~11% |
| Dropped words | 25 | ~8% |
| False strikes (active words marked as crossed out) | 20 | ~6% |

Reproduce this breakdown at any time:

```bash
source ./script_checking/bin/activate
python scripts/attribute_transcription_errors.py \
  --gt-dir data/ground_truth/transcripts/english_se_10_q1 \
  --extracted-dir outputs/extracted/english/se_10_q1_v2 --examples 10
```

## Ground rules for every step

- Try every change on the **dev** pages (0001–0028) only. Score the **test** pages (0029–0033) once, at
  the very end.
- Add each change as a row in the ablation table, with a config switch so it can be turned off.
- No per-word or per-letter rules and no hand-set thresholds. Decisions must come from the image, a
  learned or measured signal, or the question paper.

---

## 1. Measure before changing anything (cheap, do first)

**Why.** With only 56 dev pages, differences like 20 vs 36 false strikes may be within run-to-run noise.

1. **Proofread the 20 test pages** in `data/ground_truth/transcripts/english_se_10_q1_test/` and set
   `"status": "CORRECTED"` in each `.meta.json`. Then score once:
   ```bash
   python scripts/evaluate_transcription.py --lang english \
     --gt-dir data/ground_truth/transcripts/english_se_10_q1_test \
     --extracted-dir outputs/extracted/english/se_10_q1_v2 --tag se10_test_final
   ```
2. **Measure run-to-run variance.** Run Stage 1 two or three times on the dev pages into separate
   output folders and compare CER. Any later improvement smaller than this spread is not a real result.
3. **Add confidence intervals.** Bootstrap over pages (resample the 56 pages about 1,000 times) and report
   the 95% interval for CER/WER in the thesis.

## 2. Ablate the question-vocabulary block in the prompts (cheapest likely win)

**Why.** The Stage 1 and Stage 2 prompts list up to 250 words from the question paper and tell the
model to use them "to decipher ambiguous strokes". That pushes the model toward dictionary and
question words, which is the same direction as silent autocorrection ("suffuring" → "suffering").

**How.**
- Run Stage 1 on the dev pages with the vocabulary block, then without it (`build_stage1_prompt` in
  `src/prompts/stage1_verbatim.py`; same for `src/prompts/stage2_verification.py`).
- Compare CER, the autocorrection count and the silent-correction rate.
- Repeat for the syllabus and reference-numeral sections, one at a time.

**Cost.** About 1 h of GPU per variant. No new code beyond a config switch.

## 3. A second reader from a different model family, then vote (largest error buckets)

**Why.** Misreads (128) and autocorrections (34) are the largest buckets. Stage 2b showed that Gemma
judging Gemma's own readings is at chance on letters, because both readings share the same mistakes.
A reader from a different model family makes different mistakes, so agreement between the two
carries real information.

**How.**
1. Have **Qwen3.6-35B-A3B** (already loaded in LM Studio) read the same line crops that Stage 2b
   already produces.
2. First check how much its errors overlap with Gemma's on dev. If they mostly coincide, voting cannot
   help; stop here.
3. If they are mostly independent: align Gemma-page, Gemma-line and Qwen-line word by word and take the
   majority (ROVER, a standard combination method from speech recognition).
4. Where all three disagree, output `[unclear: a | b]` instead of guessing, so Stage 3/4 can treat the
   word as uncertain.

**Cost.** One extra model call per line. Both models must share the GPU (LM Studio API), so plan the
runs together with the Stage 4 work.

## 4. Use the model's own confidence to decide where to spend extra reads

**Why.** Re-reading everything is expensive and mostly confirms correct words. Token probabilities
are a learned signal of where the model is unsure.

**How.**
- Have the CUDA engine (`src/engine/gemma_cuda_engine.py`) return per-token log-probabilities and turn
  them into a per-word confidence.
- On dev, check calibration: do low-confidence words actually contain most of the errors?
- If they do, send only low-confidence words or lines to the second reader (step 3) or to a
  closer-zoom crop. The cut-off is chosen from the dev calibration curve, not set by hand.

## 5. Fine-tune Gemma for verbatim reading (most principled fix for autocorrection; bigger effort)

**Why.** Autocorrection comes from Gemma's language prior (it "knows" how words should be spelled).
Prompting fights the prior; fine-tuning changes it.

**How.**
- Build line-crop → ground-truth-line pairs. Line alignment already exists in
  `src/pipeline/line_reconciler.py`. Use class-11 ground truth plus class-10 **dev** pages, about 1,500 lines.
- QLoRA on the 4-bit model fits the RTX 5090.
- Keep the class-10 test pages completely out of training. Watch for overfitting: this is a small
  dataset, and handwriting from the same students appears on several pages, so split by **script**,
  not by page.

**Cost.** Several days including data preparation. Highest potential payoff on autocorrection.

## 6. Missed strikes: use the stroke detector's real coordinates

**Why.** 58 crossed-out words are still read as active text. Stage 0
(`src/pipeline/stage0_strikethrough_detector.py`) finds pen strokes with pixel positions, but its
precision and recall have never been measured.

**How.**
1. Measure Stage 0 on dev: annotate stroke boxes for a few pages, or check how many ground-truth
   `[struck: ...]` words have a detected stroke on the same text line.
2. If it is reliable, crop tightly around each detected stroke, zoom in, and ask which words the stroke
   crosses. Feed the answer through the same order-swapped check Stage 2b uses.

## 7. Extra and dropped words: check where pages are split into bands

**Why.** 73 errors together. Stage 1 reads tall pages in overlapping horizontal bands
(`src/pipeline/band_tiler.py`) and joins the results. Overlaps can duplicate or lose words.

**How.**
1. Using the attribution script with `--per-page --examples`, check whether extra and dropped words
   cluster at band boundaries.
2. If they do, join bands by aligning the overlapping text rather than the current heuristics.
3. Separately, try rendering pages at 300 DPI instead of 200 (`dpi=200` in `orchestrator.py`) and
   compare.

## 8. Make uncertainty part of the output (useful even where accuracy cannot improve further)

**Why.** Some handwriting is genuinely ambiguous. Instead of guessing, the pipeline can say which
words it is unsure about.

**How.**
- Mark contested words: Stage 2b disagreements, voting splits (step 3), low confidence (step 4).
- Let Stage 3/4 skip penalties on uncertain words.
- Report how accuracy rises if a human checks only the flagged words (accuracy vs. % of words
  reviewed). That is a defensible thesis result in its own right.

---

## Suggested order

| Priority | Step | Effort | Targets |
|---|---|---|---|
| 1 | Measurement (test set, variance, confidence intervals) | Low | Trustworthy numbers |
| 2 | Ablate prompt vocabulary | Low | Autocorrection, misreads |
| 3 | Second reader + voting | Medium | Misreads, autocorrection |
| 4 | Confidence-targeted re-reads | Medium | Cost of step 3 |
| 5 | Band-stitching check, 300 DPI | Low–medium | Extra / dropped words |
| 6 | Stroke-detector-guided strike checks | Medium | Missed strikes |
| 7 | Uncertainty output for Stage 3/4 | Medium | Fair grading |
| 8 | QLoRA fine-tuning | High | Autocorrection |

If time allows only three: steps 1, 2 and 3.

## Out of scope here

Stage 3's text-only error analysis (audit item P6, second half) affects grading, not transcription
CER. It belongs with the Stage 3/4 work.
