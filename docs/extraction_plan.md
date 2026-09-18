# Extraction Plan: Dynamic Handwriting Calibration, Long-Answer Recall & Pipeline Optimization

## Goal Description
Resolve the critical failure mode where Stage 3 returns 0 errors on long essay, chart, and narrative questions (Q7, Q8, Q9) due to context/token limits and over-cautious negative prompting. Replace brittle hardcoded cursive character rules with dynamic statistical character-alignment discovery across all 26 alphabets. Concurrently de-bloat the arbitration pipeline by replacing heavy 5x consensus sampling with single deterministic visual checks, cutting script processing time from ~15 minutes down to ~3 minutes.

---

## Key Strategic Principles

> [!IMPORTANT]
> **Stage 3 Prompt Decoupling**: We remove the heavy "Benefit of the Doubt / Do NOT penalize handwriting" warnings from the **Stage 3 Text Prompt**. Stage 3's mandate is strictly **inclusive candidate error extraction** (catching all grammar, syntax, and spelling irregularities). 
> The **Benefit of the Doubt doctrine is preserved in Stage 3b (`gate.py`)**, where it inspects the actual physical ink crops at native resolution. This prevents text-level caution from suppressing genuine grammar and tense errors.

> [!TIP]
> **Answer Chunking**: For subjective answers exceeding 150 words (such as Q7 Paragraph, Q8 Chart, and Q9 Story), the orchestrator evaluates the text in ~100–120 word sentence-bounded chunks and merges the results. Our empirical audit showed that answers under 140 words have near 100% error recall, while monolithic 200–300 word texts frequently collapse to 0 errors.

> [!TIP]
> **Pipeline De-Bloating**: Prune the 5-sample stochastic consensus re-reading loop in `gate.py` (which caused up to 263 VLM calls per script). A single deterministic forced-choice call on the crop provides identical visual fidelity while cutting runtime by ~75%.

---

## Proposed Changes

### Component 1: Prompt & Error Extraction (Stage 3)

#### [MODIFY] [stage3_errors.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/prompts/stage3_errors.py)
* Refocus `STAGE3_PROMPT_TEMPLATE`:
  * Remove the extensive wall of negative handwriting instructions that causes the model to suppress valid errors.
  * Direct the model to extract all grammatical, syntactic, and spelling deviations.
  * Add explicit rule for narrative past contexts:
    > *"In narrative writing (e.g. Q9 Story Completion), actions set in the past MUST use past tense verbs. Base forms (e.g. 'wake' for 'woke', 'reply' for 'replied', 'be' for 'was') must be extracted as grammar errors. Do NOT flag present-tense verbs occurring inside direct speech quotation marks (e.g. 'I will help you')."*
* Preserve existing filters for exam headers and punctuation marks.

#### [MODIFY] [stage3_error_analyzer.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage3_error_analyzer.py)
* In `run()`:
  * Add diagnostic logging: If a transcript $>50$ words returns 0 errors, log a warning with a snippet of the raw model response to detect JSON truncation or malformed generation immediately.
  * Ensure `_extract_json_from_text` fallback reports a parsing warning instead of silently returning empty errors.

---

### Component 2: Orchestration & Long-Answer Chunking

#### [MODIFY] [orchestrator.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/orchestrator.py)
* In the Stage 3 evaluation loop over `aligned_answers`:
  * Implement sentence-aware chunking for long subjective answers where `len(ans_text.split()) > 150`:
    * Split into chunks of ~100–120 words strictly along sentence boundaries (`[.!?]\s+`).
    * Run Stage 3 on each chunk and deduplicate/merge extracted errors.
  * Increase `max_new_tokens` passed to `stage3.run()` from `1536` to `3072` so that long error lists do not get truncated mid-JSON.

---

### Component 3: Dynamic Allograph Discovery (All 26 Alphabets)

#### [MODIFY] [allograph_calibrator.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/allograph_calibrator.py)
* Replace the hardcoded `Hypothesis A, B, C, D` checks with a **generalized Levenshtein character-substitution counter**:
  * For all Out-Of-Vocabulary (OOV) words in the script:
    1. Find closest dictionary words within Levenshtein edit distance $\le 2$.
    2. Compute character edit operations via `align_chars(oov, dict_word)`.
    3. Accumulate `pair_counts[(src_char, tgt_char)]` across the entire script.
    4. Automatically promote any character substitution that appears across $\ge 2$ distinct words in this script as an active allograph for this writer.
  * Ensures coverage across all 26 alphabets (e.g. `f` $\leftrightarrow$ `b`, `k` $\leftrightarrow$ `h`, `r` $\leftrightarrow$ `re`, `d` $\leftrightarrow$ `t`) without manual hardcoded rules.
  * Protects real dictionary words via the Lexicon Gate so valid words are never falsely rewritten.

---

### Component 4: Stroke-Level vs Structural Grammar Gating

#### [MODIFY] [candidate_selector.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/arbitration/candidate_selector.py)
* Ensure inflectional verb affix changes (`wake` $\leftrightarrow$ `woke`, `reply` $\leftrightarrow$ `replied`, `live` $\leftrightarrow$ `lives`, `walk` $\leftrightarrow$ `walked`) are always classified as gate candidates for Stage 3b visual arbitration.

#### [MODIFY] [gate.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/arbitration/gate.py)
* Maintain Visual Evidence Supremacy:
  * If the high-res crop shows the student physically penned the incorrect letter (e.g. clear 'a' in `wake`), mark `GENUINE_ERROR` (deduction applied).
  * If the crop shows a ligature or closed loop (e.g. `woke` misread as `wake`), mark `HANDWRITING_AMBIGUITY` (deduction dropped, transcript normalized).

---

### Component 5: Pipeline De-Bloating & Speed Optimization

#### [MODIFY] [configs/pipeline_config.yaml](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/configs/pipeline_config.yaml) & [gate.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/arbitration/gate.py)
* **Disable 5x Consensus Sampling**: Set `consensus_samples: 1` or bypass sampling in favor of direct deterministic forced choice.
* **Disable Speculative Lexicon Flooding**: Turn off `lexicon_scan` in arbitration config (`lexicon_scan: false`) so Stage 3b only arbitrates errors actually flagged by Stage 3, rather than fabricating dozens of unflagged dictionary candidates.
* **Expected Impact**: Reduces total VLM calls per script from ~260 calls to ~30 calls. Script execution time drops from 12–15 minutes down to 2–3 minutes.

---

## Verification Plan

### Automated Tests
1. **Dynamic Allograph Unit Test**:
   Verify arbitrary letter pairs (e.g. `f` $\leftrightarrow$ `b`, `k` $\leftrightarrow$ `h`) are dynamically discovered when repeated across multiple OOV words, while isolated single misspellings are NOT registered as allographs.
2. **Chunking Unit Test**:
   Verify that a 300-word answer is partitioned into overlapping sentence chunks without losing words or sentence integrity.

### Manual Verification on Ground-Truth Script `SE_11_Q1_0002`
1. Re-run Stage 3 & Stage 3b on `SE_11_Q1_0002`:
   * **Question 9 (Lion & Mouse)**: Verify error count goes from **0** to **~7** genuine tense errors (`wake` $\rightarrow$ `woke`, `reply` $\rightarrow$ `replied`, `be free` $\rightarrow$ `was freed`).
   * **Question 7 (AI Paragraph)**: Verify real errors (`AI have`, `works life`) are detected, while OCR noise is safely arbitrated by Stage 3b.
   * **Question 8 (Chart Analysis)**: Verify genuine errors (`plants also a important`, `source for produce`) are extracted.
2. Inspect `stage3b_arbitration.json` to confirm:
   * Genuine tense errors (`wake`) receive `GENUINE_ERROR`.
   * OCR stroke ambiguities receive `HANDWRITING_AMBIGUITY`.
   * Total model calls drop from 200+ to under 40.
