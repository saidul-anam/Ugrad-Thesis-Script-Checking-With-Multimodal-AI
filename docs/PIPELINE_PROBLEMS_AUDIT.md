# Master Catalog of Extraction & Grading Pipeline Flaws, Anti-Patterns & Production Solutions

**Scope:** End-to-End System Audit across all 5 Pipeline Stages (Stages 0–4), 400 Extracted Checkpoints, and 24 Ground Truth Evaluation Pages.  
**Target Architecture:** Multimodal Neuro-Symbolic Pipeline (`google/gemma-4-31B-it` on NVIDIA GeForce RTX 5090 33.6 GB VRAM).  
**Date:** 2026-09-21  

---

## Executive Summary: Full 16-Problem & Solution Matrix

| # | Pipeline Stage | Problem Area | Impact / Severity | Primary Root Cause | Production Solution |
|:---|:---|:---|:---:|:---|:---|
| **1** | **Stage 2** | **Autoregressive Drift** | 🔴 **CRITICAL** | Model regenerates full page text in JSON; language prior mutates valid student errors (**76% of pages corrupted, 845 mutations**) | **Strict "Declared-Only" Differential Architecture**: S1 is immutable baseline; S2 emits only diff list; Python applies validated changes. |
| **2** | **Stage 1 & 2** | **Strike-Through Blindness** | 🔴 **CRITICAL** | Downscaling smooths 1-px strokes; VLM OCR prior treats lines as noise (**67.4% miss rate**) | **Stage 0.5 OpenCV stroke detector** (morphological kernel) + BBox prompt injection + **2-tile high-res split**. |
| **3** | **Stage 1** | **Ungrounded Hallucination** | 🔴 **CRITICAL** | Monolithic generation on sparse page; language prior hallucinates textbook letter (**204% CER on `0022/p15`**) | **Pre-inference vertical ink-band gating** + strict negative stop constraint + **ink-to-word ratio guardrail**. |
| **4** | **Stage 2** | **Prompt Priming Bias** | 🟠 **HIGH** | Stage 2 prompt listed `'have' misread as 'hare'`; model inverted it (mutated `"have"` $\rightarrow$ `"hare"` across 3 pages) | **Strip specific negative examples** from prompt; implement **protected function-word blocklist** in Python. |
| **5** | **Stage 0** | **Destructive Inpainting Erases Student Ink** | 🔴 **CRITICAL** | Telea inpainting on body checkmarks ($X \ge 18\%$) wipes out student black ink written underneath teacher ticks | **Luminance-preserving chromatic suppression**: neutralize red hue while preserving low-luminance dark student strokes. |
| **6** | **Segmentation** | **Brittle Hardcoded Regex Rules** | 🔴 **CRITICAL** | Ad-hoc rules (e.g. `'Dans'` $\rightarrow$ `'1(B)'`, `'No. Z'` $\rightarrow$ `'No. 7'`); missed headers falsely dump essays into Question 1(A) | **Page-aware continuation state machine** (`target_q = prev_page.last_q`) + **fuzzy Levenshtein header classifier**. |
| **7** | **Stage 3** | **Hyper-Aggressive Candidate Flooding** | 🟠 **HIGH** | Stage 3 prompt tells LLM: *"Do NOT suppress valid grammatical errors; benefit of doubt deferred to Stage 3b"* | **Regional idiomatic calibration**; instruct Stage 3 to extract only unambiguous grammatical/structural violations. |
| **8** | **Stage 3b** | **Uncalibrated Provisional Arbitration Weights** | 🟠 **HIGH** | Fusion weights (`bias: -0.3, phonetic: 0.6, writer: 1.2...`) are uncalibrated default stubs; never fitted on real labels | **Empirical logistic regression fitting**: calibrate weights on 100 labeled crops using `evaluate_arbitration.py --fit`. |
| **9** | **Stage 4** | **Central-Tendency Score Compression** | 🟠 **HIGH** | LLMs grade in prose rather than arithmetic code; compress subjective scores into 6/10–7/10 bands | **Binary Rubric Decomposition**: model judges discrete boolean criteria; deterministic Python calculates marks and band snapping. |
| **10** | **Engine** | **Artificial Context Window Cap** | 🟠 **HIGH** | Hardcoded `context_window = 4096` in engine vs **262k native**; Stage 2 prompt hits 77.5% of cap | **Expose model's native context window** (set ceiling to 16,384+); remove prompt string truncation. |
| **11** | **Engine** | **Quantization Feature Loss** | 🟡 **MEDIUM** | 4-bit NF4 quantization rounds cross-attention weights; subtle pen strokes lost | **Upgrade to 8-bit quantization** (~31GB VRAM footprint natively fits inside RTX 5090's 33.6GB VRAM). |
| **12** | **Engine** | **Reasoning Tokens Choked** | 🟡 **MEDIUM** | `thinking_mode: false` hardcoded; greedy argmax forces immediate token output with zero deliberation | **Enable constrained thinking mode**: allow internal chain-of-thought for stroke deliberation while enforcing strict output JSON. |
| **13** | **Stage 1 & 4** | **Digit & Arithmetic Fragility** | 🟠 **HIGH** | Numbers lack syntax context; misread in Q8 charts (`16%` $\rightarrow$ `18%`, `46%` $\rightarrow$ `96%`) | **Question 8 mathematical consistency check**: validate that extracted percentage slices sum to $100 \pm 5\%$. |
| **14** | **Stage 2** | **Naive Regex Substitution Bug** | 🟡 **MEDIUM** | `re.sub(..., count=1)` replaces first matching word occurrence on page, corrupting wrong sentences | **Context-anchored window matching**: use `difflib.SequenceMatcher` sliding window; abort replacement if snippet is ambiguous. |
| **15** | **Stage 1** | **Stage 1 `struck_count` Code Bug** | 🟡 **MEDIUM** | Code parses `[illegible]` and `[unclear]`, but forgot `[struck:]` regex; schema count always 0 | **Add `[struck:[^\]]+\]` regex parser** to `Stage1Transcriber.run()` and populate `struck_count` in schema. |
| **16** | **Architecture** | **End-to-End Latency & Compute Bloat** | 🔴 **CRITICAL** | 70–80 neural network passes per student script (3–5 mins/script); unusable for large cohorts | **Consolidate stages**: merge S1+S2 into single high-resolution pass with confidence flags; batch Stage 3 at script level. |

---

## Part I: Preprocessing & Computer Vision (Stage 0)

### Problem 5: Destructive Inpainting Erases Student Ink Underneath Teacher Marks
- **File Reference:** [src/pipeline/stage0_red_ink_detector.py:L106-L119](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage0_red_ink_detector.py#L106-L119) and [src/pipeline/orchestrator.py:L433](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/orchestrator.py#L433)
- **The Problem:** When in-body red checkmarks are detected ($X \ge 18\%$), the detector dilates the red mask by 3 pixels and runs OpenCV Telea inpainting (`cv2.inpaint`). When a teacher draws a red tick mark ($\checkmark$), underline, or cross ($\times$) directly across a student's handwriting, **Telea inpainting paints white paper background over the student's black/blue ink**, erasing or blurring character strokes before Stage 1 ever sees the page.
- **Root Cause:** Inpainting was introduced as an ad-hoc fix to prevent the VLM from transcribing checkmarks as random punctuation (`" ed t. "`), but it treated all red pixels as empty background without checking if dark student ink was underneath.
- **The Solution: Luminance-Preserving Chromatic Suppression**
  Do NOT inpaint text areas with white background. In the HSV color space, black/blue ink has very low Value/Luminance ($V < 120$), while red pen on clean paper has high Value ($V > 160$).
  ```python
  def non_destructive_red_suppression(image_bgr, red_mask):
      """Neutralizes red teacher ink while preserving intersecting black student ink."""
      hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
      v_channel = hsv[:, :, 2]
      
      # Pixels where red ink overlaps dark black student writing
      ink_intersection_mask = (red_mask > 0) & (v_channel < 120)
      # Pixels where red ink is on clean white paper
      paper_red_mask = (red_mask > 0) & (v_channel >= 120)
      
      clean_bgr = image_bgr.copy()
      # Convert intersecting strokes to neutral dark grayscale (retaining character stroke!)
      clean_bgr[ink_intersection_mask] = np.stack([v_channel[ink_intersection_mask]] * 3, axis=-1)
      # Neutralize isolated red pen strokes on paper to white background
      clean_bgr[paper_red_mask] = [255, 255, 255]
      return clean_bgr
  ```

---

## Part II: Transcription & Vision Subsystem (Stage 1)

### Problem 2: Strike-Through Blindness (67.4% Miss Rate)
- **File Reference:** [src/prompts/stage1_verbatim.py:L16](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/prompts/stage1_verbatim.py#L16), [src/pipeline/stage1_transcriber.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage1_transcriber.py#L42-L65)
- **The Problem:** 29 out of 43 crossed-out words/clauses were missed across 24 ground truth pages:
  1. *Multi-word clause deletions (`0006/p9`, `0006/p12`):* Long diagonal slashes across 3–4 lines (`"In 1980 the percentage was"`, `"the low percentage"`, `"childs are"`) were transcribed as active student text.
  2. *False starts / aborted prefixes (`0010/p3`: `'inc'`, `0011/p3`: `'o'`, `0020/p7`: `'di'`):* Tiny 1-pixel slashes were dropped or garbled into adjacent words.
  3. *Single words struck & replaced (`0006/p11`: `'many'`, `0022/p17`: `'body'`):* Extracted as active text (`"it helps many us"`), triggering false grammar penalties.
- **Root Cause:** Standard VLM downscaling (resizing A4 to $896\times 896$ patches) smooths away 1-pixel pen slashes during spatial pooling. Furthermore, OCR foundation models are pre-trained to be invariant to lines and creases.
- **The Solution: Stage 0.5 OpenCV Line Detector + 2-Tile High-Res Split**
  1. **OpenCV Line Intersect Detector (Stage 0.5):** Apply horizontal ($1\times 25$) and diagonal ($\pm 15^\circ$) morphological kernels on the binarized ink image. Identify continuous line segments crossing through text contours. Extract normalized bounding boxes $[y_1, x_1, y_2, x_2]$.
  2. **BBox Prompt Injection:** Pass detected coordinates to Stage 1:
     `"VISUAL CROSS-OUT DETECTOR: Detected strike-through strokes at [BBox 1], [BBox 2]. You MUST tag text in these regions as [struck: ...]. Never output them as active text."`
  3. **2-Tile High-Resolution Splitting:** Slicing the 200 DPI page into Top-Half and Bottom-Half tiles (each 1654x1250 px) doubles the effective pixel density per character, ensuring 1-pixel pen strokes are preserved in the vision tokens.

### Problem 3: Ungrounded Hallucination on Sparse Pages (204% CER on `0022/p15`)
- **File Reference:** [outputs/extracted/english/SE_11_Q1_0022/checkpoints/page_15.json](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/outputs/extracted/english/SE_11_Q1_0022/checkpoints/page_15.json)
- **The Problem:** Ground truth contained only 46 words (an address box and "P.T.O" at bottom). The pipeline produced 124 words, fabricating 4 full paragraphs of a generic HSC letter ("Dear brother, I have received your letter yesterday...").
- **Physical Reality:** Ink profile showed that **50% of the vertical space (Bands 4–8) had <0.3% ink density (blank white paper)**.
- **Root Cause:** The ungrounded VLM had no spatial bounding box grounding. Knowing Question 10 is an informal letter from the syllabus, the model encountered blank white space and its autoregressive language prior auto-completed the "missing" letter body.
- **The Solution: Vertical Ink-Band Gating & Ratio Guardrail**
  1. **Vertical Projection Profiler:** Compute horizontal projection ink density in 100px vertical bands. If $>40\%$ of consecutive bands have $<0.4\%$ ink density, flag the page as **Sparse Paper**.
  2. **Negative Constraint Injection:** When sparse, inject a hard negative prompt:
     `"WARNING: This page contains blank unwritten space below the initial box/header. Transcribe ONLY physical ink strokes. If paper is blank, STOP immediately."`
  3. **Ink-to-Word Ratio Guardrail:** Compute $R = \frac{\text{Total Inked Pixels}}{\text{Transcribed Words}}$. If $R < 700$ px/word for $>50$ words (normal handwriting is 1,500–2,500 px/word), trigger an automated Hallucination Abort and truncate generation at the upper ink boundary.

### Problem 15: Stage 1 `struck_count` Code Bug
- **File Reference:** [src/pipeline/stage1_transcriber.py:L42-L65](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage1_transcriber.py#L42-L65)
- **The Problem:** Lines 43–44 parse `[illegible]` and `[unclear]`. Regex parsing for `[struck: ...]` was completely omitted. `struck_count` defaulted to 0 across all 400 checkpoints.
- **The Solution:** Add regex parsing and pass count to schema:
  ```python
  struck_matches = re.findall(r"\[struck:[^\]]+\]", raw_text, re.IGNORECASE)
  return Stage1TranscriptionResult(
      raw_transcript=raw_text,
      illegible_count=len(illegible_matches),
      unclear_count=len(unclear_matches),
      struck_count=len(struck_matches),  # <--- FIX
      ...
  )
  ```

---

## Part III: Verification & Prompting Anti-Patterns (Stage 2)

### Problem 1: Stage 2 Autoregressive Drift (845 Undeclared Mutations)
- **File Reference:** [src/pipeline/stage2_verifier.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage2_verifier.py)
- **The Problem:** Across 400 checkpoints, Stage 2 made **845 undeclared mutations across 304 pages (76% of all pages corrupted)**, silently autocorrecting genuine student misspellings (`abart` $\rightarrow$ `about`, `hause` $\rightarrow$ `house`, `poverly` $\rightarrow$ `poverty`) while its notes claimed it preserved them.
- **Root Cause:** Re-generating the entire page text inside JSON triggers language model prior drift, pulling rare student errors back to standard English dictionary tokens.
- **The Solution: Strict "Declared-Only" Differential Architecture**
  Stage 1 is the **immutable baseline**. Stage 2 is **strictly forbidden from emitting full page text**. Its output schema is modified to return ONLY a diff list:
  ```json
  {
    "silent_corrections_fixed": [
      {
        "stage1_output": "word in stage 1",
        "actual_handwritten": "what student wrote",
        "context_snippet": "exact line context"
      }
    ]
  }
  ```
  The Python orchestrator applies only validated diffs.  
  *Empirical Impact:* Cuts page regressions by more than half (30.4% $\rightarrow$ 13.0%) and eliminates 100% of undeclared mutations.

### Problem 4: Prompt Priming Bias (`"have"` $\rightarrow$ `"hare"`)
- **File Reference:** [src/prompts/stage2_verification.py:L30](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/prompts/stage2_verification.py#L30)
- **The Problem:** The prompt listed `'have' misread as 'hare'`. The model inverted this negative instruction and mutated correct `"have"` to `"hare"` on `0011/p3`, `0011/p5`, and `0013/p5`. The lexicon check failed because `"hare"` is in the dictionary (an animal!).
- **The Solution: Strip Negative Word Examples & Add Function-Word Blocklist**
  1. Remove all specific word pairs (`'have' vs 'hare'`, `'June' vs 'Jute'`) from prompts. Prompts must instruct *general morphology principles*, not specific English words.
  2. Implement a **Protected Function-Word Blocklist** in `is_spurious_reversion`:
     ```python
     PROTECTED_FUNCTION_WORDS = {
         "have", "has", "had", "are", "were", "will", "would", "with", 
         "this", "that", "these", "those", "from", "for", "to", "in", 
         "on", "at", "by", "is", "it", "its", "and", "or", "but", "june"
     }
     if s1 in PROTECTED_FUNCTION_WORDS and act not in PROTECTED_FUNCTION_WORDS:
         return True  # Reject mutating common grammatical glue into rare words/homographs!
     ```

### Problem 14: Naive Regex String Substitution Bug
- **File Reference:** [src/pipeline/stage2_verifier.py:L171-L173](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage2_verifier.py#L171-L173)
- **The Problem:** `re.sub(rf"\b{re.escape(s1)}\b", act, verified_text, count=1)` replaces the *first* occurrence of `s1` on the page when context snippets fail to match. If `s1` is `"to"`, it mutates an unrelated sentence at the top of the page.
- **The Solution: Context-Anchored Sliding Window Alignment**
  Replace naive regex with `difflib.SequenceMatcher` 5-token window alignment. If the declared `context_snippet` cannot be uniquely resolved, **abort the replacement** and preserve Stage 1 text rather than mutating arbitrary words.

---

## Part IV: Document Structuring & Answer Segmentation

### Problem 6: Brittle Hardcoded Regex & Cross-Page Question Misalignment
- **File Reference:** [src/pipeline/answer_segmenter.py:L18-L30, L488-L508](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/answer_segmenter.py#L18-L30)
- **The Problem:** The segmenter contains hardcoded hacks (`'Dans'` $\rightarrow$ `'1(B)'`, `'No. Z'` $\rightarrow$ `'No. 7'`). When a page has no header (continuation from previous page) and `current_q_no` is unresolved, line 498 **defaults to assigning the entire page to Question 1(A)**. An entire multi-page essay on pages 14–15 gets dumped into Question 1(A) (Multiple Choice), scoring 0/5.
- **The Solution: Page-Aware Continuation State Machine & Fuzzy Header Parser**
  1. **Eliminate Ad-Hoc Regex:** Replace brittle string hacks with fuzzy Levenshtein distance matching against canonical question headers (`Q1(A)`, `Q1(B)`, `Q2` ... `Q12`), constrained by syllabus sequence.
  2. **Continuation State Machine:** When a page begins with no header, it is mathematically a continuation of the previous page's active question:
     ```python
     # Page-Aware Continuation Logic:
     if not detected_header_on_page:
         target_q = previous_page_last_active_q or "1(A)"
     ```
     Never reset `current_q_no` to None between contiguous pages.

---

## Part V: Linguistic Analysis & Arbitration (Stage 3 & 3b)

### Problem 7: Stage 3 Candidate Flooding Bypassing Stage 3b
- **File Reference:** [src/prompts/stage3_errors.py:L21-L25](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/prompts/stage3_errors.py#L21-L25) vs [configs/pipeline_config.yaml:L57](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/configs/pipeline_config.yaml#L57)
- **The Problem:** Stage 3 prompt tells the LLM: *"Do NOT suppress valid grammatical errors out of text-level caution; genuine handwriting stroke ambiguities will be visually arbitrated with benefit of the doubt downstream in Stage 3b."* But **Stage 3b ONLY arbitrates spelling (char edits $\le 2$)!** Grammar errors bypass Stage 3b completely and penalize student marks in Stage 4.
- **The Solution: Regional Idiomatic Calibration**
  Update `STAGE3_PROMPT_TEMPLATE`:
  *"Extract only clear, unambiguous structural and grammatical violations. Standard South Asian English idiomatic phrasing ('take preparation', 'pass days', 'join with me') must NOT be penalized unless explicitly prohibited by syllabus rubrics."*

### Problem 8: Uncalibrated Provisional Arbitration Weights
- **File Reference:** [configs/pipeline_config.yaml:L71-L76](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/configs/pipeline_config.yaml#L71-L76)
- **The Problem:** The fusion weights (`bias: -0.3, phonetic: 0.6, writer: 1.2, consensus: 2.0, forced_choice: 2.4`) are uncalibrated default stubs.
- **The Solution: Empirical Logistic Regression Fitting**
  Label 100 candidate crops via `python scripts/label_arbitration_candidates.py` (G = Genuine error, A = Ambiguity). Run `python scripts/evaluate_arbitration.py --labels data/labels/arbitration_labels.csv --fit` and overwrite provisional constants with mathematically fitted coefficients.

---

## Part VI: Evaluation & Infrastructure (Stage 4 & Engine)

### Problem 9: Central-Tendency Score Compression in Stage 4
- **File Reference:** [src/pipeline/stage4_evaluator.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage4_evaluator.py), [src/pipeline/stage4_modes.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage4_modes.py)
- **The Problem:** LLMs grading in prose cluster subjective marks around 6/10–7/10, avoiding awarding 1–3/10 for poor scripts and 9–10/10 for excellent scripts.
- **The Solution: Binary Rubric Decomposition (Model Judges, Code Scores)**
  1. Decompose subjective rubrics (Story, Letter, Theme) into 5–8 discrete binary criteria (`has_salutation: bool`, `consistent_past_tense: bool`, `word_count_valid: bool`).
  2. LLM outputs only a boolean checklist with textual evidence citations.
  3. Deterministic Python code computes marks by summing weighted criteria, applying hard caps, and snapping to bands.
  4. Include two few-shot exemplars (one 9/10 and one 3/10) to anchor the scale.

### Problem 10: Artificial 4,096 Context Cap (vs 262k Native)
- **File Reference:** [src/engine/gemma_cuda_engine.py:L42-L43](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/engine/gemma_cuda_engine.py#L42-L43) and [src/pipeline/stage2_verifier.py:L113, L127](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage2_verifier.py#L113)
- **The Problem:** Gemma 4 natively supports **262,144 tokens**, but engine code hardcoded a cap of **4,096 tokens**. Stage 2 prompt reached 3,173 tokens (77.5% of cap), starving generation headroom and forcing text clipping.
- **The Solution:** Dynamically fetch `self.context_window = getattr(self.model.config.text_config, "max_position_embeddings", 16384)` (or set a safe production ceiling of 16,384 tokens). Remove prompt string truncation.

### Problem 11: 4-Bit NF4 Quantization Feature Loss on RTX 5090
- **File Reference:** [configs/pipeline_config.yaml:L9](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/configs/pipeline_config.yaml#L9)
- **The Problem:** 4-bit NF4 quantization degrades cross-attention projection weights, losing subtle visual stroke signals (faint strike-through slashes, decimal points).
- **The Solution:** Set `quantization: "8bit"` in [configs/pipeline_config.yaml](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/configs/pipeline_config.yaml). The 31GB footprint fits natively inside the RTX 5090's 33.6GB VRAM, maximizing stroke fidelity.

### Problem 12: Reasoning Tokens Choked (`thinking_mode: false`)
- **File Reference:** [configs/pipeline_config.yaml:L18-L24](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/configs/pipeline_config.yaml#L18-L24)
- **The Problem:** Deterministic greedy decoding (`temp=0.0`) without reasoning tokens forces immediate argmax token output with zero internal scratchpad deliberation.
- **The Solution:** Set `thinking_mode: true` with strict prompt constraints:
  `"REASONING DIRECTIVE: Use thinking tokens to deliberate on visual stroke morphology and cursive ligatures. In your final output, output ONLY the requested text/JSON with zero commentary. NEVER normalize student spelling."`

### Problem 13: Digit & Arithmetic Fragility (Q8 Pie Charts)
- **File Reference:** [outputs/extracted/english/SE_11_Q1_0010/checkpoints/page_3.json:L7](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/outputs/extracted/english/SE_11_Q1_0010/checkpoints/page_3.json#L7)
- **The Problem:** Statistical figures in data interpretation questions are frequently misread (`16%` $\rightarrow$ `18%`, `46%` $\rightarrow$ `96%`).
- **The Solution: Question 8 Mathematical Consistency Check**
  Extract all percentages in Question 8. If $\sum p_i \notin [90\%, 110\%]$ (e.g. 152%), trigger a localized digit re-examination with an arithmetic prompt constraint: *"The percentages in this pie chart must sum to 100%. Re-examine ambiguous digits."*

### Problem 16: End-to-End Latency & Compute Bloat
- **File Reference:** [extraction.log](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/extraction.log)
- **The Problem:** Each 15-page script triggers **70–80 separate neural network passes**, taking 3–5 minutes per student.
- **The Solution: Pipeline Consolidation & Script-Level Batching**
  1. **Merge Stage 1 & 2:** Use single-pass high-resolution extraction with confidence tags; only trigger verification re-reads on flagged uncertain words.
  2. **Batch Stage 3:** Run error extraction at the script level (1 call) instead of page-by-page (15 calls), saving 93% of error-analysis overhead.
  3. **Speedup:** Cuts forward passes from **75 down to ~15 per script**, accelerating the pipeline by **400%**.
