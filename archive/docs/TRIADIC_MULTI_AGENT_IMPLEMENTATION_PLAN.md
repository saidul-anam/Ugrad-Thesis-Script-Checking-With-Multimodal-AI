# Triadic Multi-Agent Architecture Implementation Plan
## Autonomous School-Level English Exam Script Evaluation

> **System Paradigm**: Triadic Collaborative Multi-Agent Architecture (Extractor ➔ Negative Critic ➔ Examiner Supervisor)  
> **Theoretical Foundation**: AAAI 2024 (*AutoSCORE*), ACL 2024 (*CAFES*), Cambridge Assessment & Pearson Edexcel Benefit-of-the-Doubt (BOD) Standards  
> **Target Engine**: `google/gemma-4-31b-it` @ 4-bit on NVIDIA RTX 5090  
> **Goal**: 100% autonomous, explainable, and scientifically rigorous evaluation of handwritten exam scripts across arbitrary schools and question types, with zero ad-hoc regex patches.

---

## 1. Architectural System Overview

The system transitions from a fragile sequential pipeline into a **Two-Stream Triadic Multi-Agent Architecture**:

```
══════════════════════════════════════════════════════════════════════════════════════════════
               STREAM 1: FORENSIC PERCEPTION STREAM (Vision ➔ Immutable Surface)
══════════════════════════════════════════════════════════════════════════════════════════════

   [Raw Script PDF] 
          │
          ▼
   1. Document Canvas Cleaner (clean_pdf.py)
      • Inpaints red examiner grading marks and reverse-side bleed-through.
      • Produces a pristine white canvas with authentic student black/blue pen strokes.
          │
          ▼
   2. Forensic Paleographer Agent (stage1_transcriber.py + BandTiler)
      • Role: Perceptual Grounding Agent (Gemma 4 31B IT, T=0.0 greedy, thinking OFF).
      • Native-DPI horizontal band tiling preserves fine ascenders/descenders.
      • Strict zero-autocorrect paleography prompt: transcribes physical ink character-by-character.
      • Decoupled from OpenCV coordinate prompts (no lined-paper false strikethrough bias).
          │
          ▼
   3. Deterministic Structural Tokenizer Tool (split_token_stitcher.py - Pure CPU)
      • Replaces legacy Stage 2 generative VLM: 0 VRAM, 0 hallucinations, <50ms latency.
      • Recombines pen-lift broken syllables (day dreaming ➔ daydreaming) via English lexicon.
      • Recombines line-wrap hyphenated words at physical page margins.
      • Strict Non-Word Invariance: STRUCTURALLY FORBIDDEN from converting non-words into dictionary words.
          │
          ▼
   ==========================================================================================
   ★ THE IMMUTABLE TRANSCRIPT OF RECORD (FROZEN SURFACE) ★
   The transcript is permanently locked into stage2_verified_transcript.txt.
   No downstream component is ever permitted to edit or back-mutate it.
   ==========================================================================================

                                              │
                                              ▼
══════════════════════════════════════════════════════════════════════════════════════════════
           STREAM 2: PEDAGOGICAL EVALUATION STREAM (Text + Vision ➔ Audited Marks)
══════════════════════════════════════════════════════════════════════════════════════════════

   4. Pedagogical Schema Router Agent (orchestrator.py)
      • Context-aware routing based on syllabus schema (data/questions/*.json).
      • Channel A (Objective MCQs Q1 Part A, Cloze Q4): Bypasses linguistic criticism.
        Forwarded directly to Key Matcher (0 false sentence-fragment errors).
      • Channel B (Subjective Essays Q2, Q3, Q7, Q8, Q10, Q11): Forwarded to GEC.
          │
          ▼
   5. Linguistic Diagnostic Critic Agent (stage3_error_analyzer.py - Negative Bias Policy)
      • Role: Adversarial Auditor / Skeptic (Gemma 4 31B IT, Discourse GEC Prompt).
      • Syntactic Sentence Chunking: Segments strictly by complete sentences (?<=[.!?\n])\s+.
        Sentences are NEVER cut in half mid-clause; 0 false clause-fragment errors.
      • Consolidated Taxonomy: Evaluates spelling, morphosyntax, and coherence.
      • Single-Penalty Invariant: A single clause span can incur at most one penalty.
          │
          ▼
      [Candidate Error Ledger]
          │
          ▼
   6. Examiner Ombudsman Agent (gate.py - Cambridge / Edexcel BOD Referee)
      • Role: Impartial Adjudicator & Student Advocate.
      • Formal Cambridge BOD Policy:
        - Phonetic misspelling of English word (gnowledge, libary, destruyed) ➔ GENUINE_ERROR.
        - Grammatical tense switch (become ➔ became, produce ➔ produced) ➔ GENUINE_ERROR.
        - Writer allograph confusion (cursive ligature, unclosed 'a', Palmer 'r') ➔ HANDWRITING_AMBIGUITY.
      • Ruling-Line Differential Subtraction: Ignores notebook paper horizontal ruling lines.
      • Multi-Page Coordinate Resolution: Crops candidate image from the exact page of occurrence.
      • ZERO TRANSCRIPT MUTATION: Waives deductions in stage3_errors.json (deduction = 0.0),
        but NEVER alters the frozen transcript.
      • Bounded Budget: Evaluates at most 8–10 high-value candidates per script (<60s runtime).
          │
          ▼
      [Audited Diagnostic Ledger]
          │
          ▼
   7. Rubric Evaluation Supervisor Agent (stage4_rubric_evaluator.py)
      • Synthesizes objective answer keys, essay content criteria bands, and audited
        linguistic deductions into the final grade report and student feedback.
```

---

## 2. Core Architectural Invariants (Zero Hardcoding)

To ensure this framework generalizes across any school exam script in the world, the implementation enforces **four universal invariants**:

1. **The Non-Word Invariance Invariant**:
   $$\forall t_{\text{cand}} \notin \mathcal{V}_{\text{lexicon}}, \quad t_{\text{replacement}} \in \mathcal{V}_{\text{lexicon}} \implies \mathbf{REJECT\ MUTATION}$$
   A student's out-of-vocabulary non-word (`illustrodes`, `strensth`, `softwor`, `momeneterm`) is diagnostic proof of an orthographic error. Post-processing must never rewrite it into a dictionary word.
2. **The Surface Immutability Invariant**:
   $$\text{Transcript}_{\text{final}} \equiv \text{Stream1}(\text{Canvas})$$
   $$\nabla_{\text{Stream2}} (\text{Transcript}_{\text{final}}) = 0$$
   Downstream evaluation stages evaluate and annotate errors, but cannot modify the transcript string.
3. **The Single-Deduction Clause Invariant**:
   For any syntactic clause span $C_i$, the total penalty deduction satisfies:
   $$\text{Deduction}(C_i) \le \max_{e \in \text{Errors}(C_i)} \text{Penalty}(e)$$
   Prevents double-penalizing the student under overlapping "Grammar" and "Syntax" categories for the same phrase.
4. **The Cambridge / Edexcel BOD Distinction**:
   * If $\text{PhoneticSim}(w_{\text{cand}}, w_{\text{intended}}) \ge 0.80$ and $w_{\text{cand}} \notin \mathcal{V}_{\text{lexicon}} \implies \mathbf{GENUINE\_ERROR}$ (student knew the sound but failed orthography).
   * If character edit distance $= 1$ and edit matches proven Writer Allograph confusion $\implies \mathbf{HANDWRITING\_AMBIGUITY}$ (deduction waived to $0.0$).

---

## 3. Step-by-Step Implementation Phases

### Phase 1: Stream 1 Hardening & Transcript Immutability
**Objective**: Guarantee pristine, un-mutated transcription with zero VLM autocorrection and zero back-mutation leakage.

1. **Purge Ruling Line Coordinates from Stage 1 Prompt**:
   * File: [`src/prompts/stage1_verbatim.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/prompts/stage1_verbatim.py)
   * Action: Remove lines 63–94 that inject `--- OPTICAL STRIKETHROUGH ADVISORY ---` and OpenCV candidate line coordinates into the Stage 1 prompt.
   * Rationale: Gemma 4's high-DPI visual encoder natively sees real student cross-outs; injecting OpenCV ruling-line coordinates causes false `[struck: ...]` tagging on clean notebook text.
2. **Sever Stage 3b Transcript Back-Mutation**:
   * File: [`src/pipeline/orchestrator.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/orchestrator.py)
   * Action: Completely remove the call and logic of `_apply_arbitration_to_transcript` (lines 280–292).
   * Rationale: Prevents arbitration decisions from retroactively rewriting `verified_transcript` and injecting false deletions that double CER.
3. **Lock Stage 2 as Deterministic CPU Tokenizer**:
   * File: [`src/pipeline/orchestrator.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/orchestrator.py) & [`scripts/extract_scripts.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/scripts/extract_scripts.py)
   * Action: Ensure Stage 2 runs strictly as the CPU normalizer (`split_token_stitcher.py` + `edge_truncation_detector.py`) with zero full-page VLM calls.
   * Rationale: Saves 50s/page latency and preserves 100% of student non-words.
4. **Verification**:
   * Re-evaluate `SE_11_Q1_0002` via `scripts/evaluate_transcription.py`.
   * **Success Criteria**: Page 5 CER drops from 2.5% back to **1.1%**; Page 10 CER drops from 2.8% back to **1.1%**; Page 11 CER remains $\le 9.6\%$.

---

### Phase 2: Pedagogical Routing & Negative Critic Agent (GEC)
**Objective**: Prevent objective questions from being graded as essays and prevent sentence-fragment errors caused by sliding windows.

1. **Schema-Driven Pedagogical Router**:
   * File: [`src/pipeline/orchestrator.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/orchestrator.py)
   * Action: In `extract_script()`, inspect the matched `question_obj` schema for each answer:
     ```python
     q_type = str(matched_sq.get("question_type") or "").lower()
     if "objective" in q_type or "mcq" in q_type or "matching" in q_type or "cloze" in q_type:
         # Bypass linguistic error extraction (Channel A)
         continue
     ```
   * Rationale: MCQs and cloze blanks are evaluated against the answer key, never penalizing single-letter answers like `(a) Ans: i` as sentence fragments.
2. **Syntactic Sentence-Boundary Chunking**:
   * File: [`src/pipeline/orchestrator.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/orchestrator.py#L118-L148)
   * Action: Refactor `_chunk_text_by_sentences(text)` to segment text strictly using complete sentence boundaries:
     ```python
     sentences = re.split(r"(?<=[.!?\n])\s+(?=[A-Z0-9\"'(\[])", text.strip())
     ```
     Accumulate complete sentences up to ~250–350 words per chunk. Sentences are never sliced in half.
3. **Negative Critic Prompt Calibration**:
   * File: [`src/prompts/stage3_errors.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/prompts/stage3_errors.py)
   * Action: Calibrate the system prompt with the "Adversarial Linguistic Auditor" persona:
     * Uncompromising scrutiny for spelling, morphosyntax (agreement/tense), and coherence.
     * Enforce single-deduction-per-clause constraint.
4. **Verification**:
   * Run extraction on `SE_11_Q1_0002`.
   * **Success Criteria**: 0 false syntax errors on Q1 Part A (MCQ); 0 fragmented sentence deductions in Q10 & Q11.

---

### Phase 3: Examiner Ombudsman Agent (Cambridge BOD Calibration)
**Objective**: Eliminate over-forgiveness leakage (46.9% $\to$ $<15\%$) and eliminate notebook ruling line false strikethroughs.

1. **Ruling-Line Differential Strikethrough Subtraction**:
   * File: [`src/pipeline/stage0_strikethrough_detector.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage0_strikethrough_detector.py) & [`src/pipeline/arbitration/gate.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/arbitration/gate.py#L350-L363)
   * Action: Detect continuous horizontal lines spanning $>70\%$ page width at uniform vertical spacing (printed notebook ruling lines). In candidate word crops, ignore horizontal strokes that coincide with the notebook ruling line baseline.
   * Rationale: Eliminates false optical strikethroughs on clean words (`the`, `preparation`).
2. **Formal Cambridge & Edexcel BOD Decision Engine**:
   * File: [`src/pipeline/arbitration/gate.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/arbitration/gate.py)
   * Action: Implement the three formal tests in `_score_candidate()`:
     * **Test 1**: If candidate is an out-of-vocabulary word with phonetic distance $\le 2$ to intended (`gnowledge`, `libary`, `destruyed`, `answear`, `accroding`) $\implies$ enforce `GENUINE_ERROR` (`score < threshold_genuine`).
     * **Test 2**: If candidate is a morphosyntactic tense switch (`become` $\to$ `became`, `produce` $\to$ `produced`) $\implies$ enforce `GENUINE_ERROR`.
     * **Test 3**: If candidate matches a proven Writer Allograph confusion (Palmer cursive 'r', terminal looped 's', unclosed 'a') $\implies$ grant `HANDWRITING_AMBIGUITY` (`score >= threshold_ambiguity`).
3. **Multi-Page Image Resolution & Bounded Budget**:
   * File: [`src/pipeline/orchestrator.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/orchestrator.py#L949) & [`src/pipeline/arbitration/gate.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/arbitration/gate.py)
   * Action: Map each candidate token to its actual page number in `page_images` before cropping. Cap visual crop calls to at most **8–10 high-value candidates per script**.
4. **Verification**:
   * Test on `SE_11_Q1_0002`.
   * **Success Criteria**: `gnowledge`, `libary`, `destruyed`, `become` are correctly retained as `GENUINE_ERROR`; crop calls drop from 252 to $\le 10$; arbitration runtime $< 45$s.

---

### Phase 4: Rubric Evaluation Supervisor (Stage 4 Synthesis)
**Objective**: Connect objective key-matching and essay rubric criteria with the audited error ledger.

1. **Objective Key Scorer Integration**:
   * File: [`src/pipeline/stage4_rubric_evaluator.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage4_rubric_evaluator.py)
   * Action: Score Channel A objective questions directly against question schema answer keys (`expected_answer`, `options`).
2. **Audited Error Deduction Aggregation**:
   * File: [`src/pipeline/stage4_rubric_evaluator.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage4_rubric_evaluator.py)
   * Action: Aggregate deductions from `stage3_errors.json` (where waived BOD errors have deduction = 0.0), apply syllabus linguistic penalty caps, and subtract from essay content scores.
3. **Audit Trail & Feedback Generation**:
   * Emit `evaluation_report.md` with full traceability: Criterion $\to$ Band Score $\to$ Itemized Errors $\to$ Final Mark.

---

### Phase 5: Corpus-Wide Empirical Benchmark Validation
**Objective**: Scientifically validate the entire architecture across the 15 Ground Truth pages from 5 scripts (`0002`, `0006`, `0010`, `0011`, `0013`).

1. **Batch Clean Remaining Ground Truth Scripts**:
   * Clean `0006`, `0010`, `0011`, `0013` (already generated in `data/cleaned_pdfs/english/`).
2. **Batch Extract with Triadic Multi-Agent Architecture**:
   * Run extraction across all 5 GT scripts.
3. **Execute Benchmark Evaluation**:
   * Run `scripts/evaluate_transcription.py`.
4. **Target Thesis Metrics**:
   * **Macro CER**: $\le 3.5\%$ (down from 8.07%).
   * **Macro WER**: $\le 6.5\%$ (down from 11.12%).
   * **Student Non-Words Preserved**: **$100\%$** (up from 75.9%).
   * **Silent-Correction Rate**: **$0.0\%$** (down from 24.14%).
   * **Average Script Runtime**: **$\le 2.5$ minutes** (down from 18–25 minutes).

---

## 4. Work Breakdown & Execution Sequence

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EXECUTION TIMELINE                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [Step 1] Stream 1 Hardening (Prompt Unbiasing & Transcript Lock)       │
│           • Purge OpenCV prompt advisory in stage1_verbatim.py         │
│           • Delete back-mutation in orchestrator.py                    │
│           • Empirically test on SE_11_Q1_0002 (Expect Page 5 CER 1.1%) │
│                                                                        │
│  [Step 2] Stream 2 Routing & Sentence Chunking                         │
│           • Implement MCQ bypass filter in orchestrator.py             │
│           • Implement syntactic sentence chunking in orchestrator.py   │
│           • Update Negative Critic prompt in stage3_errors.py          │
│                                                                        │
│  [Step 3] Examiner Ombudsman (Cambridge BOD Gate)                      │
│           • Add ruling line subtraction in strikethrough detector      │
│           • Implement Cambridge BOD rules in gate.py                   │
│           • Cap crop calls to <= 10 per script                         │
│                                                                        │
│  [Step 4] Full Corpus Validation & Benchmarking                        │
│           • Extract & evaluate 5 GT scripts (15 pages)                 │
│           • Generate definitive before-and-after benchmark tables      │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```
