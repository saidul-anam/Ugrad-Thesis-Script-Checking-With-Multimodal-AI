# Audit Plan: Verifying Pipeline Bloat & Identifying Unnecessary Heavy Patches
*(Without Ground-Truth Text Extractions)*

## Goal Description
Conduct a rigorous, scientific audit across the existing 24 script extraction results to determine with hard empirical evidence which pipeline patches are **essential**, which are **redundant**, and which are **unnecessary heavy bloat** causing the 12–15+ minute per-script processing time. 

Crucially, because **we do not have ground-truth text transcriptions**, this plan uses **non-reference evaluation methodologies**: Amdahl latency profiling, patch mutation rates, consensus agreement redundancy, candidate origin disaggregation, and downstream teacher mark sensitivity against [`gt.txt`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/gt.txt).

---

## The Core Challenge: Auditing Without Ground-Truth Transcriptions

Standard OCR evaluation relies on Word Error Rate (WER) against human-typed text transcripts. Since human text transcripts do not exist for these 24 exam scripts, standard WER is impossible.

Instead, we audit the pipeline using **5 Reference-Free Scientific Protocols**:

```mermaid
flowchart TD
    subgraph Data Sources Already Available (Zero GPU Needed)
        A["24 x extraction_result.json<br>(Stage 1, Stage 2, Stage 3, Stage 4)"]
        B["24 x stage3b_arbitration.json<br>(Raw signals, 5x consensus samples, forced-choice)"]
        C["24 x writer_profile.json<br>(Allographs, stitched splits, pair counts)"]
        D["gt.txt<br>(Ground-truth teacher marks for scripts 0002, 0010, 0011, 0015)"]
    end

    subgraph The 5 Audit Protocols
        E["Protocol 1: Amdahl Latency & VLM Call Profiler"]
        F["Protocol 2: Patch Mutation Rate & Impact vs. No-Op Audit"]
        G["Protocol 3: 5x Consensus Redundancy & Agreement Audit"]
        H["Protocol 4: Candidate Origin Audit (Stage 3 vs Lexicon Flooding)"]
        I["Protocol 5: Downstream Teacher Mark Sensitivity Audit (gt.txt)"]
    end

    A --> E
    B --> E
    A --> F
    C --> F
    B --> G
    B --> H
    A --> I
    D --> I

    E --> J["Final Bloat Audit Report:<br>Essential vs. Redundant vs. Definite Bloat"]
    F --> J
    G --> J
    H --> J
    I --> J
```

---

## The 5 Audit Protocols

### Protocol 1: Amdahl Latency & VLM Call Profiler (Where Does the Time Go?)
* **Objective**: Measure the exact wall-clock time and VLM/LLM inference calls consumed by each component per script.
* **Target Components**:
  1. `Stage 0 / 0b`: Red ink detection & teacher mark extraction.
  2. `Stage 1`: Full-page verbatim transcription (15–20 calls/script).
  3. `Stage 2`: Full-page verification re-reading (15–20 calls/script).
  4. `Allograph Calibration & Split Stitching`: CPU string passes.
  5. `Stage 3`: Linguistic error extraction (10 calls/script).
  6. `Stage 3b Arbitration`:
     - Line Localizer (BBox calls).
     - Consensus Transcriber (5 stochastic samples per candidate).
     - Forced Choice Judge (1–2 calls per candidate).
  7. `Stage 4`: Rubric evaluation (10–12 calls/script).
* **Metrics to Compute**:
  * Total VLM calls per stage.
  * % of total runtime per stage.
  * Cost-per-word-extracted.

---

### Protocol 2: Patch Mutation Rate & "Impact vs. No-Op" Audit
* **Objective**: Determine whether intermediate transformation patches actually alter text meaningfully, or if they are effectively expensive "no-ops" that change almost nothing.
* **Audit Checks**:
  1. **Stage 2 Verification Mutation Rate**:
     * Compare `stage1_raw_transcript.txt` vs. `stage2_verified_transcript.txt` across all 24 scripts.
     * Compute:
       $$\text{Token Mutation Rate} = \frac{\text{Tokens Changed by Stage 2}}{\text{Total Tokens in Script}} \times 100\%$$
     * **Audit Question**: If Stage 2 consumes 15 heavy VLM calls (~45–60s) but changes $<1\%$ of words, what is its cost-to-benefit ratio?
     * **Semantic Drift Check**: In the words Stage 2 *did* change, how many were:
       - (a) Legitimate OCR corrections (e.g. correcting a mangled letter).
       - (b) **Harmful silent auto-corrections** (e.g. student wrote `eingineer`, and Stage 2 quietly "fixed" it to `engineer`, hiding a student error from Stage 3!).
  2. **Allograph Calibrator Activity Rate**:
     * Parse all 24 `writer_profile.json` files:
     * Count how many scripts actually triggered active allograph rules (`discovered_allographs`).
     * Count the total number of words rewritten by `adapted_words` across all 24 scripts.
  3. **Split-Token Stitcher Activity Rate**:
     * Count how many pen-lift splits were stitched per script.
     * Measure if any stitched tokens were already valid standalone words (over-stitching risk).

---

### Protocol 3: 5x Consensus Redundancy & Agreement Audit
* **Objective**: Test whether running 5 stochastic re-reads per candidate in Stage 3b provides any real discriminative power over a single deterministic visual call.
* **Data**: Every candidate record in all 24 `stage3b_arbitration.json` files stores:
  - `consensus_samples` (the 5 raw text strings).
  - `consensus_token_agreement` (e.g. 1.0, 0.8, 0.6).
  - `forced_choice` (e.g. `"candidate"` or `"intended"`).
  - `forced_choice_confidence` (e.g. 1.0).
  - `verdict` (`GENUINE_ERROR` vs `HANDWRITING_AMBIGUITY` vs `UNCERTAIN`).
* **Audit Checks**:
  1. **Sample 1 vs. 5-Sample Majority Agreement**:
     * How often does the first sample ($N=1$) predict the exact same majority outcome as the full 5 samples ($N=5$)?
  2. **Consensus vs. Forced-Choice Agreement**:
     * How often does the 5-sample consensus verdict disagree with the single `forced_choice` judge verdict?
     * When they disagree, which one was correct according to the physical handwriting?
  3. **Threshold Sensitivity**:
     * If consensus sampling is ablated entirely ($N=1$), what percentage of final candidate verdicts (`GENUINE` vs `AMBIGUITY`) change?
     * **Hypothesis**: If $<3\%$ of verdicts change, the remaining 4 samples (accounting for up to 150+ VLM calls per script) are pure redundant overhead.

---

### Protocol 4: Candidate Origin Audit (Stage 3 vs. Lexicon Scan Flooding)
* **Objective**: Determine whether `lexicon_candidates()` is adding genuine value or flooding Stage 3b with phantom candidates.
* **Mechanism**:
  - Currently, Stage 3b gates two kinds of candidates:
    - **Source A (Stage 3 Errors)**: Linguistic errors flagged by the Stage 3 LLM.
    - **Source B (Lexicon Scan Injections)**: Any word in the transcript that is not in the dictionary, even if Stage 3 never thought it was an error.
* **Audit Checks across 24 Scripts**:
  1. Count of Source A candidates vs. Source B candidates per script.
  2. Final verdict breakdown for Source B:
     * How many Source B candidates were confirmed as `GENUINE_ERROR`?
     * How many were cleared as `HANDWRITING_AMBIGUITY`?
     * How many were cleared as `UNCERTAIN` or NCTB cultural terms / proper nouns?
  3. **Audit Question**: If 90%+ of Source B candidates are discarded or cleared, is `lexicon_scan` unnecessarily doubling the candidate pool and doubling arbitration runtime?

---

### Protocol 5: Downstream Teacher Mark Sensitivity Audit (Using `gt.txt`)
* **Objective**: Measure whether intermediate pipeline patches actually affect the final marks awarded to students, compared against ground-truth teacher marks.
* **Ground Truth Available**:
  - `gt.txt` contains official teacher marks per question for 4 complete scripts (`SE_11_Q1_0002`, `SE_11_Q1_0010`, `SE_11_Q1_0011`, `SE_11_Q1_0015`).
* **Counterfactual Mark Simulation**:
  Using the existing evaluation records, simulate final question marks under counterfactual patch ablations:
  1. **Baseline**: Full pipeline (all patches active).
  2. **Ablation 1 (No Consensus Sampling)**: Verdicts scored with single forced-choice call.
  3. **Ablation 2 (No Lexicon Scan)**: Deductions scored only on Stage 3 candidate errors.
  4. **Ablation 3 (No Stage 2 Verification)**: Errors extracted directly from Stage 1 transcript.
* **Metrics**:
  * Mean Absolute Error (MAE) vs. Teacher Marks:
    $$\text{MAE} = \frac{1}{Q} \sum_{q=1}^Q |\text{Mark}_{\text{AI}}(q) - \text{Mark}_{\text{Teacher}}(q)|$$
  * Pearson Correlation ($r$) with Teacher Marks.
  * **The Bloat Rule**: If removing a patch causes $|\Delta \text{MAE}| < 0.15$ marks and no drop in correlation, but saves 30% of pipeline runtime, that patch is **quantifiably proven to be bloat**.

---

## Audit Execution Script: `scripts/audit_pipeline_bloat.py`

We will implement an automated, offline audit tool that requires **ZERO GPU hours** (it reads existing JSON outputs in `outputs/extracted/english/` in under 15 seconds):

### Deliverables of the Audit Script:
1. **Summary Table**:
   | Stage / Patch | Avg VLM Calls | Avg Latency | Mutation / Hit Rate | Agreement w/ Simple | $\Delta \text{MAE}$ (gt.txt) | Verdict |
   |:---|:---:|:---:|:---:|:---:|:---:|:---:|
   | Stage 1 Transcriber | ~16 calls | ~50s | Baseline | - | - | **ESSENTIAL** |
   | Stage 2 Verifier | ~16 calls | ~50s | *[Computed %]* | - | *[Computed]* | *[To be audited]* |
   | Allograph Calibrator | 0 calls | <0.1s | *[Computed]* | - | *[Computed]* | *[To be audited]* |
   | Split Stitcher | 0 calls | <0.1s | *[Computed]* | - | *[Computed]* | *[To be audited]* |
   | Stage 3 Error Analyzer | ~10 calls | ~25s | Baseline | - | - | **ESSENTIAL** |
   | Stage 3b: 5x Consensus | ~120 calls | ~360s | - | *[Computed %]* | *[Computed]* | *[To be audited]* |
   | Stage 3b: Lexicon Scan | ~80 calls | ~240s | *[Computed %]* | - | *[Computed]* | *[To be audited]* |
   | Stage 3b: Forced Choice | ~25 calls | ~75s | High | - | *[Computed]* | **ESSENTIAL** |
   | Stage 4 Evaluator | ~11 calls | ~45s | Baseline | - | - | **ESSENTIAL** |

2. **Categorized Action Recommendations**:
   * **ESSENTIAL**: Must keep; high impact on accuracy and ground-truth alignment.
   * **LIGHTWEIGHT HELPER**: Negligible CPU cost (<0.1s); safe to keep regardless of impact.
   * **HEAVY REDUNDANT BLOAT**: High GPU call count / latency, but $<3\%$ impact on verdicts or marks. Strongly recommended for pruning.
   * **COUNTER-PRODUCTIVE PATCH**: Actually hurts accuracy (e.g. Stage 2 auto-correcting genuine student errors).

---

## Verification Plan

### Automated Run
Execute the audit script offline across all 24 script outputs:
```bash
python3 scripts/audit_pipeline_bloat.py --extracted-dir outputs/extracted/english --gt gt.txt
```

### Verification Criteria
1. Does the report quantify the exact % of latency consumed by each stage?
2. Does the report measure the 5x consensus agreement rate vs. single forced-choice?
3. Does the report determine the true hit-rate of `lexicon_scan` candidates?
4. Does the report compute the downstream mark impact on `gt.txt`?
