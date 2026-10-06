# Implementation Plan: Clean-PDF Integration, Standalone Teacher Marks & Pipeline Logic Hardening

> **Target Goal**: Decouple teacher red-mark extraction from script transcription, integrate `clean_pdf.py` as the upstream document cleaner, streamline Stage 2 based on empirical audit findings, and resolve remaining code-level pipeline defects (Group B) cataloged in [`docs/PIPELINE_PROBLEMS_AUDIT.md`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/docs/PIPELINE_PROBLEMS_AUDIT.md).

---

## 1. System Architecture Overview

```
                                  ┌──────────────────────────┐
                                  │   data/raw_pdfs/{lang}/  │  (Immutable Ground Truth)
                                  └─────────────┬────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
  ┌─────────────────────────────┐                               ┌─────────────────────────────┐
  │     1. Document Cleaner     │                               │  2. Standalone Marker Tool  │
  │       (clean_pdf.py)        │                               │ (extract_teacher_marks.py)  │
  │ • Calibrate Red Pen Hue     │                               │ • Gemma 4 on Raw Red Ink    │
  │ • Inpaint Red Marks & Halos │                               │ • Margin Question Routing   │
  │ • Inter-Page Homography &   │                               │ • Conflict Reconciliation   │
  │   Bleed-Through Removal     │                               └──────────────┬──────────────┘
  └──────────────┬──────────────┘                                              │
                 │                                                             │
                 ▼                                                             ▼
  ┌─────────────────────────────┐                               ┌─────────────────────────────┐
  │   data/cleaned_pdfs/{lang}/ │                               │  outputs/extracted/{lang}/  │
  │   (Pristine Student Ink)    │                               │     <script_id>/            │
  └──────────────┬──────────────┘                               │  stage0b_teacher_marks.json │
                 │                                              └──────────────┬──────────────┘
                 ▼                                                             │
  ┌─────────────────────────────┐                                              │
  │    3. Transcription Engine  │                                              │
  │     (extract_scripts.py)    │                                              │
  │ • Clean Canvas Auto-Detect  │                                              │
  │ • Stage 0.5 Cross-Out Prior │                                              │
  │ • Stage 1 Verbatim (No Loops│                                              │
  │ • Stage 2 Deterministic/Fast│                                              │
  │ • Stage 3 Question-Chunked  │                                              │
  │ • Stage 3b Arbitration Gate │                                              │
  └──────────────┬──────────────┘                                              │
                 │                                                             │
                 ▼                                                             │
  ┌─────────────────────────────┐                                              │
  │  outputs/extracted/{lang}/  │                                              │
  │     <script_id>/            │                                              │
  │  stage2_verified_transcript │                                              │
  │  stage3_errors.json         │                                              │
  │  raw_tier_records.csv       │◄─────────────────────────────────────────────┘
  └──────────────┬──────────────┘          (Attaches Teacher Marks & Examiner ID)
                 │
                 ▼
  ┌─────────────────────────────┐
  │ 4. Stage 4 Rubric Evaluator │
  │    (evaluate_scripts.py)    │
  │ • Grades Clean Transcript   │
  │ • Benchmarks vs. Human Marks│
  └─────────────────────────────┘
```

---

## 2. Baseline Audit Findings (October 3, 2026)

Before modifying the pipeline, `scripts/evaluate_transcription.py` was executed across 15 human-corrected ground truth pages from 5 scripts (`SE_11_Q1_0002`, `0006`, `0010`, `0011`, `0013`):

* **Stage 1 (Verbatim Transcription)**:
  * Micro CER: **5.61%** | Macro WER: **10.08%** | Micro WER: **7.61%**
  * Student non-words preserved: **75 / 87 (86.2%)**
  * Silent-correction rate: **13.79%**
* **Stage 1 + Stage 2 (VLM Verified)**:
  * Micro CER: **6.58% (Worse)** | Macro WER: **11.12% (Worse)** | Micro WER: **9.07% (Worse)**
  * Student non-words preserved: **66 / 87 (75.9% - Lost 9 genuine student errors)**
  * Silent-correction rate: **24.14% (Nearly Doubled)**
* **Root Cause**: Stage 2's VLM prompt acts as a language model autocorrect that "fixes" student spelling mistakes (`intelligane` → `intelligence`, `softwor` → `software`, `feak` → `fake`). This prevents Stage 3 from seeing those errors, creating false-negative spelling evaluations.
* **Architectural Decision**: Fast mode (`--fast`, skipping Stage 2 VLM and relying on deterministic CPU stitchers) should be the default, or Stage 2 must be strictly prohibited from replacing non-dictionary tokens with dictionary words.

---

## 3. Proposed Changes & Implementation Phases

### Phase 1: Pipeline Logic Hardening (Fix Group B Issues)

#### 1.1 Fix Punctuation Loops & Nested Tags (`stage1_transcriber.py`)
* **Problem**: Single-character punctuation runs (`’s’’’’’’’’...`) and recursive brackets (`[struck: [struck: ...`) bypass the current word-boundary regex filter.
* **File to Modify**: [`src/pipeline/stage1_transcriber.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage1_transcriber.py)
* **Changes**:
  In `_clean_transcription_artifacts(text)`:
  ```python
  # 5b. Suppress runaway punctuation runs (e.g. ’s’’’’’’’’ -> ’s)
  text = re.sub(r'([^\w\s])\1{3,}', r'\1', text)

  # 5c. Flatten nested struck tags: [struck: [struck: text]] -> [struck: text]
  while re.search(r'\[struck:\s*\[struck:', text):
      text = re.sub(r'\[struck:\s*\[struck:\s*([^\]]+)\]\s*\]', r'[struck: \1]', text)
  ```

#### 1.2 Fix Sub-Question Header Merge (`answer_segmenter.py`)
* **Problem**: Sub-question labels like `(b) Ans:` or `(d) Ans:` are merged by the VLM into `Dans: The author...` without question digits, preventing subpart segmentation.
* **File to Modify**: [`src/pipeline/answer_segmenter.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/answer_segmenter.py)
* **Changes**:
  In `normalize_transcript_headers(text)` and `HEADER_NORMALIZATION_RULES`:
  ```python
  # Unpack fused subpart markers: e.g. "Bans:" -> "(b) Ans:", "Dans:" -> "(d) Ans:"
  text = re.sub(r'(?im)^([b-gB-G])\s*ans[:\s]', r'(\1) Ans: ', text)
  text = re.sub(r'(?im)^\bDans[:\s]', r'(d) Ans: ', text)
  ```

#### 1.3 Fix Stage 3b Strikeout Phonetic Signal (`gate.py`)
* **Problem**: When a strikethrough suspect is arbitrated against `"[struck]"`, phonetic distance calculates against the literal word `"struck"`, yielding `phonetic_signal = 1.00` and artificially granting `HANDWRITING_AMBIGUITY` benefit of the doubt.
* **File to Modify**: [`src/pipeline/arbitration/gate.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/arbitration/gate.py)
* **Changes**:
  In `signals_from_evidence(ev)`:
  ```python
  # Strikeout suspects are not phonetic substitutions; neutralise phonetic signal
  if ev.intended == "[struck]" or getattr(ev, "is_strike_suspect", False):
      phonetic_val = None
  else:
      phonetic_val = ev.phonetic_signal
  ```

---

### Phase 2: Standalone Teacher Mark Extraction Script

Create a dedicated CLI tool that handles human examiner marks independently of the student transcription pipeline.

#### 2.1 Create `scripts/extract_teacher_marks.py`
* **File to Create**: `scripts/extract_teacher_marks.py`
* **Features**:
  * Scans `data/raw_pdfs/{lang}/` for scripts with red examiner ink.
  * Uses `Stage0bTeacherMarkExtractor` ([`src/pipeline/stage0b_teacher_marks.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage0b_teacher_marks.py)).
  * Dynamically matches question rubrics and resolves mark conflicts.
  * Writes to `outputs/extracted/{lang}/{script_id}/stage0b_teacher_marks.json`.
  * CLI arguments:
    * `--lang`: `english` or `bangla`
    * `--top`: Process top $N$ or all scripts
    * `--script`: Target single script ID
    * `--force`: Overwrite existing marks
    * `--quant`: Quantization level (`4bit` or `8bit`)

---

### Phase 3: Upstream Pre-Cleaning with `clean_pdf.py`

#### 3.1 Batch Process Corpus PDFs
Execute [`clean_pdf.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/clean_pdf.py) across all raw exam scripts:
```bash
python3 clean_pdf.py data/raw_pdfs/english/se_11_q1/ \
  -o data/cleaned_pdfs/english/ \
  --qa data/cleaned_qa/english/ \
  --report data/cleaned_pdfs/english_clean_report.csv
```
* **Verification**: Inspect generated QA thumbnails in `data/cleaned_qa/english/*.jpg` to verify that:
  1. All red examiner marks are inpainted smoothly.
  2. Reverse-page bleed-through ink is eliminated on sparse/blank pages.
  3. Student blue/black handwriting and ruled paper lines are preserved.

---

### Phase 4: Core Pipeline Clean-Canvas Auto-Discovery & Stage 2 Streamlining

Update the core pipeline orchestrator to automatically detect and consume pre-cleaned PDFs without breaking backward compatibility.

#### 4.1 Update `orchestrator.py` & `extract_scripts.py`
* **Files to Modify**:
  * [`src/pipeline/orchestrator.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/orchestrator.py)
  * [`scripts/extract_scripts.py`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/scripts/extract_scripts.py)
* **Changes**:
  1. **Clean-PDF Resolution**:
     When `extract_script(input_source)` is called:
     ```python
     clean_candidate = os.path.join("data/cleaned_pdfs", paper, f"{script_id}.pdf")
     if os.path.exists(clean_candidate):
         print(f"[Extraction] 🧼 Using pre-cleaned PDF: {clean_candidate}")
         pdf_to_render = clean_candidate
         is_pre_cleaned = True
     else:
         pdf_to_render = input_source
         is_pre_cleaned = False
     ```
  2. **Stage 0 / 0b Bypass on Clean Canvas**:
     * If `is_pre_cleaned` is True:
       * Bypass Stage 0 red-ink detection and Stage 0b inline VLM marks extraction (`0 compute, 0 tokens`).
       * Check if `outputs/extracted/{paper}/{script_id}/stage0b_teacher_marks.json` exists on disk. If found, load it into `extraction_result.teacher_marks` so downstream exports (`raw_tier_dataset.csv`, `raw_tier_records.csv`) remain complete.
  3. **Stage 2 Streamlining**:
     * Make fast mode default or enforce strict CPU non-word protection so Stage 2 cannot overwrite student spelling errors.
  4. **Stage 0.5 Simplification**:
     * When operating on a pre-cleaned canvas, `stage0_strike_res` runs directly without needing red-ink teacher masking.

---

## 4. Verification & Validation Plan

| Step | Target Component | Test Command / Action | Success Criteria |
|---|---|---|---|
| **V1** | Unit Tests | `pytest tests/test_linguistic_sanitizer.py tests/test_repair_checkpoints.py` | All existing unit tests pass |
| **V2** | Logic Fix Unit Test | Run test script on `SE_11_Q1_0002` transcript snippet containing `and’s’’’’’’’’` and `Dans:` | Punctuation collapsed to `’s`; `Dans:` parsed to `(d) Ans:` |
| **V3** | Batch Clean `SE_11_Q1_0002` | `python3 clean_pdf.py data/raw_pdfs/english/se_11_q1/SE_11_Q1_0002.pdf -o data/cleaned_pdfs/english/SE_11_Q1_0002.pdf --qa data/cleaned_qa/english/` | Pages 11 & 13 have white background; red marks removed |
| **V4** | Standalone Teacher Marks | `python3 scripts/extract_teacher_marks.py --lang english --script SE_11_Q1_0002` | Generates valid `stage0b_teacher_marks.json` with reconciled marks |
| **V5** | Clean Transcription | `python3 scripts/extract_scripts.py --lang english --script SE_11_Q1_0002 -y` | 0 bleed loops, 0 punctuation runs, full Stage 3 completion |
| **V6** | Comparative Benchmark | Run `scripts/evaluate_transcription.py` on clean output | WER drops below 7%, silent-correction rate drops below 8% |
| **V7** | Full Corpus Run | Run batch extraction for all 24 English scripts | 24/24 scripts extracted with 8/8 Ground Truth coverage |
