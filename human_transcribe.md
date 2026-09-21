# End-to-End Workflow Guide: Extraction, Human CER, and Evaluation

This guide details the complete, step-by-step workflow for:
1. **Running Automated Script Extraction** (Stages 0–3b)
2. **Generating and Computing Human Extraction CER / WER** (Zero external uploads needed)
3. **Running Rubric Grading & Alignment against Ground-Truth Teacher Marks** (Zero transcription needed)

---

## 🧭 Workflow Decision Matrix

| Your Immediate Goal | Do You Need Human Transcription? | Time Required | Steps to Follow |
| :--- | :--- | :--- | :--- |
| **Evaluate AI Marking Accuracy against Teachers** | **No** (Uses `gt.txt` marks) | ~1 minute per script | Follow **Track A (Sections 1 & 3)** |
| **Validate Transcription & Model Autocorrection (CER/WER)** | **Yes** (Proofread 24 continuous pages, 1/script) | ~2–3 mins/page (~1 hr total) | Follow **Track B (Section 2)** |
| **Validate Visual Arbitration Gate (Penmanship vs Error)** | **No** (Glance-level crop tags) | ~15 minutes for 50 crops | Follow **Track C (Section 4)** |
| **Fix Question Segmentation / LaTeX / Red Marks** | **No** (Automated script tools) | ~1 minute | Follow **Section 5** |

---

## 📋 Section 1: Running Multimodal Script Extraction (Automated)

The extraction pipeline transcribes handwriting character-for-character, detects red teacher ink, cleans canvas checkmarks, and segments student answers by question.

### Step 1.1: Activate the Environment
```bash
# 1. Navigate to repo root
cd /mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI

# 2. Activate virtualenv
source ./script_checking/bin/activate

# 3. Enable PyTorch CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Step 1.2: Run Extraction on a Script
You can extract a single script or a batch of scripts. 

> [!NOTE]
> Ensure LM Studio or your local API server is running on `http://localhost:1234/v1` if using `--api`.

```bash
# Single script extraction (Fast single-pass):
python3 scripts/extract_scripts.py --image data/raw_pdfs/english/SE_11_Q1_0010.pdf --lang english --api --fast -y

# Single script extraction (Full 2-pass verification with visual auditor):
python3 scripts/extract_scripts.py --image data/raw_pdfs/english/SE_11_Q1_0010.pdf --lang english --api -y

# Batch extract top 2 scripts in data/raw_pdfs/english/:
python3 scripts/extract_scripts.py --lang english --top 2 --api --fast -y
```

### Step 1.3: Inspect Extraction Outputs
Check that the extraction generated the required checkpoints and artifacts:
```bash
# Check per-page checkpoints
ls outputs/extracted/english/SE_11_Q1_0010/checkpoints/

# View human audit summary
cat outputs/extracted/english/SE_11_Q1_0010/extraction_summary.md
```

---

## ✍️ Section 2: How to Do Human Extraction CER (Step-by-Step)

Character Error Rate (CER) measures how closely the model's raw (Stage 1) and audited (Stage 1+2) transcripts match what the student actually wrote, and measures the **silent-correction rate** (verifying that the model didn't silently "fix" student misspellings).

> [!IMPORTANT]
> **Zero External Uploads Needed!**
> You do not need to transcribe from scratch or upload files from Google Drive. The system generates side-by-side draft text files and image crops directly from the extracted checkpoints.

### Step 2.1: Generate the Ground Truth Workspace

#### Recommended Sampling Strategy: 24 Pages Total (1 Page per Script)
- **100% Handwriting Diversity**: Transcribe exactly 1 continuous-writing page from each of the 24 student scripts. This guarantees every single student's handwriting style, penmanship quirks, and slant are represented in your CER/WER and Allograph calibration benchmarks.
- **Statistical Power**: 24 pages $\times$ ~150 words $\approx$ 3,600 words / 20,000 characters. In academic HTR/OCR literature (e.g., ICDAR, IAM), a test set of 3,500+ words across 24 distinct writers provides solid confidence intervals.
- **Time Required**: Because the tool gives you pre-filled machine drafts, proofreading 1 page takes only ~2–3 minutes. You can finish all 24 scripts in ~1 hour.

Run `make_transcription_gt.py` to automatically extract page images and pre-fill transcript drafts:

```bash
# 🎯 Recommended for Thesis: Automatically select 1 continuous-writing page across all 24 scripts
python3 scripts/make_transcription_gt.py --lang english --max-pages 24

# Option B: Create drafts for specific continuous-writing pages
python3 scripts/make_transcription_gt.py --lang english --pages SE_11_Q1_0010:3,4,5
```

This populates:
```text
data/ground_truth/transcripts/english/SE_11_Q1_0010/
├── page_3.png         # Reference image of the handwritten page
├── page_3.txt         # Pre-filled machine draft transcript
├── page_3.meta.json   # Metadata tracking correction status
├── page_4.png
├── page_4.txt
├── page_4.meta.json
...
```

### Step 2.2: Human Proofreading
Open `data/ground_truth/transcripts/english/SE_11_Q1_0010/` in your IDE:

1. Open `page_3.png` and `page_3.txt` side-by-side.
2. Read the text against the student handwriting and correct any mistakes:
   - **Do NOT correct the student's errors**: If the student wrote `decruise`, leave it as `decruise`. If the student wrote `he see`, leave it as `he see`.
   - **Tags for unreadable ink**: Use `[illegible]` for unreadable words, and `[unclear: word]` if partially readable.
   - **Struck text**: Use `[struck: text]` for crossed-out handwriting.
   - **Line breaks**: Keep line breaks matching the original page.
   - **Teacher marks**: Ignore red teacher checkmarks, crosses, and margin numbers.
3. Save `page_3.txt`.

### Step 2.3: Update the Metadata Status
In the same folder, open `page_3.meta.json`:
```json
{
  "script_id": "SE_11_Q1_0010",
  "page_no": 3,
  "source_checkpoint": "outputs/extracted/english/SE_11_Q1_0010/checkpoints/page_3.json",
  "draft_origin": "stage2_verified",
  "status": "CORRECTED"
}
```
Change `"status"` from `"DRAFT - needs human correction"` to `"CORRECTED"`.

> [!NOTE]
> The evaluation script skips any file where status starts with `DRAFT` so that machine drafts are never scored against themselves.

### Step 2.4: Calculate CER, WER, and Silent-Correction Rate
Run the evaluation script across all corrected scripts, or focus on a single script:
```bash
# Evaluate all corrected ground truth scripts
python3 scripts/evaluate_transcription.py --lang english

# Evaluate a single script only (e.g. SE_11_Q1_0002 or shorthand 0002)
python3 scripts/evaluate_transcription.py --lang english --script SE_11_Q1_0002
```

#### What this produces:
1. **Terminal output** showing per-page and aggregate scores:
   - **CER (Macro & Micro)**: Character error rate for Stage 1 vs Stage 1+2.
   - **WER (Macro & Micro)**: Word error rate for Stage 1 vs Stage 1+2.
   - **Student Non-Words**: Count of true misspellings written by the student.
   - **Silent-Correction Rate**: Percentage of student mistakes the model incorrectly autocorrected.
2. **Benchmark reports** saved to:
   - `outputs/benchmarks/transcription_english_<timestamp>.json`
   - `outputs/benchmarks/transcription_english_<timestamp>.md`

---

## 🎯 Section 3: Rubric Evaluation & Alignment against Ground Truth (`gt.txt`)

You can evaluate student answers against exam rubrics and compute **Mean Absolute Error (MAE)** against human examiner marks **without doing any human page transcription**.

### Step 3.1: Ensure Teacher Marks are in `gt.txt`
Verify or add teacher marks in [`gt.txt`](gt.txt):
```text
SE_11_Q1_0010
1(A)-5
1(B)-7
2-10
3-5
4-1
5-3
6-3
8-6
10-3
9-3
```

### Step 3.2: Synchronize Marks to Metadata
Propagate `gt.txt` marks to all extracted directories and datasets:
```bash
python3 scripts/sync_ground_truth.py
```

### Step 3.3: Run Modular Rubric Evaluation (Stage 4)
Run question-by-question grading:
```bash
python3 scripts/evaluate_scripts.py --script-name SE_11_Q1_0010 --lang english --api --eval-mode modular --force-evaluate -y
```

### Step 3.4: Inspect the Score Alignment Report
Open the evaluation report:
```bash
cat outputs/evaluated/english/SE_11_Q1_0010/evaluation_report.md
```
This generates the **AI vs. Human Ground Truth Alignment Table**, comparing AI awarded marks against teacher marks question-by-question, calculating $\Delta$ and overall MAE.

---

## 🔍 Section 4: Visual Arbitration Crop Labeling (Optional)

If you want to validate the **Stage 3b Visual Arbitration Gate** (which decides whether a candidate error is a genuine spelling mistake or handwriting penmanship ambiguity):

1. **Launch the labeling tool**:
   ```bash
   python3 scripts/label_arbitration_candidates.py --lang english
   ```
2. **Review the contact sheets / image crops**:
   - `G` = Genuine student spelling mistake (e.g. `decruise`, `eingineer`).
   - `A` = Handwriting / OCR stroke ambiguity (e.g. uncrossed $t$, cursive loop).
   - `U` = Unclear / unreadable.
3. **Score the arbitration gate**:
   ```bash
   python3 scripts/evaluate_arbitration.py --lang english
   ```

---

## 🔧 Section 5: Fixing Extractions Properly Across All 24 Scripts

If you notice extraction issues, broken formatting, or segmentation misalignments across the 24 scripts, **never re-run the 10+ hour VLM pipeline from scratch**. Use these targeted, instant utilities:

### 5.1 Re-segment Answer Boundaries (Segmentation Fixes)
If the model merged two questions or missed a cursive question header (e.g., `Ans to Question No - 3`):
1. Open the script's `outputs/extracted/english/<script_id>/stage2_verified_transcript.txt` or `checkpoints/page_<n>.json` and ensure the `Ans to Question No - X` header line is clean.
2. Run the dynamic answer segmenter to instantly recalculate aligned answers:
```bash
# Re-segment a single script:
python3 scripts/resegment_extraction.py --lang english --script SE_11_Q1_0010

# Re-segment ALL 24 scripts in batch:
python3 scripts/resegment_extraction.py --lang english
```

### 5.2 Repair LaTeX Formatting & Dropped Struck Tags
If flowchart questions (Q2) contain malformed LaTeX arrows (e.g., `\rightarrow` parsing glitches) or Stage 2 dropped student `[struck: ...]` strike-through tags:
```bash
# Scan and repair checkpoints across all 24 scripts:
python3 scripts/repair_stage2_checkpoints.py --lang english
```

### 5.3 Correct Teacher Red-Ink Marks (Stage 0b Override)
If the VLM missed or misread a teacher mark in the margins:
1. Simply add the correct score in `gt.txt` under the script ID.
2. Sync the marks into all extracted directories, metadata, and research CSV datasets:
```bash
python3 scripts/sync_ground_truth.py
```

### 5.4 Correcting OCR Glitches on a Specific Page
If the model hallucinated or misread handwriting on a messy page:
1. Open `outputs/extracted/english/<script_id>/checkpoints/page_<n>.json`.
2. Edit `stage2_verification.verified_transcript` directly to match the student's actual ink.
3. Re-run `python3 scripts/resegment_extraction.py --lang english --script <script_id>`.

---

## 🚀 Quick Command Cheat-Sheet (Full 24-Script Suite)

| Task | Scope | Command |
| :--- | :--- | :--- |
| **Generate CER Workspace** | 24 Scripts (1 pg/script) | `python3 scripts/make_transcription_gt.py --lang english --max-pages 24` |
| **Calculate CER / WER Benchmark** | All Corrected Pages | `python3 scripts/evaluate_transcription.py --lang english` |
| **Repair LaTeX & Struck Tags** | All 24 Scripts | `python3 scripts/repair_stage2_checkpoints.py --lang english` |
| **Re-segment Answer Boundaries** | All 24 Scripts | `python3 scripts/resegment_extraction.py --lang english` |
| **Sync `gt.txt` Marks to Datasets** | All 24 Scripts | `python3 scripts/sync_ground_truth.py` |
| **Grade Single Script vs `gt.txt`** | Single Script | `python3 scripts/evaluate_scripts.py --script-name SE_11_Q1_0010 --lang english --api --eval-mode modular --force-evaluate -y` |
| **Batch Evaluate All Scripts** | All 24 Scripts | `python3 scripts/evaluate_scripts.py --top 24 --lang english --api --eval-mode modular -y` |

