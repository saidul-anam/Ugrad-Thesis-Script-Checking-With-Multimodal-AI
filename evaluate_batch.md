# Batch Evaluation Guide: Setting Up Question Text & Script Text

This guide details how to set up **Question Papers (`.txt`)** and **Student Exam Scripts (`.txt`)** to evaluate transcribed/digitized student answers directly using the **Stage 4 Modular Evaluator** without running the vision/OCR pipeline (Stage 1 to 3).

---
# Evaluate Class 6 (22 scripts)
./script_checking/bin/python -u scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_06_q1 \
  --paper SE_06_Q1 \
  --non-interactive

# Evaluate Class 7 (34 scripts)
./script_checking/bin/python -u scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_07_q1 \
  --paper SE_07_Q1 \
  --non-interactive

# Evaluate Class 10 (33 scripts)
./script_checking/bin/python -u scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_10_q1 \
  --paper SE_10_Q1 \
  --non-interactive

# -------------------------------------------------------------
# 🚀 Force Re-Evaluate (Overwrites existing reports with updated segmenter)
# -------------------------------------------------------------
# Run all 4 classes sequentially in background:
nohup ./scripts/force_evaluate_all.sh > logs/force_eval_all.log 2>&1 &

# Or evaluate only diagnosed segmentation issue scripts first:
./scripts/force_evaluate_all.sh --issues-only

# Or force re-evaluate class-by-class:
# SE_06_Q1 (Class 6 - 22 scripts)
nohup ./script_checking/bin/python -u scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_06_q1 \
  --paper SE_06_Q1 \
  --force-evaluate \
  --non-interactive > eval_se_06_q1.log 2>&1 &

# SE_07_Q1 (Class 7 - 34 scripts)
nohup ./script_checking/bin/python -u scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_07_q1 \
  --paper SE_07_Q1 \
  --force-evaluate \
  --non-interactive > eval_se_07_q1.log 2>&1 &

# SE_08_Q1 (Class 8 - 35 scripts)
nohup ./script_checking/bin/python -u scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_08_q1 \
  --paper SE_08_Q1 \
  --force-evaluate \
  --non-interactive > eval_se_08_q1.log 2>&1 &

# SE_10_Q1 (Class 10 - 33 scripts)
nohup ./script_checking/bin/python -u scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_10_q1 \
  --paper SE_10_Q1 \
  --force-evaluate \
  --non-interactive > eval_se_10_q1.log 2>&1 &



## ⚡ 1. Environment Activation

Always activate the project virtual environment before executing scripts:

```bash
# Navigate to the repository root
cd /mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI

# Activate the dedicated Python virtual environment
source script_checking/bin/activate

# (Optional for CUDA GPU evaluation) Prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 📋 2. Workflow Overview

```
[Raw Transcriptions / Digitized Data]
  ├── Question Paper Text (.txt) ──> data/questions/<PAPER_ID>.txt
  │                                  └── outputs/questions/<lang>/<PAPER_ID>.json
  │                                  └── configs/answer_keys/<PAPER_ID>.yaml
  │                                  └── configs/rubrics/<rubric>.yaml
  │
  ├── Student Scripts (.txt)     ──> <INPUT_DIR>/<PAPER_ID>_<SCRIPT_ID>.txt
  └── Examiner Marks (.csv)      ──> data/ground_truth/transcripts/teacher_marks/<PAPER_ID>_teacher_marks.csv
                                             │
                                             ▼
                             scripts/setup_data_without_pipeline.py
                                             │
                                             ▼
                             data_without_pipeline/<lang>/<paper_id>/<script_id>/
                                             │
                                             ▼
                             scripts/evaluate_scripts.py
                                             │
                                             ▼
                             outputs/evaluated/<lang>/<paper_id>/<script_id>/
                                   ├── evaluation_report.md
                                   ├── stage4_evaluation.json
                                   └── complete_report.json
```

---

## 📝 3. How to Set Up the Question Paper (`.txt`)

The question paper establishes the exact prompts, instructions, sub-questions, and mark distributions against which scripts are evaluated.

### 3.1 File Location & Naming Convention

Place raw question text files in:
- `data/questions/<PAPER_ID>.txt` (e.g. [data/questions/SE_08_Q1.txt](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/data/questions/SE_08_Q1.txt))
- Or language-specific: `data/questions/<lang>/<PAPER_ID>.txt` (e.g. [data/questions/english/SE_08_Q1.txt](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/data/questions/english/SE_08_Q1.txt))

**Naming Standard**:
- **English**: `SE_<CLASS>_Q<NUM>.txt` (e.g. `SE_08_Q1.txt`, `SE_10_Q1.txt`, `SE_11_Q1.txt`)
- **Bangla**: `SB_<CLASS>_Q<NUM>.txt` (e.g. `SB_07_Q1.txt`, `SB_11_Q1.txt`)

### 3.2 Question Text Formatting Standard

Format the `.txt` file with clear sections and page delimiters:

1. **Page Delimiters**: Use `--- PAGE 1 ---`, `--- PAGE 2 ---` across pages.
2. **Exam Header**: School/Institution name, examination year, class, subject, full marks, and duration.
3. **Reading Passages / Stimuli**: Full text of any comprehension passages or poems.
4. **Numbered Sub-Questions**: Distinct sub-question numbers with mark allocations (e.g. `1. Choose the best answer from the alternatives. $1 \times 7 = 7$`).
5. **Item Lists**: Label items cleanly (`a.`, `b.`, `c.`, etc.) and options (`i.`, `ii.`, `iii.`, `iv.`).
6. **Writing Section**: Distinct prompts for creative writing (e.g. Story, Paragraph, Dialogue).

#### Example: `data/questions/SE_08_Q1.txt`
```text
--- PAGE 1 ---
Engineering University School and College, Dhaka
Half-Yearly Examination—2026
Class VIII
Subject: English 1st Paper
Time : 3 Hours
Full Marks—100

Part-A: Reading
Read the text and answer the questions 1 & 2 :
The word 'hygiene' means the practice of keeping ourselves clean...

1. Choose the best answer from the alternatives. $1 \times 7 = 7$
a. The term 'hygiene' refers to—
i. The use of medicine
ii. the act of buying new clothes
iii. The process of cooking food
iv. the practice of keeping ourselves clean
...

--- PAGE 2 ---
2. Write short answers to the following questions : $2 \times 5 = 10$
a. What does the word 'hygiene' mean?
b. What is thought to be godliness?
...

Part-B : Writing
11. Read the beginning of the following story and complete it. 7
    Once there lived a wood cutter. He was very poor but honest...
12. Write a paragraph on "Tree Plantation." 7
```

### 3.3 Registering Structured Question Artifacts

Stage 4 loads question definitions from `outputs/questions/<lang>/<PAPER_ID>.json`.

Create or verify the question JSON at [outputs/questions/english/SE_08_Q1.json](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/outputs/questions/english/SE_08_Q1.json):

```json
{
  "question_id": "SE_08_Q1",
  "language": "english",
  "title": "Engineering University School and College, Dhaka - Class VIII - English 1st Paper",
  "question_text": "...",
  "total_marks": 100.0,
  "sub_questions": [
    {
      "q_no": "1",
      "name": "Q1: MCQ (Hygiene Passage)",
      "title": "Q1: MCQ (Hygiene Passage)",
      "marks": 7.0,
      "max_marks": 7.0,
      "task_type": "MCQ",
      "part": "A",
      "instructions": "Choose the best answer from the alternatives (a to g). 1 x 7 = 7",
      "text": "1. Choose the best answer from the alternatives..."
    },
    {
      "q_no": "2",
      "name": "Q2: Short Answer Questions (Hygiene Passage)",
      "title": "Q2: Short Answer Questions (Hygiene Passage)",
      "marks": 10.0,
      "max_marks": 10.0,
      "task_type": "Short_Answer",
      "part": "A",
      "instructions": "Write short answers to the following questions : 2 x 5 = 10",
      "text": "2. Write short answers to the following questions..."
    }
  ]
}
```

> [!TIP]
> Also save a companion markdown preview at `outputs/questions/<lang>/<PAPER_ID>.md` for human inspection.

### 3.4 Rubric & Answer Key Configuration

1. **Class-Level Rubric** (`configs/rubrics/`):
   - **Class 8**: [configs/rubrics/english_junior.yaml](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/configs/rubrics/english_junior.yaml)
   - **Class 10**: [configs/rubrics/english_secondary.yaml](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/configs/rubrics/english_secondary.yaml)
   - **Class 11/12**: [configs/rubrics/english_higher_secondary.yaml](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/configs/rubrics/english_higher_secondary.yaml)
   - Define sub-question scoring modes: `A_ITEM` (objective/MCQ/Cloze), `B_POINT` (short answer), `C_BAND` (creative writing).

2. **Deterministic Answer Key** (`configs/answer_keys/<PAPER_ID>.yaml`):
   - Example: [configs/answer_keys/SE_08_Q1.yaml](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/configs/answer_keys/SE_08_Q1.yaml)
   - Stores accepted answer variants for objective questions, casing tolerance, regex patterns, and rearrangement keys.

### 3.5 Validating Question & Rubric Integrity

Verify that the question JSON, rubric YAML, and answer keys are fully consistent:

```bash
# Run unit tests validating question resolution and rubric integrity
pytest tests/test_question_utils.py -v
```

---

## 📄 4. How to Set Up the Student Scripts (`.txt`)

### 4.1 Folder Organization & File Naming

Transcribed or digitized student scripts can reside in any staging folder (e.g. `SE_08_Q1_extraction/`):

```
SE_08_Q1_extraction/
├── SE_08_Q1.txt            <-- Question paper (ignored by student script parser)
├── SE_08_Q1_0001.txt       <-- Student Script 1
├── SE_08_Q1_0002.txt       <-- Student Script 2
├── ...
└── SE_08_Q1_0036.txt       <-- Student Script 36
```

**Naming Convention**:
- Must follow `<PAPER_ID>_<SCRIPT_ID>.txt` (e.g. `SE_08_Q1_0001.txt`).
- The parser automatically differentiates student scripts from question papers by verifying the numeric/code suffix (e.g. `_[0-9]{3,}`).

### 4.2 Script Text Format & Page Delimiters

Transcribed student scripts should include page markers to maintain multi-page fidelity:

```text
--- PAGE 1 ---
Answer to the question number - 01

(a) iv. the practice of keeping ourselves clean
(b) ii. next to godliness
(c) iii. our body
(d) iii. dirt and bacteria
(e) ii. they give off bad smell
(f) i. by wearing socks and shoes
(g) iv. wash our hands

--- PAGE 2 ---
Answer to the question number - 02
(a)
The word hygiene means the practice of keeping ourselves clean. It also means to keep our home and work places clean.
(b)
Hygiene is thought to be next to godliness.
...
```

**Supported Page Delimiters**:
- `--- PAGE <N> ---` (e.g. `--- PAGE 1 ---`, `--- PAGE 2 ---`)
- `--- Page Break ---` (auto-increments page index)
- If omitted, the entire file is ingested as Page 1.

**Transcriber Markup Tags**:
- `[struck: <word>]` — Text crossed out by the student.
- `[illegible]` — Unreadable handwriting.
- `[unclear: <guess>?]` — Ambiguous handwriting.

---

## 📊 5. Setting Up Examiner Ground Truth Marks (`.csv`)

Official examiner marks are synchronized from human ground truth CSV files and dynamically reported in the final evaluation report.

> [!IMPORTANT]
> The model's internal OCR mark extraction (`stage0b_teacher_marks.json`) is **NEVER** connected to or treated as ground truth. Official ground truth marks are strictly loaded from verified CSV files in `data/ground_truth/transcripts/teacher_marks/`.

### 5.1 CSV File Location & Format

Place the examiner marks CSV in:
`data/ground_truth/transcripts/teacher_marks/<PAPER_ID>_teacher_marks.csv`  
(e.g. [data/ground_truth/transcripts/teacher_marks/SE_08_Q1_teacher_marks.csv](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/data/ground_truth/transcripts/teacher_marks/SE_08_Q1_teacher_marks.csv))

**CSV Structure**:
```csv
Script ID,total_score,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12,Q13,Q14
SE_08_Q1_0001,48,7,4,4,4,4,4,4,2,3,4,4,4,2,2
SE_08_Q1_0002,40,7,2,2,3,4,2,3,3,4,3,3,2,1,1
SE_08_Q1_0003,51,7,4,4,4,3,4,3,3,4,4,4,4,2,3
```

- **Header**: `Script ID` (or `script_id`), `total_score` (or `teacher_mark`), followed by question columns `Q1`, `Q2`, ..., `Q14` (or `q1`, `q2`, ...).
- If examiner marks are missing for a particular script, the evaluator safely proceeds without error and marks ground truth as unavailable.

---

## ⚙️ 6. Converting Scripts for Evaluation (`setup_data_without_pipeline.py`)

Run [scripts/setup_data_without_pipeline.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/scripts/setup_data_without_pipeline.py) to package the `.txt` files into standard pipeline artifacts inside `data_without_pipeline/`.

### 6.1 Basic Conversion Commands

```bash
# Convert ALL student scripts in an input folder:
python scripts/setup_data_without_pipeline.py \
  --input-dir SE_08_Q1_extraction \
  --paper SE_08_Q1 \
  --lang english

# Convert specific scripts by ID or filename:
python scripts/setup_data_without_pipeline.py SE_08_Q1_0001 SE_08_Q1_0002 \
  --input-dir SE_08_Q1_extraction \
  --paper SE_08_Q1

# Convert only the first N scripts:
python scripts/setup_data_without_pipeline.py \
  --input-dir SE_08_Q1_extraction \
  --paper SE_08_Q1 \
  --top 5

# Convert scripts without syncing ground truth teacher marks:
python scripts/setup_data_without_pipeline.py \
  --input-dir SE_08_Q1_extraction \
  --paper SE_08_Q1 \
  --no-gt
```

### 6.2 CLI Parameters Reference

| Flag | Short | Default | Description |
| :--- | :---: | :--- | :--- |
| `scripts` | — | `[]` (all) | Positional list of script IDs or files to process |
| `--input-dir` | `-i` | `SE_08_Q1_extraction` | Staging directory containing `.txt` scripts |
| `--output-dir` | `-o` | `data_without_pipeline` | Target root folder for canonical extraction artifacts |
| `--paper` | `-p` | Auto-inferred | Exam paper ID (e.g. `SE_08_Q1`) |
| `--lang` | `-l` | `english` | Language/subject (`english` or `bangla`) |
| `--sync-gt` | — | `True` | Synchronizes examiner marks from `data/ground_truth/transcripts/teacher_marks/` |
| `--no-gt` | — | `False` | Disables examiner ground truth synchronization |
| `--top` | — | `None` | Limits conversion to the first N scripts |

### 6.3 Generated Output Artifacts

For each script `SE_08_Q1_0001`, the converter generates:
```
data_without_pipeline/english/se_08_q1/SE_08_Q1_0001/
├── extraction_result.json       <-- Complete canonical extraction schema
├── stage1_transcription.json    <-- Transcribed text, page count, word metrics
├── stage2_verification.json     <-- Verified text record
├── stage3_errors.json           <-- Linguistic error catalog
├── ground_truth_marks.json      <-- Synchronized official examiner marks (if in CSV)
└── source_transcript.txt        <-- Copy of raw source text
```
Additionally, a convenience reference symlink is created at `data_without_pipeline/<SCRIPT_ID>`.

### 6.4 Validating Conversion

Run the test suite to verify the conversion controller:

```bash
pytest tests/test_setup_data_without_pipeline.py -v
```

---

## 🚀 7. Running Stage 4 Evaluation (`evaluate_scripts.py`)

Once converted into `data_without_pipeline/`, evaluate scripts using [scripts/evaluate_scripts.py](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/scripts/evaluate_scripts.py).

### 7.1 Single Script Evaluation

```bash
# 1. Fast Mock Dev Mode (CPU simulation, zero GPU VRAM)
python scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_08_q1 \
  --paper SE_08_Q1 \
  --script-name SE_08_Q1_0001 \
  --mock

# 2. Local API Engine (LM Studio / vLLM on port 1234, zero extra VRAM)
python scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_08_q1 \
  --paper SE_08_Q1 \
  --script-name SE_08_Q1_0001 \
  --api

# 3. Direct CUDA GPU (Gemma 4 31B IT with 4-bit quantization)
python scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_08_q1 \
  --paper SE_08_Q1 \
  --script-name SE_08_Q1_0001 \
  --quant 4bit
```

### 7.2 Batch Evaluation

```bash
# Evaluate ALL scripts non-interactively in Mock mode:
python scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_08_q1 \
  --paper SE_08_Q1 \
  --batch \
  --mock

# Evaluate TOP 5 scripts via Local API engine:
python scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_08_q1 \
  --paper SE_08_Q1 \
  --top 5 \
  --batch \
  --api

# Force re-evaluation of previously evaluated scripts:
python scripts/evaluate_scripts.py \
  --extraction-dir data_without_pipeline/english/se_08_q1 \
  --paper SE_08_Q1 \
  --force-evaluate \
  --batch \
  --mock
```

---

## 📈 8. Inspecting Evaluation Reports & Teacher Marks

Outputs are saved in:
`outputs/evaluated/english/<paper_id>/<script_id>/`

Files generated:
- **`evaluation_report.md`**: Human-readable Markdown report containing:
  - Total Score Summary & Percentage
  - Question-by-Question Evaluation Table
  - Criteria Breakdown & Content Justification
  - Linguistic Deductions (Spelling, Grammar, Syntax)
  - Pedagogical Strengths & Weaknesses
  - **Official Human Examiner Marks Comparison Table** (dynamically shown whenever ground truth exists):
    ```markdown
    ### Official Human Examiner Marks (Ground Truth)
    - Source: `SE_08_Q1_teacher_marks.csv`
    - Examiner Total Mark: **48.0 / 100.0**
    - Sub-Question Breakdown:
      | Question | Examiner Mark |
      | :--- | :--- |
      | Q1 | 7.0 |
      | Q2 | 4.0 |
      | Q3 | 4.0 |
      ...
    ```
- **`stage4_evaluation.json`**: Structured evaluation JSON schema.
- **`complete_report.json`**: Consolidated pipeline record containing stages 1–4 and metadata.

---

## ❓ 9. Troubleshooting & FAQ

1. **Q: Does the pipeline use OCR to guess the teacher marks for ground truth?**  
   **A:** No. The model's visual teacher mark extraction (`stage0b_teacher_marks.json`) is never used for ground truth. Ground truth is exclusively sourced from the verified human examiner CSV in `data/ground_truth/transcripts/teacher_marks/`.

2. **Q: What if a student script does not have teacher marks in the CSV?**  
   **A:** The script converts and evaluates completely normally. The evaluator simply notes that official examiner marks are unavailable and does not display an erroneous comparison.

3. **Q: What if a student script has no `--- PAGE N ---` markers?**  
   **A:** The converter treats the text as a single page (Page 1) and generates complete extraction artifacts.

4. **Q: How can I transfer a batch of transcript `.txt` files from Windows PowerShell?**  
   **A:** Use `scp` with directory recursive copy:
   ```powershell
   scp -r C:\Users\<Username>\Downloads\SE_08_Q1_extraction ubuntu@<HOST>:/mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/
   ```
