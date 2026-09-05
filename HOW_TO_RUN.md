# Quickstart Guide: Running the Multimodal Script Checking Pipeline

This guide provides clean, copy-pasteable instructions to run the **Multimodal Gemma 4 31B IT Script Checking & Evaluation Pipeline**.

---

## ⚡ 1. Setup & Environment Activation

```bash
# 1. Navigate to the repository root
cd /mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI

# 2. Activate the Python virtual environment
source ./script_checking/bin/activate

# 3. Configure PyTorch memory management (RTX 5090 / CUDA)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

> [!TIP]
> **Execution Engines**:
> - **Shared PC / LM Studio (`--api`)**: Connects to the already loaded Gemma 4 31B server on `http://localhost:1234/v1`. Consumes **0 additional GPU VRAM**.
> - **Direct CUDA GPU (`--quant 4bit`)**: Loads Gemma 4 31B IT directly onto the RTX 5090 using bitsandbytes NF4 quantization (requires ~18–20 GB free VRAM).
> - **Dev Mock Mode (`--mock`)**: Zero-weight CPU simulation for quick pipeline and workflow testing.

---

## 📂 2. Recommended Workflow: Decoupled Multi-Step Execution

The evaluation pipeline separates **Exam Questions**, **Student Scripts**, and **Rubric Evaluation** into modular, self-contained stages:

```
[Google Drive / Local Storage]
   ├── Questions (SE_11_Q1.pdf, SB_11_Q1.pdf) ──> scripts/download_questions.py ──> scripts/extract_questions.py ──> outputs/questions/<lang>/<qid>.json ──┐
   └── Scripts   (SE_11_Q1_0001.pdf)          ──> scripts/download_drive_pdfs.py  ──> scripts/extract_scripts.py   ──> outputs/extracted/<lang>/<sid>/     ──┴──> scripts/evaluate_scripts.py ──> outputs/evaluated/<lang>/<sid>/
```

---

### 📝 Stage A: Question Papers (Download & Extract)

Exam questions define the prompt, stimulus, instructions, and marks against which scripts are evaluated.
- **Naming Standard**: Question names uniquely identify subject, exam code, and question:
  - English: `SE_11_Q1` (automatically matches scripts like `SE_11_Q1_0001.pdf`)
  - Bangla: `SB_11_Q1` (automatically matches scripts like `SB_11_Q1_0002.pdf`)

#### 📥 Step 2.1: Download Question Papers
Download question PDFs from Google Drive (`GDRIVE_ENGLISH_QUESTION` / `GDRIVE_BANGLA_QUESTION` in `.env`) into `data/questions/<lang>/`:

```bash
# Download English questions (defaults to data/questions/english)
python3 scripts/download_questions.py --lang english

# Download Bangla questions (defaults to data/questions/bangla)
python3 scripts/download_questions.py --lang bangla

# Local check: list existing downloaded question files without contacting GDrive
python3 scripts/download_questions.py --lang english --local-only
```

> [!TIP]
> **Manual Placement**: You can also place question PDFs directly into `data/questions/english/` or `data/questions/bangla/` (e.g. `SE_11_Q1.pdf`, `SB_11_Q1.pdf`).

#### 🔍 Step 2.2: Extract Exam Questions
Extracts the question prompt text, instructions, and marks into structured question artifacts (`outputs/questions/<lang>/<question_id>.json`):

```bash
# Extract all questions in data/questions/english/
python3 scripts/extract_questions.py --lang english

# Extract all questions in data/questions/bangla/
python3 scripts/extract_questions.py --lang bangla

# Extract a specific question PDF with explicit ID
python3 scripts/extract_questions.py --pdf data/questions/english/SE_11_Q1.pdf --question-id SE_11_Q1 --lang english
```

> [!TIP]
> **Instant Digital Extraction**: Digital PDFs extract text instantly with zero VLM token cost. For scanned image PDFs, pass `--vlm` or `--api` to perform multimodal transcription.

---

### 📄 Stage B: Student Exam Scripts (Download & Extract)

#### 📥 Step 2.3: Download / Sync Exam Scripts from Google Drive
Download student exam script PDFs from Google Drive into `data/raw_pdfs/<lang>`. The smart caching layer scans existing files first and skips downloading anything already present locally.

```bash
# Download English exam scripts (defaults to data/raw_pdfs/english)
python3 scripts/download_drive_pdfs.py --lang english --top 5

# Download Bangla exam scripts (defaults to data/raw_pdfs/bangla)
python3 scripts/download_drive_pdfs.py --lang bangla --top 5

# Force re-download even if PDFs already exist locally
python3 scripts/download_drive_pdfs.py --lang english --force-download

# Local check: list existing downloaded scripts without contacting Google Drive
python3 scripts/download_drive_pdfs.py --lang english --local-only
```

#### 🔍 Step 2.4: Run Multimodal Extraction (Stages 0–3 & 0b)
Extracts student handwriting, detects teacher red ink, reverts silent model corrections, and exports the 13-column `raw_tier_dataset.csv`.

> [!IMPORTANT]
> **Strictly Local by Default**: Extraction strictly scans `data/raw_pdfs/<lang>/` (or `--image`) on your local drive. It **never** contacts Google Drive, checks network status, or triggers downloads.

```bash
# Fast Single-Pass Extraction on Local Scripts (RECOMMENDED: Local API, ~100s per 19-page PDF)
python3 scripts/extract_scripts.py --lang english --top 2 --api --fast -y

# Full 2-Pass Verification Extraction (includes visual auditor cross-check)
python3 scripts/extract_scripts.py --lang english --top 2 --api -y

# Direct CUDA GPU Execution (requires >=20 GB free VRAM)
python3 scripts/extract_scripts.py --lang english --top 2 --quant 4bit --fast -y

# Single PDF or Image File
python3 scripts/extract_scripts.py --image data/raw_pdfs/english/SE_11_Q1_0001.pdf --lang english --api --fast -y

# Fast CPU Mock Mode (Development & testing)
python3 scripts/extract_scripts.py --lang bangla --top 3 --mock -y
```

> [!NOTE]
> **Page Checkpointing & Resume**: Extracted pages are saved to disk immediately (`checkpoints/page_<N>.json`). If an extraction is interrupted, re-running the command automatically resumes from the last completed page.

---

### ⚖️ Stage C: Rubric Evaluation with Question Matching

#### ⚖️ Step 2.5: Run Rubric Evaluation (Stage 4)
Grades the pre-extracted transcripts against official rubrics, matching the student's answer to the exact extracted question paper (e.g. script `SE_11_Q1_0001` matches question `SE_11_Q1`). Teacher marks remain strictly isolated from grading inputs to prevent evaluation bias.

```bash
# Auto-matches script SE_11_Q1_0001 to question SE_11_Q1 (~12-15 seconds)
python3 scripts/evaluate_scripts.py --script-name SE_11_Q1_0001 --lang english --api -y

# Batch evaluate the top N extracted scripts (each script auto-matches its question)
python3 scripts/evaluate_scripts.py --top 5 --lang english --api -y

# Explicitly specify a question ID or question JSON override
python3 scripts/evaluate_scripts.py --script-name SE_11_Q1_0001 --question SE_11_Q1 --lang english --api -y

# Evaluate using a custom rubric file
python3 scripts/evaluate_scripts.py --script-name SE_11_Q1_0001 --rubric configs/rubrics/english_writing.yaml --api -y
```

---

## 🚀 3. Unified Controller (Stages 0–4 End-to-End)

To run the entire pipeline (Extraction + Evaluation) in a single command:

```bash
# End-to-end processing on local scripts with Local API & Fast Mode
python3 scripts/process_scripts.py --lang english --top 2 --local-only --api --fast -y

# End-to-end processing directly on CUDA RTX 5090
python3 scripts/process_scripts.py --lang bangla --top 5 --local-only --quant 4bit --fast -y

# Interactive Wizard (interactive terminal walkthrough)
python3 scripts/process_scripts.py
```

---

## 📊 4. Inspecting Outputs & Datasets

Extraction artifacts and Evaluation reports are kept in completely separate directory trees:

### 📄 Extraction Outputs (`outputs/extracted/<lang>/`)
Contains verified student handwriting transcripts, error catalogs, teacher mark audits, and research datasets (isolated from grading scores):
```
outputs/extracted/<lang>/
  ├── raw_tier_dataset.csv             # 13-column consolidated research dataset CSV
  └── <script_id>/
      ├── checkpoints/                 # Per-page checkpoints for instant resume
      │   ├── page_1.json
      │   └── page_2.json
      ├── stage0b_teacher_marks.json   # Extracted red-ink teacher marks (Q_no, marks, location)
      ├── stage1_raw_transcript.txt    # Verbatim student transcript (preserving errors)
      ├── stage1_transcription.json    # Stage 1 metrics & character stats
      ├── stage2_verification.json     # Silent autocorrection audit diffs
      ├── stage2_verified_transcript.txt # Canonical verified text
      ├── stage3_errors.json           # Global linguistic error catalog
      ├── stage3_errors.csv            # Tabular error list (spelling, grammar, syntax)
      ├── extraction_result.json       # Complete consolidated extraction package
      ├── extraction_summary.md        # Human-readable extraction overview
      └── raw_tier_records.csv         # Script-level raw-tier CSV rows
```

### ⚖️ Evaluation Outputs (`outputs/evaluated/<lang>/`)
Contains rubric criterion scores, content scoring, linguistic deductions, pedagogical feedback, and final grade reports:
```
outputs/evaluated/<lang>/
  └── <script_id>/
      ├── stage4_evaluation.json       # Rubric scores & penalty breakdown
      ├── complete_report.json         # Complete consolidated evaluation artifact
      └── evaluation_report.md         # Formatted teacher report & pedagogical feedback
```

### Quick Commands to View Outputs:
```bash
# View human-readable extraction summary
cat outputs/extracted/english/SE_11_Q1_0001/extraction_summary.md

# View teacher marks extracted from red ink
cat outputs/extracted/english/SE_11_Q1_0001/stage0b_teacher_marks.json

# View the 13-column research dataset CSV
cat outputs/extracted/english/raw_tier_dataset.csv

# View complete pedagogical evaluation report (Stage 4)
cat outputs/evaluated/english/SE_11_Q1_0001/evaluation_report.md
```

---

## 🧠 5. Context Token Budget & Usage Breakdown (Per Stage)

The active context window in LM Studio / Local API is **4,096 tokens** (`n_ctx = 4096`). In direct CUDA GPU mode, the native Gemma 4 31B window is **8,192 tokens**.

Below is the exact input/output token breakdown for each stage across **Extraction** and **Evaluation**:

| Phase | Stage & Name | Modality | Input Tokens (Prompt + Image) | Output Tokens (Generation) | Total Tokens Used | Context Usage (out of 4,096) | Notes & Optimization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Extraction** | **Stage 0**: OpenCV Red-Ink Detection | Pure CV (NumPy) | 0 tokens | 0 tokens | **0** | **0%** | Non-LLM CPU computation; consumes zero context. |
| **Extraction** | **Stage 1**: Verbatim Transcription | Multimodal (Vision + Text) | ~1,050 – 1,150 tokens<br>*(~250 prompt + ~800 vision)* | ~100 – 400 tokens *(up to 1,536)* | **~1,200 – 1,800** | **29% – 44%** | Downscales high-res image to 1280px to cap vision tokens at ~800. |
| **Extraction** | **Stage 2**: Autocorrection Verification | Multimodal (Vision + Text) | ~1,300 – 1,900 tokens<br>*(~300 prompt + ~800 vision + ~500 St 1 text)* | ~150 – 400 tokens *(diff JSON)* | **~1,500 – 2,300** | **37% – 56%** | **0 tokens (0%)** when `--fast` mode is used. |
| **Extraction** | **Stage 0b**: Teacher Mark Extraction | Multimodal (Vision + Text) | ~1,020 tokens<br>*(~220 prompt + ~800 vision)* | ~50 – 200 tokens *(JSON marks)* | **~1,100 – 1,250** | **27% – 30%** | Runs conditionally *only* on pages with red pen strokes (≥1,200 px). |
| **Extraction** | **Stage 3**: Linguistic Error Extraction | Text-Only | ~1,650 – 2,150 tokens<br>*(~350 prompt + ~1,500 full script text)* | ~400 – 1,000 tokens *(error catalog)* | **~2,200 – 3,100** | **54% – 75%** | Runs **once per script** globally on aggregated text. |
| **Evaluation** | **Stage 4**: Rubric Scoring & Feedback | Text-Only | ~2,000 – 2,400 tokens<br>*(~400 prompt + ~350 rubric + ~250 error summary + ~1,200 verified text)* | ~600 – 1,200 tokens *(scores + justifications)* | **~2,600 – 3,500** | **64% – 85%** | Linguistic errors are compactly summarized to ensure ample headroom. |

> [!TIP]
> **Key Safeguards Applied**:
> 1. **Image Resolution Clamped**: Images are resized to max 1280px in [`LocalAPIEngine`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/engine/local_api_engine.py), capping visual embeddings to ~800 tokens and preventing context overflows.
> 2. **Stage 4 Error Compaction**: Stage 3 errors are condensed into high-level counts and sample snippets rather than dumping thousands of raw JSON lines into the rubric prompt.
> 3. **Non-Thinking Mode (`reasoning_effort="none"`)**: In standard non-thinking mode, reasoning tokens are suppressed so output budgets go directly to structured JSON without wasteful context exhaustion.
> 4. **Persistent Token Tracking**: Live context usage is not just printed to the console—it is permanently recorded in:
>    - `extraction_result.json` (under `metadata["token_usage"]` and per-page in `PageExtractionResult.token_usage`)
>    - `checkpoints/page_<N>.json` (cached per page for resumable extraction)
>    - `extraction_summary.md` (in a dedicated "Context & Token Usage Breakdown" table)
>    - `complete_report.json` (under `metadata["stage4_token_usage"]` and `metadata["extraction_token_usage"]`)
>    - `evaluation_report.md` (in the Executive Summary and a dedicated "Pipeline Context & Token Usage Breakdown" table)
>    - `raw_tier_dataset.csv` & `raw_tier_records.csv` (inside `ocr_flags` preserving the strict 13-column schema, e.g. `illegible: 0, unclear: 0, struck: 0 | tokens: 742/4096 (18.1%)`)

---

## 🛠️ 6. CLI Flags Reference

### Question Download Flags (`scripts/download_questions.py`)
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--lang` | Question dataset: `english` or `bangla` | `english` |
| `--url URL` | Google Drive folder or file URL (overrides `.env`) | From `.env` |
| `--target-dir PATH` | Destination directory for question PDFs | `data/questions/<lang>` |
| `--force-download` | Re-download question PDFs even if existing locally | Skips existing |
| `--local-only` | List local question files without checking Google Drive | `False` |

### Question Extraction Flags (`scripts/extract_questions.py`)
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--lang` | Question subject: `english` or `bangla` | `english` |
| `--pdf PATH` / `--image PATH` | Path to a single question PDF or image | All in questions dir |
| `--question-id ID` | Explicit canonical question ID (e.g. `SE_11_Q1`) | Deduces from filename |
| `--questions-dir PATH` | Directory to scan for question PDFs | `data/questions/<lang>` |
| `--output-dir PATH` | Output directory for question JSONs | `outputs/questions` |
| `--vlm` | Force using Vision LLM even if digital text is present | `False` (digital text first) |

### Script Download Flags (`scripts/download_drive_pdfs.py`)
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--lang` | Exam subject dataset: `english` or `bangla` | `bangla` |
| `--top N` | Limit download to the first `N` PDFs | All available in folder |
| `--target-dir PATH` | Custom destination directory for PDFs | `data/raw_pdfs/<lang>` |
| `--url URL` | Custom Google Drive folder URL override | Language default GDrive URL |
| `--force-download` | Re-download from Google Drive even if PDFs already exist locally | Skips existing |
| `--local-only` / `--skip-download` | List existing local scripts without contacting Google Drive | `False` |

### Script Extraction Flags (`scripts/extract_scripts.py`)
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--api` | Use local OpenAI-compatible API (LM Studio on port 1234, 0 extra VRAM) | `False` |
| `--api-url` | URL for local API server | `http://localhost:1234/v1` |
| `--fast` / `--skip-stage2` | Fast single-pass mode; skips duplicate Stage 2 visual pass | `False` |
| `--lang` | Exam subject: `english` or `bangla` | `bangla` |
| `--top N` | Limit processing to the first `N` PDFs in local directory | All available |
| `--image PATH` | Path to a single PDF or image to extract | `None` |
| `--quant` | Direct CUDA quantization: `4bit`, `8bit`, or `none` | `4bit` |
| `--mock` | Fast zero-weight CPU simulation for dev testing | `False` |
| `-y` / `--non-interactive` | Bypass interactive setup prompts and use CLI values | Interactive |
| `--force-extract` | Re-extract scripts even if output already exists | Skips existing |
| `--sync-drive` | Explicitly check and download missing PDFs from Google Drive | `False` (strictly local) |
| `--force-download` | Force re-downloading PDFs from Google Drive (requires `--sync-drive`) | Skips existing |

### Evaluation Flags (`scripts/evaluate_scripts.py`)
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--script-name ID` | Name/ID of the script to evaluate (e.g. `SE_11_Q1_0001`) | `None` |
| `--question ID` | Question ID (`SE_11_Q1`) or path to JSON override | Auto-matched by script ID |
| `--questions-dir PATH` | Directory containing extracted question JSONs | `outputs/questions` |
| `--extraction-dir PATH` | Directory containing extracted scripts to evaluate | `outputs/extracted/<lang>` |
| `--output-dir PATH` | Directory to save evaluation reports | `outputs/evaluated/<lang>` |
| `--top N` | Evaluate the first `N` pre-extracted scripts in directory | All available |
| `--lang` | Subject/language rubric: `english` or `bangla` | `bangla` |
| `--rubric PATH` | Custom rubric YAML file path | Language default |
| `--api` | Use local OpenAI-compatible API for evaluation | `False` |
| `--skip-evaluated` | Skip scripts that already have completed evaluation reports | `False` |
| `-y` / `--non-interactive` | Bypass interactive setup prompts | Interactive |
