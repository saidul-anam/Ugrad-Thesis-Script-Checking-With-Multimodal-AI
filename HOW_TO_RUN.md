# Quickstart: How to Run the Pipeline Overall

This cheatsheet provides all copy-pasteable commands to run the **4-Stage Multimodal Gemma 4 31B IT Script Checking Pipeline** across development and production environments.

---

## ⚡ 1. Setup & Environment Activation

```powershell
# 1. Activate your virtual environment (e.g. on Windows PowerShell)
D:\environments\myenv\scripts\Activate.ps1

# 2. Ensure dependencies are installed
pip install -r requirements.txt

# 3. Configure Hugging Face Token in .env
# Gemma models are gated: accept terms at https://huggingface.co/google/gemma-4-31b-it
# Create a token at https://huggingface.co/settings/tokens and add to .env:
# HF_TOKEN=hf_your_token_here
```

---

## 📂 2. Separate Extraction and Evaluation Workflows (Recommended)

To keep extraction (multimodal vision processing) and evaluation (rubric grading & feedback) separate:

### 🔍 Step 2.1: Run Extraction (Stages 1 to 3)
Extracts handwriting verbatim, verifies against visual images, audits silent corrections, extracts spelling/grammar errors, and saves artifacts to `outputs/extracted/<lang>/<script_id>/`.

```powershell
# Interactive Wizard (select language, top limit, GPU mode):
python scripts/extract_scripts.py

# Extract Top 5 Bangla scripts on GPU:
python scripts/extract_scripts.py --lang bangla --top 5 --quant 4bit

# Extract on Development PC (Mock mode):
python scripts/extract_scripts.py --lang bangla --top 3 --mock
```

---

### ⚖️ Step 2.2: Run Evaluation (Stage 4)
Loads pre-extracted scripts and evaluates them against rubrics. Accepts **either** a specific script name/ID **or** a count (`--top N`).

```powershell
# Option A: Evaluate by specific script name:
python scripts/evaluate_scripts.py --script-name sample_bangla_01 --lang bangla --quant 4bit

# Option B: Evaluate Top N scripts from the extraction directory:
python scripts/evaluate_scripts.py --top 5 --lang bangla --quant 4bit

# Option C: Evaluate all extracted scripts in directory:
python scripts/evaluate_scripts.py --lang bangla --quant 4bit

# Option D: Development PC (Mock mode):
python scripts/evaluate_scripts.py --script-name sample_bangla_01 --lang bangla --mock
```

---

## 🚀 3. Running Unified Controller on Target NVIDIA RTX 5090 (32GB VRAM)

> [!IMPORTANT]
> On the RTX 5090 machine, the pipeline uses **Gemma 4 31B IT** with **4-bit NF4 Quantization** (~18–20 GB VRAM footprint), running in bfloat16 precision with CUDA acceleration.

### Step 1: Verify CUDA & GPU Environment
```bash
python scripts/setup_env.py
```
*(Check that CUDA is available and that total VRAM shows ~32 GB)*

### Step 2: Interactive Terminal Mode (Wizard)
```bash
# Unified end-to-end runner (Stages 1–4 together):
python scripts/process_scripts.py
```

### Step 3: Run Bangla Exam Scripts Directly
```bash
# Top 5 Bangla scripts with Creative Question rubric
python scripts/process_scripts.py --lang bangla --top 5 --quant 4bit
```

---

## 📊 4. How to Inspect Output Results

Extracted artifacts and evaluation reports are stored per script:

```
outputs/extracted/<lang>/<script_id>/
  ├── stage1_transcription.json        # Stage 1 metrics & character stats
  ├── stage1_raw_transcript.txt        # Raw verbatim transcription
  ├── stage2_verification.json         # Silent autocorrection audit diffs
  ├── stage2_verified_transcript.txt   # Verified canonical text
  ├── stage3_errors.json               # Extracted linguistic error catalog
  ├── stage3_errors.csv                # Tabular error list (spelling/grammar/syntax)
  ├── extraction_result.json           # Consolidated extraction package
  ├── extraction_summary.md            # Human-readable extraction summary
  ├── stage4_evaluation.json           # Rubric marks & penalty deductions
  ├── complete_report.json             # Consolidated 4-stage report
  └── evaluation_report.md             # Teacher evaluation & pedagogical report
```

To quickly view the summary markdown report of an evaluated script:
```powershell
cat outputs/extracted/bangla/<script_id>/evaluation_report.md
```

---

## 🛠️ 5. Command-Line Options Reference

### Extraction Options (`scripts/extract_scripts.py`)
| Argument | Example | Purpose |
| :--- | :--- | :--- |
| `--lang` | `--lang bangla` or `--lang english` | Select subject/language. |
| `--top N` | `--top 5` | Limit extraction to first `N` scripts. |
| `--image` | `--image path/to/script.pdf` | Extract a single PDF or image file. |
| `--pdf-dir` | `--pdf-dir data/raw_pdfs/bangla` | Source directory containing script PDFs. |
| `--output-dir` | `--output-dir outputs/extracted/bangla` | Target directory for extraction artifacts. |
| `--mock` | `--mock` | Run simulated CPU engine for development. |
| `--quant` | `--quant 4bit` | `4bit` (recommended for RTX 5090), `8bit`, or `none`. |
| `--thinking` | `--thinking` | Enable reasoning/thinking mode ablation. |
| `--force-extract` | `--force-extract` | Re-extract even if artifacts already exist. |

### Evaluation Options (`scripts/evaluate_scripts.py`)
| Argument | Example | Purpose |
| :--- | :--- | :--- |
| `--script-name` | `--script-name script_01` | Evaluate a specific script by name/ID. |
| `--top N` | `--top 5` | Evaluate first `N` extracted scripts in directory. |
| `--extraction-dir` | `--extraction-dir outputs/extracted/bangla` | Directory containing extracted script folders. |
| `--lang` | `--lang bangla` or `--lang english` | Select subject/language rubric default. |
| `--rubric` | `--rubric path/to/rubric.yaml` | Select custom grading rubric. |
| `--force-evaluate` | `--force-evaluate` | Force re-evaluation even if report exists. |
