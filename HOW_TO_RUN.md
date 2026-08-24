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

## 💻 2. Running on Your Current PC (Development / Mock Mode)

> [!NOTE]
> Use the `--mock` flag to run the full 4-stage pipeline locally without downloading model weights or requiring a GPU.

### Scenario A: Process Top 3 PDFs from Google Drive
```bash
python scripts/process_scripts.py --top 3 --mock
```

### Scenario B: Download Script PDFs from Drive Only (No Evaluation)
```bash
python scripts/process_scripts.py --download-only
```

### Scenario C: Process a Single PDF or Image File
```bash
python scripts/run_pipeline.py --image data/raw_pdfs/sample_script.pdf --mock
```

### Scenario D: Run with Thinking Mode Ablation (Reasoning ON)
```bash
python scripts/process_scripts.py --top 3 --mock --thinking
```

### Scenario E: Run Unit Tests
```bash
pytest
```

---

## 🚀 3. Running on the Target NVIDIA RTX 5090 (32GB VRAM)

> [!IMPORTANT]
> On the RTX 5090 machine, the pipeline uses **Gemma 4 31B IT** with **4-bit NF4 Quantization** (~18–20 GB VRAM footprint), running in bfloat16 precision with CUDA acceleration.

### Step 1: Verify CUDA & GPU Environment
```bash
python scripts/setup_env.py
```
*(Check that CUDA is available and that total VRAM shows ~32 GB)*

### Step 2: Interactive Terminal Mode (Wizard)
```bash
# Simply run the script - it will ask whether you are processing Bangla or English, how many scripts, rubric, etc.:
python scripts/process_scripts.py
```

### Step 3: Run Bangla Exam Scripts Directly
```bash
# Top 5 Bangla scripts with Creative Question rubric (PDFs stored in data/raw_pdfs/bangla, outputs in outputs/runs/bangla)
python scripts/process_scripts.py --lang bangla --top 5 --quant 4bit

# All available Bangla scripts:
python scripts/process_scripts.py --lang bangla --quant 4bit
```

### Step 4: Run English Writing Exam Scripts Directly
```bash
# Top 5 English scripts with Essay Writing rubric (PDFs stored in data/raw_pdfs/english, outputs in outputs/runs/english)
python scripts/process_scripts.py --lang english --top 5 --quant 4bit
```

### Step 5: Run Thinking Mode Benchmark / Ablation Study
```bash
python scripts/evaluate_benchmark.py --image-dir data/samples/
```

---

## 📊 4. How to Inspect Output Results

Evaluated scripts are stored separately by language under `outputs/runs/<lang>/<script_id>/`:

```
outputs/runs/bangla/<script_id>/  (or outputs/runs/english/<script_id>/)
  ├── stage1_transcription.json        # Stage 1 metrics & character stats
  ├── stage1_raw_transcript.txt        # Raw verbatim transcription
  ├── stage2_verification.json         # Silent autocorrection audit diffs
  ├── stage2_verified_transcript.txt   # Verified canonical text
  ├── stage3_errors.json               # Extracted linguistic error catalog
  ├── stage3_errors.csv                # Tabular error list (spelling/grammar)
  ├── stage4_evaluation.json           # Rubric marks & penalty deductions
  ├── complete_report.json             # Consolidated 4-stage report
  └── evaluation_report.md             # Human-readable Markdown report
```

To quickly view the summary markdown report of a script:
```bash
# On Windows PowerShell
cat outputs/runs/bangla/<script_id>/evaluation_report.md
```

---

## 🛠️ 5. Command-Line Options Reference (`scripts/process_scripts.py`)

| Argument | Example | Purpose |
| :--- | :--- | :--- |
| `--lang` | `--lang bangla` or `--lang english` | Select subject/language (`bangla` or `english`). |
| `--top N` | `--top 5` | Limit download and processing to the first `N` scripts. |
| `--mock` | `--mock` | Use simulated CPU engine (no GPU / no model download). |
| `--model` | `--model google/gemma-4-31b-it` | Model checkpoint identifier or local directory. |
| `--quant` | `--quant 4bit` | `4bit` (recommended for RTX 5090), `8bit`, or `none`. |
| `--thinking` | `--thinking` | Enable reasoning/thinking mode ablation. |
| `--rubric` | `--rubric path/to/rubric.yaml` | Select grading rubric. |
| `--download-only` | `--download-only` | Download PDFs from Google Drive without evaluating. |
| `--local-only` | `--local-only` (or `--skip-download`) | Skip Google Drive download and use local PDFs only. |
| `--force-download`| `--force-download` | Force re-download even if PDFs already exist locally. |
