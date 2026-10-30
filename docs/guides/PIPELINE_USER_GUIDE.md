# Multimodal Script Checking Pipeline

This guide is for someone using the project for the first time. It explains what
the pipeline does and the normal, full-quality way to run it.

## What the pipeline does

The pipeline reads scanned handwritten exam scripts, extracts the student's
answers, and evaluates them against the matching question paper and rubric.
Teacher marks are extracted separately and are not used to influence the AI
grading.

```text
Question PDF ──> Question JSON ───────────────┐
                                              ├─> Question-aware evaluation
Student script PDF ──> Extraction ────────────┘       └─> Report
```

## Extraction stages (Stages 0–3)

Extraction turns each scanned page into a verified, question-aligned answer
record. It combines computer vision, multimodal Gemma calls, text analysis, and
saved checkpoints.

1. **Stage 0 — red-ink detection and image cleaning**  
   OpenCV uses HSV colour thresholding to locate red teacher ink. The page is
   divided into a left margin zone and a student-answer body zone. Body
   checkmarks can be removed with inpainting before transcription; margin marks
   are retained for teacher-mark extraction.
2. **Stage 0b — teacher-mark extraction**  
   When numeric red marks are present, Gemma reads the margin. The marks are
   aligned to questions using their vertical position on the page. Pages with
   only checkmarks can skip this extra VLM call.
3. **Stage 1 — verbatim transcription**  
   Gemma reads the cleaned page image and transcribes the student's writing.
   The prompt requires the original spelling and grammar to be preserved;
   uncertain writing is represented with markers such as `[unclear: ...]` or
   `[illegible]`. Large pages may be processed in horizontal bands and stitched
   back together.
4. **Stage 2 — visual verification**  
   A second multimodal pass compares the Stage 1 text with the original image.
   It corrects visual/OCR noise and reverts silent autocorrections that changed
   an authentic student error. Lexicon and handwriting-shape safeguards prevent
   the verifier from introducing implausible changes.
5. **Question segmentation**  
   The verified transcript is split at detected answer headers, question
   numbers, and sub-parts. Continuation pages are joined, and the extracted
   question-paper schema is used to align answer blocks with the correct
   questions.
6. **Stage 3 — linguistic error extraction**  
   A text-only Gemma call reviews the aligned transcript and identifies
   meaningful spelling, grammar, and syntax errors. Punctuation noise and
   objective-answer content are not treated like essay errors.
7. **Stage 3b — handwriting arbitration**  
   Candidate errors are checked against the page image and available
   handwriting evidence. A mark is removed when the apparent error is
   reasonably explained by an ambiguous pen stroke; only confirmed errors are
   sent to evaluation.

## Evaluation stage (Stage 4)

1. **Question-aware rubric scoring**  
   The evaluator loads the matching question JSON, answer key, rubric, verified
   answer segment, and confirmed error summary. In the recommended **modular**
   mode, each question or sub-question is evaluated in a separate context.
   This keeps unrelated answers from affecting one another and produces
   question-level marks, deductions, reasons, and feedback.
2. **Penalty and report generation**  
   Linguistic penalties are applied according to the configured rubric and
   marking rules. Content marks and language deductions remain separate, and
   teacher marks are kept out of the AI grading input. The results are exported
   as JSON and as a readable Markdown report.

The project also supports **monolithic** evaluation, where all answers are sent
in one context. This is mainly a comparison or thesis-ablation method; use
`--eval-mode modular` for the normal run.

## Before starting

The normal full run uses direct CUDA inference with the Gemma model. A machine
should have:

- Python 3.10 or newer
- NVIDIA CUDA with enough free VRAM for 4-bit Gemma inference (approximately
  18–20 GB)
- The project dependencies installed
- Question PDFs and student script PDFs available locally, or access to the
  configured Google Drive folders

From the repository root:

```bash
cd /mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI
source ./script_checking/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If the virtual environment does not exist, install the project and development
dependencies with:

```bash
python3 -m pip install -e ".[dev]"
```

## Full run: prepare the inputs

The script ID identifies both the student script and its question paper. For
example, `SE_11_Q1_0001` belongs to question paper `SE_11_Q1`.

### 1. Download or place question papers

Question PDFs belong under `data/questions/<language>/`. They can be downloaded
from the configured source:

```bash
python3 scripts/download_questions.py --lang english
```

Or place files such as `SE_11_Q1.pdf` in
`data/questions/english/` manually.

### 2. Extract question metadata

Create the structured question files used during evaluation:

```bash
python3 scripts/extract_questions.py --lang english
```

The results are written to `outputs/questions/english/`.

### 3. Download or place student scripts

Student script PDFs belong under `data/raw_pdfs/<language>/`. The downloader
keeps existing PDFs and only downloads missing files:

```bash
python3 scripts/download_drive_pdfs.py --lang english --top 2
```

Alternatively, place PDFs directly in a local paper directory, for example:

```text
data/raw_pdfs/english/se_11_q1/SE_11_Q1_0001.pdf
```

## Full run: extract, then evaluate

Run extraction first. This command deliberately does **not** use `--api` or
`--fast`; it runs the full direct-CUDA extraction, including the visual
verification pass.

```bash
python3 scripts/extract_scripts.py \
  --lang english \
  --pdf-dir data/raw_pdfs/english/se_11_q1 \
  --top 2 \
  --quant 4bit \
  -y
```

Extraction is checkpointed page by page. If it is interrupted, rerun the same
command to resume from completed checkpoints.

Then run the question-by-question rubric evaluation:

```bash
python3 scripts/evaluate_scripts.py \
  --lang english \
  --script-name SE_11_Q1_0001 \
  --quant 4bit \
  --eval-mode modular \
  -y
```

For several already-extracted scripts:

```bash
python3 scripts/evaluate_scripts.py \
  --lang english \
  --top 2 \
  --quant 4bit \
  --eval-mode modular \
  -y
```

Use `--force-evaluate` when a report already exists and must be regenerated.
Change `english` to `bangla` and use the corresponding input directories and
rubrics for Bangla scripts.

## Where to find the results

For a script such as `SE_11_Q1_0001`, extraction artifacts are stored under:

```text
outputs/extracted/english/se_11_q1/SE_11_Q1_0001/
├── checkpoints/                    # resumable page results
├── stage0b_teacher_marks.json     # detected teacher marks
├── stage1_raw_transcript.txt      # initial verbatim transcription
├── stage2_verified_transcript.txt # visually verified transcript
├── stage3_errors.json             # linguistic error catalog
├── extraction_result.json         # consolidated extraction data
└── extraction_summary.md          # readable extraction summary
```

Evaluation artifacts are stored separately:

```text
outputs/evaluated/english/se_11_q1/SE_11_Q1_0001/
├── stage4_evaluation.json         # marks and criterion breakdown
├── complete_report.json           # consolidated evaluation data
└── evaluation_report.md           # human-readable final report
```

The main file to share with an examiner or reviewer is
`evaluation_report.md`. If something looks incorrect, inspect
`extraction_summary.md` and `stage2_verified_transcript.txt` first to determine
whether the issue occurred during extraction or grading.

## Important distinction

The full run is intended for final extraction and evaluation. `--fast` skips
the visual verification pass, while `--api` uses an external local
OpenAI-compatible server instead of direct CUDA inference. Those modes are
useful for development or constrained machines, but they are not the standard
full-quality run described here.
