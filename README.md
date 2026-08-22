# Stage 1: English Handwritten Examination OCR + Confidence Subsystem

A research subsystem for high-fidelity transcription and confidence calibration of handwritten English examination answer scripts.

---

## 🏛️ System Architecture

```text
                  HANDWRITTEN ENGLISH EXAM IMAGE
                                │
                                ▼
                    Image Preprocessor (RGB/Resize)
                                │
                                ▼
                       SINGLE-PASS OCR BACKEND
                   (Infinite-OCR / TrOCR / EasyOCR)
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │                              │
                 ▼                              ▼
            Extracted Text              OCR Confidence Score
         (Raw & Normalized)              [0.0 to 1.0 Signal]
                 │                              │
                 └──────────────┬───────────────┘
                                ▼
                       STRUCTURED OCR RESULT
                       (Typed JSON Metadata)
                                │
                                ▼
                       CONFIDENCE ROUTING
                          /           \
                         /             \
        High Confidence (C >= τ)      Low Confidence (C < τ)
                   │                             │
                   ▼                             ▼
        Fast Text-Only Inference       Multimodal Image + OCR Text
         (Future Gemma 4 Stage)           (Future Gemma 4 Stage)
```

---

## 🎯 Scope & Boundaries for Stage 1

In accordance with [`English Handwritten Exam OCR + Confidence.md`](file:///e:/thesis/English%20Handwritten%20Exam%20OCR%20+%20Confidence.md):

- **English Handwriting Only**: Focused on handwritten English answer scripts.
- **Single OCR Pass**: Strictly eliminates multi-model cascades, OCR-to-OCR corrections, or LLM post-processing loops.
- **Uncompromised Raw Output**: Raw model predictions are preserved verbatim; safe normalization only handles Unicode NFC and whitespace standardization.
- **Measurable Confidence Calibration**: Focuses on whether OCR confidence predicts actual correctness (CER/WER against human ground truth).

---

## 🛠️ Hardware & Execution Support

- **Automatic Device Resolution**: GPU acceleration (`CUDA`, `bfloat16`/`float16`) on NVIDIA RTX 5090 / CUDA devices; graceful CPU fallback.
- **Stable Image Hashing**: SHA-256 image hashes embedded in all outputs to detect duplicates and prevent train/test contamination.

---

## 🚀 Quick Start & CLI Usage

### 1. Run Single-Image OCR
Process a single handwritten answer script image:

```powershell
python scripts/run_ocr.py --input data/samples/sample_exam_001.jpg --backend mock
```

**Example Terminal Output:**
```text
========================================
OCR RESULT
========================================
Script ID:       sample_exam_001
Backend:         mock (mock-deterministic-v1)
Language:        English
Confidence:      0.9400 (type: raw)
Processing time: 0.02 seconds
Device:          cpu
Extracted text:
----------------------------------------
The student answers that honesty is the best policy in human life.
----------------------------------------
Saved to:        outputs/ocr/sample_exam_001.json
```

### 2. Run Batch OCR Directory Processing
Process an entire directory of handwritten exam images:

```powershell
python scripts/run_ocr.py --input data/samples/ --output outputs/ocr/ --backend infinite_ocr
```

---

## 📊 Ground-Truth Benchmarking & Metrics

### 1. Benchmark OCR Accuracy (CER & WER)
Evaluate single-pass OCR against human ground-truth transcriptions:

```powershell
python scripts/benchmark_ocr.py --dataset data/ground_truth.jsonl --backend infinite_ocr
```

Generates:
- `outputs/reports/ocr_benchmark.csv`: Per-sample CER, WER, confidence, and binary acceptability.
- `outputs/reports/ocr_benchmark_summary.json`: Aggregate dataset statistics.

### 2. Analyze Confidence Calibration & Bins
Calculate Expected Calibration Error (ECE), Brier score, and confidence bins:

```powershell
python scripts/analyze_confidence.py --benchmark_csv outputs/reports/ocr_benchmark.csv
```

### 3. Evaluate Routing Thresholds (4 Quadrants)
Assess candidate confidence thresholds $\tau \in [0.50, 0.95]$ across the four routing quadrants:

```powershell
python scripts/threshold_analysis.py --benchmark_csv outputs/reports/ocr_benchmark.csv
```

**Four-Quadrant Routing Matrix:**

| Category | Definition | System Action |
|---|---|---|
| **Q1: True High** | $C \ge \tau \land \text{Acceptable}$ | Fast Text-Only Grading |
| **Q2: False High** | $C \ge \tau \land \text{Unacceptable}$ | **CRITICAL RISK** (Erroneous Text Grading) |
| **Q3: False Low** | $C < \tau \land \text{Acceptable}$ | Unnecessary Multimodal Inference |
| **Q4: True Low** | $C < \tau \land \text{Unacceptable}$ | **Safe Multimodal Visual Fallback** |

---

## ⚙️ Configuration Files

All parameters are configured via `configs/`:
- **`configs/ocr.yaml`**: Model backends (`infinite_ocr`, `trocr`, `easyocr`, `mock`), checkpoints, token limits, and device mapping.
- **`configs/preprocessing.yaml`**: Safe stroke-preserving image resizing and RGB standardization.
- **`configs/confidence.yaml`**: Aggregation algorithms (`length_weighted_mean`, `mean`, `minimum`, `geometric_mean`), calibration methods, and CER/WER acceptability thresholds.

---

## 🧪 Unit Test Suite

Run all automated unit tests:

```powershell
python -m unittest discover tests
```

Tests cover:
- Schema validation, serialization, and boundary conditions.
- Confidence aggregation algorithms and derived confidence estimators.
- Character Error Rate (CER) and Word Error Rate (WER) precision against ground truths.
- Isotonic regression and Platt scaling calibrators.
