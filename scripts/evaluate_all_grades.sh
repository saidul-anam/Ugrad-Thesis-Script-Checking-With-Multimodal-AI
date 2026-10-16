#!/bin/bash
# ==============================================================================
# Sequential Batch Evaluation for All Classes: 6, 7, 8, and 10
# Executes Stage 4 modular evaluation with updated rubrics and fixed segmenter
# ==============================================================================

set -e

REPO_DIR="/mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI"
cd "$REPO_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source script_checking/bin/activate

PYTHON_BIN="./script_checking/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

echo "=================================================================="
echo "🚀 Starting Full Sequential Evaluation Run"
echo "Start Time: $(date)"
echo "Host: $(hostname)"
echo "Python: $($PYTHON_BIN --version)"
echo "=================================================================="

# --- 1. Class 6 (SE_06_Q1) ---
echo ""
echo ">>> [1/4] Starting Class 6 Evaluation (SE_06_Q1)..."
echo "Timestamp: $(date)"
$PYTHON_BIN -u scripts/evaluate_scripts.py \
    --extraction-dir data_without_pipeline/english/se_06_q1 \
    --paper SE_06_Q1 \
    --non-interactive
echo ">>> Class 6 Completed at $(date)"

# --- 2. Class 7 (SE_07_Q1) ---
echo ""
echo ">>> [2/4] Starting Class 7 Evaluation (SE_07_Q1)..."
echo "Timestamp: $(date)"
$PYTHON_BIN -u scripts/evaluate_scripts.py \
    --extraction-dir data_without_pipeline/english/se_07_q1 \
    --paper SE_07_Q1 \
    --non-interactive
echo ">>> Class 7 Completed at $(date)"

# --- 3. Class 8 (SE_08_Q1) ---
echo ""
echo ">>> [3/4] Starting Class 8 Evaluation (SE_08_Q1)..."
echo "Timestamp: $(date)"
$PYTHON_BIN -u scripts/evaluate_scripts.py \
    --extraction-dir data_without_pipeline/english/se_08_q1 \
    --paper SE_08_Q1 \
    --non-interactive
echo ">>> Class 8 Completed at $(date)"

# --- 4. Class 10 (SE_10_Q1) ---
echo ""
echo ">>> [4/4] Starting Class 10 Evaluation (SE_10_Q1)..."
echo "Timestamp: $(date)"
$PYTHON_BIN -u scripts/evaluate_scripts.py \
    --extraction-dir data_without_pipeline/english/se_10_q1 \
    --paper SE_10_Q1 \
    --non-interactive
echo ">>> Class 10 Completed at $(date)"

echo ""
echo "=================================================================="
echo "🎉 ALL EVALUATIONS COMPLETED SUCCESSFULLY (Classes 6, 7, 8, 10)"
echo "End Time: $(date)"
echo "=================================================================="
