#!/usr/bin/env bash
# ==============================================================================
# Force Re-Evaluation Runner for All Exam Scripts
# ==============================================================================
# Overwrites existing evaluation reports by passing `--force-evaluate`
# Applies updated segmenter rules, safe boundary transitions, and continuation stitching.
#
# Usage:
#   1. Force evaluate EVERYTHING (Classes 6, 7, 8, 10):
#        ./scripts/force_evaluate_all.sh
#      or in background:
#        nohup ./scripts/force_evaluate_all.sh > logs/force_eval_all.log 2>&1 &
#
#   2. Force evaluate a single class:
#        ./scripts/force_evaluate_all.sh --cohort se_07_q1
#
#   3. Force evaluate ONLY scripts diagnosed with segmentation issues:
#        ./scripts/force_evaluate_all.sh --issues-only
#
#   4. Force evaluate a single script:
#        ./scripts/force_evaluate_all.sh --script SE_07_Q1_0002 --paper SE_07_Q1
# ==============================================================================

set -eo pipefail

REPO_DIR="/mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI"
cd "$REPO_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure logs directory exists
mkdir -p logs

PYTHON_BIN="./script_checking/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEFAULT_LOG="logs/force_eval_${TIMESTAMP}.log"

COHORT=""
SINGLE_SCRIPT=""
PAPER_ID=""
ISSUES_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cohort|-c)
            COHORT=$(echo "$2" | tr '[:upper:]' '[:lower:]')
            shift 2
            ;;
        --script|-s)
            SINGLE_SCRIPT="$2"
            shift 2
            ;;
        --paper|-p)
            PAPER_ID=$(echo "$2" | tr '[:lower:]' '[:upper:]')
            shift 2
            ;;
        --issues-only|-i)
            ISSUES_ONLY=true
            shift 1
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cohort <name>     Evaluate specific cohort (se_06_q1, se_07_q1, se_08_q1, se_10_q1)"
            echo "  --issues-only       Evaluate only scripts diagnosed with segmentation/dropped-page issues"
            echo "  --script <id>       Evaluate a single script ID (e.g., SE_07_Q1_0002)"
            echo "  --paper <id>        Exam paper ID (e.g., SE_07_Q1, needed if running single script)"
            echo "  --help              Display this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage instructions."
            exit 1
            ;;
    esac
done

echo "=================================================================="
echo "🚀 Force Re-Evaluation Pipeline Initialized"
echo "Start Time: $(date)"
echo "Host:       $(hostname)"
echo "Python:     $($PYTHON_BIN --version 2>&1)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CUDA N/A')"
echo "=================================================================="

# Mode 1: Single Script
if [ -n "$SINGLE_SCRIPT" ]; then
    if [ -z "$PAPER_ID" ]; then
        # Infer paper ID from script ID prefix
        case "$SINGLE_SCRIPT" in
            SE_06_Q1*|se_06_q1*) PAPER_ID="SE_06_Q1" ;;
            SE_07_Q1*|se_07_q1*) PAPER_ID="SE_07_Q1" ;;
            SE_08_Q1*|se_08_q1*) PAPER_ID="SE_08_Q1" ;;
            SE_10_Q1*|se_10_q1*) PAPER_ID="SE_10_Q1" ;;
            *)
                echo "Error: Could not infer paper ID. Please specify --paper <ID>."
                exit 1
                ;;
        esac
    fi
    EXT_DIR="data_without_pipeline/english/$(echo "$PAPER_ID" | tr '[:upper:]' '[:lower:]')"
    echo ">>> Running single script: $SINGLE_SCRIPT (Paper: $PAPER_ID)"
    $PYTHON_BIN -u scripts/evaluate_scripts.py \
        --extraction-dir "$EXT_DIR" \
        --paper "$PAPER_ID" \
        --script-name "$SINGLE_SCRIPT" \
        --force-evaluate \
        --non-interactive
    echo ">>> Completed $SINGLE_SCRIPT at $(date)"
    exit 0
fi

# Mode 2: Issues-Only (Re-evaluates diagnosed problematic scripts)
if [ "$ISSUES_ONLY" = true ]; then
    echo ">>> Running in ISSUES-ONLY mode for scripts with segmentation/dropped-page issues..."

    SE_08_ISSUES=("SE_08_Q1_0010" "SE_08_Q1_0012" "SE_08_Q1_0013" "SE_08_Q1_0022" "SE_08_Q1_0027" "SE_08_Q1_0033")
    SE_10_ISSUES=("SE_10_Q1_0012" "SE_10_Q1_0013")

    echo ">>> [Class 6 Issues] Evaluating: ${SE_06_ISSUES[*]}"
    for s in "${SE_06_ISSUES[@]}"; do
        $PYTHON_BIN -u scripts/evaluate_scripts.py \
            --extraction-dir "data_without_pipeline/english/se_06_q1" \
            --paper "SE_06_Q1" \
            --script-name "$s" \
            --force-evaluate \
            --non-interactive
    done

    echo ">>> [Class 7 Issues] Evaluating: ${SE_07_ISSUES[*]}"
    for s in "${SE_07_ISSUES[@]}"; do
        $PYTHON_BIN -u scripts/evaluate_scripts.py \
            --extraction-dir "data_without_pipeline/english/se_07_q1" \
            --paper "SE_07_Q1" \
            --script-name "$s" \
            --force-evaluate \
            --non-interactive
    done

    echo ">>> [Class 8 Issues] Evaluating: ${SE_08_ISSUES[@]}"
    for s in "${SE_08_ISSUES[@]}"; do
        $PYTHON_BIN -u scripts/evaluate_scripts.py \
            --extraction-dir "data_without_pipeline/english/se_08_q1" \
            --paper "SE_08_Q1" \
            --script-name "$s" \
            --force-evaluate \
            --non-interactive
    done

    echo ">>> [Class 10 Issues] Evaluating: ${SE_10_ISSUES[@]}"
    for s in "${SE_10_ISSUES[@]}"; do
        $PYTHON_BIN -u scripts/evaluate_scripts.py \
            --extraction-dir "data_without_pipeline/english/se_10_q1" \
            --paper "SE_10_Q1" \
            --script-name "$s" \
            --force-evaluate \
            --non-interactive
    done

    echo ""
    echo "=================================================================="
    echo "🎉 All Diagnosed Problematic Scripts Re-Evaluated Successfully!"
    echo "End Time: $(date)"
    echo "=================================================================="
    exit 0
fi

# Mode 3: Specific Cohort
if [ -n "$COHORT" ]; then
    PAPER_NAME=$(echo "$COHORT" | tr '[:lower:]' '[:upper:]')
    echo ">>> Running single cohort: $PAPER_NAME ($COHORT)..."
    $PYTHON_BIN -u scripts/evaluate_scripts.py \
        --extraction-dir "data_without_pipeline/english/$COHORT" \
        --paper "$PAPER_NAME" \
        --force-evaluate \
        --non-interactive
    echo ">>> Completed cohort $PAPER_NAME at $(date)"
    exit 0
fi

# Mode 4: Full Sequential Re-Evaluation (All 4 Classes)
echo ""
echo ">>> [1/4] Force Re-Evaluating Class 6 (SE_06_Q1 - 22 scripts)..."
echo "Timestamp: $(date)"
$PYTHON_BIN -u scripts/evaluate_scripts.py \
    --extraction-dir data_without_pipeline/english/se_06_q1 \
    --paper SE_06_Q1 \
    --force-evaluate \
    --non-interactive

echo ""
echo ">>> [2/4] Force Re-Evaluating Class 7 (SE_07_Q1 - 34 scripts)..."
echo "Timestamp: $(date)"
$PYTHON_BIN -u scripts/evaluate_scripts.py \
    --extraction-dir data_without_pipeline/english/se_07_q1 \
    --paper SE_07_Q1 \
    --force-evaluate \
    --non-interactive

echo ""
echo ">>> [3/4] Force Re-Evaluating Class 8 (SE_08_Q1 - 35 scripts)..."
echo "Timestamp: $(date)"
$PYTHON_BIN -u scripts/evaluate_scripts.py \
    --extraction-dir data_without_pipeline/english/se_08_q1 \
    --paper SE_08_Q1 \
    --force-evaluate \
    --non-interactive

echo ""
echo ">>> [4/4] Force Re-Evaluating Class 10 (SE_10_Q1 - 33 scripts)..."
echo "Timestamp: $(date)"
$PYTHON_BIN -u scripts/evaluate_scripts.py \
    --extraction-dir data_without_pipeline/english/se_10_q1 \
    --paper SE_10_Q1 \
    --force-evaluate \
    --non-interactive

echo ""
echo "=================================================================="
echo "🎉 ALL 124 EXAM SCRIPTS FORCE RE-EVALUATED SUCCESSFULLY!"
echo "End Time: $(date)"
echo "=================================================================="
