#!/bin/bash

# ==============================================================================
# Script for Component-wise Performance Evaluation
# ==============================================================================
#
# This script runs the evaluation logic for each individual component of the
# framework to reproduce the component-specific results reported in the paper.
#
# It will execute:
# 1. Perception Module (LLaVA) Evaluation: (Placeholder - as this is often
#    integrated into the fine-tuning script's validation loop).
# 2. Knowledge Module (TransEx) Evaluation: Calculates MRR and Hits@K for
#    the link prediction task.
# 3. Reasoning Module (DRL) Evaluation: (Placeholder - DRL agent's primary
#    evaluation is its impact on the end-to-end task).
#
# Pre-requisites:
# - Stage 2 training must be complete for Knowledge Module evaluation.
#
# Usage:
# Run from the project root: > bash scripts/run_component_evaluation.sh
#
# ==============================================================================

# --- Configuration ---
set -e
set -u

# --- Announce Start ---
echo "============================================================"
echo "         RUNNING COMPONENT-WISE EVALUATION          "
echo "============================================================"
echo

# --- 1. Perception Module Evaluation ---
echo "--- [Component 1: Perception Module (LLaVA)] ---"
echo "INFO: The evaluation metrics for the Perception Module (PRA and EMA) are"
echo "      typically calculated on a validation set during the fine-tuning"
echo "      process. The final fine-tuned model's performance is reported"
echo "      in the training logs. Please refer to the output of"
echo "      'scripts/run_stage1_finetune.sh' for these results."
echo "      (Skipping separate execution for this component)."
echo "------------------------------------------------------------"
echo

# --- 2. Knowledge Module Evaluation ---
echo "--- [Component 2: Knowledge Module (TransEx)] ---"
echo "INFO: Evaluating the TransEx model on the link prediction task..."

# Variables
KG_EVAL_SCRIPT="src/knowledge_module/evaluate_link_prediction.py"
KG_EMBEDDINGS_DIR="trained_models/stage2_transex_embeddings"
KG_DATA_DIR="data/kg_embedding_dataset"

# Check if the evaluation script and trained embeddings exist
if [ ! -f "$KG_EVAL_SCRIPT" ]; then
    echo "ERROR: Knowledge module evaluation script not found at $KG_EVAL_SCRIPT"
    exit 1
fi
if [ ! -d "$KG_EMBEDDINGS_DIR" ] || [ -z "$(ls -A $KG_EMBEDDINGS_DIR)" ]; then
    echo "ERROR: Trained TransEx embeddings not found in $KG_EMBEDDINGS_DIR."
    echo "Please run Stage 2 training (run_stage2_train_kg.sh) first."
    exit 1
fi

# Execute the evaluation
python "$KG_EVAL_SCRIPT" \
    --embeddings_dir "$KG_EMBEDDINGS_DIR" \
    --data_dir "$KG_DATA_DIR"

echo "------------------------------------------------------------"
echo

# --- 3. Reasoning Module Evaluation ---
echo "--- [Component 3: Reasoning Module (DRL)] ---"
echo "INFO: The DRL agent's performance (KIA, Cumulative Reward) is evaluated"
echo "      during its training phase (see logs from run_stage3_train_drl.sh)."
echo "      Its primary contribution is measured through the end-to-end task"
echo "      performance (SeqAcc and KPA)."
echo "      (Skipping separate execution for this component)."
echo "------------------------------------------------------------"
echo

# --- Announce Completion ---
echo "============================================================"
echo "          COMPONENT-WISE EVALUATION COMPLETE          "
echo "============================================================"
echo