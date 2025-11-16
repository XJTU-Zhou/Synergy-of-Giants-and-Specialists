#!/bin/bash

# ==============================================================================
# Script for Running Ablation Studies
# ==============================================================================
#
# This script automates the execution of the ablation studies described in
# Table 9 of the paper. It calls the main end-to-end evaluation script
# with different flags to simulate the removal or replacement of core
# components.
#
# Pre-requisites:
# - All training stages must be complete.
# - The end-to-end evaluation script must be functional.
#
# Usage:
# Run from the project root: > bash scripts/run_ablation_studies.sh
#
# ==============================================================================

# --- Configuration ---
set -e
set -u

EVALUATION_SCRIPT="scripts/run_end_to_end_evaluation.py"

# --- Announce Start ---
echo "============================================================"
echo "             STARTING ABLATION STUDIES              "
echo "============================================================"
echo

# --- Pre-run Check ---
if [ ! -f "$EVALUATION_SCRIPT" ]; then
    echo "ERROR: The main evaluation script was not found at: $EVALUATION_SCRIPT"
    exit 1
fi

# --- 1. Run Full Framework (Baseline) ---
echo "--- [Baseline: Our Full Framework] ---"
python "$EVALUATION_SCRIPT"
echo "------------------------------------------------------------"
echo

# --- 2. Variant A: w/o VLM Fine-tuning ---
echo "--- [Variant A: Using generic LLaVA-7B] ---"
echo "INFO: Running evaluation with the '--use_generic_vlm' flag."
python "$EVALUATION_SCRIPT" --use_generic_vlm
echo "------------------------------------------------------------"
echo

# --- 3. Variant B: Replacing TransEx (Using ComplEx) ---
echo "--- [Variant B: Replacing TransEx with ComplEx] ---"
echo "INFO: Running evaluation with the '--reasoning_variant complex' flag."
# The python script needs to be implemented to handle this variant
python "$EVALUATION_SCRIPT" --reasoning_variant complex
echo "------------------------------------------------------------"
echo

# --- 4. Variant C: Replacing DRL (Using Beam Search) ---
echo "--- [Variant C: Replacing DRL with Beam Search] ---"
echo "INFO: Running evaluation with the '--reasoning_variant beam_search' flag."
# Note: KPA is expected to be N/A for this variant.
python "$EVALUATION_SCRIPT" --reasoning_variant beam_search
echo "------------------------------------------------------------"
echo

# --- 5. Variant D: Replacing DRL (Using A* Search) ---
echo "--- [Variant D: Replacing DRL with A* Search] ---"
echo "INFO: Running evaluation with the '--reasoning_variant a_star' flag."
# Note: KPA is expected to be N/A for this variant.
python "$EVALUATION_SCRIPT" --reasoning_variant a_star
echo "------------------------------------------------------------"
echo

# --- Announce Completion ---
echo "============================================================"
echo "               ABLATION STUDIES COMPLETE              "
echo "============================================================"
echo