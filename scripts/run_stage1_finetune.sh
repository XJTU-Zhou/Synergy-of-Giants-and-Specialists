#!/bin/bash

# ==============================================================================
# Script for Stage 1: Fine-tuning the Perception Module (LLaVA)
# ==============================================================================
#
# This script initiates the domain-adaptive fine-tuning process for the LLaVA model.
# It uses the configuration file to load hyperparameters and the dataset specified
# in 'data/vlm_finetuning_dataset/'.
#
# Pre-requisites:
# 1. A properly configured Conda environment.
# 2. The dataset must be prepared in 'data/vlm_finetuning_dataset/'.
# 3. A GPU with sufficient VRAM (e.g., >= 48GB) is required.
#
# Usage:
# Run this script from the project's root directory:
# > bash scripts/run_stage1_finetune.sh
#
# ==============================================================================

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u

# --- Announce Stage ---
echo "============================================================"
echo "  STARTING STAGE 1: PERCEPTION MODULE (LLaVA) FINE-TUNING  "
echo "============================================================"

# --- Variables ---
# Path to the Python script that runs the fine-tuning logic
PYTHON_SCRIPT="src/perception_module/finetune.py"
# Path to the configuration file for this stage
CONFIG_FILE="configs/stage1_llava_config.yaml"
# Directory to save the fine-tuned model weights and training logs
OUTPUT_DIR="trained_models/stage1_llava_finetuned"

# --- Pre-run Checks ---
echo "INFO: Checking for necessary files and directories..."
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: The training script was not found at: $PYTHON_SCRIPT"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: The configuration file was not found at: $CONFIG_FILE"
    exit 1
fi

# Create the output directory if it doesn't exist
echo "INFO: Ensuring output directory exists at: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# --- Execute Training ---
echo "INFO: Starting the fine-tuning process..."
echo " - Python Script: $PYTHON_SCRIPT"
echo " - Config File:   $CONFIG_FILE"
echo " - Output Dir:    $OUTPUT_DIR"
echo "------------------------------------------------------------"

# The 'accelerate launch' command is often used for multi-GPU or advanced training setups.
# For simplicity in this script, we call the Python script directly. The logic to handle
# device placement should be within the Python script itself.
python "$PYTHON_SCRIPT" --config "$CONFIG_FILE" --output_dir "$OUTPUT_DIR"

# --- Announce Completion ---
echo "------------------------------------------------------------"
echo "SUCCESS: Stage 1 Fine-tuning complete."
echo "Fine-tuned model and logs have been saved to: $OUTPUT_DIR"
echo "============================================================"
echo