#!/bin/bash

# ==============================================================================
# Script for Stage 3: Training the Reasoning Module (DRL Agent)
# ==============================================================================
#
# This script trains the GAE-PPO based Deep Reinforcement Learning agent for
# interpretable pathfinding on the knowledge graph.
#
# Pre-requisites:
# 1. A properly configured Conda environment.
# 2. Stage 2 must be completed, as this script requires the pre-trained KG
#    embeddings from 'trained_models/stage2_transex_embeddings/'.
#
# Usage:
# Run this script from the project's root directory:
# > bash scripts/run_stage3_train_drl.sh
#
# ==============================================================================

# --- Configuration ---
set -e
set -u

# --- Announce Stage ---
echo "================================================================"
echo "  STARTING STAGE 3: REASONING MODULE (DRL AGENT) TRAINING  "
echo "================================================================"

# --- Variables ---
PYTHON_SCRIPT="src/reasoning_module/train.py"
CONFIG_FILE="configs/stage3_drl_config.yaml"
# This stage DEPENDS on the output of Stage 2
KG_EMBEDDING_PATH="trained_models/stage2_transex_embeddings"
OUTPUT_DIR="trained_models/stage3_drl_agent"

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

# Crucial check: ensure that Stage 2 has been run and its output exists
if [ ! -d "$KG_EMBEDDING_PATH" ] || [ -z "$(ls -A $KG_EMBEDDING_PATH)" ]; then
    echo "ERROR: The KG embeddings directory is missing or empty at: $KG_EMBEDDING_PATH"
    echo "Please run Stage 2 (scripts/run_stage2_train_kg.sh) successfully before this script."
    exit 1
fi
echo "INFO: Found required KG embeddings from Stage 2."

echo "INFO: Ensuring output directory exists at: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# --- Execute Training ---
echo "INFO: Starting the DRL agent training process..."
echo " - Python Script:    $PYTHON_SCRIPT"
echo " - Config File:      $CONFIG_FILE"
echo " - Embedding Path:   $KG_EMBEDDING_PATH"
echo " - Output Dir:       $OUTPUT_DIR"
echo "------------------------------------------------------------"

python "$PYTHON_SCRIPT" \
    --config "$CONFIG_FILE" \
    --embedding_path "$KG_EMBEDDING_PATH" \
    --output_dir "$OUTPUT_DIR"

# --- Announce Completion ---
echo "------------------------------------------------------------"
echo "SUCCESS: Stage 3 Training complete."
echo "Trained DRL agent (actor and critic networks) has been saved to: $OUTPUT_DIR"
echo "================================================================"
echo