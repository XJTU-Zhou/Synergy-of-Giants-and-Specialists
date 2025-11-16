#!/bin/bash

# ==============================================================================
# Script for Stage 2: Training the Knowledge Representation Module (TransEx)
# ==============================================================================
#
# This script trains the TransEx Knowledge Graph Embedding model. It learns vector
# representations for all entities and relations in the manufacturing KG.
#
# Pre-requisites:
# 1. A properly configured Conda environment.
# 2. The KG dataset must be prepared in 'data/kg_embedding_dataset/'.
#
# Usage:
# Run this script from the project's root directory:
# > bash scripts/run_stage2_train_kg.sh
#
# ==============================================================================

# --- Configuration ---
set -e
set -u

# --- Announce Stage ---
echo "================================================================"
echo "  STARTING STAGE 2: KNOWLEDGE MODULE (TransEx) TRAINING  "
echo "================================================================"

# --- Variables ---
PYTHON_SCRIPT="src/knowledge_module/train.py"
CONFIG_FILE="configs/stage2_transex_config.yaml"
OUTPUT_DIR="trained_models/stage2_transex_embeddings"

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

echo "INFO: Ensuring output directory exists at: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# --- Execute Training ---
echo "INFO: Starting the KG embedding training process..."
echo " - Python Script: $PYTHON_SCRIPT"
echo " - Config File:   $CONFIG_FILE"
echo " - Output Dir:    $OUTPUT_DIR"
echo "------------------------------------------------------------"

python "$PYTHON_SCRIPT" --config "$CONFIG_FILE" --output_dir "$OUTPUT_DIR"

# --- Announce Completion ---
echo "------------------------------------------------------------"
echo "SUCCESS: Stage 2 Training complete."
echo "Entity and relation embeddings have been saved to: $OUTPUT_DIR"
echo "================================================================"
echo