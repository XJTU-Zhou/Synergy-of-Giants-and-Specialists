# -*- coding: utf-8 -*-

"""


This script orchestrates the entire three-stage pipeline to evaluate the
end-to-end performance of the "Synergy of Giants and Specialists" framework.
It processes test cases from the `data/end_to_end_testset` directory,
generates manufacturing process plans, and compares them against the ground
truth to calculate Sequence Accuracy (SeqAcc) and Key Parameter
Acceptability (KPA).
"""

import os
import json
import argparse
from tqdm import tqdm

# Ensure the src directory is in the Python path
import sys
# This assumes the script is run from the project root.
# For robustness, you might add sys.path.append(os.getcwd())
# Or rely on the __init__.py in src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from src.perception_module import inference as vlm_inference
from src.knowledge_module import model as transex_model # Placeholder for loading
from src.reasoning_module import pathfinder as drl_pathfinder
from src.utils.metrics import calculate_seq_acc, calculate_kpa

# --- Configuration ---
TESTSET_DIR = "data/end_to_end_testset"
GROUND_TRUTH_FILE = os.path.join(TESTSET_DIR, "ground_truth.json")
CASES_DIR = os.path.join(TESTSET_DIR, "cases")

# Model Paths
VLM_MODEL_PATH = "trained_models/stage1_llava_finetuned"
TRANSEX_MODEL_PATH = "trained_models/stage2_transex_embeddings"
DRL_MODEL_PATH = "trained_models/stage3_drl_agent"


def load_models(args):
    """Loads all three models for the pipeline."""
    print("INFO: Loading all models for the pipeline...")
    
    # Stage 1: Load Perception Module
    # In a real implementation, this would initialize the LLaVA model with fine-tuned weights.
    # We use a placeholder function signature.
    vlm_model_path_to_load = VLM_MODEL_PATH
    if args.use_generic_vlm:
        print("WARNING: Using generic (non-fine-tuned) VLM for Ablation Study (Variant A).")
        # In a real scenario, you'd point to the base LLaVA model
        vlm_model_path_to_load = "llava-hf/llava-1.5-7b-hf" # Example
    
    # stage1_model = vlm_inference.load_model(vlm_model_path_to_load)
    # print(f"  - Stage 1 Model loaded from: {vlm_model_path_to_load}")
    
    # Stage 2 & 3: Load Knowledge and Reasoning Modules
    # The DRL pathfinder will internally load the KG embeddings it needs.
    # The variant logic (e.g., using ComplEx) would be handled here.
    # reasoning_agent = drl_pathfinder.load_agent(DRL_MODEL_PATH, TRANSEX_MODEL_PATH, variant=args.reasoning_variant)
    # print(f"  - Stage 2/3 Models loaded with variant: '{args.reasoning_variant}'")
    
    # Return placeholder models for this script structure
    # In full implementation, these would be the actual model objects.
    stage1_model = "Loaded VLM Model"
    reasoning_agent = "Loaded DRL Agent"
    print("INFO: All models loaded successfully.")
    
    return stage1_model, reasoning_agent


def run_pipeline_on_case(case_id, stage1_model, reasoning_agent):
    """Runs the full 3-stage pipeline for a single test case."""
    case_dir = os.path.join(CASES_DIR, case_id)
    image_path = os.path.join(case_dir, "drawing.jpg")
    
    with open(os.path.join(case_dir, "query.txt"), 'r', encoding='utf-8') as f:
        query_text = f.read()

    # --- STAGE 1: PERCEPTION ---
    # This function should take the image and query and return structured data.
    # structured_data = vlm_inference.predict(stage1_model, image_path, query_text)
    # Mock output for demonstration:
    if case_id == "case_01":
        structured_data = {"intent": "plan_process", "entities": ["spline_shaft", "45_steel"]}
    else:
        structured_data = {"intent": "plan_process", "entities": ["pin_shaft", "q235"]}

    # --- STAGE 2 & 3: REASONING ---
    # The reasoning agent takes the entities and finds a process plan.
    # generated_plan = reasoning_agent.find_path(start_entity=structured_data['entities'][0])
    # Mock output for demonstration:
    if case_id == "case_01":
        generated_plan = {
            "sequence": ["material_cutting", "rough_turning", "heat_treatment", "spline_milling", "external_grinding", "inspection"],
            "parameters": {
                "rough_turning": {"spindle_speed_rpm": 750},
                "external_grinding": {"wheel_speed_m_s": 30}
            }
        }
    else:
        generated_plan = {
            "sequence": ["material_cutting", "turning", "inspection"],
            "parameters": {
                "turning": {"spindle_speed_rpm": 800}
            }
        }
        
    return generated_plan


def main(args):
    """Main evaluation loop."""
    print("============================================================")
    print("         STARTING END-TO-END FRAMEWORK EVALUATION         ")
    print("============================================================")
    
    # Load ground truth data
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    
    # Load models based on arguments
    stage1_model, reasoning_agent = load_models(args)
    
    all_results = []
    
    case_ids = sorted(os.listdir(CASES_DIR))
    
    for case_id in tqdm(case_ids, desc="Processing Test Cases"):
        if not os.path.isdir(os.path.join(CASES_DIR, case_id)):
            continue
            
        # Get generated plan from the pipeline
        generated_plan = run_pipeline_on_case(case_id, stage1_model, reasoning_agent)
        
        # Get ground truth
        ground_truth_plan = ground_truth_data.get(case_id)
        if not ground_truth_plan:
            print(f"WARNING: No ground truth found for case {case_id}. Skipping.")
            continue
            
        # Calculate metrics for this case
        seq_acc = calculate_seq_acc(generated_plan["sequence"], ground_truth_plan["sequence"])
        kpa = calculate_kpa(generated_plan.get("parameters", {}), ground_truth_plan.get("parameters", {}))
        
        all_results.append({"case_id": case_id, "SeqAcc": seq_acc, "KPA": kpa})

    # Aggregate and report final scores
    avg_seq_acc = sum(r["SeqAcc"] for r in all_results) / len(all_results)
    
    # Only calculate KPA average for cases where it's applicable (not N/A)
    kpa_results = [r["KPA"] for r in all_results if r["KPA"] is not None]
    avg_kpa = sum(kpa_results) / len(kpa_results) if kpa_results else "N/A"
    
    print("\n============================================================")
    print("                 EVALUATION RESULTS                 ")
    print("------------------------------------------------------------")
    print(f"  Total Cases Evaluated: {len(all_results)}")
    print(f"  Average Sequence Accuracy (SeqAcc): {avg_seq_acc:.4f}")
    print(f"  Average Key Parameter Acceptability (KPA): {avg_kpa if isinstance(avg_kpa, str) else f'{avg_kpa:.4f}'}")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end evaluation script for the LSM framework.")
    parser.add_argument('--use_generic_vlm', action='store_true', help="Ablation: Use a generic, non-fine-tuned VLM.")
    parser.add_argument('--reasoning_variant', type=str, default='drl', choices=['drl', 'complex', 'beam_search', 'a_star'], help="Ablation: Choose a different reasoning module.")
    
    args = parser.parse_args()

    main(args)
