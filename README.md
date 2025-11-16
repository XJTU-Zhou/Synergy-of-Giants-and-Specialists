# Synergy of Giants and Specialists: **: A Large-and-Small Model Integration Framework Driven by Generative AI for Smart Manufacturing**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

## Framework Overview

The core idea of this research is to combine the perceptual capabilities of "Giants" (Large Foundation Models) with the precise reasoning abilities of "Specialists" (Small, Domain-Specific Models) to solve complex decision-making problems in smart manufacturing. The framework consists of three synergistic stages:

1. **Stage 1: Multimodal Perception & Translation**

   * **Model**: A domain-adapted **LLaVA** model.
   * **Task**: To parse unstructured inputs (e.g., engineering drawings and natural language) into structured entities and intents.
2. **Stage 2: Knowledge Representation**

   * **Model**: A Transformer-based Knowledge Graph (KG) embedding model, **TransEx**.
   * **Task**: To encode a discrete manufacturing KG into a rich, continuous semantic space (the "map").
3. **Stage 3: Interpretable Reasoning**

   * **Model**: A Deep Reinforcement Learning (DRL) agent based on the **GAE-PPO** strategy.
   * **Task**: To autonomously navigate the knowledge map and generate a transparent, traceable process plan.

## Environment Setup

**Hardware Requirements**:

* **Perception Module Training**: A high-VRAM NVIDIA GPU (>= 48GB VRAM, e.g., A800, A100) is recommended for fine-tuning the LLaVA model.
* **Other Modules**: An NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090/4090) is recommended.

**Software Requirements**:

* OS: Linux (Recommended) or Windows
* CUDA Version: 11.8 or higher
* Python Version: 3.12
* Conda

Please follow these steps to set up your environment:

1. **Create and activate the Conda environment**

   ```sh
   conda create -n lsm_smart_mfg python=3.12 -y
   conda activate lsm_smart_mfg
   ```
2. **Install PyTorch** (Please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command corresponding to your CUDA version)

   ```sh
   # Example for CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Install all other dependencies**

   ```sh
   pip install -r requirements.txt
   ```

## Data Preparation

1. **VLM Fine-tuning Dataset (`data/vlm_finetuning_dataset/`)**

   * `images/`: Contains all engineering drawings in `.jpg` or `.png` format.
   * `annotations.jsonl`: A JSON Lines file where each line is an image-text pair.
   * **Format Example**:
     ```json
     {"image_file": "gear_shaft_01.jpg", "conversations": [{"from": "human", "value": "Plan the machining process for the part in this drawing."}, {"from": "gpt", "value": "Intent: Develop machining plan. Entities: [part_name: gear shaft, material: 45 steel, process: rough turning, finish turning]"}]}
     ```
2. **KG Embedding Dataset (`data/kg_embedding_dataset/`)**

   * `train.txt`, `valid.txt`, `test.txt`: Files containing KG triples.
   * **Format Example** (head, relation, tail, separated by `\t`):
     ```
     gear_shaft	hasMaterial	45_steel
     rough_turning	precedes	finish_turning
     cnc_lathe	capableOf	rough_turning
     ```
3. **Evaluation Testset (`data/end_to_end_testset/`)**

   * `cases/`: Contains 50 subdirectories, each representing a test case.
     * `case_01/drawing.jpg`: Engineering drawing for the case.
     * `case_01/query.txt`: Natural language query for the case.
   * `ground_truth.json`: Contains the ground truth answers for all test cases.
   * **Format Example**:
     ```json
     {
       "case_01": {
         "sequence": ["forging", "rough_turning", "heat_treatment", "finish_turning"],
         "parameters": {
           "rough_turning": {
             "spindle_speed_rpm": [400, 800],
             "feed_rate_mm_rev": [0.2, 0.4]
           },
           "finish_turning": {
             "spindle_speed_rpm": [1000, 1500],
             "feed_rate_mm_rev": [0.05, 0.15]
           }
         }
       }
     }
     ```

## Workflow

We provide convenient shell scripts to execute the entire training and evaluation pipeline.

### Step 1: Train the Perception Module (LLaVA)

This script fine-tunes the LLaVA model on the domain-specific data from `data/vlm_finetuning_dataset/`.

```sh
bash scripts/run_stage1_finetune.sh
```

* **Input**: The configuration file `configs/stage1_llava_config.yaml` and the VLM dataset.
* **Output**: Fine-tuned model weights will be saved to `trained_models/stage1_llava_finetuned/`.

### Step 2: Train the Knowledge Representation Module (TransEx)

This script trains the TransEx model using the `data/kg_embedding_dataset/` to generate KG embeddings.

```sh
bash scripts/run_stage2_train_kg.sh
```

* **Input**: The configuration file `configs/stage2_transex_config.yaml` and the KG dataset.
* **Output**: Trained entity and relation embeddings will be saved to `trained_models/stage2_transex_embeddings/`.

### Step 3: Train the Reasoning Module (DRL Agent)

This script loads the KG embeddings from Step 2 and trains the GAE-PPO agent.

```sh
bash scripts/run_stage3_train_drl.sh
```

* **Input**: The configuration file `configs/stage3_drl_config.yaml` and the KG embeddings.
* **Output**: Trained DRL agent weights (Actor & Critic networks) will be saved to `trained_models/stage3_drl_agent/`.

### Step 4: Performance Evaluation

```sh
python scripts/run_end_to_end_evaluation.py
```

* **Process**:
  1. Loads all trained models from `trained_models/`.
  2. Iterates through each test case in `data/end_to_end_testset/cases/`.
  3. Sequentially invokes the three modules to generate a process plan.
  4. Compares the generated output against the ground truth in `data/end_to_end_testset/ground_truth.json`.
  5. Calculates and prints the final **SeqAcc** and **KPA** scores.

## Citation

If you use the code or ideas from this work in your research, please cite our paper:

```bibtex
@article{Xu2025Synergy,   title={Synergy of Giants and Specialists: A Large-and-Small Model Integration Framework Driven by Generative AI for Smart Manufacturing},   author={Qingfeng Xu and Chao Zhang and Dongxu Ma and Yan Cao and Guanghui Zhou}}
```

## License

This project is licensed under the MIT License. See the \`LICENSE\` file for details.
