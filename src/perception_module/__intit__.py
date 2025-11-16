# -*- coding: utf-8 -*-

"""
Perception Module for the Synergy of Giants and Specialists Framework.

This module is responsible for Stage 1 of the pipeline: Multimodal Perception and Translation.
It uses a fine-tuned Vision-Language Model (VLM), specifically LLaVA, to process
unstructured inputs like engineering drawings and natural language queries, and convert
them into a structured, machine-readable format.

Key components:
- dataset.py: A custom PyTorch Dataset for loading the multimodal manufacturing data.
- model.py: A utility function to load the LLaVA model with PEFT (LoRA) and quantization for efficient fine-tuning.
- finetune.py: The main script to run the fine-tuning process.
- inference.py: A script to use the fine-tuned model for prediction.
"""

# Provide convenient imports for users of this package
from .dataset import CustomVLMDataset
from .model import load_llava_model_for_finetuning

__all__ = [
    "CustomVLMDataset",
    "load_llava_model_for_finetuning",
]