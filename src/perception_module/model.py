# -*- coding: utf-8 -*-

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def load_llava_model_for_finetuning(
    model_id: str = "llava-hf/llava-1.5-7b-hf",
    use_4bit_quantization: bool = True
) -> (LlavaForConditionalGeneration, AutoProcessor):
    """
    Loads the LLaVA model and its processor, preparing it for efficient fine-tuning.

    This function implements two key techniques for efficiency:
    1.  4-bit Quantization: Loads the model with 4-bit precision using bitsandbytes,
        significantly reducing GPU memory footprint.
    2.  LoRA (Low-Rank Adaptation): Applies LoRA adapters to the model's linear layers,
        making only a small fraction of parameters trainable.

    Args:
        model_id (str): The model identifier from the Hugging Face Hub.
        use_4bit_quantization (bool): Whether to use 4-bit quantization. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - model (peft.PeftModel): The LoRA-adapted, quantized model ready for training.
            - processor (transformers.AutoProcessor): The processor for data preprocessing.
    """
    print(f"Loading base model '{model_id}'...")

    # 1. Configure Quantization
    bnb_config = None
    if use_4bit_quantization:
        print("Using 4-bit quantization to reduce memory usage.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    # 2. Load the base LLaVA model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16, # Use float16 for memory efficiency
        device_map="auto",        # Automatically distribute model across available GPUs
        trust_remote_code=True
    )
    
    # 3. Load the processor (handles both text and images)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Set the pad token to be the same as the unknown token if it's not defined
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.unk_token
        model.config.pad_token_id = processor.tokenizer.unk_token_id

    # 4. Configure LoRA for PEFT
    # According to the LLaVA paper, applying LoRA to all linear layers is effective.
    lora_config = LoraConfig(
        r=16,  # Rank of the update matrices.
        lora_alpha=32,  # Alpha parameter for scaling.
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Target modules can be specific, but often targeting all linear layers is a good start.
        # This can be fine-tuned for better performance.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head"
        ]
    )

    # 5. Apply LoRA adapters to the model
    # model.config modifications are often needed before applying PEFT
    model.config.use_cache = False  # Caching is not needed during training
    
    peft_model = get_peft_model(model, lora_config)

    print("\nModel preparation complete. LoRA adapters have been applied.")
    peft_model.print_trainable_parameters()
    
    return peft_model, processor

if __name__ == '__main__':
    # This block allows you to run the script directly for testing purposes
    # It will download the model and print the number of trainable parameters.
    print("--- Testing model loading script ---")
    model, processor = load_llava_model_for_finetuning()
    print("\nModel and processor loaded successfully.")