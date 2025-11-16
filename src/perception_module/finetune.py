# -*- coding: utf-8 -*-

import os
import yaml
from transformers import Trainer, TrainingArguments
from src.perception_module.dataset import CustomVLMDataset
from src.perception_module.model import load_llava_model_for_finetuning

def finetune_llava(config_path: str):
    """
    Main function to run the fine-tuning process for the LLaVA model.

    This function performs the following steps:
    1. Loads the configuration file.
    2. Loads the LLaVA model and processor, prepared for fine-tuning with PEFT/LoRA.
    3. Loads the training and validation datasets.
    4. Sets up Hugging Face TrainingArguments.
    5. Initializes and runs the Trainer.
    6. Saves the trained model adapters.

    Args:
        config_path (str): Path to the YAML configuration file for fine-tuning.
    """
    # 1. Load configuration
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model_params']
    data_config = config['data_params']
    training_config = config['training_params']

    # 2. Load model and processor
    print("Loading LLaVA model for fine-tuning...")
    model, processor = load_llava_model_for_finetuning(
        model_id=model_config['model_id'],
        use_4bit_quantization=model_config.get('use_4bit_quantization', True)
    )

    # 3. Load datasets
    print("Loading and preprocessing datasets...")
    full_dataset = CustomVLMDataset(
        annotations_file=data_config['annotations_file'],
        image_dir=data_config['image_dir'],
        processor=processor
    )
    
    # Split dataset into training and validation sets
    if data_config['val_split_size'] > 0:
        print(f"Splitting dataset into {1 - data_config['val_split_size']:.0%} training and {data_config['val_split_size']:.0%} validation.")
        total_size = len(full_dataset)
        val_size = int(total_size * data_config['val_split_size'])
        train_size = total_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    else:
        train_dataset = full_dataset
        val_dataset = None # No validation

    # 4. Set up Training Arguments
    print("Setting up training arguments...")
    output_dir = training_config['output_dir']
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', training_config['per_device_train_batch_size']),
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.01),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=training_config.get('warmup_ratio', 0.03),
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_steps', 100),
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=training_config.get('eval_steps', 100) if val_dataset else None,
        save_total_limit=training_config.get('save_total_limit', 3),
        fp16=True, # Use mixed precision training
        remove_unused_columns=False, # Important for custom datasets
        report_to="tensorboard",
        load_best_model_at_end=True if val_dataset else False,
    )

    # 5. Initialize and run Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed.")

    # 6. Save the final model (LoRA adapters)
    final_model_path = os.path.join(output_dir, "final_checkpoint")
    print(f"Saving final LoRA adapters to: {final_model_path}")
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    print("Model and processor saved successfully.")

if __name__ == '__main__':
    # To run this script, execute from the project's root directory:
    # python src/perception_module/finetune.py --config configs/stage1_llava_config.yaml
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune the LLaVA model for manufacturing domain.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    
    finetune_llava(args.config)