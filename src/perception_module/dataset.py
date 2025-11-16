# -*- coding: utf-8 -*-

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomVLMDataset(Dataset):
    """
    A custom PyTorch Dataset for fine-tuning the LLaVA model on the manufacturing dataset.

    This class loads engineering drawings and their corresponding conversational annotations
    from the format specified in `data/vlm_finetuning_dataset/`. It preprocesses the
    images and tokenizes the text conversations to prepare them for the model.
    """
    def __init__(self, annotations_file: str, image_dir: str, processor):
        """
        Initializes the dataset.

        Args:
            annotations_file (str): Path to the .jsonl file containing annotations.
            image_dir (str): Path to the directory where images are stored.
            processor: The Hugging Face processor (e.g., AutoProcessor) for the VLM,
                       which handles both image and text preprocessing.
        """
        self.image_dir = image_dir
        self.processor = processor
        self.data = self._load_data(annotations_file)

    def _load_data(self, annotations_file: str) -> list:
        """Loads the data from the JSON Lines file."""
        data_list = []
        with open(annotations_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line: {line}")
        return data_list

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample from the dataset and prepares it for the model.

        This method performs the following steps:
        1. Loads the image file.
        2. Constructs the full conversational prompt from the annotations.
        3. Uses the processor to convert the image and text into model inputs
           (pixel_values, input_ids, attention_mask).
        4. Creates the `labels` tensor for calculating the loss, masking out the
           user's prompt part so the model is only trained to predict the assistant's response.
        """
        item = self.data[idx]
        
        # 1. Load the image
        image_file = item['image_file']
        image_path = os.path.join(self.image_dir, os.path.basename(image_file))
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}. Skipping sample.")
            # Return a dummy sample or handle this case appropriately
            # For simplicity, we can try getting the next sample
            return self.__getitem__((idx + 1) % len(self))

        # 2. Construct the prompt
        # LLaVA uses a specific chat template. We format the prompt accordingly.
        # Example: "USER: <image>\n{question} ASSISTANT: {answer}"
        conversation = item['conversations']
        user_prompt = conversation[0]['value'].replace('<image>', '').strip()
        assistant_response = conversation[1]['value']
        
        # We add the <image> token placeholder for the processor.
        full_prompt = f"USER: <image>\n{user_prompt} ASSISTANT: {assistant_response}"

        # 3. Process inputs using the multimodal processor
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048 # A common max length for VLMs
        )
        
        # The processor returns tensors in a batch dimension, so we squeeze it.
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # 4. Create labels for loss calculation
        # We only want to compute loss on the assistant's response.
        # We achieve this by setting the token IDs of the user's prompt to -100.
        
        # Tokenize the prompt part without the answer to find its length
        prompt_only = f"USER: <image>\n{user_prompt} ASSISTANT:"
        prompt_only_tokens = self.processor(
            text=prompt_only,
            return_tensors="pt",
            padding=False,
            truncation=True
        ).input_ids.squeeze(0)
        
        labels = inputs["input_ids"].clone()
        labels[:len(prompt_only_tokens)] = -100 # Mask out the prompt part
        
        # Replace pad token id in labels with -100
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        inputs["labels"] = labels

        return inputs