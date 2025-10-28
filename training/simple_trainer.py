#!/usr/bin/env python3
"""
Simplified Moondream 2 Training Script for Fire Detection
This is a streamlined version focused on getting started quickly.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFireDataset(Dataset):
    """Simplified dataset for fire detection"""
    
    def __init__(self, json_path: str, tokenizer, max_samples: int = None, image_base_path: str = "/home/hice1/mchidambaram7/scratch/datasets"):
        self.tokenizer = tokenizer
        self.image_base_path = Path(image_base_path)
        
        # Load data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Limit samples for testing
        if max_samples:
            self.data = self.data[:max_samples]
        
        # Filter out samples where images don't exist
        self.valid_data = []
        for item in tqdm(self.data, desc="Validating images"):
            if self._image_exists(item['image'], item['metadata']['original_path']):
                self.valid_data.append(item)
        
        logger.info(f"Loaded {len(self.valid_data)} valid samples with existing images (from {len(self.data)} total)")
    
    def _image_exists(self, image_name: str, original_path: str) -> bool:
        """Check if image exists in the filesystem"""
        # Try original path first
        if Path(original_path).exists():
            return True
        
        # Try to find image in common locations
        possible_paths = [
            self.image_base_path / "downloads" / "Training" / "Fire" / image_name,
            self.image_base_path / "downloads" / "Training" / "NoFire" / image_name,
            self.image_base_path / "Classification" / "train" / "fire" / image_name,
            self.image_base_path / "Classification" / "train" / "nofire" / image_name,
            self.image_base_path / "Classification" / "test" / "fire" / image_name,
            self.image_base_path / "Classification" / "test" / "nofire" / image_name,
            self.image_base_path / "FLAME 3 CV Dataset (Sycan Marsh)" / "Fire" / "RGB" / "Raw" / image_name,
            self.image_base_path / "FLAME 3 CV Dataset (Sycan Marsh)" / "No Fire" / "RGB" / "Raw" / image_name,
        ]
        
        return any(path.exists() for path in possible_paths)
    
    def _get_image_path(self, item: Dict) -> str:
        """Get the correct path for an image"""
        original_path = item['metadata']['original_path']
        if Path(original_path).exists():
            return original_path
        
        # Fallback to searching in common locations
        image_name = item['image']
        possible_paths = [
            self.image_base_path / "downloads" / "Training" / "Fire" / image_name,
            self.image_base_path / "downloads" / "Training" / "NoFire" / image_name,
            self.image_base_path / "Classification" / "train" / "fire" / image_name,
            self.image_base_path / "Classification" / "train" / "nofire" / image_name,
            self.image_base_path / "Classification" / "test" / "fire" / image_name,
            self.image_base_path / "Classification" / "test" / "nofire" / image_name,
            self.image_base_path / "FLAME 3 CV Dataset (Sycan Marsh)" / "Fire" / "RGB" / "Raw" / image_name,
            self.image_base_path / "FLAME 3 CV Dataset (Sycan Marsh)" / "No Fire" / "RGB" / "Raw" / image_name,
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(f"Image not found: {image_name}")
    
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        item = self.valid_data[idx]
        
        # Load and preprocess image
        image_path = self._get_image_path(item)
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))  # Standard size for Moondream 2
        
        # Create simple text input
        question = item['conversations'][0]['question']
        answer = item['conversations'][0]['answer']
        
        # Format for Moondream 2
        text = f"<image>\nQuestion: {question}\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'pixel_values': torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0,
            'labels': encoding['input_ids'].squeeze()
        }

def main():
    """Main training function"""
    
    # Configuration
    MODEL_NAME = "vikhyatk/moondream2"
    TRAIN_DATA = "/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_train.json"
    VAL_DATA = "/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_val.json"
    OUTPUT_DIR = "./moondream2_fire_detection_simple"
    
    # For testing, use only a small subset
    MAX_TRAIN_SAMPLES = 1000
    MAX_VAL_SAMPLES = 200
    
    logger.info("Loading tokenizer and model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info("Loading datasets...")
    
    # Load datasets
    train_dataset = SimpleFireDataset(TRAIN_DATA, tokenizer, MAX_TRAIN_SAMPLES)
    val_dataset = SimpleFireDataset(VAL_DATA, tokenizer, MAX_VAL_SAMPLES)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,  # Start with 1 epoch for testing
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb for simplicity
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training...")
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info(f"Training completed! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
