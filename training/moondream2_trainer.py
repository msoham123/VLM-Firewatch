#!/usr/bin/env python3
"""
Moondream 2 Fine-tuning Script for Fire Detection VQA Dataset
This script fine-tunes Moondream 2 on the unified FLAME dataset for fire detection.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import wandb
from dataclasses import dataclass, field
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name: str = field(default="vikhyatk/moondream2")
    cache_dir: Optional[str] = field(default=None)
    use_4bit: bool = field(default=True)
    bnb_4bit_compute_dtype: str = field(default="float16")
    bnb_4bit_quant_type: str = field(default="nf4")
    use_nested_quant: bool = field(default=False)

@dataclass
class DataArguments:
    """Arguments for data configuration"""
    train_data_path: str = field(default="/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_train.json")
    val_data_path: str = field(default="/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_val.json")
    test_data_path: str = field(default="/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_test.json")
    image_base_path: str = field(default="/home/hice1/mchidambaram7/scratch/datasets")
    max_length: int = field(default=512)
    image_size: int = field(default=224)

@dataclass
class TrainingArguments:
    """Arguments for training configuration"""
    output_dir: str = field(default="./moondream2_fire_detection")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.03)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    fp16: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="wandb")
    run_name: str = field(default="moondream2-fire-detection")

class FireDetectionDataset(Dataset):
    """Dataset class for fire detection VQA data"""
    
    def __init__(self, data_path: str, image_base_path: str, tokenizer, max_length: int = 512, image_size: int = 224):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.image_base_path = Path(image_base_path)
        
        # Load the VQA dataset
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        
        # Filter out samples where images don't exist
        self.valid_data = []
        for item in tqdm(self.data, desc="Validating images"):
            if self._image_exists(item['image'], item['metadata']['original_path']):
                self.valid_data.append(item)
        
        logger.info(f"Found {len(self.valid_data)} valid samples with existing images")
    
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
        image = image.resize((self.image_size, self.image_size))
        
        # Get question and answer
        conversation = item['conversations'][0]
        question = conversation['question']
        answer = conversation['answer']
        
        # Create input text
        input_text = f"<image>\nQuestion: {question}\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'pixel_values': torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0,
            'labels': encoding['input_ids'].squeeze()
        }

class Moondream2Trainer:
    """Main trainer class for Moondream 2 fine-tuning"""
    
    def __init__(self, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if needed
        quantization_config = None
        if model_args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, model_args.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=model_args.use_nested_quant,
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name,
            quantization_config=quantization_config,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch.float16 if model_args.use_4bit else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare datasets
        self.train_dataset = FireDetectionDataset(
            data_args.train_data_path,
            data_args.image_base_path,
            self.tokenizer,
            data_args.max_length,
            data_args.image_size
        )
        
        self.val_dataset = FireDetectionDataset(
            data_args.val_data_path,
            data_args.image_base_path,
            self.tokenizer,
            data_args.max_length,
            data_args.image_size
        )
        
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Validation dataset size: {len(self.val_dataset)}")
    
    def train(self):
        """Start training"""
        # Initialize wandb if specified
        if self.training_args.report_to == "wandb":
            wandb.init(
                project="moondream2-fire-detection",
                name=self.training_args.run_name,
                config={
                    "model_name": self.model_args.model_name,
                    "learning_rate": self.training_args.learning_rate,
                    "batch_size": self.training_args.per_device_train_batch_size,
                    "epochs": self.training_args.num_train_epochs,
                }
            )
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.training_args.output_dir,
            num_train_epochs=self.training_args.num_train_epochs,
            per_device_train_batch_size=self.training_args.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
            learning_rate=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay,
            warmup_ratio=self.training_args.warmup_ratio,
            logging_steps=self.training_args.logging_steps,
            save_steps=self.training_args.save_steps,
            eval_steps=self.training_args.eval_steps,
            eval_strategy=self.training_args.evaluation_strategy,
            save_strategy=self.training_args.save_strategy,
            load_best_model_at_end=self.training_args.load_best_model_at_end,
            metric_for_best_model=self.training_args.metric_for_best_model,
            greater_is_better=self.training_args.greater_is_better,
            fp16=self.training_args.fp16,
            dataloader_num_workers=self.training_args.dataloader_num_workers,
            remove_unused_columns=self.training_args.remove_unused_columns,
            report_to=self.training_args.report_to,
            run_name=self.training_args.run_name,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_args.output_dir)
        
        logger.info(f"Training completed! Model saved to {self.training_args.output_dir}")
        
        if self.training_args.report_to == "wandb":
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Moondream 2 for fire detection")
    parser.add_argument("--train_data", type=str, default="/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_train.json")
    parser.add_argument("--val_data", type=str, default="/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_val.json")
    parser.add_argument("--output_dir", type=str, default="./moondream2_fire_detection")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Create configuration objects
    model_args = ModelArguments(use_4bit=args.use_4bit)
    data_args = DataArguments(
        train_data_path=args.train_data,
        val_data_path=args.val_data
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        report_to="none" if args.no_wandb else "wandb"
    )
    
    # Initialize and start training
    trainer = Moondream2Trainer(model_args, data_args, training_args)
    trainer.train()

if __name__ == "__main__":
    main()
