#!/usr/bin/env python3
"""
Moondream 2 Simple Training Script
This script uses a custom training loop that works with Moondream 2's architecture.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Moondream2Dataset(Dataset):
    """Dataset for Moondream 2 training"""
    
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
        image = image.resize((224, 224))
        
        # Get question and answer
        question = item['conversations'][0]['question']
        answer = item['conversations'][0]['answer']
        
        return {
            'image': image,
            'question': question,
            'answer': answer,
            'has_fire': item['metadata']['has_fire']
        }

def custom_collate_fn(batch):
    """Custom collate function to handle PIL Images"""
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    has_fire = [item['has_fire'] for item in batch]
    
    return {
        'image': images,
        'question': questions,
        'answer': answers,
        'has_fire': has_fire
    }

class Moondream2Trainer:
    """Custom trainer for Moondream 2"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, dataloader, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Get images and text
                images = batch['image']
                questions = batch['question']
                answers = batch['answer']
                
                # Process each item in the batch
                batch_loss = 0
                for i in range(len(images)):
                    image = images[i]
                    question = questions[i]
                    answer = answers[i]
                    
                    # Format the input
                    prompt = f"<image>\nQuestion: {question}\nAnswer: {answer}"
                    
                    # Tokenize
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Prepare image
                    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
                    image_tensor = image_tensor.unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    try:
                        # Moondream 2 expects specific input format
                        outputs = self.model.forward(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            pixel_values=image_tensor
                        )
                        
                        # Calculate loss (simplified - using next token prediction)
                        if hasattr(outputs, 'loss') and outputs.loss is not None:
                            loss = outputs.loss
                        else:
                            # Fallback: use cross-entropy on logits
                            logits = outputs.logits
                            labels = inputs['input_ids']
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        
                        batch_loss += loss.item()
                        
                    except Exception as e:
                        logger.warning(f"Error in forward pass: {e}")
                        continue
                
                # Backward pass
                if batch_loss > 0:
                    optimizer.zero_grad()
                    loss = torch.tensor(batch_loss, requires_grad=True, device=self.device)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss
                    num_batches += 1
                    
                    progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})
                
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

def main():
    """Main training function"""
    
    # Configuration
    MODEL_NAME = "vikhyatk/moondream2"
    TRAIN_DATA = "/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_train.json"
    VAL_DATA = "/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_val.json"
    OUTPUT_DIR = "./moondream2_fire_detection_custom"
    
    # For testing, use only a small subset
    MAX_TRAIN_SAMPLES = 100
    MAX_VAL_SAMPLES = 20
    
    # Training parameters
    BATCH_SIZE = 1  # Small batch size for testing
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 1
    
    logger.info("Loading tokenizer and model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use float32 for CPU
        trust_remote_code=True
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info("Loading datasets...")
    
    # Load datasets
    train_dataset = Moondream2Dataset(TRAIN_DATA, tokenizer, MAX_TRAIN_SAMPLES)
    val_dataset = Moondream2Dataset(VAL_DATA, tokenizer, MAX_VAL_SAMPLES)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # Create trainer
    trainer = Moondream2Trainer(model, tokenizer, device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    logger.info("Starting training...")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, epoch + 1)
        
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {train_loss:.4f}")
    
    # Save model
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"Training completed! Model saved to {output_path}")

if __name__ == "__main__":
    main()
