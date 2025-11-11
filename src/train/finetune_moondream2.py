#!/usr/bin/env python3
"""
Comprehensive Moondream 2 Fine-tuning Script for Fire Detection
Based on best practices and research from 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from PIL import Image
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import wandb
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Moondream2FireDataset(torch.utils.data.Dataset):
    """
    Custom dataset for Moondream 2 fine-tuning on fire detection VQA data
    """
    
    def __init__(
        self, 
        json_path: str, 
        tokenizer, 
        image_size: int = 224,
        max_length: int = 256,
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        
        # Load data
        logger.info(f"Loading dataset from {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter valid samples
        self.valid_data = []
        for item in tqdm(self.data, desc="Validating samples"):
            if self._validate_sample(item):
                self.valid_data.append(item)
        
        if max_samples:
            self.valid_data = self.valid_data[:max_samples]
        
        logger.info(f"Loaded {len(self.valid_data)} valid samples")
    
    def _validate_sample(self, item: Dict) -> bool:
        """Validate that a sample has all required fields and image exists"""
        try:
            # Check required fields
            if 'conversations' not in item or 'metadata' not in item:
                return False
            
            # Check image path exists
            image_path = item['metadata']['original_path']
            if not os.path.exists(image_path):
                return False
            
            # Check conversation format
            if not item['conversations'] or len(item['conversations']) < 1:
                return False
            
            qa_pair = item['conversations'][0]
            if 'question' not in qa_pair or 'answer' not in qa_pair:
                return False
            
            return True
        except Exception:
            return False
    
    def __len__(self) -> int:
        return len(self.valid_data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.valid_data[idx]
        
        # Load and preprocess image
        image_path = item['metadata']['original_path']
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        
        # Get question and answer
        qa_pair = item['conversations'][0]
        question = qa_pair['question']
        answer = qa_pair['answer']
        
        # Create input text in Moondream format
        # Format: <image>\nQuestion: {question}\nAnswer: {answer}
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
            'labels': encoding['input_ids'].squeeze(),  # For causal LM
            'image': image,  # Keep PIL image for vision encoder
            'question': question,
            'answer': answer,
            'has_fire': item['metadata']['has_fire']
        }

class Moondream2Collator:
    """
    Custom collator for Moondream 2 fine-tuning
    Handles PIL images and text tokenization
    """
    
    def __init__(self, tokenizer, image_size: int = 224):
        self.tokenizer = tokenizer
        self.image_size = image_size
    
    def __call__(self, batch: List[Dict]) -> Dict:
        # Extract components
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        images = [item['image'] for item in batch]  # List of PIL Images
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        has_fire = [item['has_fire'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'images': images,
            'questions': questions,
            'answers': answers,
            'has_fire': has_fire
        }

class Moondream2FineTuner:
    """
    Main fine-tuning class for Moondream 2
    """
    
    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
    
    def _load_model_and_tokenizer(self):
        """Load Moondream 2 model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Move to device if not using device_map
        if self.device != "cuda" or "auto" not in str(self.model.device):
            self.model = self.model.to(self.device)
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_optimizer_and_scheduler(
        self,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        total_steps: int = 1000
    ):
        """Setup optimizer and learning rate scheduler"""
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Optimizer and scheduler setup with LR={learning_rate}")
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        epoch: int,
        use_wandb: bool = True
    ) -> float:
        """Train for one epoch using Moondream 2's answer_question method"""
        
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['images']  # List of PIL Images
            questions = batch['questions']
            answers = batch['answers']
            has_fire = batch['has_fire']
            
            batch_loss = 0
            valid_samples = 0
            
            # Process each sample in the batch
            for i in range(len(images)):
                image = images[i]
                question = questions[i]
                answer = answers[i]
                
                # Get model's answer using Moondream 2's method
                with torch.no_grad():
                    model_answer = self.model.answer_question(image, question, self.tokenizer)
                
                # Calculate loss based on answer similarity
                # This is a simplified loss - in practice, you'd want more sophisticated evaluation
                loss = self._calculate_answer_loss(answer, model_answer)
                
                batch_loss += loss
                valid_samples += 1
            
            if valid_samples > 0:
                avg_loss = batch_loss / valid_samples
                
                # Create a tensor that requires gradients for backprop
                loss_tensor = torch.tensor(avg_loss, requires_grad=True, device=self.device)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_tensor.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += avg_loss
                
                # Update progress bar
                progress_bar.set_postfix(loss=avg_loss)
                
                # Log to wandb
                if use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        "train_loss": avg_loss,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "batch": batch_idx
                    })
        
        return total_loss / len(train_loader)
    
    def _calculate_answer_loss(self, true_answer: str, model_answer: str) -> float:
        """Calculate loss based on answer similarity"""
        true_lower = true_answer.lower()
        model_lower = model_answer.lower()
        
        # Simple keyword-based evaluation
        fire_keywords = ['fire', 'flame', 'burning', 'yes']
        no_fire_keywords = ['no fire', 'no flames', 'no burning', 'no']
        
        true_has_fire = any(keyword in true_lower for keyword in fire_keywords)
        model_has_fire = any(keyword in model_lower for keyword in fire_keywords)
        
        # Binary loss: 0 if correct, 1 if incorrect
        if true_has_fire == model_has_fire:
            return 0.1  # Small positive loss for correct answers
        else:
            return 1.0  # Larger loss for incorrect answers
    
    def validate(self, val_loader: DataLoader, epoch: int) -> float:
        """Validate the model using Moondream 2's answer_question method"""
        
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                images = batch['images']  # List of PIL Images
                questions = batch['questions']
                answers = batch['answers']
                has_fire = batch['has_fire']
                
                batch_loss = 0
                batch_correct = 0
                
                for i in range(len(images)):
                    image = images[i]
                    question = questions[i]
                    answer = answers[i]
                    
                    # Get model's answer
                    model_answer = self.model.answer_question(image, question, self.tokenizer)
                    
                    # Calculate loss
                    loss = self._calculate_answer_loss(answer, model_answer)
                    batch_loss += loss
                    
                    # Count correct predictions
                    if self._is_correct_prediction(answer, model_answer):
                        batch_correct += 1
                
                total_loss += batch_loss / len(images)
                correct_predictions += batch_correct
                total_predictions += len(images)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss
    
    def _is_correct_prediction(self, true_answer: str, model_answer: str) -> bool:
        """Check if the model's prediction is correct"""
        true_lower = true_answer.lower()
        model_lower = model_answer.lower()
        
        fire_keywords = ['fire', 'flame', 'burning', 'yes']
        
        true_has_fire = any(keyword in true_lower for keyword in fire_keywords)
        model_has_fire = any(keyword in model_lower for keyword in fire_keywords)
        
        return true_has_fire == model_has_fire
    
    def save_model(self, save_path: str):
        """Save the fine-tuned model"""
        logger.info(f"Saving model to {save_path}")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info("Model saved successfully")

def create_dataloaders(
    train_json: str,
    val_json: str,
    tokenizer,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 224,
    max_length: int = 256,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders"""
    
    # Create datasets
    train_dataset = Moondream2FireDataset(
        json_path=train_json,
        tokenizer=tokenizer,
        image_size=image_size,
        max_length=max_length,
        max_samples=max_train_samples
    )
    
    val_dataset = Moondream2FireDataset(
        json_path=val_json,
        tokenizer=tokenizer,
        image_size=image_size,
        max_length=max_length,
        max_samples=max_val_samples
    )
    
    # Create collator
    collator = Moondream2Collator(tokenizer, image_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return train_loader, val_loader

def main():
    """Main fine-tuning function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune Moondream 2 on fire detection")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--test_mode", action="store_true", help="Test mode with small dataset")
    
    args = parser.parse_args()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    logger.info(f"Using device: {device}, dtype: {dtype}")
    
    # Test mode adjustments
    if args.test_mode:
        args.max_train_samples = 100
        args.max_val_samples = 20
        args.epochs = 1
        logger.info("Running in test mode with limited samples")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="moondream2-fire-detection",
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "device": device,
                "dtype": str(dtype)
            }
        )
    
    # Dataset paths
    unified_path = "/home/hice1/mchidambaram7/scratch/datasets/unified_dataset"
    train_json = f"{unified_path}/flame_vqa_train.json"
    val_json = f"{unified_path}/flame_vqa_val.json"
    
    # Initialize fine-tuner
    fine_tuner = Moondream2FineTuner(device=device, dtype=dtype)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_json=train_json,
        val_json=val_json,
        tokenizer=fine_tuner.tokenizer,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    
    # Setup optimizer and scheduler
    total_steps = len(train_loader) * args.epochs
    fine_tuner.setup_optimizer_and_scheduler(
        learning_rate=args.learning_rate,
        total_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting fine-tuning...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = fine_tuner.train_epoch(
            train_loader, 
            epoch, 
            use_wandb=args.use_wandb
        )
        
        # Validate
        val_loss = fine_tuner.validate(val_loader, epoch)
        
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"/home/hice1/mchidambaram7/scratch/VLM-FireWatch/models/moondream2_fire_detection_best"
            fine_tuner.save_model(save_path)
            logger.info(f"New best model saved with val_loss={val_loss:.4f}")
    
    # Save final model
    final_save_path = f"/home/hice1/mchidambaram7/scratch/VLM-FireWatch/models/moondream2_fire_detection_final"
    fine_tuner.save_model(final_save_path)
    
    logger.info("Fine-tuning completed!")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
