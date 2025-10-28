#!/usr/bin/env python3
"""
Specialized dataloader for Moondream 2 training that provides PIL images directly.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import random
from typing import Optional, Callable, Dict, List, Union
from pathlib import Path

class Moondream2Dataset(Dataset):
    """
    PyTorch Dataset for Moondream 2 training that provides PIL images directly.
    """
    
    def __init__(
        self,
        json_path: str,
        image_size: int = 224,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            json_path: Path to JSON file containing annotations
            image_size: Size to resize images to
            max_samples: Maximum number of samples to load (for testing)
        """
        self.json_path = json_path
        self.image_size = image_size
        
        # Load JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Limit samples for testing
        if max_samples:
            self.data = self.data[:max_samples]
        
        print(f"Loaded {len(self.data)} samples from {json_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        
        # Load image using original_path from metadata
        image_path = sample['metadata']['original_path']
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        image = image.resize((self.image_size, self.image_size))
        
        # Get Q&A pair
        qa_pair = sample['conversations'][0]
        
        return {
            'image': image,  # PIL Image
            'question': qa_pair['question'],
            'answer': qa_pair['answer'],
            'image_name': sample['image'],
            'metadata': sample['metadata']
        }

def moondream2_collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for Moondream 2 that handles PIL images.
    """
    images = [item['image'] for item in batch]  # List of PIL Images
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    image_names = [item['image_name'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
    return {
        'images': images,  # List of PIL Images
        'questions': questions,
        'answers': answers,
        'image_names': image_names,
        'metadata': metadata
    }

def create_moondream2_dataloaders(
    train_json: str,
    val_json: str,
    test_json: str,
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: int = 224,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None
):
    """
    Create train, validation, and test dataloaders for Moondream 2.
    
    Args:
        train_json: Path to training JSON file
        val_json: Path to validation JSON file
        test_json: Path to test JSON file
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        image_size: Size to resize images to
        max_train_samples: Maximum training samples (for testing)
        max_val_samples: Maximum validation samples (for testing)
        max_test_samples: Maximum test samples (for testing)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = Moondream2Dataset(
        json_path=train_json,
        image_size=image_size,
        max_samples=max_train_samples
    )
    
    val_dataset = Moondream2Dataset(
        json_path=val_json,
        image_size=image_size,
        max_samples=max_val_samples
    )
    
    test_dataset = Moondream2Dataset(
        json_path=test_json,
        image_size=image_size,
        max_samples=max_test_samples
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=moondream2_collate_fn,
        pin_memory=True,  # Enable for GPU
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=moondream2_collate_fn,
        pin_memory=True,  # Enable for GPU
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=moondream2_collate_fn,
        pin_memory=True,  # Enable for GPU
        persistent_workers=num_workers > 0
    )
    
    print(f"Created Moondream 2 dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

# Test function
if __name__ == "__main__":
    import sys
    sys.path.append('/home/hice1/mchidambaram7/scratch/VLM-FireWatch')
    from src.data.dataset_configs import unified_config
    
    # Test the dataloader
    unified_path = unified_config['src']
    
    train_loader, val_loader, test_loader = create_moondream2_dataloaders(
        train_json=f'{unified_path}/flame_vqa_train.json',
        val_json=f'{unified_path}/flame_vqa_val.json',
        test_json=f'{unified_path}/flame_vqa_test.json',
        batch_size=2,
        num_workers=0,
        image_size=224,
        max_train_samples=10,
        max_val_samples=5,
        max_test_samples=5
    )
    
    # Test a batch
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Number of images: {len(batch['images'])}")
        print(f"Image type: {type(batch['images'][0])}")
        print(f"Image size: {batch['images'][0].size}")
        print(f"Sample question: {batch['questions'][0]}")
        print(f"Sample answer: {batch['answers'][0]}")
        break
