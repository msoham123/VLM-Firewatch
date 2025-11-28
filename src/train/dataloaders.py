import torch
from torchvision import transforms
from typing import List
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.torch_datasets import FireDataset, FireDatasetMultiQA, FireDatasetWithThermal
from torch.utils.data import DataLoader

# Custom collate function for VQA tasks
def vqa_collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for batching VQA data.
    """
    images = torch.stack([item['image'] for item in batch])
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    image_names = [item['image_name'] for item in batch]
    
    result = {
        'images': images,
        'questions': questions,
        'answers': answers,
        'image_names': image_names
    }
    
    # Include metadata if available
    if 'metadata' in batch[0]:
        result['metadata'] = [item['metadata'] for item in batch]
    
    return result


# Example usage and helper functions
def get_data_transforms(image_size: int = 224, augment: bool = True):
    """
    Returns appropriate transforms for training/validation.
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(
    train_json: str,
    val_json: str,
    test_json: str,
    batch_size: int = 32,
    num_workers: int = 4,
    mode: str = 'classification',
    image_size: int = 224,
    pin_memory: bool = True
):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_json: Path to training JSON file
        val_json: Path to validation JSON file
        test_json: Path to test JSON file
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        mode: 'classification' or 'vqa'
        image_size: Size to resize images to (default: 224)
        pin_memory: Whether to use pinned memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Get transforms
    train_transform = get_data_transforms(image_size=image_size, augment=True)
    val_transform = get_data_transforms(image_size=image_size, augment=False)
    
    # Create datasets
    train_dataset = FireDataset(
        json_path=train_json,
        transform=train_transform,
        mode=mode,
        return_metadata=True 
    )
    
    val_dataset = FireDataset(
        json_path=val_json,
        transform=val_transform,
        mode=mode,
        return_metadata=True 
    )
    
    test_dataset = FireDataset(
        json_path=test_json,
        transform=val_transform,
        mode=mode,
        return_metadata=True 
    )
    
    # Create dataloaders
    if mode == 'vqa':
        # Use custom collate function for VQA

        # # ← ADD THIS before creating train_loader
        # from torch.utils.data import WeightedRandomSampler
        # train_labels = [1 if item['metadata']['has_fire'] else 0 
        #             for item in train_dataset.valid_data]
        # fire_count = sum(train_labels)
        # nofire_count = len(train_labels) - fire_count
        
        # # Weight fire samples 2x more
        # sample_weights = [2.0 if label == 1 else 1.0 for label in train_labels]
        
        # sampler = WeightedRandomSampler(
        #     weights=sample_weights,
        #     num_samples=len(train_dataset),
        #     replacement=True
        # )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # sampler=sampler,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=vqa_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=vqa_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=vqa_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    else:
        # Standard dataloaders for classification
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    
    print(f"Created dataloaders for {mode} mode:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


# # Example usage
# if __name__ == '__main__':
#     # Example 1: Create dataloaders for classification
#     train_loader, val_loader, test_loader = create_dataloaders(
#         train_json='path/to/train.json',
#         val_json='path/to/val.json',
#         test_json='path/to/test.json',
#         batch_size=32,
#         num_workers=4,
#         mode='classification',
#         image_size=224
#     )
    
#     # Iterate through a batch
#     for images, labels in train_loader:
#         print(f"Image batch shape: {images.shape}")
#         print(f"Label batch shape: {labels.shape}")
#         break
    
#     # Example 2: Create dataloaders for VQA
#     vqa_train_loader, vqa_val_loader, vqa_test_loader = create_dataloaders(
#         train_json='path/to/train.json',
#         val_json='path/to/val.json',
#         test_json='path/to/test.json',
#         batch_size=16,
#         num_workers=4,
#         mode='vqa',
#         image_size=384
#     )
    
#     # Iterate through a VQA batch
#     for batch in vqa_train_loader:
#         print(f"Image batch shape: {batch['images'].shape}")
#         print(f"Number of questions: {len(batch['questions'])}")
#         print(f"Sample question: {batch['questions'][0]}")
#         print(f"Sample answer: {batch['answers'][0]}")
#         break
