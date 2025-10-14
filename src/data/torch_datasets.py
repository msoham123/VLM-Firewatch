import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import random
from typing import Optional, Callable, Dict, List, Union


class FireDataset(Dataset):
    """
    PyTorch Dataset for loading fire detection data from JSON file.
    Supports both classification and VQA tasks.
    Uses the original_path from metadata to load images.
    """
    
    def __init__(
        self,
        json_path: str,
        transform: Optional[Callable] = None,
        mode: str = 'classification',
        max_qa_pairs: int = 1,
        return_metadata: bool = False
    ):
        """
        Args:
            json_path: Path to JSON file containing annotations
            transform: Optional transform to apply to images
            mode: 'classification' or 'vqa'
            max_qa_pairs: Maximum number of Q&A pairs to return per sample (for VQA mode)
            return_metadata: Whether to return metadata dict
        """
        self.json_path = json_path
        self.transform = transform
        self.mode = mode
        self.max_qa_pairs = max_qa_pairs
        self.return_metadata = return_metadata
        
        # Load JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {json_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[tuple, dict]:
        """
        Returns data based on mode:
        - classification: (image, label)
        - vqa: dict with image, question, answer, and optional metadata
        """
        sample = self.data[idx]
        
        # Load image using original_path from metadata
        image_path = sample['metadata']['original_path']
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mode == 'classification':
            # Binary classification: fire (1) or non-fire (0)
            label = 1 if sample['metadata']['has_fire'] else 0
            
            if self.return_metadata:
                return {
                    'image': image,
                    'label': label,
                    'metadata': sample['metadata'],
                    'image_name': sample['image']
                }
            return image, label
        
        elif self.mode == 'vqa':
            # For VQA, return Q&A pairs
            conversations = sample['conversations']
            
            # If multiple Q&A pairs, select up to max_qa_pairs
            if len(conversations) > self.max_qa_pairs:
                conversations = random.sample(conversations, self.max_qa_pairs)
            
            # Return first Q&A pair (can be extended for multiple)
            qa_pair = conversations[0]
            
            result = {
                'image': image,
                'question': qa_pair['question'],
                'answer': qa_pair['answer'],
                'image_name': sample['image']
            }
            
            if self.return_metadata:
                result['metadata'] = sample['metadata']
            
            return result
        
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose 'classification' or 'vqa'")


class FireDatasetMultiQA(Dataset):
    """
    PyTorch Dataset that returns ALL Q&A pairs for each image.
    Useful for comprehensive VQA evaluation.
    """
    
    def __init__(
        self,
        json_path: str,
        transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        self.json_path = json_path
        self.transform = transform
        self.return_metadata = return_metadata
        
        # Load JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Flatten data: create one entry per Q&A pair
        self.samples = []
        for item in self.data:
            for qa in item['conversations']:
                self.samples.append({
                    'image_path': item['metadata']['original_path'],
                    'image_name': item['image'],
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'metadata': item['metadata']
                })
        
        print(f"Loaded {len(self.samples)} Q&A pairs from {len(self.data)} images")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Load image using original_path
        image_path = sample['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        result = {
            'image': image,
            'question': sample['question'],
            'answer': sample['answer'],
            'image_name': sample['image_name']
        }
        
        if self.return_metadata:
            result['metadata'] = sample['metadata']
        
        return result


class FireDatasetWithThermal(Dataset):
    """
    Extended dataset that handles both RGB and thermal images when available.
    Assumes thermal images have similar path structure or naming convention.
    """
    
    def __init__(
        self,
        json_path: str,
        thermal_suffix: str = '_thermal',
        transform: Optional[Callable] = None,
        mode: str = 'classification',
        use_both_modalities: bool = False
    ):
        """
        Args:
            json_path: Path to JSON file
            thermal_suffix: Suffix or pattern to find thermal images (e.g., '_thermal' or 'thermal/')
            transform: Image transforms
            mode: 'classification' or 'vqa'
            use_both_modalities: Whether to load both RGB and thermal
        """
        self.json_path = json_path
        self.thermal_suffix = thermal_suffix
        self.transform = transform
        self.mode = mode
        self.use_both_modalities = use_both_modalities
        
        # Load JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {json_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _get_thermal_path(self, rgb_path: str) -> Optional[str]:
        """
        Construct thermal image path from RGB path.
        Adjust this logic based on your thermal image naming convention.
        """
        # Example: /path/to/image.jpg -> /path/to/image_thermal.jpg
        base, ext = os.path.splitext(rgb_path)
        thermal_path = f"{base}{self.thermal_suffix}{ext}"
        
        if os.path.exists(thermal_path):
            return thermal_path
        
        # Alternative: replace directory name
        # /path/rgb/image.jpg -> /path/thermal/image.jpg
        thermal_path_alt = rgb_path.replace('/rgb/', '/thermal/')
        if os.path.exists(thermal_path_alt):
            return thermal_path_alt
        
        return None
    
    def __getitem__(self, idx: int) -> Union[tuple, dict]:
        sample = self.data[idx]
        
        # Load RGB image
        rgb_path = sample['metadata']['original_path']
        rgb_image = Image.open(rgb_path).convert('RGB')
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        # Load thermal image if available
        thermal_image = None
        if self.use_both_modalities and sample['metadata']['thermal_available']:
            thermal_path = self._get_thermal_path(rgb_path)
            if thermal_path:
                thermal_image = Image.open(thermal_path).convert('RGB')
                if self.transform:
                    thermal_image = self.transform(thermal_image)
        
        if self.mode == 'classification':
            label = 1 if sample['metadata']['has_fire'] else 0
            
            if thermal_image is not None:
                # Concatenate RGB and thermal as 6-channel input
                combined = torch.cat([rgb_image, thermal_image], dim=0)
                return combined, label
            
            return rgb_image, label
        
        elif self.mode == 'vqa':
            qa_pair = sample['conversations'][0]
            return {
                'image': rgb_image,
                'thermal': thermal_image,
                'question': qa_pair['question'],
                'answer': qa_pair['answer'],
                'image_name': sample['image'],
                'metadata': sample['metadata']
            }
