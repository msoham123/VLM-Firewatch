"""
Configuration file for Moondream 2 fine-tuning
"""

import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class FineTuningConfig:
    """Configuration class for Moondream 2 fine-tuning"""
    
    # Model settings
    model_name: str = "vikhyatk/moondream2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Training hyperparameters
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Data settings
    image_size: int = 224
    max_length: int = 256
    num_workers: int = 4
    
    # Dataset limits (for testing)
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    
    # Logging and monitoring
    use_wandb: bool = True
    log_interval: int = 10
    save_interval: int = 1
    
    # Paths
    unified_dataset_path: str = "/home/hice1/mchidambaram7/scratch/datasets/unified_dataset"
    output_path: str = "/home/hice1/mchidambaram7/scratch/VLM-FireWatch/models"
    
    # Test mode settings
    test_mode: bool = False
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.test_mode:
            self.max_train_samples = 100
            self.max_val_samples = 20
            self.epochs = 1
            self.batch_size = 4
            self.use_wandb = False
        
        # Ensure output directory exists
        import os
        os.makedirs(self.output_path, exist_ok=True)

# Predefined configurations
CONFIGS = {
    "test": FineTuningConfig(test_mode=True),
    "small": FineTuningConfig(
        batch_size=4,
        epochs=2,
        max_train_samples=1000,
        max_val_samples=200
    ),
    "medium": FineTuningConfig(
        batch_size=8,
        epochs=3,
        max_train_samples=5000,
        max_val_samples=1000
    ),
    "full": FineTuningConfig(
        batch_size=8,
        epochs=5,
        learning_rate=2e-5
    ),
    "gpu_optimized": FineTuningConfig(
        batch_size=16,
        epochs=3,
        learning_rate=3e-5,
        num_workers=8
    )
}

def get_config(config_name: str = "medium") -> FineTuningConfig:
    """Get a predefined configuration"""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]
