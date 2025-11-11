#!/usr/bin/env python3
"""
Quick test script to verify Moondream 2 fine-tuning setup
"""

import torch
import logging
from pathlib import Path
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from finetuning_config import get_config
from finetune_moondream2 import Moondream2FineTuner, create_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_setup():
    """Test the fine-tuning setup with minimal data"""
    
    logger.info("Testing Moondream 2 fine-tuning setup...")
    
    # Use test configuration
    config = get_config("test")
    logger.info(f"Using test configuration: {config}")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check dataset files
    train_json = f"{config.unified_dataset_path}/flame_vqa_train.json"
    val_json = f"{config.unified_dataset_path}/flame_vqa_val.json"
    
    if not Path(train_json).exists():
        logger.error(f"Training dataset not found: {train_json}")
        return False
    
    if not Path(val_json).exists():
        logger.error(f"Validation dataset not found: {val_json}")
        return False
    
    logger.info("Dataset files found ✓")
    
    try:
        # Test model loading
        logger.info("Testing model loading...")
        fine_tuner = Moondream2FineTuner(
            model_name=config.model_name,
            device=config.device,
            dtype=config.dtype
        )
        logger.info("Model loaded successfully ✓")
        
        # Test dataloader creation
        logger.info("Testing dataloader creation...")
        train_loader, val_loader = create_dataloaders(
            train_json=train_json,
            val_json=val_json,
            tokenizer=fine_tuner.tokenizer,
            batch_size=config.batch_size,
            max_train_samples=config.max_train_samples,
            max_val_samples=config.max_val_samples
        )
        logger.info("Dataloaders created successfully ✓")
        
        # Test a single batch
        logger.info("Testing single batch processing...")
        batch = next(iter(train_loader))
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Images count: {len(batch['images'])}")
        logger.info("Batch processing successful ✓")
        
        # Test optimizer setup
        logger.info("Testing optimizer setup...")
        total_steps = len(train_loader) * config.epochs
        fine_tuner.setup_optimizer_and_scheduler(
            learning_rate=config.learning_rate,
            total_steps=total_steps
        )
        logger.info("Optimizer setup successful ✓")
        
        logger.info("All tests passed! Fine-tuning setup is ready.")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_setup()
    if success:
        print("\n🎉 Setup test passed! You can now run fine-tuning.")
        print("\nTo start fine-tuning, run:")
        print("python train_moondream2_simple.py --config test")
    else:
        print("\n❌ Setup test failed. Please check the errors above.")
        sys.exit(1)
