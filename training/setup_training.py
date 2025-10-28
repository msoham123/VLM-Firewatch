#!/usr/bin/env python3
"""
Setup script for Moondream 2 training environment
This script checks dependencies and prepares the training environment.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            logger.warning("CUDA is not available. Training will be slower on CPU.")
            return True  # Don't treat as failure, just warning
    except ImportError:
        logger.error("PyTorch is not installed")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch',
        'transformers',
        'PIL',
        'numpy',
        'tqdm',
        'datasets'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_dataset_files():
    """Check if dataset files exist"""
    from config import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH
    
    dataset_files = {
        "Train": TRAIN_DATA_PATH,
        "Validation": VAL_DATA_PATH,
        "Test": TEST_DATA_PATH
    }
    
    all_exist = True
    for name, path in dataset_files.items():
        if path.exists():
            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ {name} dataset: {path} ({size_mb:.1f} MB)")
        else:
            logger.error(f"❌ {name} dataset not found: {path}")
            all_exist = False
    
    return all_exist

def check_dataset_structure():
    """Check the structure of the dataset files"""
    from config import TRAIN_DATA_PATH
    
    try:
        with open(TRAIN_DATA_PATH, 'r') as f:
            data = json.load(f)
        
        if not data:
            logger.error("Dataset is empty")
            return False
        
        # Check structure of first item
        first_item = data[0]
        required_keys = ['image', 'conversations', 'metadata']
        
        for key in required_keys:
            if key not in first_item:
                logger.error(f"Missing key '{key}' in dataset structure")
                return False
        
        logger.info(f"✅ Dataset structure is valid. Sample count: {len(data)}")
        
        # Show sample
        sample = first_item
        logger.info("Sample data structure:")
        logger.info(f"  Image: {sample['image']}")
        logger.info(f"  Question: {sample['conversations'][0]['question']}")
        logger.info(f"  Answer: {sample['conversations'][0]['answer']}")
        logger.info(f"  Source: {sample['metadata']['source_dataset']}")
        logger.info(f"  Has fire: {sample['metadata']['has_fire']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking dataset structure: {e}")
        return False

def test_model_loading():
    """Test if we can load the Moondream 2 model"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("Testing model loading...")
        tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
        logger.info("✅ Tokenizer loaded successfully")
        
        # Test with a small model first
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("✅ Model loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def create_output_directory():
    """Create output directory for training"""
    from config import TRAINING_CONFIG
    
    output_dir = Path(TRAINING_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Output directory created: {output_dir}")
    return True

def main():
    """Main setup function"""
    logger.info("🚀 Setting up Moondream 2 training environment...")
    
    checks = [
        ("Python version", check_python_version),
        ("CUDA availability", check_cuda),
        ("Dependencies", check_dependencies),
        ("Dataset files", check_dataset_files),
        ("Dataset structure", check_dataset_structure),
        ("Model loading", test_model_loading),
        ("Output directory", create_output_directory),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        logger.info(f"\n🔍 Checking {check_name}...")
        if not check_func():
            all_passed = False
            logger.error(f"❌ {check_name} check failed")
        else:
            logger.info(f"✅ {check_name} check passed")
    
    if all_passed:
        logger.info("\n🎉 All checks passed! You're ready to start training.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python simple_trainer.py (for quick test)")
        logger.info("2. Run: python moondream2_trainer.py (for full training)")
    else:
        logger.error("\n❌ Some checks failed. Please fix the issues before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()
