"""
Configuration file for Moondream 2 training
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path("/home/hice1/mchidambaram7/scratch/VLM-FireWatch")
DATASET_DIR = Path("/home/hice1/mchidambaram7/scratch/datasets")
UNIFIED_DATASET_DIR = DATASET_DIR / "unified_dataset"

# Model configuration
MODEL_NAME = "vikhyatk/moondream2"
CACHE_DIR = None  # Use default cache

# Dataset paths
TRAIN_DATA_PATH = UNIFIED_DATASET_DIR / "flame_vqa_train.json"
VAL_DATA_PATH = UNIFIED_DATASET_DIR / "flame_vqa_val.json"
TEST_DATA_PATH = UNIFIED_DATASET_DIR / "flame_vqa_test.json"

# Image base paths (where the actual images are stored)
IMAGE_BASE_PATHS = [
    DATASET_DIR / "downloads" / "Training",
    DATASET_DIR / "Classification",
    DATASET_DIR / "FLAME 3 CV Dataset (Sycan Marsh)"
]

# Training configuration
TRAINING_CONFIG = {
    "output_dir": BASE_DIR / "training" / "outputs",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "fp16": True,
    "dataloader_num_workers": 4,
    "remove_unused_columns": False,
    "report_to": "none",  # Change to "wandb" if you want to use wandb
    "run_name": "moondream2-fire-detection",
}

# Model configuration
MODEL_CONFIG = {
    "use_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "use_nested_quant": False,
}

# Data configuration
DATA_CONFIG = {
    "max_length": 512,
    "image_size": 224,
    "max_train_samples": None,  # Use all data
    "max_val_samples": None,    # Use all data
}

# Quick test configuration (for initial testing)
QUICK_TEST_CONFIG = {
    "max_train_samples": 1000,
    "max_val_samples": 200,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "save_steps": 100,
    "eval_steps": 100,
}
