#!/usr/bin/env python3
"""
Simplified Moondream 2 Fine-tuning Script
Uses configuration-based approach for easy experimentation
"""

import torch
import logging
from pathlib import Path
import argparse
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from finetuning_config import get_config, FineTuningConfig
from finetune_moondream2 import Moondream2FineTuner, create_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function for simplified fine-tuning"""
    
    parser = argparse.ArgumentParser(description="Fine-tune Moondream 2 on fire detection")
    parser.add_argument(
        "--config", 
        type=str, 
        default="test", 
        choices=["test", "small", "medium", "full", "gpu_optimized"],
        help="Configuration preset to use"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None, 
        help="Override batch size"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None, 
        help="Override number of epochs"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=None, 
        help="Override learning rate"
    )
    parser.add_argument(
        "--max_train_samples", 
        type=int, 
        default=None, 
        help="Override max training samples"
    )
    parser.add_argument(
        "--use_wandb", 
        action="store_true", 
        help="Enable Weights & Biases logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples
    if args.use_wandb:
        config.use_wandb = True
    
    logger.info(f"Using configuration: {args.config}")
    logger.info(f"Device: {config.device}, Batch size: {config.batch_size}, Epochs: {config.epochs}")
    
    # Initialize wandb if enabled
    if config.use_wandb:
        import wandb
        wandb.init(
            project="moondream2-fire-detection",
            config=vars(config),
            name=f"moondream2-{args.config}-{torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}"
        )
    
    try:
        # Dataset paths
        train_json = f"{config.unified_dataset_path}/flame_vqa_train.json"
        val_json = f"{config.unified_dataset_path}/flame_vqa_val.json"
        
        # Verify dataset files exist
        if not Path(train_json).exists():
            raise FileNotFoundError(f"Training dataset not found: {train_json}")
        if not Path(val_json).exists():
            raise FileNotFoundError(f"Validation dataset not found: {val_json}")
        
        # Initialize fine-tuner
        logger.info("Initializing Moondream 2 fine-tuner...")
        fine_tuner = Moondream2FineTuner(
            model_name=config.model_name,
            device=config.device,
            dtype=config.dtype
        )
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            train_json=train_json,
            val_json=val_json,
            tokenizer=fine_tuner.tokenizer,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            max_length=config.max_length,
            max_train_samples=config.max_train_samples,
            max_val_samples=config.max_val_samples
        )
        
        # Setup optimizer and scheduler
        total_steps = len(train_loader) * config.epochs
        fine_tuner.setup_optimizer_and_scheduler(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps
        )
        
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Training loop
        logger.info("Starting fine-tuning...")
        best_val_loss = float('inf')
        
        for epoch in range(1, config.epochs + 1):
            logger.info(f"Starting epoch {epoch}/{config.epochs}")
            
            # Train
            train_loss = fine_tuner.train_epoch(
                train_loader, 
                epoch, 
                use_wandb=config.use_wandb
            )
            
            # Validate
            val_loss = fine_tuner.validate(val_loader, epoch)
            
            logger.info(f"Epoch {epoch} completed:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            
            # Log to wandb
            if config.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": fine_tuner.scheduler.get_last_lr()[0]
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f"{config.output_path}/moondream2_fire_detection_best"
                fine_tuner.save_model(save_path)
                logger.info(f"New best model saved (val_loss={val_loss:.4f})")
            
            # Save checkpoint every save_interval epochs
            if epoch % config.save_interval == 0:
                checkpoint_path = f"{config.output_path}/moondream2_fire_detection_epoch_{epoch}"
                fine_tuner.save_model(checkpoint_path)
                logger.info(f"Checkpoint saved at epoch {epoch}")
        
        # Save final model
        final_save_path = f"{config.output_path}/moondream2_fire_detection_final"
        fine_tuner.save_model(final_save_path)
        
        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Models saved to: {config.output_path}")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise
    
    finally:
        if config.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()
