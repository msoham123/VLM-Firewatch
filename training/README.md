# Moondream 2 Fire Detection Training

This directory contains scripts to fine-tune Moondream 2 on the unified fire detection dataset.

## Overview

We have successfully aggregated three fire detection datasets (FLAME, FLAME3, FlameVision) into a unified VQA format. Now we'll fine-tune Moondream 2 to perform fire detection through visual question answering.

## Dataset

The unified dataset contains:
- **Training set**: ~200,000 samples
- **Validation set**: ~43,000 samples  
- **Test set**: ~43,000 samples

Each sample contains:
- Image filename
- Question-answer pair for fire detection
- Metadata (source dataset, fire presence, thermal availability)

## Files

- `setup_training.py` - Environment setup and validation script
- `simple_trainer.py` - Simplified training script for quick testing
- `moondream2_trainer.py` - Full-featured training script with advanced options
- `config.py` - Configuration file with all training parameters
- `README.md` - This file

## Quick Start

### 1. Setup Environment

First, activate your conda environment and install additional dependencies:

```bash
conda activate vlm_firewatch_env
cd /home/hice1/mchidambaram7/scratch/VLM-FireWatch/training
pip install datasets timm peft trl wandb
```

### 2. Validate Setup

Run the setup script to check everything is ready:

```bash
python setup_training.py
```

This will check:
- Python version and dependencies
- CUDA availability
- Dataset files and structure
- Model loading capability

### 3. Quick Test Training

Start with a small test to make sure everything works:

```bash
python simple_trainer.py
```

This will:
- Use only 1,000 training samples and 200 validation samples
- Train for 1 epoch
- Save the model to `./moondream2_fire_detection_simple/`

### 4. Full Training

Once the test works, run the full training:

```bash
python moondream2_trainer.py
```

This will:
- Use the complete dataset
- Train for 3 epochs
- Use 4-bit quantization for memory efficiency
- Save checkpoints and best model

## Configuration

Edit `config.py` to modify training parameters:

- **Model settings**: quantization, precision
- **Training settings**: epochs, batch size, learning rate
- **Data settings**: max length, image size
- **Output settings**: save directory, logging

## Expected Results

After training, you should have a fine-tuned Moondream 2 model that can:

1. **Answer fire detection questions** like:
   - "Is there fire visible in this image?"
   - "Can you detect any fire or flames in this aerial image?"
   - "Does this image contain active fire?"

2. **Provide accurate responses**:
   - "Yes, there is active fire visible in the image."
   - "No, there is no fire visible in this image."

## Monitoring Training

The training script will show:
- Training loss and validation loss
- Learning rate schedule
- Model checkpoints saved

If you enable wandb logging, you can monitor training progress in the web interface.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config
2. **Dataset not found**: Check paths in `config.py`
3. **Model loading fails**: Ensure you have enough disk space for model cache

### Memory Requirements

- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **With 4-bit quantization**: Can work with 8GB

## Fine-tuning from Last Checkpoint

If training was interrupted or you want to continue training from a saved checkpoint:

### Option 1: Resume from Best Model

To continue training from the best saved checkpoint:

```bash
cd /home/hice1/mchidambaram7/scratch/VLM-FireWatch/src/train

python finetune_moondream2.py \
    --train_json /home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_train.json \
    --val_json /home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_val.json \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 1e-5 \
    --output_path /home/hice1/mchidambaram7/scratch/VLM-FireWatch/models
```

**Note:** The current `finetune_moondream2.py` script starts from the base model. To resume from a checkpoint, you would need to modify the script to load the checkpoint first:

```python
# In finetune_moondream2.py, modify the model loading section:
checkpoint_path = "/home/hice1/mchidambaram7/scratch/VLM-FireWatch/models/moondream2_fire_detection_best"
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,  # Use checkpoint instead of base model
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto"
)
```

### Option 2: Continue Training with Additional Epochs

If you want to add more epochs to an already trained model:

1. Load the fine-tuned model from `models/moondream2_fire_detection_best/`
2. Continue training with the same or adjusted hyperparameters
3. The script will save a new best model if validation loss improves

### Running Fine-tuning in Background

For long training runs on a remote server:

```bash
cd /home/hice1/mchidambaram7/scratch/VLM-FireWatch/src/train

nohup python finetune_moondream2.py \
    --train_json /home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_train.json \
    --val_json /home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_val.json \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 1e-5 \
    > finetuning.log 2>&1 &
```

Monitor progress:
```bash
tail -f finetuning.log
```

Check if training is running:
```bash
ps aux | grep finetune_moondream2
```

## Next Steps

After training completes:

1. **Evaluate the model** on the test set
2. **Run object detection evaluation** (see `src/train/evaluate_object_detection.py`)
3. **Create inference script** for real-time fire detection
4. **Deploy the model** for production use
5. **Fine-tune further** if needed based on performance

## Support

If you encounter issues:
1. Check the setup validation output
2. Verify dataset paths and structure
3. Ensure sufficient GPU memory
4. Check CUDA and PyTorch installation
