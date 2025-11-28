# Step-by-Step Guide to Train Moondream 2 for Fire Detection

This guide walks you through the complete process of training a Moondream 2 model on the unified fire detection dataset.

## Prerequisites

1. **Hardware Requirements:**
   - GPU with at least 8GB VRAM (recommended: 16GB+)
   - Sufficient disk space for datasets (~50GB+) and model checkpoints (~10GB)

2. **Software Requirements:**
   - Python 3.8+
   - CUDA-capable GPU (if using GPU)
   - Conda or pip for package management

## Step 1: Set Up Environment

### Option A: Using Conda (Recommended)

```bash
# Navigate to project directory
cd /path/to/VLM-FireWatch

# Create conda environment (if environment.yml exists)
conda env create -f env/environment.yml
conda activate vlm-firewatch  # or the name specified in environment.yml

## Step 2: Prepare Datasets

### 2.1 Download Raw Datasets

You need to obtain the following datasets:

1. **FLAME Dataset**: Original FLAME training data
2. **FLAME 3 Dataset**: FLAME 3 CV Dataset (Sycan Marsh)
3. **FlameVision Dataset**: Classification dataset
4. **Places365 Dataset** (optional): For negative samples

### 2.2 Configure Dataset Paths

Edit `src/data/dataset_configs.py` to point to your dataset locations:

```python
flame_config = {
    "src": "/path/to/your/FLAME/Training",
}

flame3_config = {
    "src": "/path/to/your/FLAME 3 CV Dataset (Sycan Marsh)",
}

flamevision_config = {
    "src": "/path/to/your/Classification",
}

places_365_config = {
    "src": "/path/to/your/places365",
    "processed": "/path/to/your/places365/processed_nofire_samples"
}

unified_config = {
    "src": "/path/to/output/unified_dataset"
}
```

### 2.3 Process and Aggregate Datasets

Run the dataset processing script to aggregate all datasets into a unified VQA format:

```bash
cd src/data
python process_datasets.py
```

This will:
- Load all available datasets
- Convert them to VQA (Visual Question Answering) format
- Create train/validation/test splits (70/15/15)
- Save JSON files to the unified dataset directory:
  - `flame_vqa_train.json`
  - `flame_vqa_val.json`
  - `flame_vqa_test.json`

**Expected Output:**
- Total samples: ~45,000+ (depending on available datasets)
- Train: ~70% of samples
- Validation: ~15% of samples
- Test: ~15% of samples

## Step 3: Train the Model

### 3.1 Choose Training Script

You have two main training scripts:

1. **`src/train/finetune_moondream2.py`** - Full-featured training with best model saving
2. **`src/train/train_moondream2.py`** - Simpler training script

### 3.2 Run Training

#### Option A: Using finetune_moondream2.py (Recommended)

```bash
cd /path/to/VLM-FireWatch/src/train

python finetune_moondream2.py \
    --train_json /path/to/unified_dataset/flame_vqa_train.json \
    --val_json /path/to/unified_dataset/flame_vqa_val.json \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 1e-5 \
    --output_path /path/to/VLM-FireWatch/models
```

#### Option B: Using train_moondream2.py

```bash
cd /path/to/VLM-FireWatch/src/train
python train_moondream2.py
```

**Note:** This script uses hardcoded paths from `dataset_configs.py`, so make sure those are configured correctly.

### 3.3 Training in Background (for long runs)

To run training in the background on a remote server:

```bash
nohup python finetune_moondream2.py \
    --train_json /path/to/unified_dataset/flame_vqa_train.json \
    --val_json /path/to/unified_dataset/flame_vqa_val.json \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 1e-5 \
    --output_path /path/to/VLM-FireWatch/models \
    > training.log 2>&1 &
```

Monitor progress:
```bash
tail -f training.log
```

## Step 4: Locate Trained Models

After training completes, models are saved in the specified output directory (default: `VLM-FireWatch/models/`):

### Model Checkpoints:

1. **`moondream2_fire_detection_best/`** - Best model based on validation loss
   - This is the recommended model to use for inference
   - Contains: `model.safetensors`, `config.json`, tokenizer files, and custom code

2. **`moondream2_fire_detection_final/`** - Final model after all epochs

3. **`moondream2_fire_detection_epoch_N/`** - Checkpoints saved at specific epochs (if configured)

### Model Directory Structure:

```
models/moondream2_fire_detection_best/
├── model.safetensors          # Model weights
├── config.json                # Model configuration
├── tokenizer.json             # Tokenizer
├── tokenizer_config.json      # Tokenizer config
├── vocab.json                 # Vocabulary
├── merges.txt                 # BPE merges
├── moondream.py               # Custom model code
├── vision.py                  # Vision encoder code
├── layers.py                  # Custom layers
└── ... (other supporting files)
```

## Step 5: Verify Model Training

### 5.1 Check Training Logs

Review the training log file or console output for:
- Training loss decreasing over epochs
- Validation accuracy improving
- No critical errors

### 5.2 Test the Model

Run inference on the test set:

```bash
cd /path/to/VLM-FireWatch/src/train

python simple_inference.py \
    --model_path /path/to/VLM-FireWatch/models/moondream2_fire_detection_best \
    --test_json /path/to/unified_dataset/flame_vqa_test.json \
    --output_file inference_results.json
```

## Step 6: Use the Trained Model

### For Inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/models/moondream2_fire_detection_best",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/models/moondream2_fire_detection_best",
    trust_remote_code=True
)

# Run inference
image = Image.open("path/to/image.jpg")
question = "Is there a fire in this image?"
answer = model.answer_question(image, question, tokenizer)
print(answer)
```

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory:**
   - Reduce `batch_size` (try 4 or 2)
   - Use gradient accumulation
   - Use `torch.float16` instead of `float32`

2. **Dataset Not Found:**
   - Verify paths in `dataset_configs.py`
   - Ensure datasets are downloaded and extracted
   - Check file permissions

3. **Model Loading Errors:**
   - Ensure `trust_remote_code=True` is set
   - Check that all model files are present
   - Verify Python and transformers library versions

4. **Slow Training:**
   - Ensure GPU is being used: `torch.cuda.is_available()` should return `True`
   - Increase `num_workers` in DataLoader
   - Use mixed precision training

## Expected Training Time

- **With GPU (RTX 3090/4090 or similar):** ~2-4 hours for 3 epochs
- **With GPU (lower-end):** ~6-12 hours for 3 epochs
- **CPU only:** Not recommended (would take days)

## Summary

The complete workflow:
1. ✅ Set up Python environment with required packages
2. ✅ Download and configure dataset paths
3. ✅ Run `process_datasets.py` to create unified VQA dataset
4. ✅ Run `finetune_moondream2.py` to train the model
5. ✅ Find trained model in `models/moondream2_fire_detection_best/`
6. ✅ Test with `simple_inference.py`
7. ✅ Use the model for your fire detection tasks!

---

**Note:** Make sure to update all file paths in the commands above to match your actual directory structure.


