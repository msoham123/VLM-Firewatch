#!/bin/bash
# Script to run object detection evaluation on allocated GPU
# Run this script AFTER allocating GPU with: salloc -N 1 -c 2 -G H100:1 --mem-per-cpu=65536 -t 120

# Load required modules
module load nvhpc/24.5

# Navigate to script directory
cd /home/hice1/mchidambaram7/scratch/VLM-FireWatch/src/train

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Run object detection evaluation
echo "Starting object detection evaluation..."
python evaluate_object_detection.py \
    --model_path /home/hice1/mchidambaram7/scratch/VLM-FireWatch/models/moondream2_fire_detection_best \
    --test_json /home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_test.json \
    --output_file object_detection_results.json \
    --max_samples 100 \
    --device cuda

echo "Evaluation complete! Results saved to object_detection_results.json"

