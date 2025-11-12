#!/bin/bash
# Script to run object detection evaluation on allocated GPU

# Allocate GPU (if not already allocated)
# salloc -N 1 -c 2 -G H100:1 --mem-per-cpu=65536 -t 120

# Navigate to script directory
cd /home/hice1/mchidambaram7/scratch/VLM-FireWatch/src/train

# Run object detection evaluation
python evaluate_object_detection.py \
    --model_path /home/hice1/mchidambaram7/scratch/VLM-FireWatch/models/moondream2_fire_detection_best \
    --test_json /home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_test.json \
    --output_file object_detection_results.json \
    --max_samples 100 \
    --device cuda

echo "Evaluation complete! Results saved to object_detection_results.json"

