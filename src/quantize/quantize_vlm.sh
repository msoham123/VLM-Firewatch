#!/bin/bash
set -e

# Set paths
model_dir="/home/hice1/smanoli3/scratch/moondream2_fire_detection_best"
output_dir="/home/hice1/smanoli3/scratch/moondream2_fire_q4f16_1"

echo "=== MoonDream2 INT4 Quantization ==="

# Step 1: Convert weights (warning is safe to ignore)
echo "[1/3] Converting weights..."
python -m mlc_llm convert_weight \
    $model_dir \
    --quantization q4f16_1 \
    --device cuda \
    -o $output_dir

# Step 2: Generate config
echo "[2/3] Generating config..."
python -m mlc_llm gen_config \
    $model_dir \
    --quantization q4f16_1 \
    --conv-template LM \
    -o $output_dir

# Step 3: Compile
mkdir -p "${output_dir}/libs"
echo "[3/3] Compiling..."
python -m mlc_llm compile \
    "${output_dir}/mlc-chat-config.json" \
    --device cuda \
    --opt "cuda_max_num_threads=512;max_num_sequence=2048" \
    -o "${output_dir}/libs/moondream2_fire_q4f16_1-cuda.so"

echo "✓ Complete!"
