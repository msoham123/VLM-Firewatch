#!/bin/bash
set -e

rm -rf ~/.cache/torch_extensions/

echo "=== MoonDream2 INT4 Quantization (CPU Mode) ==="

model_dir="/home/hice1/smanoli3/scratch/moondream2_fire_detection_best"
output_dir="/home/hice1/smanoli3/scratch/moondream2_fire_q4f16_1"

echo "Model: $model_dir"
echo "Output: $output_dir"
echo ""

# Step 1: Convert weights - SPECIFY safetensors format explicitly
echo "[1/3] Converting weights..."
python -m mlc_llm convert_weight \
    $model_dir \
    --quantization q4f16_1 \
    --source-format huggingface-safetensor \
    --device cpu \
    -o $output_dir

# Step 2: Generate config
echo ""
echo "[2/3] Generating config..."
python -m mlc_llm gen_config \
    $model_dir \
    --quantization q4f16_1 \
    --conv-template LM \
    -o $output_dir

# Step 3: Compile for Jetson
mkdir -p "${output_dir}/libs"
echo ""
echo "[3/3] Compiling for Jetson..."
python -m mlc_llm compile \
    "${output_dir}/mlc-chat-config.json" \
    --device cuda \
    --target cuda \
    --opt "cuda_max_num_threads=512;max_num_sequence=2048" \
    -o "${output_dir}/libs/moondream2_fire_q4f16_1-cuda.so"

echo ""
echo "✓ Complete!"
du -sh $model_dir $output_dir
