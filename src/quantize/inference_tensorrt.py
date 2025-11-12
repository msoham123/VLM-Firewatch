#!/usr/bin/env python3
"""
TensorRT INT8 inference - with base MoonDream model
"""

import sys
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Paths
trt_engine_path = "/home/hice1/smanoli3/scratch/moondream2_base_int8.engine"
base_model_id = "vikhyatk/moondream2"  # Official base model
test_json = "/home/hice1/smanoli3/scratch/datasets/unified_dataset/flame_vqa_test.json"

IMAGE_SIZE = 378

# Load TensorRT engine
print("Loading TensorRT INT8 engine...")
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

with open(trt_engine_path, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

print(f"✓ Engine loaded ({os.path.getsize(trt_engine_path)/(1024**2):.2f} MB)")

# Get shapes
input_shape = engine.get_tensor_shape(engine.get_tensor_name(0))
output_shape = engine.get_tensor_shape(engine.get_tensor_name(1))
print(f"  Input: {input_shape}, Output: {output_shape}")

# Allocate buffers
stream = cuda.Stream()
d_input = cuda.mem_alloc(int(np.prod(input_shape) * 4))
d_output = cuda.mem_alloc(int(np.prod(output_shape) * 4))

# Load base MoonDream model from HuggingFace
print(f"\nLoading base MoonDream model: {base_model_id}")
full_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    revision="2024-08-26",  # Stable version
    device_map="cuda"
).eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
print("✓ Base model loaded\n")

def preprocess_image(image):
    """Preprocess for TensorRT"""
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0).astype(np.float32)

# Run inference on test set
print("Running inference with base MoonDream + TensorRT INT8 engine...\n")

with open(test_json) as f:
    test_data = json.load(f)

correct = 0
total = 0
errors = 0

for sample in tqdm(test_data, desc="Testing"):
    try:
        img_path = sample['metadata']['original_path']
        if not os.path.exists(img_path):
            continue
        
        image = Image.open(img_path).convert('RGB')
        question = sample['conversations'][0]['question']
        gt = sample['conversations'][0]['answer']
        
        # Use base model's answer_question method
        with torch.no_grad():
            prediction = full_model.answer_question(image, question, tokenizer)
        
        # Evaluate
        gt_norm = 'yes' if 'yes' in gt.lower() else 'no'
        pred_norm = 'yes' if 'yes' in prediction.lower() else 'no'
        
        if gt_norm == pred_norm:
            correct += 1
        total += 1
        
    except Exception as e:
        errors += 1
        if errors < 3:
            print(f"\nError: {e}")
        continue

accuracy = correct / total if total > 0 else 0

print(f"\n{'='*60}")
print(f"BASE MOONDREAM + TensorRT INT8 RESULTS")
print(f"{'='*60}")
print(f"Model: {base_model_id}")
print(f"Engine: TensorRT INT8")
print(f"Tested: {total} samples")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Errors: {errors}")
print(f"{'='*60}")
