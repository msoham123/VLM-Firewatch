#!/usr/bin/env python3
"""
Convert ONNX to TensorRT with INT8 - with optimization profile
"""

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
import numpy as np
from PIL import Image
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset_configs import unified_config, moondream_config
from src.train.dataloaders import create_dataloaders
import random

# Paths
onnx_path = moondream_config["onnx"]
trt_engine_path = moondream_config["quantized"]
calibration_json = os.path.join(unified_config["src"], "flame_vqa_train.json")

IMAGE_SIZE = 378

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_size=1):  # Use batch_size=1 for calibration
        trt.IInt8EntropyCalibrator2.__init__(self)

        num_calibration_images = 5000
        
        # Create temporary dataloader just to get image paths
        train_loader, _, _ = create_dataloaders(
            train_json=calibration_json,
            val_json=calibration_json,  # Doesn't matter, we only use train
            test_json=calibration_json,
            batch_size=1,
            num_workers=0,
            mode='vqa',
            image_size=378,
            pin_memory=False
        )
        
        # Extract all image paths from dataset
        all_image_paths = []
        for item in train_loader.dataset.data:
            all_image_paths.append(item['metadata']['original_path'])
        
        # Shuffle and take first N
        random.seed(42)  # For reproducibility
        random.shuffle(all_image_paths)
        self.image_files = all_image_paths[:num_calibration_images]
        
        print(f"Loaded {len(self.image_files)} shuffled images for calibration")
        
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = cuda.mem_alloc(batch_size * 3 * IMAGE_SIZE * IMAGE_SIZE * 4)
        
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_files):
            return None
        
        batch = []
        for i in range(self.batch_size):
            try:
                img_path = self.image_files[self.current_index + i]
                
                if not os.path.exists(img_path):
                    batch.append(np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
                    continue
                
                img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
                img = np.array(img).astype(np.float32) / 255.0
                img = (img - 0.5) / 0.5
                img = np.transpose(img, (2, 0, 1))  # ← ADD THIS: HWC to CHW
                batch.append(img)
            except Exception as e:
                print(f"Error: {e}")
                batch.append(np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
        
        batch = np.array(batch).astype(np.float32)
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        
        if self.current_index % 100 == 0:
            print(f"Calibration: {self.current_index}/{len(self.image_files)}")
        
        return [self.device_input]
    
    def read_calibration_cache(self):
        if os.path.exists('calibration.cache'):
            print("Using cached calibration")
            with open('calibration.cache', 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open('calibration.cache', 'wb') as f:
            f.write(cache)

# Build
print("Checking CUDA...")
print(f"Using: {cuda.Device(0).name()}\n")

print("Building TensorRT INT8 engine...")
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

print(f"Parsing: {onnx_path}")
with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit(1)

print("✓ ONNX parsed\n")

# Config
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

# ADD OPTIMIZATION PROFILE (fix for dynamic shapes)
profile = builder.create_optimization_profile()
input_name = network.get_input(0).name
print(f"Input tensor: {input_name}")

# Set shape: min, optimal, max (batch, channels, height, width)
profile.set_shape(
    input_name,
    min=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
    opt=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
    max=(1, 3, IMAGE_SIZE, IMAGE_SIZE)
)
config.add_optimization_profile(profile)

# INT8
config.set_flag(trt.BuilderFlag.INT8)

print("Setting up calibration...")
calibrator = ImageCalibrator(batch_size=1)
config.int8_calibrator = calibrator

print("\nBuilding (10-15 min)...\n")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("❌ Failed")
    exit(1)

with open(trt_engine_path, 'wb') as f:
    f.write(serialized_engine)

print(f"\n{'='*60}")
print(f"✓ INT8 engine created!")
print(f"Output: {trt_engine_path}")
print(f"Size: {os.path.getsize(trt_engine_path)/(1024**2):.2f} MB")
print(f"{'='*60}")
