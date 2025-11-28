#!/usr/bin/env python3
"""
Export base MoonDream2 model to ONNX (to test the pipeline)
"""

import torch
from transformers import AutoModelForCausalLM
import os
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset_configs import moondream_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use official MoonDream model from HuggingFace
model_id = "vikhyatk/moondream2"
output_onnx = moondream_config["onnx"]

# Step 1: Load base model architecture from HuggingFace
logger.info("Loading base Moondream2 architecture...")
base_model_name = "vikhyatk/moondream2"
md_revision = "2024-07-23"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    revision=md_revision,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Step 2: Load fine-tuned weights
logger.info("Loading fine-tuned weights...")
model_path = moondream_config["fine_tuned"]
finetuned_weights_path = os.path.join(model_path, "pytorch_model.bin")

if os.path.exists(finetuned_weights_path):
    state_dict = torch.load(finetuned_weights_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    logger.info("Fine-tuned weights loaded successfully!")
else:
    # Try alternative weight file names
    alt_paths = [
        os.path.join(model_path, "model.safetensors"),
        os.path.join(model_path, "pytorch_model.safetensors"),
    ]
    
    loaded = False
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            from safetensors.torch import load_file
            state_dict = load_file(alt_path)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Fine-tuned weights loaded from {alt_path}!")
            loaded = True
            break
    
    if not loaded:
        logger.warning("No fine-tuned weights found! Using base model.")
        logger.warning(f"Looked in: {model_path}")

# Step 3: Move to device
logger.info("Model loaded and ready for ONNX conversion!")

# Extract the vision encoder
vision_model = model.vision_encoder

# Move ONLY the vision encoder to float32 on CPU for ONNX export:
vision_model = vision_model.to(torch.float32).cpu()

print(f"Vision encoder: {type(vision_model)}")
print("Exporting to ONNX...")

# Dummy input
dummy_input = torch.randn(1, 3, 378, 378, dtype=torch.float32)

# Test inference first
with torch.no_grad():
    output = vision_model(dummy_input)
    print(f"✓ Vision encoder works, output shape: {output.shape}")

# Export to ONNX
try:
    torch.onnx.export(
        vision_model,
        dummy_input,
        output_onnx,
        input_names=['image'],
        output_names=['features'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True
    )
    print(f"✓ Exported to: {output_onnx}")
    print(f"Size: {os.path.getsize(output_onnx) / (1024**2):.2f} MB")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(output_onnx)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")
    
except Exception as e:
    print(f"ONNX export failed: {e}")
    
    print("\nTrying TorchScript instead...")
    traced = torch.jit.trace(vision_model, dummy_input)
    pt_path = output_onnx.replace('.onnx', '.pt')
    traced.save(pt_path)
    print(f"✓ Saved TorchScript to: {pt_path}")
    print(f"Size: {os.path.getsize(pt_path) / (1024**2):.2f} MB")
