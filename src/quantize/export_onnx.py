#!/usr/bin/env python3
"""
Export base MoonDream2 model to ONNX (to test the pipeline)
"""

import torch
from transformers import AutoModelForCausalLM
import os

# Use official MoonDream model from HuggingFace
model_id = "vikhyatk/moondream2"
output_onnx = "/home/hice1/smanoli3/scratch/moondream2_base.onnx"

print(f"Loading base MoonDream2 from HuggingFace: {model_id}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    revision="2024-08-26"  # Stable version
).cuda().eval()

print("✓ Model loaded")

# Get vision encoder
vision_model = model.vision_encoder

print(f"Vision encoder: {type(vision_model)}")
print("Exporting to ONNX...")

# Dummy input
dummy_input = torch.randn(1, 3, 378, 378, device='cuda', dtype=torch.float16)

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
