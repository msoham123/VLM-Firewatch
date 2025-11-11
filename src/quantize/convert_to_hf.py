# Create: convert_to_hf.py
import torch
from safetensors.torch import load_file

model_dir = "/home/hice1/smanoli3/scratch/moondream2_fire_detection_best"
output_dir = "/home/hice1/smanoli3/scratch/moondream2_fire_hf"

# Load the safetensors file
state_dict = load_file(f"{model_dir}/model-002.safetensors")

# Save as standard pytorch format
import os
os.makedirs(output_dir, exist_ok=True)
torch.save(state_dict, f"{output_dir}/pytorch_model.bin")

# Copy config files
import shutil
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", 
          "special_tokens_map.json", "vocab.json", "merges.txt"]:
    if os.path.exists(f"{model_dir}/{f}"):
        shutil.copy(f"{model_dir}/{f}", f"{output_dir}/{f}")

print(f"✓ Converted to HF format: {output_dir}")
