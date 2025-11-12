from safetensors.torch import load_file
from pathlib import Path

model_path = "/home/hice1/smanoli3/scratch/moondream2_fire_detection_best"
safetensors_files = list(Path(model_path).glob("*.safetensors"))
state_dict = load_file(str(safetensors_files[0]))

print("All tensor names and shapes:")
for name, tensor in state_dict.items():
    print(f"{name}: shape={list(tensor.shape)}, dtype={tensor.dtype}, numel={tensor.numel():,}")
