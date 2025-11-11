import torch
import os
import json
from pathlib import Path
from safetensors.torch import load_file, save_file
import sys
import shutil
import glob
from config import model_path, output_path, quantization_bits

class MoonDreamQuantizer:
    def __init__(self, model_path, output_path, quantization_bits=8):
        self.model_path = model_path
        self.output_path = output_path
        self.quantization_bits = quantization_bits
        
    def quantize_model(self):
        """Quantize MoonDream2 model to INT8 for Jetson deployment"""
        print(f"=== MoonDream2 Quantization for Jetson Orin Nano ===")
        print(f"Model: {self.model_path}")
        print(f"Output: {self.output_path}")
        print(f"Quantization: INT{self.quantization_bits}")
        print()
        
        # Step 1: Load model configuration
        print("[1/4] Loading model configuration...")
        try:
            with open(os.path.join(self.model_path, 'config.json'), 'r') as f:
                config = json.load(f)
            print(f"✓ Model config loaded: {config.get('model_type', 'unknown')}")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return False
            
        # Step 2: Load weights
        print("\n[2/4] Loading model weights...")
        try:
            safetensors_files = list(Path(self.model_path).glob("*.safetensors"))
            if not safetensors_files:
                print("❌ No safetensors files found")
                return False
                
            weight_file = safetensors_files[0]
            print(f"Loading weights from: {weight_file.name}")
            state_dict = load_file(str(weight_file))
            print(f"✓ Loaded {len(state_dict)} weight tensors")
            
            # Analyze tensor types
            dtype_counts = {}
            for tensor in state_dict.values():
                dtype_str = str(tensor.dtype)
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
            print(f"Tensor dtypes: {dtype_counts}")
            
            total_params = sum(p.numel() for p in state_dict.values())
            total_size_mb = sum(p.numel() * p.element_size() for p in state_dict.values()) / (1024**2)
            print(f"Total parameters: {total_params:,}")
            print(f"Original size: {total_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")
            return False
        
        # Step 3: Apply quantization
        print(f"\n[3/4] Applying INT{self.quantization_bits} quantization...")
        quantized_state_dict = {}
        scales_dict = {}
        
        quantized_params = 0
        skipped_params = 0
        original_bytes = 0
        quantized_bytes = 0
        
        for name, tensor in state_dict.items():
            # Quantize all float-type tensors with > 1000 elements and 2+ dimensions
            # This includes bfloat16, float16, and float32
            is_float_type = tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]
            is_large_enough = tensor.numel() > 1000
            is_multi_dim = len(tensor.shape) >= 2
            
            if is_float_type and is_large_enough and is_multi_dim:
                # Quantize to INT8
                quantized_tensor, scale = self._quantize_int8_compressed(tensor)
                quantized_state_dict[name] = quantized_tensor
                scales_dict[f"{name}.scale"] = scale
                
                orig_bytes = tensor.numel() * tensor.element_size()
                quant_bytes = tensor.numel() * 1 + scale.numel() * scale.element_size()  # INT8 + scale
                
                original_bytes += orig_bytes
                quantized_bytes += quant_bytes
                quantized_params += 1
                
                if quantized_params <= 3:  # Show first 3
                    print(f"  ✓ {name}")
                    print(f"     {list(tensor.shape)} | {tensor.dtype} → INT8")
                    print(f"     {orig_bytes/(1024**2):.2f} MB → {quant_bytes/(1024**2):.2f} MB")
            else:
                # Keep in FP16 for inference compatibility
                if tensor.dtype == torch.bfloat16:
                    # Convert bfloat16 to float16 for better hardware support
                    quantized_state_dict[name] = tensor.float().half()
                    quant_bytes = tensor.numel() * 2
                elif tensor.dtype == torch.float32:
                    quantized_state_dict[name] = tensor.half()
                    quant_bytes = tensor.numel() * 2
                else:
                    quantized_state_dict[name] = tensor
                    quant_bytes = tensor.numel() * tensor.element_size()
                
                quantized_bytes += quant_bytes
                original_bytes += tensor.numel() * tensor.element_size()
                skipped_params += 1
        
        # Add scales to state dict
        quantized_state_dict.update(scales_dict)
        
        if quantized_params > 3:
            print(f"  ... and {quantized_params - 3} more tensors")
        
        quantized_size_mb = quantized_bytes / (1024**2)
        compression_ratio = original_bytes / quantized_bytes if quantized_bytes > 0 else 1.0
        
        print(f"\n✓ Quantization summary:")
        print(f"  Quantized (INT8): {quantized_params} tensors")
        print(f"  Kept in FP16: {skipped_params} tensors")
        print(f"  Original size: {total_size_mb:.2f} MB")
        print(f"  Quantized size: {quantized_size_mb:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Memory savings: {(1 - quantized_size_mb/total_size_mb)*100:.1f}%")
        
        if quantized_params == 0:
            print("\n⚠ WARNING: No tensors were quantized!")
            return False
        
        # Step 4: Save quantized model
        print(f"\n[4/4] Saving quantized model...")
        os.makedirs(self.output_path, exist_ok=True)
        
        try:
            # Save quantized weights
            output_file = os.path.join(self.output_path, "model.safetensors")
            save_file(quantized_state_dict, output_file)
            actual_size_mb = os.path.getsize(output_file) / (1024**2)
            print(f"✓ Saved weights: {output_file}")
            print(f"  File size on disk: {actual_size_mb:.2f} MB")

            # Copy ALL files (use glob to get everything)
            print("\nCopying files:")
            for pattern in ['*.json', '*.txt', '*.py']:
                for src_file in glob.glob(os.path.join(model_path, pattern)):
                    filename = os.path.basename(src_file)
                    shutil.copy2(src_file, os.path.join(output_path, filename))
                    print(f"  ✓ {filename}")

            # Save quantization config
            quant_config = {
                "quantization_bits": self.quantization_bits,
                "quantization_method": "pytorch_int8_symmetric",
                "original_size_mb": float(total_size_mb),
                "quantized_size_mb": float(quantized_size_mb),
                "actual_file_size_mb": float(actual_size_mb),
                "compression_ratio": float(compression_ratio),
                "quantized_params": quantized_params,
                "total_params": len(state_dict),
                "target_device": "jetson_orin_nano",
                "requires_dequantization": True,
            }
            
            with open(os.path.join(self.output_path, 'quantization_config.json'), 'w') as f:
                json.dump(quant_config, f, indent=2)
            print("✓ Saved quantization config")
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n{'='*70}")
        print("✓ QUANTIZATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Original model:  {total_size_mb:.2f} MB")
        print(f"Quantized model: {actual_size_mb:.2f} MB")
        print(f"Space saved:     {total_size_mb - actual_size_mb:.2f} MB ({(1-actual_size_mb/total_size_mb)*100:.1f}%)")
        print(f"Output location: {self.output_path}")
        print(f"{'='*70}")
        return True
    
    def _quantize_int8_compressed(self, tensor):
        """Quantize tensor to INT8 (symmetric quantization)"""
        # Convert to float32 for quantization math
        tensor_float = tensor.float()
        
        # Calculate scale (symmetric quantization)
        max_val = torch.max(torch.abs(tensor_float))
        if max_val == 0:
            max_val = torch.tensor(1.0)
        scale = max_val / 127.0
        
        # Quantize to INT8
        quantized = torch.round(tensor_float / scale).clamp(-127, 127).to(torch.int8)
        
        # Return INT8 tensor and FP16 scale
        return quantized, scale.half()


def main():    
    # Clean up old output if exists
    if os.path.exists(output_path):
        print(f"Removing existing output: {output_path}\n")
        shutil.rmtree(output_path)
    
    quantizer = MoonDreamQuantizer(model_path, output_path, quantization_bits)
    success = quantizer.quantize_model()
    
    if not success:
        print("\n❌ Quantization failed")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
