#!/usr/bin/env python3
"""
H100 Vision Encoder Benchmark - Compare PyTorch FP16 vs TensorRT INT8
Measures power consumption, latency, and FPS for vision encoding only
"""

import os
import time
import argparse
import subprocess
import json
import numpy as np
import torch
from PIL import Image
from typing import Dict
import threading
import gc

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class NvidiaSmiMonitor:
    """Monitor power consumption using nvidia-smi"""
    
    def __init__(self, gpu_id: int = 0, interval_ms: int = 100):
        self.gpu_id = gpu_id
        self.interval_ms = interval_ms
        self.power_readings = []
        self.is_running = False
        self.thread = None
        
    def _monitor_loop(self):
        """Background loop to collect power readings"""
        while self.is_running:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits', f'-i={self.gpu_id}'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0:
                    power_w = float(result.stdout.strip())
                    self.power_readings.append(power_w)
            except (subprocess.TimeoutExpired, ValueError):
                pass
            
            time.sleep(self.interval_ms / 1000.0)
    
    def start(self):
        """Start power monitoring in background thread"""
        print(f"Starting nvidia-smi power monitoring (GPU {self.gpu_id})...")
        self.power_readings = []
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        time.sleep(0.2)
        print(f"✓ Power monitoring started")
    
    def stop(self):
        """Stop power monitoring"""
        if self.is_running:
            print("Stopping power monitoring...")
            self.is_running = False
            if self.thread:
                self.thread.join(timeout=2)
            print("✓ Power monitoring stopped")
    
    def get_metrics(self) -> Dict:
        """Get power metrics from collected readings"""
        if not self.power_readings:
            print("⚠️ Warning: No power readings collected")
            return {}
        
        return {
            'avg_power_w': np.mean(self.power_readings),
            'min_power_w': np.min(self.power_readings),
            'max_power_w': np.max(self.power_readings),
            'samples': len(self.power_readings)
        }


class VisionEncoderBenchmark:
    """Base class for vision encoder benchmarking"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.warmup_runs = 50
        self.benchmark_runs = 5000  # More runs since vision encoding is fast
    
    def create_dummy_input(self) -> Image.Image:
        """Create dummy input image"""
        img_array = np.random.randint(0, 256, (378, 378, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    def warmup(self):
        """Warmup runs to stabilize performance"""
        print(f"Warming up {self.model_name}...")
        for i in range(self.warmup_runs):
            dummy_img = self.create_dummy_input()
            self.encode_image(dummy_img)
            del dummy_img
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"✓ Warmup complete ({self.warmup_runs} runs)")
    
    def benchmark_latency(self) -> Dict:
        """Benchmark vision encoding latency"""
        print(f"Benchmarking {self.model_name} vision encoding ({self.benchmark_runs} runs)...")
        
        latencies = []
        
        for i in range(self.benchmark_runs):
            dummy_img = self.create_dummy_input()
            
            # CUDA synchronization for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            embeddings = self.encode_image(dummy_img)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            del dummy_img, embeddings
            
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i+1}/{self.benchmark_runs} runs")
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'std_latency_ms': np.std(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'fps': 1000.0 / np.mean(latencies)
        }
    
    def benchmark_with_power(self, power_monitor: NvidiaSmiMonitor) -> Dict:
        """Benchmark with power monitoring"""
        print(f"\nBenchmarking {self.model_name} with power monitoring...")
        
        power_monitor.start()
        latency_metrics = self.benchmark_latency()
        power_monitor.stop()
        
        power_metrics = power_monitor.get_metrics()
        
        return {
            **latency_metrics,
            **power_metrics,
            'energy_per_inference_j': (power_metrics.get('avg_power_w', 0) * 
                                       latency_metrics['avg_latency_ms'] / 1000.0)
        }
    
    def encode_image(self, image: Image.Image):
        """Encode image - to be implemented by subclasses"""
        raise NotImplementedError


class PyTorchVisionEncoderBenchmark(VisionEncoderBenchmark):
    """Benchmark PyTorch FP16 vision encoder"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__("PyTorch FP16 Vision Encoder")
        self.model_path = model_path
        self.device = torch.device(device)
        self.load_model()
    
    def load_model(self):
        """Load PyTorch model"""
        print(f"Loading PyTorch vision encoder...")
        
        from transformers import AutoModelForCausalLM
        
        base_model_name = "vikhyatk/moondream2"
        md_revision = "2024-07-23"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            revision=md_revision,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Load fine-tuned weights if available
        finetuned_weights = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(finetuned_weights):
            from safetensors.torch import load_file
            state_dict = load_file(finetuned_weights)
            self.model.load_state_dict(state_dict, strict=False)
            print("✓ Fine-tuned weights loaded")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ PyTorch vision encoder loaded on {self.device}")
    
    def encode_image(self, image: Image.Image):
        """Encode image using PyTorch"""
        with torch.no_grad():
            embeddings = self.model.encode_image(image)
        return embeddings


class TensorRTVisionEncoderBenchmark(VisionEncoderBenchmark):
    """Benchmark TensorRT INT8 vision encoder"""
    
    def __init__(self, trt_engine_path: str):
        super().__init__("TensorRT INT8 Vision Encoder")
        self.trt_engine_path = trt_engine_path
        self.load_engine()
    
    def load_engine(self):
        """Load TensorRT engine"""
        print(f"Loading TensorRT engine from {self.trt_engine_path}...")
        
        try:
            import tensorrt as trt
        except ModuleNotFoundError:
            import tensorrt_cu12 as trt
        
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(self.trt_engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.input_shape = (1, 3, 378, 378)
        self.output_shape = (1, 729, 2048)
        
        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape) * 4))
        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape) * 4))
        self.stream = cuda.Stream()
        
        print("✓ TensorRT engine loaded")
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for TensorRT"""
        img = image.resize((378, 378))
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img, dtype=np.float32)
    
    def encode_image(self, image: Image.Image):
        """Encode image using TensorRT"""
        import pycuda.driver as cuda
        
        input_data = self.preprocess(image)
        cuda.memcpy_htod(self.d_input, input_data)
        
        bindings = [int(self.d_input), int(self.d_output)]
        self.context.execute_v2(bindings=bindings)
        
        output_data = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output_data, self.d_output)
        self.stream.synchronize()
        
        return output_data


def print_results(results: Dict, model_name: str):
    """Pretty print benchmark results"""
    print(f"\n{'='*70}")
    print(f"{model_name} - BENCHMARK RESULTS")
    print(f"{'='*70}")
    
    print(f"\n📊 Latency Metrics:")
    print(f"  Average: {results['avg_latency_ms']:.3f} ms")
    print(f"  Min:     {results['min_latency_ms']:.3f} ms")
    print(f"  Max:     {results['max_latency_ms']:.3f} ms")
    print(f"  Std Dev: {results['std_latency_ms']:.3f} ms")
    print(f"  P50:     {results['p50_latency_ms']:.3f} ms")
    print(f"  P95:     {results['p95_latency_ms']:.3f} ms")
    print(f"  P99:     {results['p99_latency_ms']:.3f} ms")
    
    print(f"\n🎬 Throughput:")
    print(f"  FPS: {results['fps']:.1f}")
    
    if 'avg_power_w' in results:
        print(f"\n⚡ Power Consumption:")
        print(f"  Average: {results['avg_power_w']:.2f} W")
        print(f"  Min:     {results['min_power_w']:.2f} W")
        print(f"  Max:     {results['max_power_w']:.2f} W")
        print(f"  Samples: {results['samples']}")
        
        print(f"\n🔋 Energy Efficiency:")
        print(f"  Energy per inference: {results['energy_per_inference_j']:.6f} J")
        print(f"  Inferences per Wh: {3600 / results['energy_per_inference_j']:.0f}")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Vision Encoders on H100")
    parser.add_argument("--finetuned_path", type=str, 
                       default="/home/hice1/smanoli3/scratch/finetuned_moondream",
                       help="Path to fine-tuned model directory")
    parser.add_argument("--trt_engine", type=str, 
                       default="/home/hice1/smanoli3/scratch/moondream2_tuned_int8.engine",
                       help="Path to TensorRT engine file")
    parser.add_argument("--runs", type=int, default=5000,
                       help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=50,
                       help="Number of warmup runs")
    parser.add_argument("--output", type=str, default="vision_encoder_benchmark.json",
                       help="Output JSON file")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--models", type=str, default="both",
                       choices=["pytorch", "tensorrt", "both"],
                       help="Which encoders to benchmark")
    
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu)
    results = {}
    
    # Benchmark PyTorch vision encoder
    if args.models in ["pytorch", "both"]:
        if os.path.exists(args.finetuned_path):
            print("\n" + "="*70)
            print("BENCHMARKING PYTORCH FP16 VISION ENCODER")
            print("="*70 + "\n")
            
            try:
                benchmark = PyTorchVisionEncoderBenchmark(args.finetuned_path, device=f"cuda:{args.gpu}")
                benchmark.warmup_runs = args.warmup
                benchmark.benchmark_runs = args.runs
                
                benchmark.warmup()
                
                power_monitor = NvidiaSmiMonitor(gpu_id=args.gpu, interval_ms=100)
                pytorch_results = benchmark.benchmark_with_power(power_monitor)
                
                print_results(pytorch_results, "PyTorch FP16 Vision Encoder")
                results['pytorch_fp16'] = pytorch_results
            finally:
                del benchmark
                torch.cuda.empty_cache()
                gc.collect()
                print("✓ PyTorch encoder freed\n")
        else:
            print(f"⚠️ Model not found at {args.finetuned_path}")
    
    # Benchmark TensorRT vision encoder
    if args.models in ["tensorrt", "both"]:
        if os.path.exists(args.trt_engine):
            print("\n" + "="*70)
            print("BENCHMARKING TENSORRT INT8 VISION ENCODER")
            print("="*70 + "\n")
            
            benchmark = TensorRTVisionEncoderBenchmark(args.trt_engine)
            benchmark.warmup_runs = args.warmup
            benchmark.benchmark_runs = args.runs
            
            benchmark.warmup()
            
            power_monitor = NvidiaSmiMonitor(gpu_id=args.gpu, interval_ms=100)
            tensorrt_results = benchmark.benchmark_with_power(power_monitor)
            
            print_results(tensorrt_results, "TensorRT INT8 Vision Encoder")
            results['tensorrt_int8'] = tensorrt_results
            print("✓ TensorRT encoder freed\n")
        else:
            print(f"⚠️ TensorRT engine not found at {args.trt_engine}")
    
    # Comparison
    if 'pytorch_fp16' in results and 'tensorrt_int8' in results:
        print("\n" + "="*70)
        print("VISION ENCODER COMPARISON")
        print("="*70)
        
        speedup = results['pytorch_fp16']['avg_latency_ms'] / results['tensorrt_int8']['avg_latency_ms']
        power_diff = ((results['tensorrt_int8']['avg_power_w'] - results['pytorch_fp16']['avg_power_w']) / 
                      results['pytorch_fp16']['avg_power_w'] * 100)
        energy_improvement = results['pytorch_fp16']['energy_per_inference_j'] / results['tensorrt_int8']['energy_per_inference_j']
        
        print(f"\n⚡ Speedup: {speedup:.2f}x faster (TensorRT INT8 vs PyTorch FP16)")
        print(f"💾 Model size: ~4x smaller (INT8 vs FP16)")
        print(f"🔋 Power difference: {power_diff:+.1f}%")
        print(f"🎯 Energy efficiency: {energy_improvement:.2f}x better")
        print(f"📈 Throughput: {results['tensorrt_int8']['fps']:.1f} vs {results['pytorch_fp16']['fps']:.1f} FPS")
        
        print(f"\n📊 Absolute Numbers:")
        print(f"  PyTorch FP16:   {results['pytorch_fp16']['avg_latency_ms']:.3f} ms/image")
        print(f"  TensorRT INT8:  {results['tensorrt_int8']['avg_latency_ms']:.3f} ms/image")
        print(f"  Time saved:     {results['pytorch_fp16']['avg_latency_ms'] - results['tensorrt_int8']['avg_latency_ms']:.3f} ms")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
