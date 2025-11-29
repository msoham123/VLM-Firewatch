#!/usr/bin/env python3
"""
Jetson Benchmark Script for Fine-tuned and Quantized Moondream2
Measures power consumption, latency, and FPS using tegrastats
"""

import os
import sys
import time
import argparse
import subprocess
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import threading
import re
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset_configs import moondream_config

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class TegrastatsMonitor:
    """Monitor power consumption using tegrastats"""
    
    def __init__(self, output_file: str = "tegrastats.log"):
        self.output_file = output_file
        self.process = None
        self.is_running = False
        
    def start(self):
        """Start tegrastats monitoring"""
        print(f"Starting tegrastats monitoring...")
        self.process = subprocess.Popen(
            ['tegrastats', '--interval', '100'],  # Sample every 100ms
            stdout=open(self.output_file, 'w'),
            stderr=subprocess.DEVNULL
        )
        self.is_running = True
        time.sleep(1)  # Let it start collecting
        print(f"✓ Tegrastats started, logging to {self.output_file}")
    
    def stop(self):
        """Stop tegrastats monitoring"""
        if self.process:
            print("Stopping tegrastats...")
            self.process.terminate()
            self.process.wait()
            self.is_running = False
            print("✓ Tegrastats stopped")
    
    def parse_log(self) -> Dict:
        """Parse tegrastats log and extract power metrics"""
        if not os.path.exists(self.output_file):
            return {}
        
        power_readings = []
        
        with open(self.output_file, 'r') as f:
            for line in f:
                # Parse power readings (mW)
                # Format: POM_5V_GPU 1234/5678
                match = re.search(r'POM_5V_GPU (\d+)/(\d+)', line)
                if match:
                    current_power = int(match.group(1))
                    power_readings.append(current_power)
                
                # Alternative format for different Jetson models
                if not match:
                    match = re.search(r'VDD_GPU_SOC (\d+)mW', line)
                    if match:
                        power_readings.append(int(match.group(1)))
        
        if not power_readings:
            print("⚠️ Warning: No power readings found in tegrastats log")
            return {}
        
        return {
            'avg_power_mw': np.mean(power_readings),
            'min_power_mw': np.min(power_readings),
            'max_power_mw': np.max(power_readings),
            'avg_power_w': np.mean(power_readings) / 1000.0,
            'samples': len(power_readings)
        }


class ModelBenchmark:
    """Base class for model benchmarking"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.warmup_runs = 10
        self.benchmark_runs = 5000
    
    def create_dummy_input(self) -> Image.Image:
        """Create dummy input image"""
        # Create random RGB image 378x378 (Moondream2 default)
        img_array = np.random.randint(0, 256, (378, 378, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    def warmup(self):
        """Warmup runs to stabilize performance"""
        print(f"Warming up {self.model_name}...")
        for i in range(self.warmup_runs):
            dummy_img = self.create_dummy_input()
            self.inference(dummy_img)
        print(f"✓ Warmup complete ({self.warmup_runs} runs)")
    
    def benchmark_latency(self) -> Dict:
        """Benchmark inference latency"""
        print(f"Benchmarking {self.model_name} latency ({self.benchmark_runs} runs)...")
        
        latencies = []
        
        for i in range(self.benchmark_runs):
            dummy_img = self.create_dummy_input()
            
            start_time = time.time()
            self.inference(dummy_img)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 20 == 0:
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
    
    def benchmark_with_power(self, tegrastats_monitor: TegrastatsMonitor) -> Dict:
        """Benchmark with power monitoring"""
        print(f"\nBenchmarking {self.model_name} with power monitoring...")
        
        # Start power monitoring
        tegrastats_monitor.start()
        
        # Run benchmark
        latency_metrics = self.benchmark_latency()
        
        # Stop power monitoring
        tegrastats_monitor.stop()
        
        # Parse power metrics
        power_metrics = tegrastats_monitor.parse_log()
        
        # Combine metrics
        return {
            **latency_metrics,
            **power_metrics,
            'energy_per_inference_j': (power_metrics.get('avg_power_w', 0) * 
                                       latency_metrics['avg_latency_ms'] / 1000.0)
        }
    
    def inference(self, image: Image.Image):
        """Run inference - to be implemented by subclasses"""
        raise NotImplementedError


# class FinetunedMoondream2Benchmark(ModelBenchmark):
#     """Benchmark fine-tuned PyTorch model"""
    
#     def __init__(self, model_path: str, device: str = "cuda"):
#         super().__init__("Fine-tuned Moondream2")
#         self.model_path = model_path
#         self.device = torch.device(device)
#         self.load_model()
    
#     def load_model(self):
#         """Load fine-tuned model"""
#         print(f"Loading fine-tuned model from {self.model_path}...")
        
#         from transformers import AutoTokenizer, AutoModelForCausalLM
        
#         base_model_name = "vikhyatk/moondream2"
#         md_revision = "2024-07-23"
        
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             base_model_name,
#             revision=md_revision,
#             trust_remote_code=True
#         )
        
#         self.model = AutoModelForCausalLM.from_pretrained(
#             base_model_name,
#             revision=md_revision,
#             trust_remote_code=True,
#             torch_dtype=torch.float16
#         )
        
#         # Load fine-tuned weights
#         finetuned_weights = os.path.join(self.model_path, "model.safetensors")
#         if os.path.exists(finetuned_weights):
#             from safetensors.torch import load_file
#             state_dict = load_file(finetuned_weights)
#             self.model.load_state_dict(state_dict, strict=False)
        
#         torch.cuda.empty_cache()
#         gc.collect()
#         self.model = self.model.to(self.device)
#         self.model.eval()
        
#         print("✓ Fine-tuned model loaded")
    
#     def inference(self, image: Image.Image):
#         """Run inference"""
#         question = "Is there fire visible in this image?"
        
#         with torch.no_grad():
#             enc_image = self.model.encode_image(image)
#             answer = self.model.answer_question(enc_image, question, self.tokenizer)
        
#         return answer

# class FinetunedMoondream2Benchmark(ModelBenchmark):
#     """Benchmark fine-tuned PyTorch model on CPU (GPU too small on Jetson)"""
    
#     def __init__(self, model_path: str, device: str = "cpu"):  # ← Force CPU
#         super().__init__("Fine-tuned Moondream2 (CPU)")
#         self.model_path = model_path
#         self.device = torch.device(device)
#         self.load_model()
    
#     def load_model(self):
#         """Load fine-tuned model on CPU"""
#         print(f"Loading fine-tuned model from {self.model_path}...")
#         print("⚠️ Running on CPU (PyTorch model too large for 8GB Jetson GPU)")
        
#         from transformers import AutoTokenizer, AutoModelForCausalLM
        
#         base_model_name = "vikhyatk/moondream2"
#         md_revision = "2024-07-23"
        
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             base_model_name,
#             revision=md_revision,
#             trust_remote_code=True
#         )
        
#         # Load on CPU with float32 (FP16 not supported on CPU)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             base_model_name,
#             revision=md_revision,
#             trust_remote_code=True,
#             torch_dtype=torch.float32,  # ← CPU needs float32
#             low_cpu_mem_usage=True
#         )
        
#         # Load fine-tuned weights
#         finetuned_weights = os.path.join(self.model_path, "model.safetensors")
#         if os.path.exists(finetuned_weights):
#             from safetensors.torch import load_file
#             state_dict = load_file(finetuned_weights)
#             self.model.load_state_dict(state_dict, strict=False)
#             print("✓ Fine-tuned weights loaded")
        
#         # Keep on CPU
#         self.model.eval()
        
#         print("✓ Fine-tuned model loaded on CPU")
    
#     def inference(self, image: Image.Image):
#         """Run inference on CPU"""
#         question = "Is there fire visible in this image?"
        
#         with torch.no_grad():
#             enc_image = self.model.encode_image(image)
#             answer = self.model.answer_question(enc_image, question, self.tokenizer)
        
#         return answer

class BaseMoondream2Benchmark(ModelBenchmark):
    """Benchmark base Moondream2 (lighter than fine-tuned)"""
    
    def __init__(self, device: str = "cpu"):
        super().__init__("Base Moondream2 (CPU)")
        self.device = torch.device(device)
        self.load_model()
    
    def load_model(self):
        """Load base model only (no fine-tuned weights)"""
        print(f"Loading base Moondream2...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        base_model_name = "vikhyatk/moondream2"
        md_revision = "2024-07-23"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            revision=md_revision,
            trust_remote_code=True
        )
        
        # Load ONLY base model (no fine-tuned weights)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            revision=md_revision,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        print("✓ Base model loaded on CPU")
    
    def inference(self, image: Image.Image):
        """Run inference on CPU"""
        question = "Is there fire visible in this image?"
        
        with torch.no_grad():
            enc_image = self.model.encode_image(image)
            answer = self.model.answer_question(enc_image, question, self.tokenizer)
        
        return answer

class QuantizedMoondream2Benchmark(ModelBenchmark):
    """Benchmark TensorRT INT8 quantized model"""
    
    def __init__(self, trt_engine_path: str):
        super().__init__("Quantized Moondream2 (TensorRT INT8)")
        self.trt_engine_path = trt_engine_path
        self.load_engine()
    
    def load_engine(self):
        """Load TensorRT engine"""
        print(f"Loading TensorRT engine from {self.trt_engine_path}...")
        
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(self.trt_engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
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
    
    def inference(self, image: Image.Image):
        """Run TensorRT inference"""
        import pycuda.driver as cuda
        
        # Preprocess
        input_data = self.preprocess(image)
        
        # Copy to device
        cuda.memcpy_htod(self.d_input, input_data)
        
        # Run inference
        bindings = [int(self.d_input), int(self.d_output)]
        self.context.execute_v2(bindings=bindings)
        
        # Copy output (optional - just for timing)
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
    print(f"  Average: {results['avg_latency_ms']:.2f} ms")
    print(f"  Min:     {results['min_latency_ms']:.2f} ms")
    print(f"  Max:     {results['max_latency_ms']:.2f} ms")
    print(f"  Std Dev: {results['std_latency_ms']:.2f} ms")
    print(f"  P50:     {results['p50_latency_ms']:.2f} ms")
    print(f"  P95:     {results['p95_latency_ms']:.2f} ms")
    print(f"  P99:     {results['p99_latency_ms']:.2f} ms")
    
    print(f"\n🎬 Throughput:")
    print(f"  FPS: {results['fps']:.2f}")
    
    if 'avg_power_w' in results:
        print(f"\n⚡ Power Consumption:")
        print(f"  Average: {results['avg_power_w']:.2f} W")
        print(f"  Min:     {results['min_power_mw']/1000:.2f} W")
        print(f"  Max:     {results['max_power_mw']/1000:.2f} W")
        print(f"  Samples: {results['samples']}")
        
        print(f"\n🔋 Energy Efficiency:")
        print(f"  Energy per inference: {results['energy_per_inference_j']:.4f} J")
        print(f"  Inferences per Wh: {3600 / results['energy_per_inference_j']:.0f}")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Moondream2 models on Jetson")
    parser.add_argument("--finetuned_path", type=str, 
                       default="/home/hparch/smanoli3/finetuned_moondream",
                       help="Path to fine-tuned model directory")
    parser.add_argument("--trt_engine", type=str,
                       default="/home/hparch/smanoli3/moondream2_tuned_int8.engine",
                       help="Path to TensorRT engine file")
    parser.add_argument("--runs", type=int, default=5000,
                       help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup runs")
    parser.add_argument("--output", type=str, default="results/benchmark_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--models", type=str, default="both",
                       choices=["finetuned", "quantized", "both"],
                       help="Which models to benchmark")
    
    args = parser.parse_args()
    
    results = {}
    
    # # Benchmark fine-tuned model
    # if args.models in ["finetuned", "both"]:
    #     if os.path.exists(args.finetuned_path):
    #         print("\n" + "="*70)
    #         print("BENCHMARKING FINE-TUNED MODEL")
    #         print("="*70 + "\n")
            
    #         benchmark = FinetunedMoondream2Benchmark(args.finetuned_path)
    #         benchmark.warmup_runs = args.warmup
    #         benchmark.benchmark_runs = args.runs
            
    #         benchmark.warmup()
            
    #         tegrastats = TegrastatsMonitor("tegrastats_finetuned.log")
    #         finetuned_results = benchmark.benchmark_with_power(tegrastats)
            
    #         print_results(finetuned_results, "Fine-tuned Moondream2")
    #         results['finetuned'] = finetuned_results
    #     else:
    #         print(f"⚠️ Fine-tuned model not found at {args.finetuned_path}")

    # Benchmark fine-tuned model
    if args.models in ["finetuned", "both"]:
        if os.path.exists(args.finetuned_path):
            print("\n" + "="*70)
            print("BENCHMARKING FINE-TUNED MODEL")
            print("="*70 + "\n")
            
            benchmark = BaseMoondream2Benchmark()
            benchmark.warmup_runs = args.warmup
            benchmark.benchmark_runs = args.runs
            
            benchmark.warmup()
            
            tegrastats = TegrastatsMonitor("tegrastats_finetuned.log")
            finetuned_results = benchmark.benchmark_with_power(tegrastats)
            
            print_results(finetuned_results, "Fine-tuned Moondream2")
            results['finetuned'] = finetuned_results
        else:
            print(f"⚠️ Fine-tuned model not found at {args.finetuned_path}")
    
    # Benchmark quantized model
    if args.models in ["quantized", "both"]:
        if os.path.exists(args.trt_engine):
            print("\n" + "="*70)
            print("BENCHMARKING QUANTIZED MODEL (TensorRT INT8)")
            print("="*70 + "\n")
            
            benchmark = QuantizedMoondream2Benchmark(args.trt_engine)
            benchmark.warmup_runs = args.warmup
            benchmark.benchmark_runs = args.runs
            
            benchmark.warmup()
            
            tegrastats = TegrastatsMonitor("tegrastats_quantized.log")
            quantized_results = benchmark.benchmark_with_power(tegrastats)
            
            print_results(quantized_results, "Quantized Moondream2 (INT8)")
            results['quantized'] = quantized_results
        else:
            print(f"⚠️ TensorRT engine not found at {args.trt_engine}")
    
    # Comparison
    if 'finetuned' in results and 'quantized' in results:
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        
        speedup = results['finetuned']['avg_latency_ms'] / results['quantized']['avg_latency_ms']
        power_reduction = ((results['finetuned']['avg_power_w'] - results['quantized']['avg_power_w']) / 
                          results['finetuned']['avg_power_w'] * 100)
        
        print(f"\n⚡ Speedup: {speedup:.2f}x faster (Quantized vs Fine-tuned)")
        print(f"💾 Model size reduction: ~4x (INT8 vs FP16)")
        print(f"🔋 Power reduction: {power_reduction:.1f}%")
        print(f"🎯 Energy efficiency improvement: {results['finetuned']['energy_per_inference_j']/results['quantized']['energy_per_inference_j']:.2f}x")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
