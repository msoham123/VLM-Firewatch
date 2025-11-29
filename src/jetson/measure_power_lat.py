#!/usr/bin/env python3
"""
Jetson Benchmark Script for Base Moondream2
Measures power consumption, latency, and FPS using tegrastats
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
import re
import gc

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
            ['tegrastats', '--interval', '100'],
            stdout=open(self.output_file, 'w'),
            stderr=subprocess.DEVNULL
        )
        self.is_running = True
        time.sleep(1)
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
                # Format: POM_5V_GPU 1234/5678 or VDD_GPU_SOC 1234mW
                match = re.search(r'POM_5V_GPU (\d+)/(\d+)', line)
                if match:
                    power_readings.append(int(match.group(1)))
                else:
                    match = re.search(r'VDD_GPU_SOC (\d+)mW', line)
                    if match:
                        power_readings.append(int(match.group(1)))
        
        if not power_readings:
            print("⚠️ Warning: No power readings found")
            return {}
        
        return {
            'avg_power_mw': np.mean(power_readings),
            'min_power_mw': np.min(power_readings),
            'max_power_mw': np.max(power_readings),
            'avg_power_w': np.mean(power_readings) / 1000.0,
            'samples': len(power_readings)
        }


class Moondream2Benchmark:
    """Benchmark base Moondream2 model"""
    
    def __init__(self):
        self.model_name = "Base Moondream2"
        self.warmup_runs = 10
        self.benchmark_runs = 100
        self.load_model()
    
    def load_model(self):
        """Load base Moondream2 model"""
        print(f"Loading {self.model_name}...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_id = "vikhyatk/moondream2"
        revision = "2024-07-23"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"  # Auto distribute across GPU/CPU
        )
        
        self.model.eval()
        
        # Check which device model is on
        device = next(self.model.parameters()).device
        print(f"✓ Model loaded on {device}")
    
    def create_dummy_input(self) -> Image.Image:
        """Create dummy input image (378x378 RGB)"""
        img_array = np.random.randint(0, 256, (378, 378, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    def inference(self, image: Image.Image):
        """Run inference"""
        question = "Is there fire visible in this image?"
        
        with torch.no_grad():
            enc_image = self.model.encode_image(image)
            answer = self.model.answer_question(enc_image, question, self.tokenizer)
        
        return answer
    
    def warmup(self):
        """Warmup runs to stabilize performance"""
        print(f"\nWarming up ({self.warmup_runs} runs)...")
        for i in range(self.warmup_runs):
            dummy_img = self.create_dummy_input()
            self.inference(dummy_img)
            
            del dummy_img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        print(f"✓ Warmup complete")
    
    def benchmark_latency(self) -> Dict:
        """Benchmark inference latency"""
        print(f"\nBenchmarking latency ({self.benchmark_runs} runs)...")
        
        latencies = []
        
        for i in range(self.benchmark_runs):
            dummy_img = self.create_dummy_input()
            
            start_time = time.time()
            self.inference(dummy_img)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Cleanup
            del dummy_img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            if (i + 1) % 10 == 0:
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
        print(f"\n{'='*70}")
        print(f"BENCHMARKING {self.model_name.upper()}")
        print(f"{'='*70}")
        
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


def print_results(results: Dict):
    """Pretty print benchmark results"""
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS")
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
    parser = argparse.ArgumentParser(description="Benchmark Moondream2 on Jetson")
    parser.add_argument("--runs", type=int, default=100,
                       help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup runs")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = Moondream2Benchmark()
    benchmark.warmup_runs = args.warmup
    benchmark.benchmark_runs = args.runs
    
    # Run warmup
    benchmark.warmup()
    
    # Run benchmark with power monitoring
    tegrastats = TegrastatsMonitor("tegrastats.log")
    results = benchmark.benchmark_with_power(tegrastats)
    
    # Print results
    print_results(results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
