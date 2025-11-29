import matplotlib.pyplot as plt
import numpy as np
import os


# Create graphs directory if it doesn't exist
os.makedirs('graphs', exist_ok=True)


# Performance data
data = {
    "finetuned": {
        "avg_latency_ms": 126.44627285003662,
        "min_latency_ms": 124.7861385345459,
        "max_latency_ms": 145.51210403442383,
        "std_latency_ms": 1.2976142709927319,
        "p50_latency_ms": 126.23727321624756,
        "p95_latency_ms": 127.34198570251463,
        "p99_latency_ms": 131.85767650604248,
        "fps": 7.908497241243204,
        "avg_power_w": 246.63575238095237,
        "min_power_w": 209.02,
        "max_power_w": 255.57,
        "samples": 1050,
        "energy_per_inference_j": 31.186171640135974
    },
    "quantized": {
        "avg_latency_ms": 127.86701083183289,
        "min_latency_ms": 126.56140327453613,
        "max_latency_ms": 136.5816593170166,
        "std_latency_ms": 0.7682515036155646,
        "p50_latency_ms": 127.79474258422852,
        "p95_latency_ms": 128.34070920944214,
        "p99_latency_ms": 130.19388437271118,
        "fps": 7.820625456828517,
        "avg_power_w": 264.92175586854466,
        "min_power_w": 223.85,
        "max_power_w": 278.96,
        "samples": 1065,
        "energy_per_inference_j": 33.87475302723139
    }
}


# Model size data (UPDATE THESE VALUES WITH YOUR ACTUAL MODEL SIZES)
model_sizes = {
    "finetuned": 3.5,  # GB - UPDATE THIS
    "quantized": 1.8   # GB - UPDATE THIS
}


# Graph configurations
graphs = [
    {
        'title': 'Model Size Comparison',
        'ylabel': 'Model Size (GB)',
        'data_key': 'model_size',
        'filename': 'model_size_comparison.png',
        'values': [model_sizes['finetuned'], model_sizes['quantized']],
        'color_ft': '#3498db',
        'color_q': '#2ecc71'
    },
    {
        'title': 'Throughput Comparison',
        'ylabel': 'Frames Per Second (FPS)',
        'data_key': 'fps',
        'filename': 'throughput_comparison.png',
        'values': [data['finetuned']['fps'], data['quantized']['fps']],
        'color_ft': '#3498db',
        'color_q': '#2ecc71'
    },
    {
        'title': 'Energy Per Inference Comparison',
        'ylabel': 'Energy (Joules)',
        'data_key': 'energy_per_inference_j',
        'filename': 'energy_per_inference_comparison.png',
        'values': [data['finetuned']['energy_per_inference_j'], data['quantized']['energy_per_inference_j']],
        'color_ft': '#3498db',
        'color_q': '#2ecc71'
    },
    {
        'title': 'P99 Latency Comparison',
        'ylabel': 'Latency (ms)',
        'data_key': 'p99_latency_ms',
        'filename': 'p99_latency_comparison.png',
        'values': [data['finetuned']['p99_latency_ms'], data['quantized']['p99_latency_ms']],
        'color_ft': '#3498db',
        'color_q': '#2ecc71'
    }
]


# Create each graph
for graph_config in graphs:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Model types
    models = ['Fine-Tuned', 'Quantized']
    x = np.arange(len(models))
    width = 0.5
    
    # Values
    values = graph_config['values']
    colors = [graph_config['color_ft'], graph_config['color_q']]
    
    # Create bars
    bars = ax.bar(x, values, width, color=colors, alpha=0.8)
    
    # Customize the plot
    ax.set_ylabel(graph_config['ylabel'], fontsize=12, fontweight='bold')
    ax.set_title(graph_config['title'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Set y-axis to start from 0 or slightly below min value
    y_min = 0 if min(values) > max(values) * 0.3 else min(values) * 0.9
    y_max = max(values) * 1.15
    ax.set_ylim(y_min, y_max)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'graphs/{graph_config["filename"]}', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created {graph_config['title']}")


print(f"\n✓ All 4 graphs saved in 'graphs/' directory")
print("\nGenerated graphs:")
print("  1. Model Size Comparison")
print("  2. Throughput Comparison")
print("  3. Energy Per Inference Comparison")
print("  4. P99 Latency Comparison")
