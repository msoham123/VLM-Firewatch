import matplotlib.pyplot as plt
import numpy as np
import os

# Create graphs directory if it doesn't exist
os.makedirs('graphs', exist_ok=True)

# Models to create graphs for
models = ['EfficientNet', 'YOLO', 'MoonDream2']

# Categories on x-axis
categories = ['End to End', 'Fire', 'No Fire']

# Placeholder accuracy data for each model
# Format: {model_name: {'base': [end_to_end, fire, no_fire], 'fine_tuned': [end_to_end, fire, no_fire]}}
accuracy_data = {
    'EfficientNet': {
        'base': [0.6191892896987727, 0.7517594369801663,  0.5167298500082413],
        'fine_tuned': [0.995909259947936, 0.9916826615483045, 0.9991758694577221,]
    },
    'YOLO': {
        'base': [0.5641, 0.0, 1.0],
        'fine_tuned': [0.9032168092227594, 0.8108338664960546, 0.9746167792978407]
    },
    'MoonDream2': {
        'base': [0.8961509854964671,  0.7747920665387076, 0.9899456073842097],
        'fine_tuned': [0.9988843436221644, 0.9974408189379399, 1.0]
    },
}

# Create a bar graph for each model
for model in models:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up bar positions
    x = np.arange(len(categories))
    width = 0.35
    
    # Get data for this model
    base_accuracies = [acc * 100 for acc in accuracy_data[model]['base']]
    fine_tuned_accuracies = [acc * 100 for acc in accuracy_data[model]['fine_tuned']]

    # Find min and max for reasonable y-axis range
    all_values = base_accuracies + fine_tuned_accuracies
    min_val = min(all_values)
    max_val = max(all_values)

    # Set y-axis range with some padding
    y_min = max(0, min_val - 10)  # At least 0
    y_max = min(100, max_val + 5)  # At most 100
    
    # Create bars
    bars1 = ax.bar(x - width/2, base_accuracies, width, label='Base', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, fine_tuned_accuracies, width, label='Fine-Tuned', color='#e74c3c', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Fire Detection Accuracy for {model}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=10)
    ax.set_ylim(y_min, y_max)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'graphs/{model}_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created graph for {model}")

print(f"\nAll graphs saved in 'graphs/' directory")
