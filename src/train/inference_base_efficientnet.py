import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json
from tqdm import tqdm
import timm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from dataloaders import create_dataloaders
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset_configs import unified_config


# EfficientNet Model
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        # Using timm library for EfficientNet
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)


def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []  # Store prediction probabilities
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Running Inference'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Detailed metrics
    print(f'\n{"="*60}')
    print("BASE MODEL EVALUATION RESULTS (No Fine-tuning)")
    print("="*60)
    
    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'\nOverall Accuracy: {accuracy:.4f}')
    
    # Class-specific metrics
    print(f'\nPer-Class Performance:')
    print(classification_report(all_labels, all_preds, 
                                target_names=['No Fire', 'Fire'],
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f'\nConfusion Matrix:')
    print(f'                Predicted')
    print(f'              No Fire  Fire')
    print(f'Actual No Fire   {cm[0][0]:5d}   {cm[0][1]:5d}')
    print(f'       Fire      {cm[1][0]:5d}   {cm[1][1]:5d}')
    
    # Calculate per-class accuracy
    no_fire_acc = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    fire_acc = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    
    print(f'\nPer-Class Accuracy:')
    print(f'  No Fire: {no_fire_acc:.4f} ({cm[0][0]}/{cm[0][0] + cm[0][1]})')
    print(f'  Fire:    {fire_acc:.4f} ({cm[1][1]}/{cm[1][0] + cm[1][1]})')
    
    # Prediction distribution
    fire_predictions = sum(all_preds)
    total_predictions = len(all_preds)
    print(f'\nPrediction Distribution:')
    print(f'  Predicted Fire: {fire_predictions}/{total_predictions} ({fire_predictions/total_predictions*100:.1f}%)')
    print(f'  Predicted No Fire: {total_predictions - fire_predictions}/{total_predictions} ({(total_predictions - fire_predictions)/total_predictions*100:.1f}%)')
    
    # Baseline comparison
    actual_fire = sum(all_labels)
    baseline_acc = max(actual_fire, total_predictions - actual_fire) / total_predictions
    print(f'\nBaseline (always predict majority): {baseline_acc:.4f}')
    print(f'Model improvement over baseline: {accuracy - baseline_acc:.4f} ({(accuracy - baseline_acc)/baseline_acc*100:.1f}%)')
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    print(f'\nDetailed Metrics per Class:')
    print(f'  No Fire - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}')
    print(f'  Fire    - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}')
    
    return {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'no_fire_accuracy': float(no_fire_acc),
        'fire_accuracy': float(fire_acc),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        # 'predictions': [int(p) for p in all_preds],  # Convert to Python int
        # 'labels': [int(l) for l in all_labels],  # Convert to Python int
        # 'probabilities': [prob.tolist() for prob in all_probs],
        'prediction_distribution': {
            'fire': int(fire_predictions),
            'no_fire': int(total_predictions - fire_predictions)
        }
    }


# Main execution for base inference
if __name__ == '__main__':
    # Configuration
    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_NAME = 'efficientnet_b0'
    
    unified_path = unified_config["src"]
    
    # Create output directories if they don't exist
    os.makedirs('results', exist_ok=True)
    
    # Get test dataloader only
    print("Loading test dataset...")
    _, _, test_loader = create_dataloaders(
        train_json=f"{unified_path}/flame_vqa_train.json",
        val_json=f"{unified_path}/flame_vqa_val.json",
        test_json=f"{unified_path}/flame_vqa_test.json",
        batch_size=BATCH_SIZE,
        num_workers=2,
        mode='classification',
        image_size=224
    )
    
    # Initialize pretrained model (no training, just pretrained weights)
    print(f"\nInitializing pretrained {MODEL_NAME}...")
    model = EfficientNetClassifier(num_classes=2, model_name=MODEL_NAME, pretrained=True)
    model = model.to(DEVICE)
    
    print(f'Running inference on device: {DEVICE}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Evaluate on test set
    results = evaluate_model(model, test_loader, DEVICE)
    
    # Save detailed results
    output_file = 'results/efficientnet_base_inference_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f'\n✅ Results saved to {output_file}')
    print("="*60)
