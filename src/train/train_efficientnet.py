import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import json
from tqdm import tqdm
import timm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
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

# Training Function
def train_efficientnet(model, train_loader, val_loader, num_epochs=10, device='cuda'):

    train_labels = [1 if item['metadata']['has_fire'] else 0 for item in train_loader.dataset.data]
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    print(f"\n📊 Class Weights Applied:")
    print(f"  No Fire (class 0): {class_weights[0]:.4f}")
    print(f"  Fire (class 1): {class_weights[1]:.4f}")
    print(f"  Ratio: {class_weights[0]/class_weights[1]:.2f}:1 (No Fire is weighted {class_weights[0]/class_weights[1]:.1f}x more)\n")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_efficientnet_classifier.pth')
            print(f'Model saved with validation accuracy: {val_acc:.4f}')
        
        scheduler.step()
    
    return history

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Detailed metrics
    print(f'\n{"="*60}')
    print("DETAILED EVALUATION RESULTS")
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
    
    # Check if model is just predicting fire for everything
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
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'no_fire_accuracy': no_fire_acc,
        'fire_accuracy': fire_acc
    }

# Main execution for EfficientNet
if __name__ == '__main__':
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unified_path = unified_config["src"]

    # Get dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_json=f"{unified_path}/flame_vqa_train.json",
        val_json=f"{unified_path}/flame_vqa_val.json",
        test_json=f"{unified_path}/flame_vqa_test.json",
        batch_size=32,
        num_workers=2,
        mode='classification',
        image_size=224
    )
    
    # Initialize model
    model = EfficientNetClassifier(num_classes=2, model_name='efficientnet_b0', pretrained=True)
    model = model.to(DEVICE)
    
    print(f'Training on device: {DEVICE}')
    print(f'Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}')
    
    # Train model
    history = train_efficientnet(model, train_loader, val_loader, NUM_EPOCHS, DEVICE)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('models/best_efficientnet_classifier.pth'))
    results = evaluate_model(model, test_loader, DEVICE)
    
    # Save results
    with open('results/efficientnet_results.json', 'w') as f:
        json.dump({
            'history': history,
            'test_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in results.items()}
        }, f, indent=4)
