import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from dataloaders import create_dataloaders
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset_configs import unified_config

# Import ultralytics YOLO
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel


# YOLO Classification Model Wrapper
class YOLOClassifier(nn.Module):
    """
    YOLO Classification wrapper for PyTorch-style training.
    Supports YOLO11n-cls, YOLOv8n-cls, YOLOv8s-cls
    """
    def __init__(self, num_classes=2, model_name='yolov8n-cls.pt', pretrained=True):
        super(YOLOClassifier, self).__init__()
        
        # Load pretrained YOLO classification model
        if pretrained:
            # Download and load pretrained weights
            yolo_model = YOLO(model_name)
            self.model = yolo_model.model
        else:
            # Create model from scratch
            self.model = ClassificationModel(model_name.replace('.pt', '.yaml'), nc=num_classes)
        
        # Replace classifier head for custom number of classes
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'linear'):
            in_features = self.model.model.linear.in_features
            self.model.model.linear = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'linear'):
            in_features = self.model.linear.in_features
            self.model.linear = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)


# Alternative: Lightweight YOLO-style Model for Classification
class YOLOLiteClassifier(nn.Module):
    """
    Lightweight YOLO-inspired classifier optimized for Jetson Orin Nano Super.
    This is a simplified version that's easier to customize.
    """
    def __init__(self, num_classes=2):
        super(YOLOLiteClassifier, self).__init__()
        
        # Use YOLO11n backbone architecture (simplified)
        self.features = nn.Sequential(
            # Initial conv block
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            
            # Bottleneck blocks
            self._make_bottleneck(16, 32, 3, 2),
            self._make_bottleneck(32, 64, 6, 2),
            self._make_bottleneck(64, 128, 6, 2),
            self._make_bottleneck(128, 256, 3, 2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def _make_bottleneck(self, in_channels, out_channels, num_blocks, stride):
        """Create bottleneck blocks similar to YOLO architecture"""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.SiLU(inplace=True))
        
        for _ in range(num_blocks - 1):
            layers.append(self._bottleneck_block(out_channels))
        
        return nn.Sequential(*layers)
    
    def _bottleneck_block(self, channels):
        """Single bottleneck block"""
        return nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, 1, 0),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Training Function
def train_yolo(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    """
    Training function for YOLO classifier with class weighting.
    """
    # Calculate class weights
    train_labels = [1 if item['metadata']['has_fire'] else 0 for item in train_loader.dataset.data]
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    print(f"\n📊 Class Weights Applied:")
    print(f"  No Fire (class 0): {class_weights[0]:.4f}")
    print(f"  Fire (class 1): {class_weights[1]:.4f}")
    print(f"  Ratio: {class_weights[0]/class_weights[1]:.2f}:1 (No Fire is weighted {class_weights[0]/class_weights[1]:.1f}x more)\n")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer with lower learning rate for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixed precision training for better performance on Jetson
    scaler = torch.amp.GradScaler('cuda')
    
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
            
            # Mixed precision training
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
                
                with torch.amp.autocast('cuda'):
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
            torch.save(model.state_dict(), 'models/best_yolo_classifier.pth')
            print(f'✅ Model saved with validation accuracy: {val_acc:.4f}')
        
        scheduler.step()
    
    return history


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model with detailed metrics.
    """
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda'):
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


# Main execution for YOLO
if __name__ == '__main__':
    # Configuration
    BATCH_SIZE = 16  # Reduced for Jetson Orin Nano Super
    NUM_EPOCHS = 1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unified_path = unified_config["src"]

    # Get dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_json=f"{unified_path}/flame_vqa_train.json",
        val_json=f"{unified_path}/flame_vqa_val.json",
        test_json=f"{unified_path}/flame_vqa_test.json",
        batch_size=BATCH_SIZE,
        num_workers=2,
        mode='classification',
        image_size=224  # Standard input size for YOLO classification
    )
    
    # Initialize model
    # Option 1: Use lightweight custom YOLO-inspired model (recommended for Jetson)
    model = YOLOLiteClassifier(num_classes=2)
    
    # Option 2: Use pretrained YOLO11n-cls (uncomment to use)
    # model = YOLOClassifier(num_classes=2, model_name='yolo11n-cls.pt', pretrained=True)
    
    # Option 3: Use pretrained YOLOv8n-cls (uncomment to use)
    # model = YOLOClassifier(num_classes=2, model_name='yolov8n-cls.pt', pretrained=True)
    
    model = model.to(DEVICE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'\n🚀 Training on device: {DEVICE}')
    print(f'📊 Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}')
    print(f'🔧 Total parameters: {total_params:,}')
    print(f'🎯 Trainable parameters: {trainable_params:,}\n')
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Train model
    history = train_yolo(model, train_loader, val_loader, NUM_EPOCHS, DEVICE)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('models/best_yolo_classifier.pth'))
    results = evaluate_model(model, test_loader, DEVICE)
    
    # Save results
    with open('results/yolo_results.json', 'w') as f:
        json.dump({
            'history': history,
            'test_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in results.items()}
        }, f, indent=4)
    
    print(f'\n✅ Training complete! Results saved to results/yolo_results.json')
    print(f'📁 Best model saved to models/best_yolo_classifier.pth')
