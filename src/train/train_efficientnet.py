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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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
def train_efficientnet(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss()
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
            torch.save(model.state_dict(), 'best_efficientnet_classifier.pth')
            print(f'Model saved with validation accuracy: {val_acc:.4f}')
        
        scheduler.step()
    
    return history

# Evaluation Function
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f'\nTest Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'\nConfusion Matrix:\n{cm}')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Main execution for EfficientNet
if __name__ == '__main__':
    # Configuration
    DATA_PATH = 'path/to/your/preprocessed/dataset'
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unified_path = unified_config["src"]

    # Get dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_json=f"{unified_path}/flame_vqa_train.json",
        val_json=f"{unified_path}/flame_vqa_val.json",
        test_json=f"{unified_path}/flame_vqa_test.json",
        batch_size=32,
        num_workers=4,
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
    model.load_state_dict(torch.load('best_efficientnet_classifier.pth'))
    results = evaluate_model(model, test_loader, DEVICE)
    
    # Save results
    with open('efficientnet_results.json', 'w') as f:
        json.dump({
            'history': history,
            'test_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in results.items()}
        }, f, indent=4)
