import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
import timm
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_loader import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger données
_, _, test_loader = get_dataloaders(batch_size=16)

# === FONCTION D'ÉVALUATION ===
def evaluate_model(model, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Métriques
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'model': model_name,
        'accuracy': acc * 100,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

# === CHARGER MODÈLES ===
results = []

# ResNet18
print("Evaluating ResNet18...")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load("models/resnet18_deforestation.pth", map_location=device))
model.to(device)
results.append(evaluate_model(model, "ResNet18"))


# ViT
print("Evaluating ViT...")
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
model.load_state_dict(torch.load("models/vit_deforestation.pth", map_location=device))
model.to(device)
results.append(evaluate_model(model, "ViT-Base"))

# === TABLEAU COMPARATIF ===
df = pd.DataFrame(results)
print("\n" + "="*80)
print("COMPARATIVE RESULTS")
print("="*80)
print(df[['model', 'accuracy', 'precision', 'recall', 'f1_score']].to_string(index=False))

# === CONFUSION MATRIX ===
fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
if len(results) == 1:
    axes = [axes]

for idx, result in enumerate(results):
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Deforested', 'Non-Deforested'],
                yticklabels=['Deforested', 'Non-Deforested'],
                ax=axes[idx])
    axes[idx].set_title(f'{result["model"]} - Acc: {result["accuracy"]:.2f}%')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('comparison_confusion_matrices.png', dpi=300)
print("\nConfusion matrices saved to comparison_confusion_matrices.png")

# === CLASSIFICATION REPORT ===
for result in results:
    print(f"\n{'='*80}")
    print(f"{result['model']} - Detailed Report")
    print('='*80)
    print(classification_report(
        result['labels'], 
        result['predictions'],
        target_names=['Deforested', 'Non-Deforested']
    ))