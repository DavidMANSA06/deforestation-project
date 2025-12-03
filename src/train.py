# src/train.py

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from data_loader import get_dataloaders

# ------------- CONFIGURATION ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
num_epochs = 5   # tu peux augmenter plus tard
learning_rate = 1e-4

# ------------- DATA LOADERS -----------------
train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

# ------------- MODEL -----------------------
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # ou ResNet18_Weights.DEFAULT

# Adapter la dernière couche pour classification binaire
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ------------- LOSS & OPTIMIZER -------------
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)  #l adapte automatiquement le taux d'apprentissage pour chaque paramètre 

# ------------- TRAINING LOOP ----------------
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    print(f"Epoch {epoch} - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")

    # ------------- VALIDATION -----------------
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / val_total
    print(f"Validation - Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_acc:.4f}")

# ------------- SAVE MODEL -----------------
torch.save(model.state_dict(), "models/resnet18_deforestation.pth")
print("Model saved to models/resnet18_deforestation.pth")
