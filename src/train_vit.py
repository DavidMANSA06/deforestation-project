import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from data_loader import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

# ViT Model
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

# Training loop (5 epochs)
for epoch in range(1, 6):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
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
    
    print(f"Epoch {epoch} - Loss: {running_loss/len(train_loader):.4f}, Acc: {correct/total:.4f}")

torch.save(model.state_dict(), "models/vit_deforestation.pth")