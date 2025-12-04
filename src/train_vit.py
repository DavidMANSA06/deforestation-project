import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from src.data_loader import get_dataloaders

# ---------------- Device ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DataLoaders ----------------
train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

# ---------------- Model ----------------
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
model.to(device)

# ---------------- Loss & Optimizer ----------------
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

# ---------------- Training Loop ----------------
EPOCHS = 5

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        # ✅ Correction : batch est un tuple (images, labels)
        images, labels = batch
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

        pbar.set_postfix({"loss": running_loss / (total or 1)})

    print(f"Epoch {epoch} - Loss: {running_loss/len(train_loader):.4f}, Acc: {correct/total:.4f}")

# ---------------- Save Model ----------------
torch.save(model.state_dict(), "models/vit_deforestation.pth")
print("Model saved to models/vit_deforestation.pth")
