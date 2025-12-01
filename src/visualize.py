from torchvision.models import resnet18, ResNet18_Weights
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from src.data_loader import get_dataloaders
import torchvision.transforms.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8

# 1️⃣ Data
_, _, test_loader = get_dataloaders(batch_size=batch_size)
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# 2️⃣ Model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/resnet18_deforestation.pth"))
model.to(device)
model.eval()

# 3️⃣ Grad-CAM
cam_extractor = SmoothGradCAMpp(model, target_layer='layer4')

# Forward pass pour hooker les features
outputs = model(images)
preds = torch.argmax(outputs, dim=1)

# 4️⃣ Visualisation
fig, axes = plt.subplots(2, 4, figsize=(15, 8))

for i, ax in enumerate(axes.flatten()):
    # on passe l'image individuellement pour Grad-CAM
    img_tensor = images[i].unsqueeze(0)
    out = model(img_tensor)
    
    # heatmap
    activation_map = cam_extractor(preds[i].item(), out)[0].cpu()
    
    # overlay
    img_vis = F.to_pil_image(img_tensor.squeeze(0).cpu())
    result = overlay_mask(img_vis, F.to_pil_image(activation_map, mode='F'), alpha=0.5)
    
    ax.imshow(result)
    ax.set_title(f"Pred: {preds[i].item()}, Label: {labels[i].item()}")
    ax.axis('off')

plt.tight_layout()
plt.show()
