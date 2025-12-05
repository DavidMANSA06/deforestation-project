import timm
import torch
import matplotlib.pyplot as plt
from data_loader import get_dataloaders
import torchvision.transforms.functional as F
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8

# 1️⃣ Data
_, _, test_loader = get_dataloaders(batch_size=batch_size)
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# 2️⃣ Model ViT
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/vit_deforestation.pth"))
model.to(device)
model.eval()

# 3️⃣ Grad-CAM pour Vision Transformer
# ViT utilise une architecture différente, on cible le dernier bloc d'attention
cam_extractor = SmoothGradCAMpp(model, target_layer='blocks.11.norm1')

# Forward pass
outputs = model(images)
preds = torch.argmax(outputs, dim=1)

# 4️⃣ Visualisation
fig, axes = plt.subplots(2, 4, figsize=(15, 8))

for i, ax in enumerate(axes.flatten()):
    # Image individuelle pour Grad-CAM
    img_tensor = images[i].unsqueeze(0)
    out = model(img_tensor)
    
    # Heatmap
    activation_map = cam_extractor(preds[i].item(), out)[0].cpu()
    
    # Overlay
    img_vis = F.to_pil_image(img_tensor.squeeze(0).cpu())
    result = overlay_mask(img_vis, F.to_pil_image(activation_map, mode='F'), alpha=0.5)
    
    ax.imshow(result)
    ax.set_title(f"ViT - Pred: {preds[i].item()}, Label: {labels[i].item()}")
    ax.axis('off')

plt.suptitle('Vision Transformer (ViT) - Grad-CAM Visualization', fontsize=16)
plt.tight_layout()
plt.savefig('vit_gradcam.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualisation ViT sauvegardée dans 'vit_gradcam.png'")
