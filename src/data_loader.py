# src/data_loader.py

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        """
        Dataset PyTorch custom à partir d'un dataset Hugging Face.
        :param hf_dataset: un split du dataset Hugging Face
        :param transform: transformations torchvision à appliquer aux images
        """
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Récupérer image et label
        img = self.dataset[idx]['image']
        label = self.dataset[idx]['label']

        # Appliquer transformation si nécessaire
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size=16):
    """
    Charge le dataset Hugging Face et retourne des DataLoaders PyTorch
    :param batch_size: taille du batch
    :return: train_loader, val_loader, test_loader
    """
    # Charger dataset
    dataset = load_dataset("Duo1111/Deforestation")

    # Transformations pour ResNet / ViT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Créer splits train / val / test
    # Ici on prend 70% train, 15% val, 15% test
    train_val = dataset['train'].train_test_split(test_size=0.3, seed=42)
    val_test = train_val['test'].train_test_split(test_size=0.5, seed=42)

    train_ds = HFDataset(train_val['train'], transform)
    val_ds = HFDataset(val_test['train'], transform)
    test_ds = HFDataset(val_test['test'], transform)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# Test rapide
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)
    for imgs, labels in train_loader:
        print(imgs.shape, labels.shape)
        break
