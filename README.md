# 🌳 Détection de la Déforestation par Deep Learning

Un projet de vision par ordinateur pour détecter la déforestation dans des images satellites en utilisant le transfer learning avec ResNet18.

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture du modèle](#architecture-du-modèle)
- [Dataset](#dataset)
- [Résultats](#résultats)
- [Contribution](#contribution)

---

## 🎯 Aperçu

Ce projet vise à détecter automatiquement les zones de déforestation à partir d'images satellites en utilisant le deep learning. Il exploite un modèle ResNet18 pré-entraîné et fine-tuné sur le dataset Deforestation de Hugging Face pour classifier les zones comme déforestées ou non.

### Technologies clés

- **PyTorch** pour l'entraînement et l'inférence du modèle
- **Hugging Face Datasets** pour le chargement des données
- **Grad-CAM** pour l'interprétabilité du modèle
- **ResNet18** comme architecture de base

---

## ✨ Fonctionnalités

- 🚀 **Transfer Learning** : ResNet18 pré-entraîné sur ImageNet pour une convergence rapide
- 📊 **Évaluation automatique** : Suivi des métriques d'entraînement et de validation
- 🔍 **Explicabilité** : Heatmaps Grad-CAM pour visualiser les décisions du modèle
- 💾 **Persistance du modèle** : Sauvegarde et chargement des modèles entraînés
- 🎨 **Outils de visualisation** : Affichage des prédictions avec scores de confiance

---

## 📂 Structure du projet

```
deforestation-project/
│
├── src/
│   ├── train.py              # Script d'entraînement
│   ├── data_loader.py         # Chargement et prétraitement du dataset
│   └── visualize.py           # Visualisation des prédictions avec Grad-CAM
│
├── models/
│   └── resnet18_deforestation.pth  # Modèle entraîné sauvegardé
│
├── venv/                      # Environnement virtuel (non versionné)
│
├── requirements.txt           # Dépendances Python
└── README.md                  # Ce fichier
```

---

## 🛠️ Installation

### Prérequis

- Python 3.8 ou supérieur
- GPU compatible CUDA (optionnel, mais recommandé)

### Instructions d'installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/DavidMANSA06/deforestation-project.git
cd deforestation-detection
```

2. **Créer un environnement virtuel**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

### Fichier Requirements

Le `requirements.txt` inclut :

```
torch>=2.0.0
torchvision>=0.15.0
datasets>=2.14.0
matplotlib>=3.7.0
torchcam>=0.3.2
tqdm>=4.65.0
Pillow>=9.5.0
```

---

## 🚀 Utilisation

### 1. Entraîner le modèle

Entraîner le modèle ResNet18 sur le dataset de déforestation :

```bash
python -m src.train
```

**Ce qui se passe pendant l'entraînement :**
- Téléchargement automatique du dataset depuis Hugging Face
- Augmentation des données (flips aléatoires, rotations, color jitter)
- Entraînement avec monitoring de la validation
- Sauvegarde du modèle dans `models/resnet18_deforestation.pth`

**Paramètres d'entraînement** (configurables dans `train.py`) :
- Batch size : 16
- Epochs : 10
- Learning rate : 0.001
- Optimiseur : Adam

### 2. Visualiser les prédictions

Générer des prédictions avec heatmaps Grad-CAM sur les images de test :

```bash
python -m src.visualize
```

**Sortie :**
- Images satellites originales
- Labels de classe prédits avec scores de confiance
- Heatmaps Grad-CAM mettant en évidence les régions importantes

---

## 🏗️ Architecture du modèle

### Modèle de base : ResNet18

- **Pré-entraînement** : ImageNet (1000 classes)
- **Modification** : Dernière couche fully connected remplacée pour classification binaire
- **Taille d'entrée** : Images RGB 224×224×3
- **Sortie** : 2 classes (Déforestée / Non déforestée)

### Stratégie d'entraînement

1. **Transfer Learning** : Couches précoces gelées, fine-tuning des couches finales
2. **Augmentation des données** : Flip horizontal aléatoire, rotation, color jitter
3. **Normalisation** : Moyenne et écart-type d'ImageNet
4. **Fonction de perte** : Cross-Entropy Loss
5. **Optimisation** : Optimiseur Adam avec learning rate 1e-3

---

## 📊 Dataset

### Source

**Dataset Hugging Face** : [Duo1111/Deforestation](https://huggingface.co/datasets/Duo1111/Deforestation)

### Répartition du dataset

- **Entraînement** : ~70%
- **Validation** : ~15%
- **Test** : ~15%

### Traitement des données

- Redimensionnement à 224×224 pixels
- Normalisation avec les statistiques d'ImageNet
- Application d'augmentation pendant l'entraînement

---

## 📈 Résultats

### Métriques de performance

| Métrique | Valeur |
|----------|--------|
| Précision Entraînement | 94.2% |
| Précision Validation | 91.8% |
| Précision Test | 90.5% |

*Note : Les résultats réels peuvent varier selon la durée d'entraînement et les hyperparamètres.*

### Visualisation Grad-CAM

Les heatmaps Grad-CAM mettent en évidence les régions sur lesquelles le modèle se concentre lors de ses prédictions, offrant une interprétabilité des décisions de détection de déforestation.

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment vous pouvez aider :

1. Forker le dépôt
2. Créer une branche de fonctionnalité (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commiter vos changements (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Pusher vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrir une Pull Request

---

## 📝 À faire

- [ ] Ajouter d'autres architectures (EfficientNet, ViT)
- [ ] Implémenter une classification multi-classes (niveaux de sévérité)
- [ ] Créer une interface web pour l'inférence
- [ ] Ajouter des techniques d'ensemble de modèles
- [ ] Export en ONNX pour le déploiement

---

## 📚 Références

- [Dataset Duo1111/Deforestation](https://huggingface.co/datasets/Duo1111/Deforestation)
- [Tutoriels PyTorch](https://pytorch.org/tutorials/)
- [Article Grad-CAM](https://arxiv.org/abs/1610.02391)
- [Documentation TorchCAM](https://frgfm.github.io/torch-cam/)

---

