# Questions/Réponses - Présentation Projet Déforestation

## 🎯 Questions Générales sur le Projet

### 1. Quel est l'objectif principal du projet ?
**Réponse :** Développer un système de détection automatique de la déforestation à partir d'images satellites, combinant classification binaire et interprétabilité visuelle pour identifier et analyser les zones critiques affectées.

### 2. Pourquoi ce projet est-il important ?
**Réponse :** La déforestation est un enjeu environnemental majeur. Un système automatisé permet :
- Une surveillance continue et à grande échelle des forêts
- Une détection précoce des zones déforestées
- Une aide à la décision pour les organismes de protection environnementale
- Une réduction des coûts et du temps d'analyse manuelle

### 3. Quels sont les deux volets principaux du projet ?
**Réponse :**
1. **Classification binaire** : Distinguer automatiquement les images "deforested" vs "non-deforested"
2. **Interprétabilité** : Identifier visuellement les zones critiques qui influencent les prédictions via Grad-CAM

---

## 📊 Questions sur les Données

### 4. Quel dataset utilisez-vous ?
**Réponse :**
- **Nom :** Duo1111/Deforestation
- **Source :** Hugging Face (open-access)
- **Taille :** ~2 010 images satellites
- **Classes :** 2 catégories (0=deforested, 1=non-deforested)
- **Caractéristiques :** Images variées de différentes zones géographiques

### 5. Comment sont réparties les données (train/val/test) ?
**Réponse :**
- **Train :** 70% (~1407 images) - pour l'entraînement du modèle
- **Validation :** 15% (~302 images) - pour ajuster les hyperparamètres
- **Test :** 15% (~301 images) - pour l'évaluation finale

### 6. Les données sont-elles équilibrées ?
**Réponse :** Cette question nécessite une vérification. Si déséquilibre détecté, nous pouvons utiliser :
- Augmentation de données (data augmentation)
- Pondération des classes dans la fonction de perte
- Techniques de sur/sous-échantillonnage

### 7. Quelle est la résolution des images ?
**Réponse :** Les images sont redimensionnées à **224×224 pixels** pour être compatibles avec les modèles pré-entraînés (ResNet, ViT).

---

## 🤖 Questions sur le Modèle

### 8. Quel(s) modèle(s) utilisez-vous ?
**Réponse :** Nous utilisons **ResNet18** pré-entraîné sur ImageNet :
- Architecture éprouvée pour la vision par ordinateur
- Poids pré-entraînés permettant le transfer learning
- Dernière couche adaptée pour classification binaire (512 → 2 neurones)

### 9. Pourquoi utiliser un modèle pré-entraîné ?
**Réponse :**
- **Transfer Learning :** Le modèle a déjà appris des features génériques sur ImageNet (1M+ images)
- **Performance :** Meilleurs résultats avec moins de données
- **Temps :** Convergence plus rapide qu'un entraînement from scratch
- **Efficacité :** Nécessite moins de ressources computationnelles

### 10. Quelles sont les alternatives à ResNet18 ?
**Réponse :**
- **Vision Transformer (ViT)** : Architecture basée sur l'attention, très performante
- **ResNet50/101** : Versions plus profondes pour plus de capacité
- **EfficientNet** : Meilleur rapport performance/efficacité
- **Comparaison possible** pour optimiser les résultats

### 11. Comment le modèle est-il adapté à votre problème ?
**Réponse :**
```python
model.fc = nn.Linear(model.fc.in_features, 2)  # 512 → 2 sorties
```
La dernière couche fully-connected est remplacée pour produire 2 probabilités (deforested, non-deforested).

---

## 🔧 Questions Techniques

### 12. Quelle fonction de perte utilisez-vous ?
**Réponse :** **CrossEntropyLoss** - Standard pour la classification multi-classes :
- Combine LogSoftmax + NLLLoss
- Pénalise les mauvaises prédictions proportionnellement à leur erreur
- Adapté aux problèmes de classification avec classes mutuellement exclusives

### 13. Quel optimiseur et pourquoi ?
**Réponse :** **Adam (Adaptive Moment Estimation)** avec learning rate = 1e-4 :
- Adapte automatiquement le taux d'apprentissage pour chaque paramètre
- Combine les avantages de RMSprop et SGD avec momentum
- Convergence rapide et stable
- Standard pour le deep learning

### 14. Combien d'epochs d'entraînement ?
**Réponse :** Actuellement **5 epochs** pour tester rapidement. Peut être augmenté (10-20) selon :
- Convergence de la loss
- Performance sur validation
- Risque d'overfitting

### 15. Quelle taille de batch ?
**Réponse :** **Batch size = 16** - Compromis entre :
- Vitesse d'entraînement (plus grand = plus rapide)
- Mémoire GPU disponible
- Stabilité de la convergence

---

## 📈 Questions sur l'Évaluation

### 16. Quelles métriques utilisez-vous ?
**Réponse :**
- **Accuracy** : Pourcentage de prédictions correctes
- **Precision** : Taux de vrais positifs parmi les prédictions positives
- **Recall** : Taux de vrais positifs détectés
- **F1-Score** : Moyenne harmonique de precision et recall
- **Matrice de confusion** : Visualisation des erreurs

### 17. Quelle métrique est la plus importante ?
**Réponse :** Dépend du contexte :
- **Recall élevé** si on veut détecter toutes les déforestations (priorité : ne rien manquer)
- **Precision élevée** si on veut éviter les fausses alertes
- **F1-Score** pour un équilibre entre les deux

### 18. Comment validez-vous que le modèle ne fait pas d'overfitting ?
**Réponse :**
- Comparaison train accuracy vs validation accuracy
- Si train >> validation → overfitting
- Solutions : dropout, régularisation, data augmentation, early stopping

---

## 🔍 Questions sur l'Interprétabilité

### 19. Qu'est-ce que Grad-CAM ?
**Réponse :** **Gradient-weighted Class Activation Mapping** :
- Technique d'explicabilité visuelle
- Génère une heatmap montrant les zones de l'image importantes pour la prédiction
- Utilise les gradients de la dernière couche de convolution
- Permet de visualiser "où" le modèle regarde pour décider

### 20. Pourquoi l'interprétabilité est-elle cruciale ?
**Réponse :**
- **Validation** : Vérifier que le modèle se concentre sur les bonnes features (végétation, zones dégagées)
- **Confiance** : Augmenter la confiance des utilisateurs dans les prédictions
- **Débug** : Identifier si le modèle apprend des biais
- **Insight** : Fournir des informations exploitables pour la surveillance environnementale

### 21. Que montrent les heatmaps générées ?
**Réponse :** Les heatmaps visualisent :
- **Rouge/Chaud** : Zones ayant le plus d'influence sur la prédiction (ex: zones déboisées)
- **Bleu/Froid** : Zones avec peu d'influence
- Permettent de valider que le modèle détecte bien la déforestation et non des artefacts

---

## 🚀 Questions sur l'Implémentation

### 22. Quelle est la structure du code ?
**Réponse :**
```
src/
├── data_loader.py    # Chargement et préparation des données
├── train.py          # Entraînement du modèle
├── evaluate.py       # Évaluation (métriques, confusion matrix)
├── interpret.py      # Grad-CAM et visualisations
└── visualize.py      # Génération de graphiques
```

### 23. Quels sont les prérequis techniques ?
**Réponse :**
- **Python 3.8+**
- **PyTorch** : Framework de deep learning
- **torchvision** : Modèles et transformations d'images
- **timm** : Bibliothèque de modèles pré-entraînés
- **Grad-CAM** : Interprétabilité
- **GPU recommandé** (CUDA) pour accélérer l'entraînement

### 24. Combien de temps prend l'entraînement ?
**Réponse :**
- **Avec GPU** : ~5-10 minutes pour 5 epochs
- **Avec CPU** : ~30-60 minutes pour 5 epochs
- Variable selon le matériel disponible

---

## 🎓 Questions d'Analyse

### 25. Quels sont les défis du projet ?
**Réponse :**
- **Déséquilibre potentiel** des classes
- **Variabilité** des images satellites (saisons, qualité, résolution)
- **Généralisation** à de nouvelles régions géographiques
- **Faux positifs** : zones naturellement dégagées vs déforestation
- **Besoin de GPU** pour entraînement efficace

### 26. Quelles améliorations futures sont possibles ?
**Réponse :**
1. **Modèles plus performants** : ViT, EfficientNet, modèles ensemble
2. **Data augmentation** : rotations, flips, ajustements de couleur
3. **Détection d'objets** : Localiser précisément les zones déforestées (YOLO, Faster R-CNN)
4. **Analyse temporelle** : Comparer des images de différentes dates
5. **Déploiement** : API web pour utilisation en production
6. **Données supplémentaires** : Augmenter le dataset

### 27. Quelles sont les applications concrètes ?
**Réponse :**
- **ONG environnementales** : Surveillance des zones protégées
- **Gouvernements** : Contrôle des activités illégales
- **Recherche** : Études sur l'évolution de la couverture forestière
- **Entreprises** : Vérification de la conformité des chaînes d'approvisionnement

---

## 💡 Questions pour la Démo

### 28. Que montrerez-vous lors de la présentation ?
**Réponse :**
1. **Architecture** du modèle et pipeline de données
2. **Résultats d'entraînement** : courbes de loss et accuracy
3. **Métriques d'évaluation** : accuracy, precision, recall, F1
4. **Matrice de confusion** : analyse des erreurs
5. **Grad-CAM** : heatmaps sur images de test
6. **Comparaison** : images correctement/incorrectement classées

### 29. Comment prouver que le modèle fonctionne ?
**Réponse :**
- **Accuracy > 85%** sur test set
- **Grad-CAM** montrant que le modèle se concentre sur les bonnes zones
- **Exemples visuels** de prédictions correctes
- **Analyse d'erreurs** : comprendre les cas difficiles

### 30. Quel est le résultat attendu final ?
**Réponse :** Un système complet capable de :
- Classifier automatiquement des images satellites avec haute précision
- Expliquer visuellement ses décisions via heatmaps
- Servir de base pour un système de surveillance environnementale
- Être étendu à d'autres problèmes de télédétection

---

## 📝 Notes pour la Présentation

**Points forts à mettre en avant :**
- ✅ Problème réel et impactant (environnement)
- ✅ Approche moderne (deep learning + transfer learning)
- ✅ Interprétabilité (pas une boîte noire)
- ✅ Résultats quantitatifs et visuels
- ✅ Potentiel d'extension et déploiement

**Éléments à préparer :**
- 📊 Graphiques de performance
- 🖼️ Exemples visuels de Grad-CAM
- 📈 Comparaison avant/après entraînement
- 🎯 Démonstration sur nouvelles images
