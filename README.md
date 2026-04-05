# Bone Marrow Cell Classification Challenge

This repository contains my solution for the **Bone Marrow Cell Classification Challenge**, where the objective is to classify microscopic bone marrow cell images into disease categories using deep learning.

The project consists of **two phases**:

* **Phase 1:** Standard supervised classification with **14 diseases** and a large labeled dataset.
* **Phase 2:** Extension where **6 new diseases (15–20)** appear with **only 5 samples each**, creating a **few-shot learning scenario**.

The solution uses **EfficientNet-B4 with transfer learning** implemented in **PyTorch**.

---

# Dataset Description

## Phase 1

Phase 1 consists of a large supervised dataset.

* **Number of diseases:** 14
* **Samples per disease:** 2400
* **Total images:** 33,600
* **Image format:** RGB
* **Image resolution:** 250 × 250

Each disease category contains **2400 labeled images** used for training the model.

---

## Phase 2

Phase 2 introduces **previously unseen diseases**.

* **New diseases:** 15 – 20
* **Support samples per disease:** 5 images
* **Testing dataset:** new diseases

The goal in Phase 2 is to **adapt the trained model to recognize the new diseases using very limited data**.

---

# Model Architecture

The model used in this project is **EfficientNet-B4 pretrained on ImageNet**.

EfficientNet was chosen because it provides:

* Strong feature extraction capability
* Efficient parameter usage
* Good performance on medical image classification tasks

The classifier layer was modified to output **14 classes** for Phase 1 training.

Example model setup:

```python
from torchvision import models
import torch.nn as nn

model = models.efficientnet_b4(
    weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    14
)
```

---

# Image Preprocessing

All images were resized from **250×250 to 224×224** to match EfficientNet input requirements.

The following transformations were used:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
```

Data augmentation techniques used:

* Random Horizontal Flip
* Random Rotation
* ImageNet normalization

These transformations help improve **generalization and robustness**.

---

# Training Configuration

| Parameter       | Value             |
| --------------- | ----------------- |
| Model           | EfficientNet-B4   |
| Loss Function   | CrossEntropyLoss  |
| Optimizer       | AdamW             |
| Learning Rate   | 2e-4              |
| Weight Decay    | 1e-4              |
| Scheduler       | CosineAnnealingLR |
| Epochs          | 30                |
| Batch Size      | 32                |
| Mixed Precision | Enabled           |

Optimizer configuration:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4
)
```

Learning rate scheduler:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30,
    eta_min=1e-6
)
```

---

# Training Strategy

## Transfer Learning

The EfficientNet-B4 backbone is **pretrained on ImageNet**.
Using pretrained weights allows the model to learn useful **visual features**, which improves performance on medical image classification tasks.

Only the **final classifier layer is modified** to match the number of disease classes.

---

## Mixed Precision Training

Training uses **PyTorch Automatic Mixed Precision (AMP)** to improve training speed and reduce memory usage.

Example:

```python
scaler = torch.amp.GradScaler("cuda")
```

Benefits:

* Faster training
* Lower GPU memory usage
* Stable training

---

# Phase 2 Strategy (Few-Shot Adaptation)

Phase 2 introduces **6 new diseases with only 5 samples each**, making it a **few-shot learning problem**.

The approach used:

1. Load **trained Phase 1 model weights**
2. Fine-tune the model using **Phase 2 support images**
3. Train with **smaller batch size** due to limited data

Example configuration:

```python
batch_size = 8
```

This allows the model to **adapt to new disease classes while retaining knowledge from Phase 1**.

---

# Inference Pipeline

For each test image:

1. Load the image
2. Apply preprocessing transformations
3. Pass the image through the trained model
4. Predict the class using `argmax`

Example inference code:

```python
image = transform(image).unsqueeze(0).to(device)

outputs = model(image)

prediction = torch.argmax(outputs, dim=1)
```

Predictions are saved into a **submission CSV file**.

---

# Libraries Used

Main libraries used in this project:

```
torch
torchvision
numpy
pandas
PIL
matplotlib
```

---

# Key Ideas

* Transfer learning using **EfficientNet-B4**
* Data augmentation for better generalization
* Mixed precision training for faster computation
* Few-shot learning adaptation for new diseases

---

# Future Improvements

Possible improvements include:

* Using **EfficientNet-V2**
* Applying **Test Time Augmentation (TTA)**
* Using **k-fold cross validation**
* Applying **class balancing methods**
* Using **ensemble models**

---
