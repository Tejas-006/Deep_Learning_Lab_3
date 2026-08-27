<div align="center">

# 🧠 Deep Learning Lab — Experiment 3
### Convolutional Neural Networks (CNNs) for Image Classification

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![torchvision](https://img.shields.io/badge/torchvision-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

**Course:** CS3807 – Deep Learning Laboratory · **Degree:** B.Tech AI & Data Science
**Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 50,000 train / 10,000 test · 10 classes · 32×32 RGB

📄 [**Read the full lab report (PDF)**](./docs/DL_LAB_3_Report.pdf) · 📓 [**Open the notebook**](./Deep_Learning_Lab_3.ipynb)

</div>

---

## 📌 Objective

Build a working understanding of the Convolutional Neural Network by implementing the convolution operation, pooling, feature map visualization, and a full image classification pipeline in PyTorch, then study how kernel size, stride, padding, pooling type, activation choice, and filter count each shape the result.

```
Input → Conv(16, 3×3) → ReLU → MaxPool → Conv(32, 3×3) → ReLU → MaxPool → Flatten → Dense(128) → ReLU → Dense(10) → Softmax
```

## 📑 Table of Contents

- [1. Imports & Setup](#1-imports--setup)
- [2. Dataset Loading & Exploration](#2-dataset-loading--exploration)
- [3. Task 2 — Effect of Kernel Size](#3-task-2--effect-of-kernel-size)
- [4. Task 3 — Effect of Stride & Padding](#4-task-3--effect-of-stride--padding)
- [5. CNN Architecture](#5-cnn-architecture)
- [6. Training Function](#6-training-function)
- [7. Task 6 — Training the Main CNN](#7-task-6--training-the-main-cnn)
- [8. Task 4 — Feature Map Visualization](#8-task-4--feature-map-visualization)
- [9. Task 5 — Max Pooling vs Average Pooling](#9-task-5--max-pooling-vs-average-pooling)
- [10. Task 7 — Model Evaluation](#10-task-7--model-evaluation)
- [11. Additional Exercises 1 & 2 — Output Size & Parameter Count](#11-additional-exercises-1--2--output-size--parameter-count)
- [12. Additional Exercise 3 — ReLU vs Sigmoid](#12-additional-exercise-3--relu-vs-sigmoid)
- [13. Additional Exercise 5 — Increasing Filters (16 → 64)](#13-additional-exercise-5--increasing-filters-16--64)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [References](#-references)

---

## 1. Imports & Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_score, recall_score, f1_score, accuracy_score)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck']
```

> **What's happening:** Loads PyTorch for the model and training loop, scikit-learn for evaluation metrics, and matplotlib for every figure. Seeds are fixed for reproducibility, and `device` picks up the GPU automatically when Colab provides one. `CLASS_NAMES` maps CIFAR-10's ten integer labels to their real category names.

---

## 2. Dataset Loading & Exploration

```python
from datasets import load_dataset

hf_data = load_dataset("uoft-cs/cifar10")

def convert_split_to_arrays(split_data):
    images_array = np.zeros((len(split_data), 32, 32, 3), dtype=np.uint8)
    labels_array = np.zeros((len(split_data),), dtype=np.int64)
    for i in range(len(split_data)):
        images_array[i] = np.array(split_data[i]['img'])
        labels_array[i] = split_data[i]['label']
    return images_array, labels_array

train_images, train_labels = convert_split_to_arrays(hf_data['train'])
test_images, test_labels = convert_split_to_arrays(hf_data['test'])

class CIFAR10ArrayDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        image_tensor = torch.from_numpy(self.images_array[index]).permute(2, 0, 1).float() / 255.0
        return image_tensor, self.labels_array[index]
```

> **What's happening:** Pulls CIFAR-10 from Hugging Face's CDN (50,000 train / 10,000 test images, `32×32×3`), decodes it once into plain NumPy arrays up front, then wraps it in a small `Dataset` class so `DataLoader` can batch it normally. Ten sample images and the per-class training count are then plotted — the training set comes out perfectly balanced at **5,000 images per class**.

---

## 3. Task 2 — Effect of Kernel Size

```python
kernel_sizes_to_test = [3, 5, 7]
for kernel_size in kernel_sizes_to_test:
    conv_layer = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=kernel_size, stride=1, padding=0)
    output = conv_layer(input_batch)
    output_size_formula = (32 - kernel_size + 2 * 0) // 1 + 1
    print("Kernel size:", kernel_size, "| Output shape:", output.shape, "| Formula:", output_size_formula)
```

> **What's happening:** Applies a single un-trained convolution layer to one `32×32` image at three kernel sizes with stride 1 and no padding, comparing the resulting tensor shape against the `(N−F+2P)/S+1` formula by hand. Results: **3×3 → 30×30**, **5×5 → 28×28**, **7×7 → 26×26** — every step up in kernel size costs exactly 2 pixels per side, matching theory exactly.

---

## 4. Task 3 — Effect of Stride & Padding

```python
for stride_value in [1, 2]:
    conv_layer = nn.Conv2d(3, 8, kernel_size=3, stride=stride_value, padding=0)
    print("Stride:", stride_value, "| Output shape:", conv_layer(input_batch).shape)

for padding_mode in ['same', 'valid']:
    conv_layer = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=padding_mode)
    print("Padding:", padding_mode, "| Output shape:", conv_layer(input_batch).shape)
```

> **What's happening:** Fixes the kernel at `3×3` and instead varies stride and padding. Stride 2 roughly halves the output (**30×30 → 15×15**); `padding='same'` keeps the output identical to the input size (**32×32**), while `padding='valid'` applies none and behaves exactly like the stride-1, padding-0 case from Task 2.

---

## 5. CNN Architecture

```python
class SimpleCNN(nn.Module):
    def __init__(self, conv1_filters=16, conv2_filters=32, pooling_type='max', activation_type='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, stride=1, padding=1)
        self.pool1, self.pool2 = (nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)) if pooling_type == 'max' \
                                   else (nn.AvgPool2d(2, 2), nn.AvgPool2d(2, 2))
        self.flatten_size = conv2_filters * 8 * 8
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.apply_activation(self.conv1(x)))
        x = self.pool2(self.apply_activation(self.conv2(x)))
        x = x.view(x.size(0), self.flatten_size)
        x = self.apply_activation(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
```

> **What's happening:** A configurable CNN — `conv1_filters`, `conv2_filters`, `pooling_type`, and `activation_type` are all parameters, so the exact same class builds every model variant compared later (main model, average-pooling variant, sigmoid variant, and the 16-vs-64-filter comparison) instead of duplicating code four times. Two `3×3` convolutions (padding 1, so only pooling shrinks the spatial size) each followed by activation and `2×2` pooling, then flatten → Dense(128) → Dense(10) → Softmax, exactly matching the manual's required architecture.

---

## 6. Training Function

```python
def train_model(model, train_loader, test_loader, num_epochs, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()  # applied to log(softmax output), since the model already ends in Softmax
    for epoch in range(num_epochs):
        for batch_images, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = loss_function(torch.log(outputs + 1e-10), batch_labels)
            loss.backward()
            optimizer.step()
        # ... validation pass, history logging, per-epoch timing ...
    return history
```

> **What's happening:** One shared training loop used by every model in this notebook — Adam optimizer, and `NLLLoss` on the log of the model's own softmax output (rather than `CrossEntropyLoss`, which expects raw logits and would double-apply softmax here, since the architecture in Section 5 already ends in an explicit `Softmax` layer). Tracks train/validation loss and accuracy plus wall-clock time per epoch, every epoch.

---

## 7. Task 6 — Training the Main CNN

```python
main_model = SimpleCNN(conv1_filters=16, conv2_filters=32, pooling_type='max', activation_type='relu')
main_history = train_model(main_model, train_loader, test_loader, num_epochs=20, learning_rate=0.001)
```

> **What's happening:** Trains the main architecture — 16-then-32 filters, max pooling, ReLU — for 20 epochs, batch size 32, exactly as the manual specifies (**268,650 trainable parameters**). Training accuracy climbs steadily to **90.72%**, but validation accuracy plateaus around epoch 10 near **68%** while validation loss starts climbing from that point on — a textbook overfitting signature, visible clearly in the training/validation accuracy and loss plots.

---

## 8. Task 4 — Feature Map Visualization

```python
main_model.eval()
with torch.no_grad():
    first_conv_output = F.relu(main_model.conv1(sample_batch_for_fmap))
# plot the first 8 of 16 channels
```

> **What's happening:** Runs one real test image through the *trained* model's first convolution layer (not a randomly initialized one) and plots 8 of its 16 output channels. The eight filters shown respond to clearly different things — some pick out broad smooth regions, others fire along sharper localized edges — evidence the layer learned differentiated, non-redundant features rather than collapsing to near-duplicates.

---

## 9. Task 5 — Max Pooling vs Average Pooling

```python
avgpool_model = SimpleCNN(conv1_filters=16, conv2_filters=32, pooling_type='avg', activation_type='relu')
avgpool_history = train_model(avgpool_model, train_loader, test_loader, num_epochs=8, learning_rate=0.001)
```

> **What's happening:** An otherwise identical model, swapping max pooling for average pooling, trained for a reduced 8 epochs (rather than repeating the full 20) to keep total notebook runtime reasonable while still producing a valid accuracy trend — a tradeoff stated explicitly here rather than left implicit. At the matched epoch-8 mark: **max pooling 0.6844 vs average pooling 0.6440** — max pooling is both faster converging and, at 20 epochs, the higher-scoring of the two (**0.6699** final). This result also answers Additional Exercise 4, which asks the same question.

---

## 10. Task 7 — Model Evaluation

```python
test_accuracy  = accuracy_score(all_true_labels, all_predictions)
test_precision = precision_score(all_true_labels, all_predictions, average='macro')
test_recall    = recall_score(all_true_labels, all_predictions, average='macro')
test_f1        = f1_score(all_true_labels, all_predictions, average='macro')
confusion_matrix_result = confusion_matrix(all_true_labels, all_predictions)
```

> **What's happening:** Evaluates the main model on all 10,000 held-out test images. Accuracy, macro-averaged precision/recall/F1, the full classification report, and the confusion matrix are all computed here. The confusion matrix shows vehicle classes (automobile, ship, truck) and frog are cleanly separated (>71% recall each), while confusion concentrates among visually similar animal classes — 236 cats predicted as dogs, 149 cats as frogs.

---

## 11. Additional Exercises 1 & 2 — Output Size & Parameter Count

```python
# Exercise 1: 64x64 image, 5x5 kernel, stride 2, padding 2
output_size = (64 - 5 + 2 * 2) // 2 + 1          # -> 32

# Exercise 2: 64 filters, 3x3 kernel, RGB input
total_conv_parameters = (3 * 3 * 3 + 1) * 64      # -> 1,792
```

> **What's happening:** Two direct applications of the formulas from Section 2/3 of the manual, done by hand rather than by building a real layer. Output size works out to **32×32**; parameter count works out to **1,792** — both cross-checked against the values PyTorch itself reports when you build the equivalent `nn.Conv2d` layer.

---

## 12. Additional Exercise 3 — ReLU vs Sigmoid

```python
sigmoid_model = SimpleCNN(conv1_filters=16, conv2_filters=32, pooling_type='max', activation_type='sigmoid')
sigmoid_history = train_model(sigmoid_model, train_loader, test_loader, num_epochs=8, learning_rate=0.001)
```

> **What's happening:** Same architecture and pooling as the main model, but every ReLU swapped for a sigmoid, trained for 8 epochs. ReLU reaches ~67% validation accuracy within just 4 epochs; sigmoid is still only at 46% at that same point and tops out at **51.9%** after all 8 of its epochs — a direct, visible consequence of sigmoid's vanishing-gradient problem, since its derivative saturates toward zero away from the origin while ReLU's stays a constant 1 for every positive input.

---

## 13. Additional Exercise 5 — Increasing Filters (16 → 64)

```python
model_16f = SimpleCNN(conv1_filters=16, conv2_filters=16, pooling_type='max', activation_type='relu')
model_64f = SimpleCNN(conv1_filters=64, conv2_filters=64, pooling_type='max', activation_type='relu')
# both trained back-to-back, same hardware, same 8 epochs, same everything else
```

> **What's happening:** To isolate the effect of filter count alone (rather than conflating it with the main model's 16-then-32 layout, or with GPU-vs-CPU timing differences), both models here use the *same* filter count in both conv layers and are trained back-to-back on identical hardware for the same 8 epochs. Quadrupling filters (16 → 64) raised validation accuracy by **~4 points** (64.6% → 68.7%) but cost **4.2× the parameters** and **3.2× the time per epoch** — a concrete, controlled look at the accuracy/compute tradeoff of a wider network.

---

## 📊 Results

| Metric | Main CNN (ReLU · Max Pool · 20 ep) |
|:---|:---:|
| Test Accuracy | **0.6699** |
| Precision (macro) | 0.6769 |
| Recall (macro) | 0.6699 |
| F1 score (macro) | 0.6686 |
| Trainable Parameters | 268,650 |

| Comparison | Config A | Config B |
|:---|:---:|:---:|
| Pooling (Task 5 / Ex. 4) | Max — **0.6699** (20 ep) | Average — 0.6440 (8 ep) |
| Activation (Ex. 3) | ReLU — **0.6699** (20 ep) | Sigmoid — 0.5188 (8 ep) |
| Filter count (Ex. 5, controlled) | 16 filters — 0.6464 (8 ep, 135K params) | 64 filters — **0.6870** (8 ep, 564K params) |

## 🔍 Key Findings

- **Overfitting sets in around epoch 10** — training accuracy keeps climbing to 90.7% while validation accuracy plateaus near 68% and validation loss starts rising, a clear generalization gap.
- **Animal classes are the hard ones** — cat, dog, deer, and bird account for most of the confusion matrix's off-diagonal mass; vehicle classes and frog are all cleanly separated at >71% recall.
- **Sigmoid loses badly to ReLU on this architecture** — sigmoid needed all 8 epochs to reach where ReLU already stood at epoch 4, a direct vanishing-gradient effect.
- **Max pooling edges out average pooling** at both a matched epoch count and at convergence, though the gap is modest (~4 points).
- **Wider isn't free** — quadrupling filters bought ~4 points of accuracy for 4.2× the parameters and 3.2× the compute per epoch; a compute-constrained setting may prefer the narrower model.

## ✅ Recommended Configuration

> ReLU activation · Max pooling · Adam optimizer · learning rate 0.001 · batch size 32 · 16-then-32 filters for a compute-efficient baseline, or 64-then-64 filters when the extra ~4 points of accuracy justify roughly 3× the training cost.

---

## 📚 References

1. Goodfellow et al., *Deep Learning*
2. Bishop, *Pattern Recognition and Machine Learning*
3. Haykin, *Neural Networks and Learning Machines*
4. CIFAR-10 Dataset, University of Toronto
5. PyTorch and torchvision Documentation
