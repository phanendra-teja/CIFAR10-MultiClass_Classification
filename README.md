# 🖼️ MultiClass CNN — CIFAR-10 Image Classifier

A Convolutional Neural Network (CNN) built from scratch and compared against Transfer Learning using MobileNetV2, both trained to classify images across **10 object categories** using TensorFlow/Keras. This project was developed as part of coursework/research at NIT Puducherry (NITPY).

---

## 📌 Project Overview

This project implements **Multi-Class Image Classification** on the CIFAR-10 benchmark dataset using two distinct deep learning strategies:

| Notebook | Approach | Description |
|---|---|---|
| `MultiClassCNN-CIFAR10.ipynb` | Custom CNN | Built and trained from scratch using a Sequential CNN |
| `MultiClassCNN-withoutTraining.ipynb` | Transfer Learning | MobileNetV2 (pretrained on ImageNet, base frozen) |

Both models take a **32×32 RGB image** as input and output one of 10 class predictions.

---

## 📂 Dataset — CIFAR-10

- **Source**: Built-in via `tf.keras.datasets.cifar10` — no manual download required
- **Total Images**: 60,000 color images (32×32 pixels, RGB)
- **Classes**: 10 object categories

| Label | Class |
|---|---|
| 0 | Airplane ✈️ |
| 1 | Automobile 🚗 |
| 2 | Bird 🐦 |
| 3 | Cat 🐱 |
| 4 | Deer 🦌 |
| 5 | Dog 🐶 |
| 6 | Frog 🐸 |
| 7 | Horse 🐴 |
| 8 | Ship 🚢 |
| 9 | Truck 🚛 |

### Dataset Split

| Split | Samples | Shape | Description |
|---|---|---|---|
| X_train | 50,000 | (50000, 32, 32, 3) | Training images, normalized |
| y_train | 50,000 | (50000,) | Integer class labels |
| X_test | 10,000 | (10000, 32, 32, 3) | Test images, normalized |
| y_test | 10,000 | (10000,) | Integer class labels |

> Pixel values are normalized to **[0, 1]** by dividing by 255.  
> For the Transfer Learning notebook, labels are **one-hot encoded** to shape `(N, 10)`.

---

## 🧠 Model Architectures

### 1. Custom CNN — Trained from Scratch (`MultiClassCNN-CIFAR10.ipynb`)

A lightweight Sequential CNN built entirely with Keras:

```
Input: (32, 32, 3) — 32×32 RGB image

Conv2D(32 filters, 3×3, ReLU)
MaxPooling2D(2×2)

Conv2D(64 filters, 3×3, ReLU)
MaxPooling2D(2×2)

Flatten()
Dense(64, ReLU)
Dense(10, Softmax)        ← 10-class output probabilities
```

- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Output Activation**: Softmax — outputs a probability distribution over all 10 classes
- **Prediction**: `argmax(output)` → predicted class index

---

### 2. Transfer Learning — MobileNetV2 (`MultiClassCNN-withoutTraining.ipynb`)

Uses **MobileNetV2** pretrained on ImageNet as a frozen feature extractor with a custom classification head:

```
MobileNetV2 Base (weights="imagenet", include_top=False, frozen)
  ↓
GlobalAveragePooling2D()
Dropout(0.3)
Dense(10, Softmax)        ← 10-class output probabilities
```

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam (learning_rate=0.0001)
- **Base Model**: Frozen — ImageNet weights are not updated during training
- **Labels**: One-hot encoded

> MobileNetV2 was designed for large, high-resolution datasets. CIFAR-10's small 32×32 images make this a challenging transfer learning scenario — which highlights the trade-offs between training from scratch vs. reusing pretrained features.

---

## 📈 Training Results — Custom CNN

| Epoch | Training Accuracy | Training Loss |
|---|---|---|
| 1 | ~44% | ~1.57 |
| 2 | ~54% | ~1.31 |
| 3 | ~58% | ~1.19 |
| 4 | ~61% | ~1.10 |
| 5 | ~64% | ~1.02 |
| 6 | ~66% | ~0.96 |
| 7 | ~68% | ~0.91 |
| 8 | ~70% | ~0.86 |
| 9 | ~71% | ~0.82 |
| 10 | ~72% | ~0.79 |

> **Test Accuracy**: ~70% | **Test Loss**: ~0.87  
> *(Update these values with your actual notebook output)*

---

## 📊 Evaluation

Both notebooks include a full evaluation pipeline:

- ✅ Test accuracy and test loss
- 📉 **Confusion Matrix** — 10×10 heatmap (Seaborn / ConfusionMatrixDisplay)
- 📋 **Classification Report** — precision, recall, F1-score per class
- 🖼️ **Random Image Prediction** — displays test image with predicted vs. true label
- 🔢 **Per-Class TP / FP / FN / TN breakdown** *(Custom CNN notebook)*

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core programming language |
| TensorFlow | 2.x | Deep learning framework |
| Keras | (via TF) | High-level neural network API |
| NumPy | latest | Array manipulation |
| Matplotlib | latest | Image & plot visualization |
| Seaborn | latest | Confusion matrix heatmap |
| scikit-learn | latest | Metrics: confusion matrix, classification report |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/phanendra-teja/MultiClassCNN-CIFAR10.git
cd MultiClassCNN-CIFAR10
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy
```

### 4. Run the Notebook

```bash
# Custom CNN
jupyter notebook MultiClassCNN-CIFAR10.ipynb

# Transfer Learning
jupyter notebook MultiClassCNN-withoutTraining.ipynb
```

> ✅ CIFAR-10 is **automatically downloaded** on first run via `tf.keras.datasets` — no manual dataset setup needed.

---

## 📁 Project Structure

```
MultiClassCNN-CIFAR10/
│
├── MultiClassCNN-CIFAR10.ipynb           # Custom CNN — trained from scratch
├── MultiClassCNN-withoutTraining.ipynb   # Transfer Learning — MobileNetV2
├── .gitignore
└── README.md                             # This file
```

---

## 🔮 Sample Prediction

After training, the model predicts on a random test image:

```python
import random, numpy as np

def predict_random_image(model, X, y, classes):
    idx = random.randint(0, len(X) - 1)
    img = X[idx]
    true_label = y[idx]

    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch)
    pred_class = np.argmax(pred[0])

    plt.imshow(img)
    plt.title(f"Predicted: {classes[pred_class]} | True: {classes[true_label]}")
    plt.axis("off")
    plt.show()

predict_random_image(cnn, X_test, y_test, classes)
```

---

## 📌 .gitignore (Recommended)

```
__pycache__/
*.pyc
.venv/
*.h5
*.keras
```

---

## 🎯 Future Improvements

- Train the Custom CNN for **more epochs (30–50+)** — accuracy shows a consistent upward trend and is likely to improve significantly with longer training
- Add **Data Augmentation** (random flips, rotations, zoom) to reduce overfitting and improve generalization
- **Fine-tune MobileNetV2** by unfreezing the top layers for CIFAR-10-specific feature adaptation
- Experiment with deeper architectures like **ResNet50** or **EfficientNetB0** for higher accuracy
- Add a **Learning Rate Scheduler** (e.g., `ReduceLROnPlateau`) for smoother convergence
- Add **Batch Normalization** and **Dropout** layers to stabilize training
- Save and reload trained models using `.h5` or `SavedModel` format
- Build a simple **web demo** using Flask or Streamlit for live image prediction

---

## 👤 Author

**Phanendra Teja V**  
B.Tech CSE — NIT Puducherry (NITPY), Batch 2024–2028

---

## 📄 License

This project is for educational purposes.  
Dataset: CIFAR-10 — available publicly via `tf.keras.datasets`.
