# ===============================================
# test.py — Evaluate Fundus Model on Test Set
# ===============================================
import os, json
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from timm import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
NUM_CLASSES = 5
DATA_DIR = "./datasets/fundus/split_dataset/test"
MODEL_PATH = "fundus_model_best.pth"


# =====================
# DATASET
# =====================
class FundusDataset(Dataset):
    def __init__(self, img_paths, labels, augment=False):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = (
            A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        label = self.labels[idx]
        img = self.transform(image=img)["image"]
        return img, label


# =====================
# LOAD DATASET
# =====================
def load_fundus_dataset():
    img_paths, labels = [], []
    class_names = sorted(os.listdir(DATA_DIR))
    for label, cls in enumerate(class_names):
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_paths.append(os.path.join(folder, file))
                labels.append(label)
    return img_paths, labels, class_names


# =====================
# MODEL
# =====================
def get_model(num_classes=NUM_CLASSES):
    model = create_model("tf_efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# =====================
# TEST LOOP
# =====================
def test_model():
    img_paths, labels, class_names = load_fundus_dataset()
    dataset = FundusDataset(img_paths, labels, augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    model = get_model()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Testing"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            predicted = outputs.argmax(1).cpu().numpy()
            preds.extend(predicted)
            targets.extend(lbls.numpy())

    # =====================
    # METRICS
    # =====================
    overall_acc = accuracy_score(targets, preds)
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")

    # Per-class metrics
    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    test_model()
