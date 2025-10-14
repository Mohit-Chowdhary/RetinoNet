# test_b4.py
import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from timm import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./datasets/funfus/split_dataset/test"
MODEL_SAVE_PATH = "./best_efficientnet_b4.pth"  # make sure B4 weights exist here
IMG_SIZE = 448
BATCH_SIZE = 16
NUM_CLASSES = 5

# -----------------------
# Build manifest
# -----------------------
def build_manifest(root_dir):
    paths, labels = [], []
    for label_name in sorted(os.listdir(root_dir)):
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        label = int(label_name)
        for fname in os.listdir(label_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                paths.append(os.path.join(label_path, fname))
                labels.append(label)
    df = pd.DataFrame({"image_path": paths, "label": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# -----------------------
# Dataset & transforms
# -----------------------
val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class FunfusDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = int(row["label"])
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

# -----------------------
# Evaluation
# -----------------------
def evaluate_model(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            batch_preds = outputs.argmax(1).cpu().numpy()
            preds.extend(batch_preds)
            labels.extend(lbls.numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    return acc, f1, qwk, preds, labels

# -----------------------
# Main
# -----------------------
def main():
    print("Building manifest from:", DATA_DIR)
    manifest = build_manifest(DATA_DIR)
    print("Total samples:", len(manifest))

    test_ds = FunfusDataset(manifest, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("Creating EfficientNet-B4 model and loading weights...")
    model = create_model("tf_efficientnet_b4", pretrained=False, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model weights not found: {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    acc, f1, qwk, preds, labels = evaluate_model(model, test_loader)

    print("\nResults on Funfus Test Set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")
    print("\nClassification Report:\n", classification_report(labels, preds, digits=4))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Funfus Test Set")
    plt.show()

if __name__ == "__main__":
    main()
