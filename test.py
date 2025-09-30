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
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Config
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "./best_efficientnet_b4.pth"
IMG_SIZE = 448
BATCH_SIZE = 16
NUM_CLASSES = 5

# Training datasets (for info, weights already trained on these)
aptos_csv = "./datasets/aptos2019-blindness-detection/train.csv"
aptos_img = "./datasets/aptos2019-blindness-detection/train_images"

messidor_csv = "./datasets/messidor/messidor_data.csv"
messidor_img = "./datasets/messidor/messidor-2/messidor-2/preprocess"

# IDRiD dataset (testing labels)
idrid_csv = "./datasets/idrid/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
idrid_img = "./datasets/idrid/B. Disease Grading/1. Original Images/b. Testing Set"

# -----------------------
# Build manifest
# -----------------------
def build_manifest():
    manifests = []

    # IDRiD
    if os.path.exists(idrid_csv):
        df = pd.read_csv(idrid_csv)
        if "Image name" in df.columns and "Retinopathy grade" in df.columns:
            df["image_path"] = df["Image name"].apply(lambda x: os.path.join(idrid_img, f"{x}.jpg"))
            df.rename(columns={"Retinopathy grade": "label"}, inplace=True)
            manifests.append(df[["image_path", "label"]])
        else:
            print("IDRiD CSV missing expected columns:", df.columns.tolist())
    else:
        print("IDRiD CSV not found:", idrid_csv)

    # Optional: add APTOS train CSV (only if you want labels, usually for analysis)
    if os.path.exists(aptos_csv):
        df = pd.read_csv(aptos_csv)
        if "id_code" in df.columns and "diagnosis" in df.columns:
            df["image_path"] = df["id_code"].apply(lambda x: os.path.join(aptos_img, f"{x}.png"))
            df.rename(columns={"diagnosis": "label"}, inplace=True)
            manifests.append(df[["image_path", "label"]])

    # Optional: add Messidor CSV (if labeled)
    if os.path.exists(messidor_csv):
        df = pd.read_csv(messidor_csv)
        # Try common column names
        if "id_code" in df.columns and "diagnosis" in df.columns:
            df["image_path"] = df["id_code"].apply(lambda x: os.path.join(messidor_img, x))
            df.rename(columns={"diagnosis": "label"}, inplace=True)
            manifests.append(df[["image_path", "label"]])
        elif "image" in df.columns and "level" in df.columns:
            df["image_path"] = df["image"].apply(lambda x: os.path.join(messidor_img, x))
            df.rename(columns={"level": "label"}, inplace=True)
            manifests.append(df[["image_path", "label"]])

    if not manifests:
        raise RuntimeError("No manifests created — check CSV paths and column names.")

    full = pd.concat(manifests, ignore_index=True)
    full = full.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return full

# -----------------------
# Dataset & transforms
# -----------------------
val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

class RetinoTestDataset(Dataset):
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
        return img, label, path

# -----------------------
# Evaluation
# -----------------------
def evaluate_model(model, loader):
    model.eval()
    preds, labels, paths = [], [], []
    with torch.no_grad():
        for imgs, lbls, pths in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            batch_preds = outputs.argmax(1).cpu().numpy()
            preds.extend(batch_preds.tolist())
            labels.extend(lbls.numpy().tolist())
            paths.extend(pths)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    return acc, f1, qwk, preds, labels, paths

# -----------------------
# Main
# -----------------------
def main():
    print("Building manifest from IDRiD + optional other datasets...")
    manifest = build_manifest()
    print("Total samples:", len(manifest))

    test_ds = RetinoTestDataset(manifest, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("Creating EfficientNet-B4 and loading weights:", MODEL_SAVE_PATH)
    model = create_model("tf_efficientnet_b4", pretrained=False, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model weights not found: {MODEL_SAVE_PATH}")

    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    model.load_state_dict(sd)

    acc, f1, qwk, preds, labels, paths = evaluate_model(model, test_loader)
    print(f"\nTest Results — samples: {len(labels)}")
    print("Accuracy:", acc)
    print("Macro F1:", f1)
    print("Quadratic Weighted Kappa:", qwk)
    print("\nClassification report:\n", classification_report(labels, preds, digits=4))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:\n", cm)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
