# ===============================================
# train_fundus.py — Full training using tuned hyperparameters
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
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
NUM_CLASSES = 5
DATA_DIR = "./datasets/fundus/split_dataset/test"


class fundusDataset(Dataset):
    def __init__(self, img_paths, labels, augment=False):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = (
            A.Compose([
                A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(0.8,1.0)),
                A.HorizontalFlip(), A.VerticalFlip(), A.Rotate(limit=15),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ]) if augment else
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1-pt)**self.gamma * ce_loss).mean()


def load_fundus_dataset():
    img_paths, labels = [], []
    class_names = sorted(os.listdir(DATA_DIR))
    for label, cls in enumerate(class_names):
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(folder): continue
        for file in os.listdir(folder):
            if file.lower().endswith((".png",".jpg",".jpeg")):
                img_paths.append(os.path.join(folder,file))
                labels.append(label)
    return img_paths, labels


def get_model(num_classes=NUM_CLASSES):
    model = create_model("tf_efficientnet_b0", pretrained=True, num_classes=num_classes)
    return model


def train_full(hyperparams, epochs=20):
    # Load dataset
    img_paths, labels = load_fundus_dataset()
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_ds = fundusDataset(train_paths, train_labels, augment=True)
    val_ds = fundusDataset(val_paths, val_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=2)

    model = get_model().to(DEVICE)
    criterion = FocalLoss(alpha=1, gamma=hyperparams["focal_gamma"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])

    scaler = GradScaler()
    best_qwk = -1
    best_model = None

    for epoch in range(epochs):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Evaluate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(imgs)
                preds.extend(outputs.argmax(1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
        qwk = cohen_kappa_score(targets, preds, weights="quadratic")
        print(f"Epoch {epoch+1}/{epochs} | Val QWK: {qwk:.4f}")

        if qwk > best_qwk:
            best_qwk = qwk
            best_model = model.state_dict()

    # Save best model
    torch.save(best_model, "fundus_model_best.pth")
    print("✅ Training complete. Model saved as fundus_model_best.pth")
    print("Best QWK:", best_qwk)


if __name__ == "__main__":
    # Load hyperparameters from tuning
    with open("best_hyperparams.json", "r") as f:
        best_hyperparams = json.load(f)
    print("Using hyperparameters:", best_hyperparams)

    # Run full training
    train_full(best_hyperparams, epochs=25)
