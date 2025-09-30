import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from timm import create_model
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================
# CONFIG
# =====================
BATCH_SIZE = 4
IMG_SIZE = 448
EPOCHS_HEAD = 2
EPOCHS_FINE = 15  # longer fine-tuning
LR_HEAD = 1e-4
LR_FINE = 1e-5
WD = 1e-5
NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# DATA AUGMENTATIONS
# =====================
train_transforms = A.Compose([
    # Use a single tuple `size=(H, W)` exactly
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), ratio=(3/4, 4/3)),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Rotate(limit=15),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
    A.RandomGamma(gamma_limit=(80,120)),
    A.CoarseDropout(max_holes=8, max_height=IMG_SIZE//8, max_width=IMG_SIZE//8, fill_value=0, p=1.0),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])


# =====================
# DATASET CLASS
# =====================
class RetinoDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df
        self.augment = augment
        self.transform = train_transforms if augment else val_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["image_path"]).convert("RGB"))
        label = int(row["label"])
        img = self.transform(image=img)["image"]
        return img, label

# =====================
# FOCAL LOSS
# =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return loss.mean() if self.reduction=="mean" else loss.sum()

# =====================
# LOAD DATA
# =====================
def load_datasets():
    manifests = []

    # APTOS
    aptos_csv = "./datasets/aptos2019-blindness-detection/train.csv"
    aptos_img = "./datasets/aptos2019-blindness-detection/train_images"
    df = pd.read_csv(aptos_csv)
    df["image_path"] = df["id_code"].apply(lambda x: os.path.join(aptos_img, f"{x}.png"))
    df.rename(columns={"diagnosis": "label"}, inplace=True)
    manifests.append(df[["image_path", "label"]])

    # Messidor
    messidor_csv = "./datasets/messidor/messidor_data.csv"
    messidor_img = "./datasets/messidor/messidor-2/messidor-2/preprocess"
    df = pd.read_csv(messidor_csv)
    df["image_path"] = df["id_code"].apply(lambda x: os.path.join(messidor_img, x))
    df.rename(columns={"diagnosis": "label"}, inplace=True)
    manifests.append(df[["image_path", "label"]])

    # IDRiD
    idrid_csv = "./datasets/idrid/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    idrid_img = "./datasets/idrid/B. Disease Grading/1. Original Images/a. Training Set"
    df = pd.read_csv(idrid_csv)
    df["image_path"] = df["Image name"].apply(lambda x: os.path.join(idrid_img, f"{x}.jpg"))
    df.rename(columns={"Retinopathy grade": "label"}, inplace=True)
    manifests.append(df[["image_path", "label"]])

    full_df = pd.concat(manifests, ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return full_df

# =====================
# MODEL
# =====================
def get_model(num_classes=NUM_CLASSES):
    model = create_model("tf_efficientnet_b4", pretrained=True, num_classes=num_classes)
    return model

# =====================
# TRAINING LOOP
# =====================
def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, scheduler=None, stage="head"):
    scaler = GradScaler()
    best_qwk, best_model = -1, None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, preds, targets = 0, [], []

        for imgs, labels in tqdm(train_loader, desc=f"{stage} Epoch {epoch}/{epochs}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(targets, preds)

        # Validation
        val_loss, vpreds, vtargets = 0, [], []
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                vpreds.extend(outputs.argmax(1).cpu().numpy())
                vtargets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(vtargets, vpreds)
        qwk = cohen_kappa_score(vtargets, vpreds, weights="quadratic")

        print(f"{stage} Epoch {epoch}/{epochs} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
              f"Val Acc {val_acc:.4f} | Val QWK {qwk:.4f}")

        if scheduler: scheduler.step()

        if qwk > best_qwk:
            best_qwk = qwk
            best_model = model.state_dict()
            torch.save(best_model, f"best_efficientnet_b4.pth")

    return best_model

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    df = load_datasets()
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)

    train_dataset = RetinoDataset(train_df, augment=True)
    val_dataset = RetinoDataset(val_df, augment=False)

    # Weighted sampler
    class_counts = train_df["label"].value_counts().sort_index().values
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[train_df["label"].values]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = get_model().to(DEVICE)
    criterion = FocalLoss(alpha=1, gamma=2)

    # Stage 1: Train head
    for p in model.parameters(): p.requires_grad = False
    for p in model.get_classifier().parameters(): p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_HEAD, weight_decay=WD)
    train_model(model, train_loader, val_loader, EPOCHS_HEAD, optimizer, criterion, stage="head")

    # Stage 2: Fine-tune full model
    for p in model.parameters(): p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINE, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FINE, eta_min=1e-6)
    best_state = train_model(model, train_loader, val_loader, EPOCHS_FINE, optimizer, criterion, scheduler, stage="finetune")

    # Final evaluation
    model.load_state_dict(torch.load("best_efficientnet_b4.pth"))
    model.eval()
    vpreds, vtargets = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            vpreds.extend(outputs.argmax(1).cpu().numpy())
            vtargets.extend(labels.cpu().numpy())

    print(classification_report(vtargets, vpreds, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(vtargets, vpreds))
    print("QWK:", cohen_kappa_score(vtargets, vpreds, weights="quadratic"))
    print("Accuracy:", accuracy_score(vtargets, vpreds))




# if __name__ == "__main__":
#     df = load_datasets()
#     train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)

#     train_dataset = RetinoDataset(train_df, augment=True)
#     img, label = train_dataset[0]
#     print("Image shape:", img.shape)
#     print("Image min/max:", img.min().item(), img.max().item())
#     print("Label:", label)
