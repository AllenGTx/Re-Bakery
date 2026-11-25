# 🥐 Re:Bakery — Google Colab Step-by-Step Guide

> **Asumsi**: Dataset sudah di-mount dari Google Drive dengan struktur 8 folder kelas.
> **Runtime**: Pastikan pakai **T4 GPU** (`Runtime > Change runtime type > T4 GPU`)

---

## 📁 Struktur Dataset di Drive (yang sudah di-mount)

```
/content/drive/MyDrive/
└── Dataset/                  ← root dataset kamu
    ├── Baguette/
    ├── Croissant/
    ├── Cromboloni/
    ├── Donut/
    ├── Hole_Donut/
    ├── Pie_Susu/
    ├── Pretzel/
    └── Sliced_Bread/
```

---

## 📓 NOTEBOOK 1 — Setup, HEIC Conversion & Data Preprocessing

---

### Cell 1 — Install Dependencies

```python
# ============================================================
# CELL 1: Install semua library yang dibutuhkan
# ============================================================
!pip install pillow-heif albumentations timm onnx onnxruntime -q
!pip install tensorboard scikit-learn matplotlib seaborn -q
print("✅ Semua library berhasil diinstall!")
```

---

### Cell 2 — Import & Konfigurasi Global

```python
# ============================================================
# CELL 2: Import semua library & konfigurasi global
# ============================================================
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from PIL import Image
import pillow_heif

# Register HEIC support
pillow_heif.register_heif_opener()

# ─── KONFIGURASI — SESUAIKAN PATH INI ─────────────────────────
DATASET_RAW   = "/content/drive/MyDrive/Dataset"        # ← folder 8 kelas kamu
DATASET_PROC  = "/content/dataset_processed"            # working dir di Colab
CHECKPOINT_DIR = "/content/drive/MyDrive/ReBakery/checkpoints"
EXPORT_DIR    = "/content/drive/MyDrive/ReBakery/exports"

CLASSES = [
    "Baguette", "Croissant", "Cromboloni",
    "Donut", "Hole_Donut", "Pie_Susu",
    "Pretzel", "Sliced_Bread"
]

IMAGE_SIZE   = 224
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)

# Buat folder output
for d in [DATASET_PROC, CHECKPOINT_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(f"{DATASET_PROC}/{split}/{cls}", exist_ok=True)

print("✅ Konfigurasi selesai!")
print(f"   Dataset raw  : {DATASET_RAW}")
print(f"   Dataset proc : {DATASET_PROC}")
print(f"   Kelas        : {CLASSES}")
```

---

### Cell 3 — Audit Dataset Raw

```python
# ============================================================
# CELL 3: Audit & tampilkan jumlah gambar per kelas
# ============================================================
print("📊 AUDIT DATASET RAW\n" + "="*40)
total = 0
class_counts = {}
for cls in CLASSES:
    folder = os.path.join(DATASET_RAW, cls)
    if os.path.isdir(folder):
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.heic','.jpg','.jpeg','.png','.webp'))]
        class_counts[cls] = len(files)
        total += len(files)
        status = "✅" if len(files) >= 30 else "⚠️"
        print(f"  {status} {cls:<15} : {len(files)} gambar")
    else:
        class_counts[cls] = 0
        print(f"  🚨 {cls:<15} : FOLDER TIDAK DITEMUKAN!")

print(f"\n  Total : {total} gambar")

# Plot distribusi
plt.figure(figsize=(10, 4))
plt.bar(class_counts.keys(), class_counts.values(),
        color='#D4A017', edgecolor='#8B4513')
plt.title("📊 Distribusi Gambar per Kelas", fontsize=13)
plt.xlabel("Kelas")
plt.ylabel("Jumlah Gambar")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()
```

---

### Cell 4 — Konversi HEIC → JPG & Copy ke Processed

```python
# ============================================================
# CELL 4: Konversi semua gambar (HEIC/JPG/PNG) → JPG 224x224
#          & Stratified split train/val/test
# ============================================================
from sklearn.model_selection import train_test_split

def convert_and_resize(src_path, dst_path, size=IMAGE_SIZE):
    """Buka gambar (HEIC/JPG/PNG), resize, simpan sebagai JPG."""
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dst_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"    ⚠️  Gagal convert {src_path}: {e}")
        return False

VALID_EXT = ('.heic', '.jpg', '.jpeg', '.png', '.webp')
processed_counts = {"train": {}, "val": {}, "test": {}}

print("🔄 Memulai konversi & split dataset...\n")

for cls in CLASSES:
    src_folder = os.path.join(DATASET_RAW, cls)
    if not os.path.isdir(src_folder):
        print(f"  🚨 Skip {cls}: folder tidak ditemukan")
        continue

    # Kumpulkan semua file valid
    files = sorted([
        f for f in os.listdir(src_folder)
        if f.lower().endswith(VALID_EXT)
    ])

    if len(files) == 0:
        print(f"  🚨 Skip {cls}: tidak ada gambar")
        continue

    # Stratified split
    train_files, temp_files = train_test_split(
        files, test_size=(1 - TRAIN_RATIO), random_state=SEED
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=0.5, random_state=SEED
    )

    splits = {"train": train_files, "val": val_files, "test": test_files}

    for split, split_files in splits.items():
        count = 0
        for fname in split_files:
            src = os.path.join(src_folder, fname)
            # Ganti ekstensi jadi .jpg
            dst_name = Path(fname).stem + ".jpg"
            dst = os.path.join(DATASET_PROC, split, cls, dst_name)
            if convert_and_resize(src, dst):
                count += 1
        processed_counts[split][cls] = count

    total_cls = len(files)
    print(f"  ✅ {cls:<15}: {len(train_files)} train | {len(val_files)} val | {len(test_files)} test  (dari {total_cls})")

print("\n✅ Konversi & split selesai!")
```

---

### Cell 5 — Verifikasi & Preview Hasil

```python
# ============================================================
# CELL 5: Verifikasi split & tampilkan sample gambar
# ============================================================
print("📊 RINGKASAN DATASET PROCESSED\n" + "="*50)
print(f"{'Kelas':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
print("-"*50)
grand_total = 0
for cls in CLASSES:
    tr = processed_counts["train"].get(cls, 0)
    vl = processed_counts["val"].get(cls, 0)
    te = processed_counts["test"].get(cls, 0)
    tot = tr + vl + te
    grand_total += tot
    print(f"{cls:<15} {tr:>8} {vl:>8} {te:>8} {tot:>8}")
print("-"*50)
print(f"{'TOTAL':<15} {sum(v.get(c,0) for c in CLASSES for v in [processed_counts['train']]):>8} "
      f"{sum(v.get(c,0) for c in CLASSES for v in [processed_counts['val']]):>8} "
      f"{sum(v.get(c,0) for c in CLASSES for v in [processed_counts['test']]):>8} "
      f"{grand_total:>8}")

# Preview sample gambar per kelas
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, cls in enumerate(CLASSES):
    train_dir = os.path.join(DATASET_PROC, "train", cls)
    imgs = os.listdir(train_dir)
    if imgs:
        img_path = os.path.join(train_dir, imgs[0])
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(cls, fontsize=11, fontweight='bold')
    axes[i].axis('off')
plt.suptitle("🥐 Sample Gambar per Kelas (dari Train Set)", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 📓 NOTEBOOK 2 — Model Training (EfficientNet-B0)

---

### Cell 6 — Import Torch & Dataset Loader

```python
# ============================================================
# CELL 6: Import PyTorch & setup Dataset class
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ─── HYPERPARAMETER CONFIG ─────────────────────────────────
CONFIG = {
    "image_size"    : 224,
    "batch_size"    : 16,
    "num_workers"   : 2,
    "num_classes"   : 8,
    "classes"       : CLASSES,

    # Phase A — frozen backbone
    "lr_phase_a"    : 1e-3,
    "epochs_phase_a": 15,

    # Phase B — fine-tuning
    "lr_phase_b"    : 1e-4,
    "epochs_phase_b": 35,
    "weight_decay"  : 1e-4,
    "unfreeze_ratio": 0.30,

    # Regularization
    "dropout_1"     : 0.35,
    "dropout_2"     : 0.20,
    "label_smoothing": 0.10,

    # Scheduler
    "lr_factor"     : 0.5,
    "lr_patience"   : 5,

    # Early stopping
    "early_stop_patience": 10,

    # Paths
    "dataset_proc"  : DATASET_PROC,
    "checkpoint_dir": CHECKPOINT_DIR,
}

print("✅ Config ready!")
```

---

### Cell 7 — Augmentasi & Dataset Class

```python
# ============================================================
# CELL 7: Augmentation pipeline & Custom Dataset
# ============================================================

# Training augmentation — agresif untuk dataset kecil
train_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3,
                  saturation=0.2, hue=0.05, p=0.5),
    A.OneOf([
        A.GaussNoise(p=1.0),
        A.GaussianBlur(blur_limit=3, p=1.0),
        A.MotionBlur(blur_limit=3, p=1.0),
    ], p=0.3),
    A.CoarseDropout(max_holes=8, max_height=20,
                    max_width=20, min_holes=1, p=0.25),
    A.RandomPerspective(distortion_scale=0.2, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                       rotate_limit=10, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Val/Test — hanya resize & normalize
val_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class BakeryDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


# Buat dataset objects
train_ds = BakeryDataset(f"{DATASET_PROC}/train", CLASSES, train_transform)
val_ds   = BakeryDataset(f"{DATASET_PROC}/val",   CLASSES, val_transform)
test_ds  = BakeryDataset(f"{DATASET_PROC}/test",  CLASSES, val_transform)

print(f"✅ Dataset loaded:")
print(f"   Train : {len(train_ds)} gambar")
print(f"   Val   : {len(val_ds)} gambar")
print(f"   Test  : {len(test_ds)} gambar")
```

---

### Cell 8 — DataLoader dengan WeightedSampler

```python
# ============================================================
# CELL 8: DataLoader + WeightedRandomSampler (handle imbalance)
# ============================================================

# Hitung bobot per sample untuk handle class imbalance
labels_train = [s[1] for s in train_ds.samples]
class_counts = Counter(labels_train)
class_weights = {
    cls_idx: 1.0 / class_counts.get(cls_idx, 1)
    for cls_idx in range(CONFIG["num_classes"])
}
sample_weights = [class_weights[lbl] for lbl in labels_train]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=CONFIG["batch_size"],
    sampler=sampler,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)
test_loader = DataLoader(
    test_ds,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
)

print(f"✅ DataLoader ready!")
print(f"   Train batches : {len(train_loader)}")
print(f"   Val batches   : {len(val_loader)}")

# Distribusi per kelas di train set
print("\n📊 Distribusi kelas di Train Set:")
for cls, idx in train_ds.class_to_idx.items():
    print(f"   {cls:<15}: {class_counts.get(idx, 0)} gambar  (weight={class_weights.get(idx, 0):.4f})")
```

---

### Cell 9 — Build Model EfficientNet-B0

```python
# ============================================================
# CELL 9: Bangun model EfficientNet-B0 dengan custom head
# ============================================================

class ReBakeryModel(nn.Module):
    def __init__(self, num_classes, dropout1=0.35, dropout2=0.20):
        super().__init__()
        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,          # hapus classifier asli
            global_pool='avg'       # global average pooling
        )
        in_features = self.backbone.num_features  # 1280

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        """Bekukan semua layer backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_top_layers(self, ratio=0.30):
        """Unfreeze top `ratio` persen layer backbone."""
        all_params = list(self.backbone.parameters())
        n_unfreeze = int(len(all_params) * ratio)
        # Unfreeze dari belakang (layer teratas)
        for param in all_params[-n_unfreeze:]:
            param.requires_grad = True
        total_frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total_unfrozen = sum(1 for p in self.backbone.parameters() if p.requires_grad)
        print(f"   Backbone — frozen: {total_frozen}, unfrozen: {total_unfrozen}")


model = ReBakeryModel(
    num_classes=CONFIG["num_classes"],
    dropout1=CONFIG["dropout_1"],
    dropout2=CONFIG["dropout_2"]
).to(DEVICE)

# Hitung total parameter
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model built!")
print(f"   Total params     : {total_params:,}")
print(f"   Trainable params : {trainable_params:,}")
```

---

### Cell 10 — Helper Functions (Training & Evaluation)

```python
# ============================================================
# CELL 10: Helper functions — train_epoch, evaluate
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#D4A017")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="#8B4513")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="#D4A017")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",   color="#8B4513")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

print("✅ Helper functions ready!")
```

---

### Cell 11 — Phase A: Training (Frozen Backbone)

```python
# ============================================================
# CELL 11: PHASE A — Feature Extraction (Frozen Backbone)
#          Epoch 1-15, hanya train custom head
# ============================================================
print("="*55)
print("🔒 PHASE A: FROZEN BACKBONE — Training Head Only")
print("="*55)

# Bekukan backbone
model.freeze_backbone()

# Optimizer hanya update head
optimizer_a = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["lr_phase_a"]
)
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
scheduler_a = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_a, mode='min',
    factor=CONFIG["lr_factor"],
    patience=CONFIG["lr_patience"]
)

history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [],  "val_acc":  []
}

best_val_acc = 0.0
best_epoch = 0

for epoch in range(1, CONFIG["epochs_phase_a"] + 1):
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer_a, criterion)
    vl_loss, vl_acc = evaluate(model, val_loader, criterion)
    scheduler_a.step(vl_loss)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    # Save best model
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_epoch = epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_acc": vl_acc,
            "config": CONFIG,
        }, os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"))

    print(f"Epoch [{epoch:2d}/{CONFIG['epochs_phase_a']}] "
          f"Loss: {tr_loss:.4f}/{vl_loss:.4f}  "
          f"Acc: {tr_acc*100:.1f}%/{vl_acc*100:.1f}%  "
          f"LR: {optimizer_a.param_groups[0]['lr']:.6f}"
          f" {'← BEST' if epoch == best_epoch else ''}")

print(f"\n✅ Phase A selesai! Best Val Acc: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
plot_history(history)
```

---

### Cell 12 — Phase B: Fine-tuning (Unfreeze Top Layers)

```python
# ============================================================
# CELL 12: PHASE B — Fine-tuning (Unfreeze Top 30% Backbone)
#          Epoch 1-35, train backbone + head bersama
# ============================================================
print("="*55)
print("🔓 PHASE B: FINE-TUNING — Unfreeze Top 30% Backbone")
print("="*55)

# Unfreeze top layers
model.unfreeze_top_layers(ratio=CONFIG["unfreeze_ratio"])

optimizer_b = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["lr_phase_b"],
    weight_decay=CONFIG["weight_decay"]
)
scheduler_b = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_b, mode='min',
    factor=CONFIG["lr_factor"],
    patience=CONFIG["lr_patience"]
)

# Early stopping
no_improve = 0

for epoch in range(1, CONFIG["epochs_phase_b"] + 1):
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer_b, criterion)
    vl_loss, vl_acc = evaluate(model, val_loader, criterion)
    scheduler_b.step(vl_loss)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    # Save best model
    improved = ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_epoch = CONFIG["epochs_phase_a"] + epoch
        no_improve = 0
        improved = " ← BEST ✨"
        torch.save({
            "epoch": best_epoch,
            "model_state": model.state_dict(),
            "val_acc": vl_acc,
            "config": CONFIG,
        }, os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"))
    else:
        no_improve += 1

    print(f"Epoch [{epoch:2d}/{CONFIG['epochs_phase_b']}] "
          f"Loss: {tr_loss:.4f}/{vl_loss:.4f}  "
          f"Acc: {tr_acc*100:.1f}%/{vl_acc*100:.1f}%"
          f"{improved}")

    # Early stopping
    if no_improve >= CONFIG["early_stop_patience"]:
        print(f"\n🛑 Early stopping di epoch {epoch} (tidak ada peningkatan {no_improve} epoch)")
        break

print(f"\n✅ Phase B selesai! Best Val Acc: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
plot_history(history)
```

---

## 📓 NOTEBOOK 3 — Evaluasi Model

---

### Cell 13 — Load Best Model & Test Set Evaluation

```python
# ============================================================
# CELL 13: Load model terbaik & evaluasi di TEST SET
# ============================================================
from sklearn.metrics import classification_report, confusion_matrix

# Load best checkpoint
checkpoint = torch.load(
    os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"),
    map_location=DEVICE
)
model.load_state_dict(checkpoint["model_state"])
print(f"✅ Loaded best model dari epoch {checkpoint['epoch']}, "
      f"Val Acc: {checkpoint['val_acc']*100:.2f}%")

# Evaluasi di test set
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Classification Report
test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"\n📊 TEST ACCURACY: {test_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=3))
```

---

### Cell 14 — Confusion Matrix

```python
# ============================================================
# CELL 14: Confusion Matrix Heatmap
# ============================================================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='YlOrBr',
    xticklabels=CLASSES, yticklabels=CLASSES,
    linewidths=0.5
)
plt.title(f"🥐 Re:Bakery — Confusion Matrix (Test Acc: {test_acc*100:.2f}%)",
          fontsize=13, fontweight='bold')
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("Actual", fontsize=11)
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(EXPORT_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
```

---

### Cell 15 — Prediksi Visual (Sanity Check)

```python
# ============================================================
# CELL 15: Visual sanity check — lihat prediksi vs label asli
# ============================================================
EMOJI = {"Baguette":"🥖","Croissant":"🥐","Cromboloni":"🌀",
         "Donut":"🍩","Hole_Donut":"⭕","Pie_Susu":"🥧",
         "Pretzel":"🥨","Sliced_Bread":"🍞"}

model.eval()
imgs_batch, labels_batch = next(iter(test_loader))
with torch.no_grad():
    outputs = model(imgs_batch.to(DEVICE))
    probs = torch.softmax(outputs, dim=1).cpu()
    preds = probs.argmax(dim=1)

# Denormalize untuk tampil
inv_mean = torch.tensor([0.485, 0.456, 0.406])
inv_std  = torch.tensor([0.229, 0.224, 0.225])

fig, axes = plt.subplots(3, 5, figsize=(18, 11))
axes = axes.flatten()

for i in range(min(15, len(imgs_batch))):
    img = imgs_batch[i].permute(1, 2, 0)
    img = img * inv_std + inv_mean
    img = img.clamp(0, 1).numpy()

    pred_cls = CLASSES[preds[i]]
    true_cls = CLASSES[labels_batch[i]]
    conf = probs[i][preds[i]].item()
    color = "green" if pred_cls == true_cls else "red"
    emoji = EMOJI.get(pred_cls, "")

    axes[i].imshow(img)
    axes[i].set_title(
        f"{emoji} {pred_cls}\n(True: {true_cls})\n{conf*100:.1f}%",
        fontsize=9, color=color, fontweight='bold'
    )
    axes[i].axis('off')

plt.suptitle("🔍 Prediksi vs Label Asli — Hijau=Benar, Merah=Salah", fontsize=12)
plt.tight_layout()
plt.show()
```

---

## 📓 NOTEBOOK 4 — Export Model

---

### Cell 16 — Export ke ONNX

```python
# ============================================================
# CELL 16: Export model ke format ONNX
# ============================================================
import onnx
import onnxruntime as ort

model.eval()
model.to("cpu")

dummy_input = torch.randn(1, 3, CONFIG["image_size"], CONFIG["image_size"])
onnx_path = os.path.join(EXPORT_DIR, "rebakery.onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

# Verifikasi ONNX
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print(f"✅ ONNX model valid! Disimpan di: {onnx_path}")

# Test inference ONNX vs PyTorch
session = ort.InferenceSession(onnx_path)
test_input = dummy_input.numpy()
onnx_out = session.run(None, {"input": test_input})[0]

model.to("cpu")
with torch.no_grad():
    torch_out = model(dummy_input).numpy()

diff = np.abs(onnx_out - torch_out).max()
print(f"✅ Max diff PyTorch vs ONNX: {diff:.6f}  ({'OK ✅' if diff < 1e-4 else 'PERLU CEK ⚠️'})")
```

---

### Cell 17 — Export ke TensorFlow.js (untuk realtime kamera di web)

```python
# ============================================================
# CELL 17: Export ONNX → TF → TF.js
# ============================================================
!pip install onnx-tf tensorflow tensorflowjs -q

import subprocess

tf_model_path = os.path.join(EXPORT_DIR, "tf_model")
tfjs_model_path = os.path.join(EXPORT_DIR, "tfjs_model")

# Step 1: ONNX → TensorFlow SavedModel
print("🔄 Step 1: ONNX → TensorFlow SavedModel...")
!onnx-tf convert -i {onnx_path} -o {tf_model_path}

# Step 2: TensorFlow → TF.js + Quantization
print("🔄 Step 2: TF SavedModel → TF.js (dengan uint8 quantization)...")
!tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --quantize_uint8="*" \
    --signature_name=serving_default \
    {tf_model_path} {tfjs_model_path}

# Cek ukuran
import os
total_size = sum(
    os.path.getsize(os.path.join(tfjs_model_path, f))
    for f in os.listdir(tfjs_model_path)
    if os.path.isfile(os.path.join(tfjs_model_path, f))
)
print(f"\n✅ TF.js model disimpan di: {tfjs_model_path}")
print(f"   Ukuran total: {total_size / 1024 / 1024:.2f} MB")
print(f"   File: {os.listdir(tfjs_model_path)}")
```

---

### Cell 18 — Simpan Class Labels & Ringkasan Final

```python
# ============================================================
# CELL 18: Simpan class labels JSON & ringkasan akhir
# ============================================================
import json

# Simpan label mapping untuk dipakai di web app
labels_info = {
    "classes": CLASSES,
    "class_to_idx": {cls: i for i, cls in enumerate(CLASSES)},
    "idx_to_class": {i: cls for i, cls in enumerate(CLASSES)},
    "emoji": {
        "Baguette"    : "🥖",
        "Croissant"   : "🥐",
        "Cromboloni"  : "🌀",
        "Donut"       : "🍩",
        "Hole_Donut"  : "⭕",
        "Pie_Susu"    : "🥧",
        "Pretzel"     : "🥨",
        "Sliced_Bread": "🍞"
    },
    "model_info": {
        "architecture" : "EfficientNet-B0",
        "image_size"   : CONFIG["image_size"],
        "best_val_acc" : round(best_val_acc, 4),
        "test_acc"     : round(test_acc, 4),
    }
}

labels_path = os.path.join(EXPORT_DIR, "labels.json")
with open(labels_path, "w") as f:
    json.dump(labels_info, f, indent=2)

print("✅ SEMUA EXPORT SELESAI!\n")
print("="*50)
print("📁 File yang siap dipakai:")
for fname in os.listdir(EXPORT_DIR):
    fpath = os.path.join(EXPORT_DIR, fname)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath) / 1024 / 1024
        print(f"   {fname:<30} {size:.2f} MB")
    else:
        print(f"   {fname}/ (folder)")

print("\n🎯 RINGKASAN PERFORMA:")
print(f"   Best Val Accuracy : {best_val_acc*100:.2f}%")
print(f"   Test Accuracy     : {test_acc*100:.2f}%")
print(f"   Status            : {'✅ Target ≥85% TERCAPAI!' if test_acc >= 0.85 else '⚠️ Perlu iterasi lebih lanjut'}")
print("\n🚀 Next step: Upload file dari EXPORT_DIR ke web app!")
```

---

## 🗂️ Ringkasan File Output

| File | Digunakan untuk |
|---|---|
| `best_model.pth` | Backup checkpoint PyTorch |
| `rebakery.onnx` | Inference di backend FastAPI |
| `tfjs_model/` | Inference realtime kamera di browser |
| `labels.json` | Mapping kelas untuk frontend |
| `confusion_matrix.png` | Dokumentasi evaluasi |

---

*Re:Bakery Colab Guide — v1.0 — October 2026*
