"""
Phase 1 step 2 — Train EfficientNet-B0 multi-label (4 dental diseases)
รัน: python train_multilabel.py
ต้องมี: data/train_labels.csv, data/val_labels.csv (จาก prepare_labels.py)
Output: models/best_model.pth, models/training_curve.png, models/metrics.json

Design:
- 4 disease heads (sigmoid multi-label): Caries / Deep Caries / Periapical Lesion / Impacted
- "Normal" = ทุก head < threshold (ไม่ใช่ class แยก เพราะ DENTEX fully-labeled)
- Focal loss แก้ class imbalance (Caries dominant ~74%)
- Transfer learning EfficientNet-B0 (timm, pretrained ImageNet)
"""
import json
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── config ─────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

DISEASES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
N_CLASSES = len(DISEASES)

CFG = {
    "img_size": 512,
    "batch_size": 8,          # RTX 5060 8GB → 8 พอ (ลดเป็น 4 ถ้า OOM)
    "epochs": 40,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "backbone": "efficientnet_b0",
    "num_workers": 4,
    "focal_gamma": 2.0,
    "early_stop_patience": 8,
    "seed": 42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ── dataset ────────────────────────────────────────────────
class DentexDataset(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["filepath"]).convert("RGB")
        img = np.array(img)
        img = self.tfm(image=img)["image"]
        label = torch.tensor([row[d] for d in DISEASES], dtype=torch.float32)
        return img, label


def get_transforms(train: bool):
    sz = CFG["img_size"]
    if train:
        return A.Compose([
            A.Resize(sz, sz),
            A.HorizontalFlip(p=0.5),          # ปากซ้าย/ขวา symmetric → flip ปลอดภัย
            A.Rotate(limit=8, p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),    # boost contrast X-ray
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── focal loss (multi-label) ──────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal = (1 - p_t) ** self.gamma * bce
        return focal.mean()


# ── model ──────────────────────────────────────────────────
def build_model():
    model = timm.create_model(
        CFG["backbone"], pretrained=True, num_classes=N_CLASSES)
    return model.to(DEVICE)


# ── eval ───────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_targets = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_targets.append(y)
    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    metrics = {"per_class": {}}
    aucs, f1s = [], []
    for i, d in enumerate(DISEASES):
        t = targets[:, i]
        if t.sum() == 0 or t.sum() == len(t):
            auc = float("nan")
        else:
            auc = roc_auc_score(t, probs[:, i])
        ap = average_precision_score(t, probs[:, i]) if t.sum() > 0 else float("nan")
        f1 = f1_score(t, preds[:, i], zero_division=0)
        metrics["per_class"][d] = {"auc": round(float(auc), 4),
                                   "ap": round(float(ap), 4),
                                   "f1": round(float(f1), 4)}
        if not np.isnan(auc):
            aucs.append(auc)
        f1s.append(f1)
    metrics["mean_auc"] = round(float(np.mean(aucs)), 4) if aucs else 0.0
    metrics["mean_f1"] = round(float(np.mean(f1s)), 4)
    return metrics, probs, targets


# ── train loop ─────────────────────────────────────────────
def main():
    set_seed(CFG["seed"])
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df = pd.read_csv(DATA / "train_labels.csv")
    val_df = pd.read_csv(DATA / "val_labels.csv")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    # pos_weight = neg/pos ต่อคลาส (แก้ imbalance)
    pos = train_df[DISEASES].sum().values
    neg = len(train_df) - pos
    pos_weight = torch.tensor(neg / np.clip(pos, 1, None),
                              dtype=torch.float32).to(DEVICE)
    print("pos_weight:", dict(zip(DISEASES, pos_weight.cpu().numpy().round(2))))

    train_ds = DentexDataset(train_df, get_transforms(True))
    val_ds = DentexDataset(val_df, get_transforms(False))
    train_ld = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True)

    model = build_model()
    criterion = FocalLoss(gamma=CFG["focal_gamma"], pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    history = {"train_loss": [], "val_auc": [], "val_f1": []}
    best_auc, best_epoch, patience = 0.0, 0, 0

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        running = 0.0
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running / len(train_ds)

        metrics, _, _ = evaluate(model, val_ld)
        history["train_loss"].append(train_loss)
        history["val_auc"].append(metrics["mean_auc"])
        history["val_f1"].append(metrics["mean_f1"])

        print(f"[{epoch:02d}/{CFG['epochs']}] loss={train_loss:.4f} "
              f"val_AUC={metrics['mean_auc']:.4f} val_F1={metrics['mean_f1']:.4f}")

        if metrics["mean_auc"] > best_auc:
            best_auc, best_epoch, patience = metrics["mean_auc"], epoch, 0
            torch.save({"model": model.state_dict(), "cfg": CFG,
                        "diseases": DISEASES, "metrics": metrics},
                       MODELS / "best_model.pth")
            with open(MODELS / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"  ✓ saved best (AUC={best_auc:.4f})")
        else:
            patience += 1
            if patience >= CFG["early_stop_patience"]:
                print(f"Early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    # ── plot curves ────────────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(history["train_loss"], label="train loss", color="#E91E63")
    a1.set_title("Training Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(history["val_auc"], label="val mean AUC", color="#2196F3")
    a2.plot(history["val_f1"], label="val mean F1", color="#4CAF50")
    a2.set_title("Validation Metrics"); a2.set_xlabel("epoch"); a2.legend()
    plt.tight_layout()
    plt.savefig(MODELS / "training_curve.png", dpi=150, bbox_inches="tight")
    print(f"\nBest AUC={best_auc:.4f} @ epoch {best_epoch}")
    print(f"Saved: models/best_model.pth, metrics.json, training_curve.png")

    # ── final per-class report ─────────────────────────────
    ckpt = torch.load(MODELS / "best_model.pth")
    model.load_state_dict(ckpt["model"])
    final_metrics, _, _ = evaluate(model, val_ld)
    print("\nPer-class (best model):")
    for d in DISEASES:
        m = final_metrics["per_class"][d]
        print(f"  {d:20s}: AUC={m['auc']:.3f}  AP={m['ap']:.3f}  F1={m['f1']:.3f}")

    # Go/No-Go
    print("\n" + "="*50)
    if final_metrics["mean_auc"] >= 0.70:
        print(f"✅ GO — mean AUC {final_metrics['mean_auc']:.3f} >= 0.70")
    else:
        print(f"⚠️ mean AUC {final_metrics['mean_auc']:.3f} < 0.70 "
              "→ พิจารณาลดเหลือ 3 โรค / เพิ่ม data (ทาง 2)")
    print("="*50)
    print("Next: python inference_gradcam.py <image.png>")


if __name__ == "__main__":
    main()
