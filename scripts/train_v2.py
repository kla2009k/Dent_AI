"""
Phase 1 v2 — Train EfficientNet-B0 multi-label (anti-overfit + aspect-ratio fix)
รัน: python train_v2.py
เก็บ v1 baseline ไว้เทียบ (models/best_model.pth) — v2 save ที่ models/v2/

เปลี่ยนจาก v1:
  1. resize 640x320 คงสัดส่วน panoramic 2:1 (v1 บีบ 512² → ฟันแบน)
  2. drop_rate 0.3 + drop_path 0.2 (regularize)
  3. augment แรงขึ้น: ShiftScaleRotate, CoarseDropout, สูงขึ้น brightness/CLAHE
  4. lr 1.5e-4 (ลดจาก 3e-4), weight_decay 1e-3 (เพิ่มจาก 1e-4)
  5. freeze backbone 3 epoch แรก (warmup head) แล้ว unfreeze
  6. label smoothing 0.05
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
import warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models" / "v2"
MODELS.mkdir(parents=True, exist_ok=True)

DISEASES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
N_CLASSES = len(DISEASES)

CFG = {
    "img_w": 640, "img_h": 320,    # aspect 2:1 คงสัดส่วน panoramic
    "batch_size": 8,
    "epochs": 50,
    "lr": 1.5e-4,
    "weight_decay": 1e-3,
    "backbone": "efficientnet_b0",
    "drop_rate": 0.3,
    "drop_path_rate": 0.2,
    "num_workers": 2,
    "focal_gamma": 2.0,
    "label_smooth": 0.05,
    "freeze_epochs": 3,
    "early_stop_patience": 12,
    "seed": 42,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class DentexDataset(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True); self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = np.array(Image.open(row["filepath"]).convert("RGB"))
        img = self.tfm(image=img)["image"]
        label = torch.tensor([row[d] for d in DISEASES], dtype=torch.float32)
        return img, label


def get_transforms(train):
    H, W = CFG["img_h"], CFG["img_w"]
    if train:
        return A.Compose([
            A.Resize(H, W),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1,
                               rotate_limit=10, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.6),
            A.CLAHE(clip_limit=2.0, p=0.4),
            A.GaussNoise(p=0.25),
            A.CoarseDropout(num_holes_range=(1, 8),
                            hole_height_range=(int(H*0.05), int(H*0.1)),
                            hole_width_range=(int(W*0.05), int(W*0.1)), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, smooth=0.0):
        super().__init__()
        self.gamma = gamma; self.pos_weight = pos_weight; self.smooth = smooth
    def forward(self, logits, targets):
        if self.smooth > 0:
            targets = targets * (1 - self.smooth) + 0.5 * self.smooth
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


def build_model():
    return timm.create_model(CFG["backbone"], pretrained=True,
                             num_classes=N_CLASSES,
                             drop_rate=CFG["drop_rate"],
                             drop_path_rate=CFG["drop_path_rate"]).to(DEVICE)


def set_backbone_frozen(model, frozen: bool):
    for name, p in model.named_parameters():
        if "classifier" not in name and "fc" not in name:
            p.requires_grad = not frozen


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits_all, targets_all = [], []
    for x, y in loader:
        logits_all.append(model(x.to(DEVICE)).cpu()); targets_all.append(y)
    logits = torch.cat(logits_all).numpy(); targets = torch.cat(targets_all).numpy()
    probs = 1 / (1 + np.exp(-logits)); preds = (probs > 0.5).astype(int)
    metrics = {"per_class": {}}; aucs, f1s = [], []
    for i, d in enumerate(DISEASES):
        t = targets[:, i]
        auc = roc_auc_score(t, probs[:, i]) if 0 < t.sum() < len(t) else float("nan")
        ap = average_precision_score(t, probs[:, i]) if t.sum() > 0 else float("nan")
        f1 = f1_score(t, preds[:, i], zero_division=0)
        metrics["per_class"][d] = {"auc": round(float(auc), 4),
                                   "ap": round(float(ap), 4), "f1": round(float(f1), 4)}
        if not np.isnan(auc): aucs.append(auc)
        f1s.append(f1)
    metrics["mean_auc"] = round(float(np.mean(aucs)), 4) if aucs else 0.0
    metrics["mean_f1"] = round(float(np.mean(f1s)), 4)
    return metrics


def main():
    set_seed(CFG["seed"])
    print(f"Device: {DEVICE} | {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'}")
    print(f"Config: {CFG['img_w']}x{CFG['img_h']}, lr={CFG['lr']}, wd={CFG['weight_decay']}, "
          f"drop={CFG['drop_rate']}/{CFG['drop_path_rate']}")

    # อ่าน cached CSV ถ้ามี (ภาพ pre-resized 640x320 → เร็วกว่ามาก)
    tr_csv = DATA / "train_labels_cached.csv"
    va_csv = DATA / "val_labels_cached.csv"
    if tr_csv.exists() and va_csv.exists():
        train_df = pd.read_csv(tr_csv); val_df = pd.read_csv(va_csv)
        print("Using PRE-RESIZED cache (640x320)")
    else:
        train_df = pd.read_csv(DATA / "train_labels.csv")
        val_df = pd.read_csv(DATA / "val_labels.csv")
        print("Using ORIGINAL images (no cache — run prep_cache.py for speed)")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    pos = train_df[DISEASES].sum().values
    pos_weight = torch.tensor((len(train_df) - pos) / np.clip(pos, 1, None),
                              dtype=torch.float32).to(DEVICE)

    train_ld = DataLoader(DentexDataset(train_df, get_transforms(True)),
                          batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    val_ld = DataLoader(DentexDataset(val_df, get_transforms(False)),
                        batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True)

    model = build_model()
    criterion = FocalLoss(CFG["focal_gamma"], pos_weight, CFG["label_smooth"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    history = {"train_loss": [], "val_auc": [], "val_f1": []}
    best_auc, best_epoch, patience = 0.0, 0, 0

    for epoch in range(1, CFG["epochs"] + 1):
        if epoch <= CFG["freeze_epochs"]:
            set_backbone_frozen(model, True)
        elif epoch == CFG["freeze_epochs"] + 1:
            set_backbone_frozen(model, False)
            print(f"  → unfroze backbone at epoch {epoch}")

        model.train(); running = 0.0
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running / len(train_ld.dataset)

        m = evaluate(model, val_ld)
        history["train_loss"].append(train_loss)
        history["val_auc"].append(m["mean_auc"]); history["val_f1"].append(m["mean_f1"])
        print(f"[{epoch:02d}/{CFG['epochs']}] loss={train_loss:.4f} "
              f"val_AUC={m['mean_auc']:.4f} val_F1={m['mean_f1']:.4f}")

        if m["mean_auc"] > best_auc:
            best_auc, best_epoch, patience = m["mean_auc"], epoch, 0
            torch.save({"model": model.state_dict(), "cfg": CFG,
                        "diseases": DISEASES, "metrics": m}, MODELS / "best_model.pth")
            with open(MODELS / "metrics.json", "w") as f:
                json.dump(m, f, indent=2, ensure_ascii=False)
            print(f"  ✓ saved best (AUC={best_auc:.4f})")
        else:
            patience += 1
            if patience >= CFG["early_stop_patience"]:
                print(f"Early stop epoch {epoch} (best {best_epoch})"); break

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(history["train_loss"], label="train loss", color="#E91E63")
    a1.set_title("v2 Training Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(history["val_auc"], label="val mean AUC", color="#2196F3")
    a2.plot(history["val_f1"], label="val mean F1", color="#4CAF50")
    a2.axhline(0.70, color="gray", ls="--", alpha=0.5, label="Go 0.70")
    a2.set_title("v2 Validation Metrics"); a2.set_xlabel("epoch"); a2.legend()
    plt.tight_layout(); plt.savefig(MODELS / "training_curve.png", dpi=150, bbox_inches="tight")

    ckpt = torch.load(MODELS / "best_model.pth")
    model.load_state_dict(ckpt["model"])
    fm = evaluate(model, val_ld)
    print(f"\n=== v2 RESULT — Best AUC={best_auc:.4f} @ epoch {best_epoch} ===")
    for d in DISEASES:
        mm = fm["per_class"][d]
        print(f"  {d:20s}: AUC={mm['auc']:.3f}  AP={mm['ap']:.3f}  F1={mm['f1']:.3f}")
    # compare v1
    v1p = ROOT / "models" / "metrics.json"
    if v1p.exists():
        v1 = json.load(open(v1p))
        print(f"\n  v1 mean AUC: {v1['mean_auc']:.4f}  →  v2 mean AUC: {fm['mean_auc']:.4f}  "
              f"({'+' if fm['mean_auc']>=v1['mean_auc'] else ''}{fm['mean_auc']-v1['mean_auc']:+.4f})")
    print("=" * 55)
    print("GO" if fm["mean_auc"] >= 0.70 else "ยังไม่ถึง 0.70", f"(mean AUC {fm['mean_auc']:.3f})")


if __name__ == "__main__":
    main()
