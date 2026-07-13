"""
Intraoral-photo track — Train EfficientNet-B0 multi-label classifier on
intraoral (smartphone/clinical camera) photos.  Separate modality from the
panoramic X-ray model; served alongside it via predictor.py modality routing.

Run:  python train_intraoral.py
Needs: data/intraoral_train.csv, data/intraoral_val.csv (from prep_intraoral.py)
Out:   models/intraoral/best_model.pth, metrics.json, training_curve.png

Design notes
------------
- Disease columns are read from the CSV header (every column except `filepath`),
  so extending from 1 class (Caries) to 3 (Caries/Gingivitis/Calculus) needs no
  code change — just regenerate the CSVs with more label columns.
- Photo augmentation differs from X-ray: real colour, variable lighting/white
  balance, no CLAHE-heavy contrast.  We add colour/hue jitter and mild blur.
- Same EfficientNet-B0 + focal-loss + Grad-CAM stack as the X-ray model so the
  two tracks stay architecturally consistent and share serving code.
"""
import argparse
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
MODELS = ROOT / "models" / "intraoral"
MODELS.mkdir(parents=True, exist_ok=True)

# CSV prefix chosen at runtime (default single-class caries; intraoral3 = 3-class)
_ap = argparse.ArgumentParser()
_ap.add_argument("--prefix", default="intraoral")
_PREFIX = _ap.parse_known_args()[0].prefix

# prefer pre-resized cache (cache_intraoral.py) if available → faster dataloader
TRAIN_CSV = DATA / f"{_PREFIX}_train_cached.csv"
VAL_CSV = DATA / f"{_PREFIX}_val_cached.csv"
if not TRAIN_CSV.exists():
    TRAIN_CSV = DATA / f"{_PREFIX}_train.csv"
if not VAL_CSV.exists():
    VAL_CSV = DATA / f"{_PREFIX}_val.csv"

CFG = {
    "img_size": 384,          # photos: 384 enough, lighter than X-ray 512
    "batch_size": 24,         # RTX 5060 8GB @ 384 b0 → 24 fits, better GPU util
    "epochs": 30,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "backbone": "efficientnet_b0",
    "num_workers": 6,
    "focal_gamma": 2.0,
    "early_stop_patience": 6,
    "seed": 42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_diseases(df) -> list:
    """Every column except `filepath` is a binary disease label."""
    return [c for c in df.columns if c != "filepath"]


# ── dataset ────────────────────────────────────────────────
class IntraoralDataset(Dataset):
    def __init__(self, df, diseases, tfm):
        self.df = df.reset_index(drop=True)
        self.diseases = diseases
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["filepath"]).convert("RGB")
        img = np.array(img)
        img = self.tfm(image=img)["image"]
        label = torch.tensor([float(row[d]) for d in self.diseases],
                             dtype=torch.float32)
        return img, label


def get_transforms(train: bool):
    sz = CFG["img_size"]
    if train:
        # kept augments cheap so the CPU dataloader doesn't starve the GPU
        # (dropped ImageCompression/GaussianBlur — JPEG re-encode was the bottleneck)
        return A.Compose([
            A.Resize(sz, sz),
            A.HorizontalFlip(p=0.5),               # mouth left/right symmetric
            A.Rotate(limit=12, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.6),   # phone lighting varies
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                                 val_shift_limit=10, p=0.4),  # white-balance drift
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


def build_model(n_classes):
    model = timm.create_model(
        CFG["backbone"], pretrained=True, num_classes=n_classes)
    return model.to(DEVICE)


@torch.no_grad()
def evaluate(model, loader, diseases):
    model.eval()
    all_logits, all_targets = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        all_logits.append(model(x).cpu())
        all_targets.append(y)
    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    metrics = {"per_class": {}}
    aucs, f1s = [], []
    for i, d in enumerate(diseases):
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
    return metrics


def main():
    set_seed(CFG["seed"])
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        raise SystemExit(
            f"Missing CSVs. Run prep_intraoral.py first.\n"
            f"  expected: {TRAIN_CSV}\n            {VAL_CSV}")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    diseases = load_diseases(train_df)
    n_classes = len(diseases)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Classes: {diseases}")

    pos = train_df[diseases].sum().values
    neg = len(train_df) - pos
    pos_weight = torch.tensor(neg / np.clip(pos, 1, None),
                              dtype=torch.float32).to(DEVICE)
    print("pos_weight:", dict(zip(diseases, pos_weight.cpu().numpy().round(2))))

    train_ds = IntraoralDataset(train_df, diseases, get_transforms(True))
    val_ds = IntraoralDataset(val_df, diseases, get_transforms(False))
    # persistent_workers avoids re-spawning workers each epoch (slow on Windows);
    # prefetch_factor keeps batches ready so the GPU isn't starved
    train_ld = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
                          persistent_workers=True, prefetch_factor=4)
    val_ld = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True,
                        persistent_workers=True, prefetch_factor=4)

    model = build_model(n_classes)
    criterion = FocalLoss(gamma=CFG["focal_gamma"], pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    history = {"train_loss": [], "val_auc": [], "val_f1": []}
    best_auc, best_epoch, patience = 0.0, 0, 0

    import time
    n_batches = len(train_ld)
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        for bi, (x, y) in enumerate(train_ld, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            if epoch == 1 and bi % 50 == 0:
                ips = bi * CFG["batch_size"] / (time.time() - t0)
                print(f"  ep1 batch {bi}/{n_batches}  {ips:.0f} img/s", flush=True)
        scheduler.step()
        train_loss = running / len(train_ds)

        metrics = evaluate(model, val_ld, diseases)
        history["train_loss"].append(train_loss)
        history["val_auc"].append(metrics["mean_auc"])
        history["val_f1"].append(metrics["mean_f1"])
        print(f"[{epoch:02d}/{CFG['epochs']}] loss={train_loss:.4f} "
              f"val_AUC={metrics['mean_auc']:.4f} val_F1={metrics['mean_f1']:.4f}")

        if metrics["mean_auc"] > best_auc:
            best_auc, best_epoch, patience = metrics["mean_auc"], epoch, 0
            torch.save({"model": model.state_dict(), "cfg": CFG,
                        "diseases": diseases, "metrics": metrics,
                        "modality": "photo"},
                       MODELS / "best_model.pth")
            with open(MODELS / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"  ✓ saved best (AUC={best_auc:.4f})")
        else:
            patience += 1
            if patience >= CFG["early_stop_patience"]:
                print(f"Early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(history["train_loss"], label="train loss", color="#E91E63")
    a1.set_title("Training Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(history["val_auc"], label="val mean AUC", color="#2196F3")
    a2.plot(history["val_f1"], label="val mean F1", color="#4CAF50")
    a2.set_title("Validation Metrics"); a2.set_xlabel("epoch"); a2.legend()
    plt.tight_layout()
    plt.savefig(MODELS / "training_curve.png", dpi=150, bbox_inches="tight")

    ckpt = torch.load(MODELS / "best_model.pth")
    model.load_state_dict(ckpt["model"])
    final = evaluate(model, val_ld, diseases)
    print(f"\nBest AUC={best_auc:.4f} @ epoch {best_epoch}")
    print("Per-class (best model):")
    for d in diseases:
        m = final["per_class"][d]
        print(f"  {d:20s}: AUC={m['auc']:.3f}  AP={m['ap']:.3f}  F1={m['f1']:.3f}")
    print(f"\nSaved: {MODELS}/best_model.pth, metrics.json, training_curve.png")
    print("Next: wire into predictor.py (modality='photo')")


if __name__ == "__main__":
    main()
