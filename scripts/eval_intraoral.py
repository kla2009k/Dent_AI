"""
Evaluate the intraoral caries model on the HELD-OUT test split and sanity-check
what it actually looks at (Grad-CAM) — guards against the model exploiting a
confound (lighting/framing/source) instead of the actual decay.

Run: python eval_intraoral.py
In:  models/intraoral/best_model.pth, data/intraoral_test_cached.csv
Out: models/intraoral/test_metrics.json
     models/intraoral/gradcam_samples/*.png  (caries-positive test images)
     prints per-class AUC/AP/F1 + confusion at threshold 0.5
"""
import argparse
import json
import pathlib
import numpy as np
import pandas as pd
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             confusion_matrix)
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = pathlib.Path(__file__).parent.parent
MODELS = ROOT / "models" / "intraoral"
DATA = ROOT / "data"
_ap = argparse.ArgumentParser()
_ap.add_argument("--prefix", default="intraoral")
_PREFIX = _ap.parse_known_args()[0].prefix
TEST_CSV = DATA / f"{_PREFIX}_test_cached.csv"
if not TEST_CSV.exists():
    TEST_CSV = DATA / f"{_PREFIX}_test.csv"
GC_DIR = MODELS / "gradcam_samples"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ck = torch.load(MODELS / "best_model.pth", map_location="cpu", weights_only=False)
    diseases = ck["diseases"]
    sz = ck["cfg"].get("img_size", 384)
    backbone = ck["cfg"].get("backbone", "efficientnet_b0")
    model = timm.create_model(backbone, pretrained=False, num_classes=len(diseases))
    model.load_state_dict(ck["model"])
    model.to(DEVICE).eval()
    print(f"Model: {backbone} classes={diseases} img={sz}")
    print(f"Train-time val AUC: {ck.get('metrics', {}).get('mean_auc')}")

    df = pd.read_csv(TEST_CSV)
    tfm = A.Compose([A.Resize(sz, sz),
                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                     ToTensorV2()])

    probs, targets, paths = [], [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            img = np.array(Image.open(row["filepath"]).convert("RGB"))
            x = tfm(image=img)["image"].unsqueeze(0).to(DEVICE)
            p = torch.sigmoid(model(x))[0].cpu().numpy()
            probs.append(p)
            targets.append([float(row[d]) for d in diseases])
            paths.append(row["filepath"])
    probs = np.array(probs); targets = np.array(targets)

    metrics = {"n_test": len(df), "per_class": {}}
    for i, d in enumerate(diseases):
        t = targets[:, i]; pr = probs[:, i]
        auc = roc_auc_score(t, pr) if 0 < t.sum() < len(t) else float("nan")
        ap = average_precision_score(t, pr) if t.sum() > 0 else float("nan")
        pred = (pr > 0.5).astype(int)
        f1 = f1_score(t, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(t, pred, labels=[0, 1]).ravel()
        metrics["per_class"][d] = {
            "auc": round(float(auc), 4), "ap": round(float(ap), 4),
            "f1": round(float(f1), 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
        }
    metrics["mean_auc"] = round(float(np.nanmean(
        [metrics["per_class"][d]["auc"] for d in diseases])), 4)

    with open(MODELS / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== TEST SET (held-out) ===")
    for d in diseases:
        m = metrics["per_class"][d]
        print(f"  {d:12s} AUC={m['auc']:.3f} AP={m['ap']:.3f} F1={m['f1']:.3f} "
              f"sens={m['sensitivity']} spec={m['specificity']} "
              f"(tp{m['tp']} fp{m['fp']} tn{m['tn']} fn{m['fn']})")
    print(f"  mean AUC = {metrics['mean_auc']}")

    # ── Grad-CAM sanity: does it look at the decay? ──
    GC_DIR.mkdir(exist_ok=True)
    layer = [model.conv_head] if hasattr(model, "conv_head") else [model.blocks[-1]]
    cam = HiResCAM(model=model, target_layers=layer)
    # pick highest-confidence true-positive caries images
    pos = [(probs[i, 0], paths[i]) for i in range(len(paths)) if targets[i, 0] == 1]
    pos.sort(reverse=True)
    for rank, (score, path) in enumerate(pos[:6], 1):
        img = np.array(Image.open(path).convert("RGB").resize((sz, sz)))
        x = tfm(image=np.array(Image.open(path).convert("RGB")))["image"].unsqueeze(0).to(DEVICE)
        gray = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])[0]
        overlay = show_cam_on_image(img / 255.0, gray, use_rgb=True)
        out = GC_DIR / f"tp_{rank}_p{score:.2f}.png"
        Image.fromarray(overlay).save(out)
    print(f"\nGrad-CAM saved: {GC_DIR} (top-6 confident caries) — inspect visually")


if __name__ == "__main__":
    main()
