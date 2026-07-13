"""
Serving optimizer — หา config เสิร์ฟที่ดีสุดโดยไม่ต้อง retrain:
  1. เทียบ: v1 เดี่ยว / v2 เดี่ยว / ensemble(v1+v2) / + hflip TTA
  2. config ที่ชนะ → tune per-class threshold (maximize F1 บน val)
  3. เซฟ models/serving_config.json → predictor.py อ่านไปใช้

หมายเหตุ: threshold จูนบน val 50 ภาพ (เล็ก) — รายงานในเล่มต้อง disclose
รัน: python -u tune_serving.py
"""
import json
import pathlib
import numpy as np
import pandas as pd
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DISEASES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANDIDATES = {
    "v1": MODELS / "best_model.pth",
    "v2": MODELS / "v2" / "best_model.pth",
}


def cfg_wh(cfg):
    if "img_w" in cfg and "img_h" in cfg:
        return cfg["img_w"], cfg["img_h"]
    return cfg.get("img_size", 512), cfg.get("img_size", 512)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["diseases"] == DISEASES, f"{path}: class mismatch"
    m = timm.create_model(ck["cfg"].get("backbone", "efficientnet_b0"),
                          pretrained=False, num_classes=len(DISEASES))
    m.load_state_dict(ck["model"])
    m.to(DEVICE).eval()
    return m, cfg_wh(ck["cfg"])


def tfm(w, h):
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


@torch.no_grad()
def probs_for(model, wh, df, tta=False):
    t = tfm(*wh)
    out = []
    for fp in df["filepath"]:
        arr = np.array(Image.open(fp).convert("RGB"))
        x = t(image=arr)["image"].unsqueeze(0).to(DEVICE)
        logit = model(x)
        if tta:
            logit = (logit + model(torch.flip(x, dims=[3]))) / 2
        out.append(torch.sigmoid(logit)[0].cpu().numpy())
    return np.array(out)


def auc_table(name, probs, targets):
    aucs = {}
    for i, d in enumerate(DISEASES):
        t = targets[:, i]
        aucs[d] = roc_auc_score(t, probs[:, i]) if 0 < t.sum() < len(t) else float("nan")
    mean = np.nanmean(list(aucs.values()))
    row = "  ".join(f"{aucs[d]:.3f}" for d in DISEASES)
    print(f"  {name:18s}: {row}  | mean {mean:.4f}")
    return mean


def main():
    df = pd.read_csv(DATA / "val_labels.csv")  # original full-res (ตรง training eval)
    targets = df[DISEASES].values.astype(int)
    print(f"Val: {len(df)} images | device {DEVICE}\n")
    print(f"  {'config':18s}: {'  '.join(d[:6] for d in DISEASES)}")

    models = {}
    for name, path in CANDIDATES.items():
        if path.exists():
            models[name] = load(path)

    P = {}
    for name, (m, wh) in models.items():
        P[name] = probs_for(m, wh, df)
        P[name + "+tta"] = probs_for(m, wh, df, tta=True)

    results = {}
    for name, p in P.items():
        results[name] = (auc_table(name, p, targets), p)
    if "v1" in P and "v2" in P:
        for suf in ["", "+tta"]:
            ens = (P["v1" + suf] + P["v2" + suf]) / 2
            results["ens(v1,v2)" + suf] = (auc_table("ens(v1,v2)" + suf, ens, targets), ens)

    best_name = max(results, key=lambda k: results[k][0])
    best_auc, best_probs = results[best_name]
    print(f"\n🏆 best config: {best_name} (mean AUC {best_auc:.4f})")

    # per-class threshold tuning (max F1)
    thresholds = {}
    print("\nPer-class threshold tuning (max F1):")
    for i, d in enumerate(DISEASES):
        t = targets[:, i]
        best_f1, best_th = 0.0, 0.5
        for th in np.arange(0.15, 0.86, 0.05):
            f1 = f1_score(t, (best_probs[:, i] > th).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, round(float(th), 2)
        # คุม threshold ไม่ให้สุดโต่ง (val 50 ภาพ overfit ง่าย) → clamp 0.30-0.65
        best_th = float(min(0.65, max(0.30, best_th)))
        thresholds[d] = best_th
        print(f"  {d:18s}: th={best_th:.2f}  F1={best_f1:.3f}")

    # ── สร้าง serving_config ──
    model_paths, tta = [], False
    if best_name.startswith("ens"):
        model_paths = [str(CANDIDATES["v1"].resolve()), str(CANDIDATES["v2"].resolve())]
    else:
        base = best_name.replace("+tta", "")
        model_paths = [str(CANDIDATES[base].resolve())]
    tta = best_name.endswith("+tta")

    config = {
        "mode": "ensemble" if len(model_paths) > 1 else "single",
        "models": model_paths,
        "tta_hflip": tta,
        "thresholds": thresholds,
        "diseases": DISEASES,
        "val_mean_auc": round(float(best_auc), 4),
        "best_config_name": best_name,
        "note": "thresholds จูนบน DENTEX val (50 ภาพ) — เล็ก ต้อง disclose ในเล่ม; "
                "ensemble+TTA = robust ไม่ overfit threshold",
    }
    out = MODELS / "serving_config.json"
    json.dump(config, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n✅ saved {out}")
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
