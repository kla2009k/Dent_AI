"""
Phase 1 v3 — 3-class merge (Caries + Deep Caries → "Caries รวม")
เหตุผล: v1/v2 สับสน Caries↔Deep Caries (spectrum ความลึกเดียวกัน, co-occur สูง)
        merge → ลด ambiguity → คาดว่า boost
3 classes: Caries(รวม) / Periapical Lesion / Impacted
อ่าน cached CSV (640x320), config คล้าย v2 แต่ลด drop_rate (v2 over-regularized → Caries แย่)
Output: models/v3/
"""
import json, pathlib, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models" / "v3"
MODELS.mkdir(parents=True, exist_ok=True)

DISEASES = ["Caries", "Periapical Lesion", "Impacted"]   # Caries = merged
N_CLASSES = len(DISEASES)
CFG = {
    "img_w": 640, "img_h": 320, "batch_size": 8, "epochs": 50,
    "lr": 1.5e-4, "weight_decay": 7e-4, "backbone": "efficientnet_b0",
    "drop_rate": 0.2, "drop_path_rate": 0.1, "num_workers": 2,
    "focal_gamma": 2.0, "label_smooth": 0.05, "freeze_epochs": 3,
    "early_stop_patience": 14, "seed": 42,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def merge_labels(df):
    """รวม Caries OR Deep Caries → Caries"""
    df = df.copy()
    df["Caries"] = ((df["Caries"] == 1) | (df["Deep Caries"] == 1)).astype(int)
    return df


class DS(Dataset):
    def __init__(self, df, tfm): self.df = df.reset_index(drop=True); self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = np.array(Image.open(row["filepath"]).convert("RGB"))
        img = self.tfm(image=img)["image"]
        return img, torch.tensor([row[d] for d in DISEASES], dtype=torch.float32)


def tfms(train):
    H, W = CFG["img_h"], CFG["img_w"]
    if train:
        return A.Compose([
            A.Resize(H, W), A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.6), A.CLAHE(clip_limit=2.0, p=0.4),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(num_holes_range=(1,6), hole_height_range=(int(H*0.05),int(H*0.1)),
                            hole_width_range=(int(W*0.05),int(W*0.1)), p=0.25),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
    return A.Compose([A.Resize(H, W), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])


class FocalLoss(nn.Module):
    def __init__(self, gamma, pos_weight, smooth): super().__init__(); self.g=gamma; self.pw=pos_weight; self.s=smooth
    def forward(self, logits, t):
        if self.s>0: t = t*(1-self.s)+0.5*self.s
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none", pos_weight=self.pw)
        p = torch.sigmoid(logits); pt = p*t+(1-p)*(1-t)
        return ((1-pt)**self.g*bce).mean()


def set_frozen(model, fr):
    for n,p in model.named_parameters():
        if "classifier" not in n and "fc" not in n: p.requires_grad = not fr


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); L,T=[],[]
    for x,y in loader: L.append(model(x.to(DEVICE)).cpu()); T.append(y)
    logits=torch.cat(L).numpy(); targets=torch.cat(T).numpy()
    probs=1/(1+np.exp(-logits)); preds=(probs>0.5).astype(int)
    m={"per_class":{}}; aucs,f1s=[],[]
    for i,d in enumerate(DISEASES):
        t=targets[:,i]
        auc=roc_auc_score(t,probs[:,i]) if 0<t.sum()<len(t) else float("nan")
        ap=average_precision_score(t,probs[:,i]) if t.sum()>0 else float("nan")
        f1=f1_score(t,preds[:,i],zero_division=0)
        m["per_class"][d]={"auc":round(float(auc),4),"ap":round(float(ap),4),"f1":round(float(f1),4)}
        if not np.isnan(auc): aucs.append(auc)
        f1s.append(f1)
    m["mean_auc"]=round(float(np.mean(aucs)),4) if aucs else 0.0
    m["mean_f1"]=round(float(np.mean(f1s)),4)
    return m


def main():
    np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"]); torch.cuda.manual_seed_all(CFG["seed"])
    print(f"Device: {DEVICE} | 3-class merge | {CFG['img_w']}x{CFG['img_h']} drop={CFG['drop_rate']}")
    tr = DATA/"train_labels_cached.csv"; va = DATA/"val_labels_cached.csv"
    train_df = merge_labels(pd.read_csv(tr)); val_df = merge_labels(pd.read_csv(va))
    print(f"Train {len(train_df)} Val {len(val_df)}")
    for d in DISEASES:
        print(f"  {d:20s}: train {int(train_df[d].sum())} val {int(val_df[d].sum())}")

    pos = train_df[DISEASES].sum().values
    pw = torch.tensor((len(train_df)-pos)/np.clip(pos,1,None), dtype=torch.float32).to(DEVICE)
    tl = DataLoader(DS(train_df,tfms(True)), batch_size=CFG["batch_size"], shuffle=True,
                    num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    vl = DataLoader(DS(val_df,tfms(False)), batch_size=CFG["batch_size"], shuffle=False,
                    num_workers=CFG["num_workers"], pin_memory=True)

    model = timm.create_model(CFG["backbone"], pretrained=True, num_classes=N_CLASSES,
                              drop_rate=CFG["drop_rate"], drop_path_rate=CFG["drop_path_rate"]).to(DEVICE)
    crit = FocalLoss(CFG["focal_gamma"], pw, CFG["label_smooth"])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type=="cuda")

    hist={"loss":[],"auc":[],"f1":[]}; best=0.0; best_ep=0; pat=0
    for ep in range(1, CFG["epochs"]+1):
        if ep<=CFG["freeze_epochs"]: set_frozen(model, True)
        elif ep==CFG["freeze_epochs"]+1: set_frozen(model, False); print(f"  unfroze @ {ep}")
        model.train(); run=0.0
        for x,y in tl:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"): loss=crit(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run+=loss.item()*x.size(0)
        sch.step(); tloss=run/len(tl.dataset)
        m=evaluate(model,vl)
        hist["loss"].append(tloss); hist["auc"].append(m["mean_auc"]); hist["f1"].append(m["mean_f1"])
        print(f"[{ep:02d}/{CFG['epochs']}] loss={tloss:.4f} val_AUC={m['mean_auc']:.4f} val_F1={m['mean_f1']:.4f}")
        if m["mean_auc"]>best:
            best,best_ep,pat=m["mean_auc"],ep,0
            torch.save({"model":model.state_dict(),"cfg":CFG,"diseases":DISEASES,"metrics":m}, MODELS/"best_model.pth")
            json.dump(m, open(MODELS/"metrics.json","w"), indent=2, ensure_ascii=False)
            print(f"  ✓ saved best (AUC={best:.4f})")
        else:
            pat+=1
            if pat>=CFG["early_stop_patience"]: print(f"Early stop {ep} (best {best_ep})"); break

    fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
    a1.plot(hist["loss"],color="#E91E63",label="loss"); a1.set_title("v3 Loss (3-class)"); a1.legend()
    a2.plot(hist["auc"],color="#2196F3",label="AUC"); a2.plot(hist["f1"],color="#4CAF50",label="F1")
    a2.axhline(0.70,color="gray",ls="--",alpha=.5,label="0.70"); a2.set_title("v3 Val"); a2.legend()
    plt.tight_layout(); plt.savefig(MODELS/"training_curve.png", dpi=150, bbox_inches="tight")

    ckpt=torch.load(MODELS/"best_model.pth"); model.load_state_dict(ckpt["model"])
    fm=evaluate(model,vl)
    print(f"\n=== v3 RESULT (3-class) — Best AUC={best:.4f} @ epoch {best_ep} ===")
    for d in DISEASES:
        mm=fm["per_class"][d]; print(f"  {d:20s}: AUC={mm['auc']:.3f} AP={mm['ap']:.3f} F1={mm['f1']:.3f}")
    v1=json.load(open(ROOT/"models"/"metrics.json")) if (ROOT/"models"/"metrics.json").exists() else None
    if v1: print(f"\n  v1(4cls): {v1['mean_auc']:.4f}  →  v3(3cls): {fm['mean_auc']:.4f}")
    print("="*55)
    print("✅ GO" if fm["mean_auc"]>=0.70 else "ยังไม่ถึง 0.70", f"(mean AUC {fm['mean_auc']:.3f})")


if __name__ == "__main__":
    main()
