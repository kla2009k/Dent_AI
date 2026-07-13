"""
Compare Grad-CAM variants x target-layer for localization (pointing-game) on the
intraoral caries model — WITHOUT retraining. Answers "how much can the heatmap
improve for free". Metric: Grad-CAM peak inside a ground-truth caries box, over
test caries images the model correctly flags (prob>0.5).
"""
import pathlib, time
import numpy as np, torch, timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_grad_cam import (GradCAM, GradCAMPlusPlus, XGradCAM, LayerCAM,
                              HiResCAM, EigenGradCAM)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT = pathlib.Path(__file__).parent.parent
MODELS = ROOT / "models" / "intraoral"
BASE = ROOT / "intraoral_data" / "extracted" / "Benchmarking Dataset" / "test"
IMG, YOLO = BASE / "images", BASE / "yolo"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGES = 150   # subsample flagged caries images for speed


def boxes(stem):
    txt = YOLO / f"{stem}.txt"
    if not txt.exists():
        return []
    out = []
    for ln in txt.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            _, cx, cy, w, h = map(float, p[:5])
            out.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))
    return out


def main():
    ck = torch.load(MODELS / "best_model.pth", map_location="cpu", weights_only=False)
    sz = ck["cfg"].get("img_size", 384)
    model = timm.create_model(ck["cfg"].get("backbone", "efficientnet_b0"),
                              pretrained=False, num_classes=len(ck["diseases"]))
    model.load_state_dict(ck["model"]); model.to(DEVICE).eval()
    tfm = A.Compose([A.Resize(sz, sz),
                     A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                     ToTensorV2()])

    # gather flagged caries images once (with their tensors + boxes)
    samples = []
    for img in sorted(IMG.iterdir()):
        if img.suffix.lower() not in (".jpg",".jpeg",".png"):
            continue
        bx = boxes(img.stem)
        if not bx:
            continue
        x = tfm(image=np.array(Image.open(img).convert("RGB")))["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if torch.sigmoid(model(x))[0,0].item() > 0.5:
                samples.append((x, bx))
        if len(samples) >= MAX_IMAGES:
            break
    print(f"flagged caries images used: {len(samples)}\n")

    layers = {
        "conv_head (12x12)": [model.conv_head],
        "blocks[-1]": [model.blocks[-1]],
        "blocks[-2] (higher-res)": [model.blocks[-2]],
    }
    methods = {"GradCAM": GradCAM, "GradCAM++": GradCAMPlusPlus, "XGradCAM": XGradCAM,
               "HiResCAM": HiResCAM, "LayerCAM": LayerCAM, "EigenGradCAM": EigenGradCAM}

    print(f"{'method':14s} {'layer':26s} pointing%")
    print("-"*52)
    results = []
    for lname, layer in layers.items():
        for mname, M in methods.items():
            cam = M(model=model, target_layers=layer)
            hit = 0
            for x, bx in samples:
                gray = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])[0]
                py, px = np.unravel_index(int(gray.argmax()), gray.shape)
                nx, ny = px/gray.shape[1], py/gray.shape[0]
                if any(x0<=nx<=x1 and y0<=ny<=y1 for x0,y0,x1,y1 in bx):
                    hit += 1
            pct = 100*hit/len(samples)
            results.append((pct, mname, lname))
            print(f"{mname:14s} {lname:26s} {pct:5.1f}")
    print("\nTOP 5:")
    for pct, m, l in sorted(results, reverse=True)[:5]:
        print(f"  {pct:5.1f}%  {m} @ {l}")


if __name__ == "__main__":
    main()
