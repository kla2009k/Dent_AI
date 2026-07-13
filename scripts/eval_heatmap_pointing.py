"""
Heatmap localization accuracy (pointing game) for the intraoral caries model.
For each caries-positive TEST image the model correctly flags, check whether the
Grad-CAM peak falls inside a ground-truth caries bounding box. Reports % hit —
a real "does the heatmap point at the actual decay" metric for judges.

Run: python eval_heatmap_pointing.py
"""
import pathlib
import numpy as np
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT = pathlib.Path(__file__).parent.parent
MODELS = ROOT / "models" / "intraoral"
BASE = ROOT / "intraoral_data" / "extracted" / "Benchmarking Dataset" / "test"
IMG, YOLO = BASE / "images", BASE / "yolo"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_boxes(stem):
    txt = YOLO / f"{stem}.txt"
    if not txt.exists():
        return []
    boxes = []
    for line in txt.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            _, cx, cy, w, h = map(float, p[:5])
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))  # x0y0x1y1 norm
    return boxes


def main():
    ck = torch.load(MODELS / "best_model.pth", map_location="cpu", weights_only=False)
    sz = ck["cfg"].get("img_size", 384)
    model = timm.create_model(ck["cfg"].get("backbone", "efficientnet_b0"),
                              pretrained=False, num_classes=len(ck["diseases"]))
    model.load_state_dict(ck["model"]); model.to(DEVICE).eval()
    layer = [model.conv_head] if hasattr(model, "conv_head") else [model.blocks[-1]]
    cam = HiResCAM(model=model, target_layers=layer)
    tfm = A.Compose([A.Resize(sz, sz),
                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                     ToTensorV2()])

    hit = flagged = total_pos = 0
    for img in IMG.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        boxes = load_boxes(img.stem)
        if not boxes:
            continue                      # only caries-positive images
        total_pos += 1
        arr = np.array(Image.open(img).convert("RGB"))
        x = tfm(image=arr)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].item()
        if prob <= 0.5:
            continue                      # only images the model actually flags
        flagged += 1
        gray = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])[0]  # sz x sz
        py, px = np.unravel_index(int(gray.argmax()), gray.shape)
        nx, ny = px / gray.shape[1], py / gray.shape[0]
        if any(x0 <= nx <= x1 and y0 <= ny <= y1 for x0, y0, x1, y1 in boxes):
            hit += 1

    print(f"caries-positive test images : {total_pos}")
    print(f"flagged by model (prob>0.5) : {flagged}")
    print(f"Grad-CAM peak inside a caries box : {hit}/{flagged} "
          f"= {100*hit/max(flagged,1):.1f}%  (pointing-game accuracy)")


if __name__ == "__main__":
    main()
