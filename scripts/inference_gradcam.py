"""
Phase 1 step 3 — Inference + Grad-CAM heatmap
รัน: python inference_gradcam.py <path/to/xray.png>
Output: <name>_result.png (X-ray + heatmap overlay + prob bars)
        + print JSON ผลทำนาย 4 โรค

ใช้ตอน Phase 2 ด้วย — ฟังก์ชัน predict_with_heatmap() เรียกจาก FastAPI ได้
"""
import sys
import json
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.parent
MODELS = ROOT / "models"

DISEASES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
DISEASES_TH = {
    "Caries": "ฟันผุ",
    "Deep Caries": "ฟันผุลึก",
    "Periapical Lesion": "รอยโรคปลายราก",
    "Impacted": "ฟันคุด",
}
COLORS = ["#2196F3", "#FF9800", "#E91E63", "#9C27B0"]
THRESHOLD = 0.5
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None  # lazy singleton


def load_model():
    global _model
    if _model is not None:
        return _model
    ckpt = torch.load(MODELS / "best_model.pth", map_location=DEVICE)
    model = timm.create_model(ckpt["cfg"]["backbone"], pretrained=False,
                              num_classes=len(DISEASES))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    _model = model
    return model


def preprocess(pil_img):
    tfm = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    arr = np.array(pil_img.convert("RGB"))
    tensor = tfm(image=arr)["image"].unsqueeze(0).to(DEVICE)
    # rgb float [0,1] สำหรับ overlay
    rgb = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0
    return tensor, rgb


def get_target_layer(model):
    # EfficientNet-B0 (timm) → conv_head เป็น layer สุดท้ายก่อน pooling
    if hasattr(model, "conv_head"):
        return [model.conv_head]
    # fallback: block สุดท้าย
    return [model.blocks[-1]]


def predict_with_heatmap(pil_img, make_heatmap=True):
    """คืน dict ผลทำนาย + (option) heatmap ต่อโรคที่ตรวจพบ
    ใช้ได้ทั้ง CLI และ FastAPI (Phase 2)"""
    model = load_model()
    tensor, rgb = preprocess(pil_img)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    result = {
        "predictions": {d: round(float(p), 4) for d, p in zip(DISEASES, probs)},
        "detected": [d for d, p in zip(DISEASES, probs) if p > THRESHOLD],
        "is_normal": bool(all(p <= THRESHOLD for p in probs)),
        "heatmaps": {},
    }

    if make_heatmap and result["detected"]:
        cam = GradCAM(model=model, target_layers=get_target_layer(model))
        for d in result["detected"]:
            idx = DISEASES.index(d)
            grayscale = cam(input_tensor=tensor,
                            targets=[ClassifierOutputTarget(idx)])[0]
            overlay = show_cam_on_image(rgb.astype(np.float32), grayscale,
                                        use_rgb=True)
            result["heatmaps"][d] = overlay  # np.uint8 HxWx3
    result["_rgb"] = (rgb * 255).astype(np.uint8)
    return result


def visualize(pil_img, result, out_path):
    detected = result["detected"]
    n = max(1, len(detected))
    fig, axes = plt.subplots(2, max(2, n), figsize=(6 * max(2, n), 11),
                             gridspec_kw={"height_ratios": [3, 2]})
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    # row 0: original + heatmaps
    axes[0, 0].imshow(result["_rgb"])
    axes[0, 0].set_title("Original X-ray", fontweight="bold")
    axes[0, 0].axis("off")
    for j, d in enumerate(detected):
        ax = axes[0, min(j + 1, axes.shape[1] - 1)] if len(detected) < axes.shape[1] else axes[0, j]
        ax.imshow(result["heatmaps"][d])
        ax.set_title(f"Grad-CAM: {DISEASES_TH[d]}\n(p={result['predictions'][d]:.2f})",
                     fontweight="bold")
        ax.axis("off")
    for j in range(len(detected) + 1, axes.shape[1]):
        axes[0, j].axis("off")

    # row 1: prob bar chart (span)
    for j in range(axes.shape[1]):
        axes[1, j].axis("off")
    bar_ax = fig.add_subplot(2, 1, 2)
    names = [DISEASES_TH[d] for d in DISEASES]
    vals = [result["predictions"][d] for d in DISEASES]
    bars = bar_ax.barh(names, vals, color=COLORS)
    bar_ax.axvline(THRESHOLD, color="red", linestyle="--", label=f"threshold {THRESHOLD}")
    bar_ax.set_xlim(0, 1)
    bar_ax.set_title("Disease Probability", fontweight="bold")
    bar_ax.legend(loc="lower right")
    for bar, v in zip(bars, vals):
        bar_ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.2f}", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_gradcam.py <xray.png>")
        sys.exit(1)
    img_path = pathlib.Path(sys.argv[1])
    pil = Image.open(img_path)
    res = predict_with_heatmap(pil, make_heatmap=True)

    print(json.dumps({
        "predictions": res["predictions"],
        "detected": res["detected"],
        "is_normal": res["is_normal"],
    }, indent=2, ensure_ascii=False))

    out = img_path.parent / f"{img_path.stem}_result.png"
    visualize(pil, res, out)
    print(f"\nSaved: {out}")
