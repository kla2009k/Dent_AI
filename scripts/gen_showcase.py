"""
สร้างภาพ showcase (poster-ready) จาก ensemble API + ฟอนต์ไทย
รัน: python gen_showcase.py   (ต้องเปิด server http://localhost:8000 ก่อน)
Output: web/samples/sampleN_showcase.png
"""
import io
import json
import base64
import pathlib
import urllib.request
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = pathlib.Path(__file__).parent.parent
SAMPLES = ROOT / "web" / "samples"
API = "http://localhost:8000/api/predict"

# ── ฟอนต์ไทย (Tahoma มี Thai glyphs ครบ) ──
for cand in ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/LEELAWAD.TTF"]:
    if pathlib.Path(cand).exists():
        fm.fontManager.addfont(cand)
        plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        break

DISEASES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
TH = {"Caries": "ฟันผุ", "Deep Caries": "ฟันผุลึก",
      "Periapical Lesion": "รอยโรคปลายราก", "Impacted": "ฟันคุด"}
COLORS = {"Caries": "#2196F3", "Deep Caries": "#FF9800",
          "Periapical Lesion": "#E91E63", "Impacted": "#8b5cf6"}


def call_api(img_path):
    b64 = "data:image/png;base64," + base64.b64encode(img_path.read_bytes()).decode()
    body = json.dumps({"image": b64, "symptoms": {}, "heatmap": True}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=90))


def b64_to_img(s):
    raw = base64.b64decode(s.split(",", 1)[1])
    return np.array(Image.open(io.BytesIO(raw)))


def make_showcase(img_path, out_path):
    res = call_api(img_path)
    preds = res["predictions"]
    detected = res["detected"]
    th = res.get("thresholds", {})
    orig = np.array(Image.open(img_path).convert("RGB"))

    n_hm = len(detected)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, max(2, n_hm + 1), height_ratios=[3, 2], hspace=0.3, wspace=0.15)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(orig); ax0.set_title("ภาพ X-ray ต้นฉบับ", fontsize=12, fontweight="bold")
    ax0.axis("off")
    for j, d in enumerate(detected):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.imshow(b64_to_img(res["heatmaps"][d]))
        ax.set_title(f"Grad-CAM: {TH[d]} ({preds[d]*100:.0f}%)",
                     fontsize=11, fontweight="bold", color=COLORS[d])
        ax.axis("off")

    axb = fig.add_subplot(gs[1, :])
    names = [TH[d] for d in DISEASES]
    vals = [preds[d] for d in DISEASES]
    cols = [COLORS[d] for d in DISEASES]
    bars = axb.barh(names, vals, color=cols, height=0.6)
    for d, bar in zip(DISEASES, bars):
        t = th.get(d, 0.5)
        axb.plot([t, t], [bar.get_y(), bar.get_y() + bar.get_height()],
                 color="red", lw=1.5, ls="--")
        axb.text(preds[d] + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{preds[d]*100:.0f}%", va="center", fontweight="bold")
    axb.set_xlim(0, 1)
    axb.set_title("ความน่าจะเป็นของแต่ละโรค (เส้นแดง = threshold)",
                  fontsize=12, fontweight="bold")
    axb.invert_yaxis()
    mode = res["model"].get("mode", "single")
    auc = res["model"].get("val_mean_auc")
    fig.suptitle(f"DentScan AI — {mode} + TTA  (val mean AUC {auc})",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  {out_path.name}: detected {detected}")


if __name__ == "__main__":
    for i in (1, 2, 3):
        p = SAMPLES / f"sample{i}.png"
        if p.exists():
            make_showcase(p, SAMPLES / f"sample{i}_showcase.png")
    print("Done.")
