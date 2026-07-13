"""
DentScan AI — Phase 0: EDA DENTEX Challenge 2023
รันได้ตรงๆ: python 00_EDA_DENTEX.py
Output: PNG charts + dentex_labels.csv ใน folder เดียวกัน
"""

import json
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ไม่ต้อง display window — save PNG แทน
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from collections import Counter, defaultdict

# ── paths ──────────────────────────────────────────────────
ROOT   = pathlib.Path(__file__).parent.parent
DATA   = ROOT / "dentex_data" / "DENTEX"
ANN    = DATA / "validation_triple.json"
IMG_DIR = DATA / "validation_data" / "quadrant_enumeration_disease" / "xrays"
OUT    = pathlib.Path(__file__).parent

# ── disease classes ────────────────────────────────────────
# DENTEX category_id_3: 0=Impacted, 1=Caries, 2=Periapical Lesion, 3=Deep Caries
DISEASE_CLASSES = ["Normal", "Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0"]

# ── load annotation ────────────────────────────────────────
print("Loading annotation...")
with open(ANN) as f:
    coco = json.load(f)

cat3_map = {c["id"]: c["name"] for c in coco["categories_3"]}
print(f"  Images:      {len(coco['images'])}")
print(f"  Annotations: {len(coco['annotations'])}")
print(f"  Diseases:    {cat3_map}")

# ── annotation-level count ─────────────────────────────────
ann_counts = Counter(cat3_map[a["category_id_3"]] for a in coco["annotations"])
print("\nAnnotation count per disease:")
for name, cnt in sorted(ann_counts.items(), key=lambda x: -x[1]):
    print(f"  {name:25s}: {cnt}")

# ── multi-label image-level conversion ────────────────────
img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
img_diseases: dict = defaultdict(set)
for ann in coco["annotations"]:
    img_diseases[ann["image_id"]].add(cat3_map[ann["category_id_3"]])

rows = []
for img in coco["images"]:
    iid = img["id"]
    diseases = img_diseases[iid]
    row = {"image_id": iid, "filename": img["file_name"]}
    for cls in DISEASE_CLASSES:
        row[cls] = 1 if (cls == "Normal" and len(diseases) == 0) else (1 if cls in diseases else 0)
    row["n_diseases"] = len(diseases)
    rows.append(row)

df = pd.DataFrame(rows)
print(f"\nImage-level labels (total {len(df)} images):")
for cls in DISEASE_CLASSES:
    cnt = int(df[cls].sum())
    pct = df[cls].mean() * 100
    bar = "█" * max(1, cnt // 2)
    print(f"  {cls:25s}: {cnt:3d} ({pct:5.1f}%)  {bar}")

# ── CHART 1: annotation distribution ──────────────────────
print("\nGenerating charts...")
fig, ax = plt.subplots(figsize=(8, 5))
names = list(ann_counts.keys())
counts = [ann_counts[n] for n in names]
c_map = {"Caries":"#2196F3","Deep Caries":"#FF9800","Periapical Lesion":"#E91E63","Impacted":"#9C27B0"}
bars = ax.bar(names, counts, color=[c_map.get(n,"#607D8B") for n in names], edgecolor="white", width=0.6)
ax.set_title("DENTEX — Bounding Box Annotation Count per Disease", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of bounding boxes")
ax.set_xlabel("Disease class")
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(cnt), ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "chart1_ann_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved chart1_ann_distribution.png")

# ── CHART 2: image-level label distribution ───────────────
fig, ax = plt.subplots(figsize=(9, 5))
cls_counts = [int(df[cls].sum()) for cls in DISEASE_CLASSES]
bars = ax.bar(DISEASE_CLASSES, cls_counts, color=COLORS, edgecolor="white", width=0.6)
ax.set_title("DENTEX — Image-Level Label Distribution (multi-label)", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of images")
ax.set_xlabel("Disease class")
ax.set_xticklabels(DISEASE_CLASSES, rotation=15, ha="right")
for bar, cnt in zip(bars, cls_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(cnt), ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "chart2_image_label_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved chart2_image_label_dist.png")

# ── CHART 3: co-occurrence heatmap ────────────────────────
cooc = df[DISEASE_CLASSES].T.dot(df[DISEASE_CLASSES])
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cooc, annot=True, fmt="d", cmap="Blues", ax=ax,
            linewidths=0.5, linecolor="white",
            annot_kws={"fontsize": 11})
ax.set_title("Disease Co-occurrence Matrix (image-level)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "chart3_cooccurrence.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved chart3_cooccurrence.png")

# ── CHART 4: diseases per image ───────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
vc = df["n_diseases"].value_counts().sort_index()
ax.bar(vc.index.astype(str), vc.values, color="#607D8B", edgecolor="white")
ax.set_title("Diseases per Image", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of diseases in one X-ray")
ax.set_ylabel("Images")
for i, (x, y) in enumerate(zip(vc.index, vc.values)):
    ax.text(i, y + 0.2, str(y), ha="center", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "chart4_diseases_per_image.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved chart4_diseases_per_image.png")

# ── CHART 5: sample X-ray per disease class ──────────────
print("  Generating sample X-ray images...")
fig, axes = plt.subplots(1, len(DISEASE_CLASSES), figsize=(24, 5))
for i, cls in enumerate(DISEASE_CLASSES):
    subset = df[df[cls] == 1]
    ax = axes[i]
    ax.set_facecolor("#1a1a2e")
    if len(subset) == 0:
        ax.text(0.5, 0.5, "No sample", ha="center", va="center",
                transform=ax.transAxes, color="white")
    else:
        fname = subset.iloc[0]["filename"]
        # DENTEX val filenames are like "val_0.png"
        img_path = IMG_DIR / fname
        if not img_path.exists():
            # try name only
            img_path = IMG_DIR / pathlib.Path(fname).name
        if img_path.exists():
            img = Image.open(img_path).convert("L")
            # resize wide panoramic to fit
            w, h = img.size
            new_w = min(w, 1600)
            new_h = int(h * new_w / w)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            ax.imshow(img, cmap="gray", aspect="auto")
        else:
            ax.text(0.5, 0.5, f"file not found\n{fname}",
                    ha="center", va="center", transform=ax.transAxes, color="red", fontsize=8)
    ax.set_title(f"{cls}\n(n={len(subset)})", fontsize=10, color="black",
                 fontweight="bold", pad=6)
    ax.axis("off")

plt.suptitle("Sample Panoramic X-ray per Disease Class — DENTEX", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "chart5_sample_xrays.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved chart5_sample_xrays.png")

# ── CHART 6: one annotated X-ray with bbox overlay ───────
print("  Generating annotated X-ray example...")
sample_img_id = coco["images"][0]["id"]
sample_fname  = coco["images"][0]["file_name"]
sample_anns   = [a for a in coco["annotations"] if a["image_id"] == sample_img_id]

img_path = IMG_DIR / pathlib.Path(sample_fname).name
if img_path.exists():
    img = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(img, aspect="auto")
    disease_colors = {0:"#9C27B0", 1:"#2196F3", 2:"#E91E63", 3:"#FF9800"}
    disease_labels = {0:"Impacted", 1:"Caries", 2:"Periapical Lesion", 3:"Deep Caries"}
    for ann in sample_anns:
        x, y, w, h = ann["bbox"]
        did = ann["category_id_3"]
        col = disease_colors.get(did, "white")
        rect = mpatches.Rectangle((x, y), w, h, linewidth=2,
                                   edgecolor=col, facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 4, disease_labels.get(did, "?"),
                color=col, fontsize=7, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1))
    patches_legend = [mpatches.Patch(color=v, label=disease_labels[k])
                      for k, v in disease_colors.items()]
    ax.legend(handles=patches_legend, loc="upper right",
              facecolor="black", labelcolor="white", fontsize=9)
    ax.set_title(f"Annotated X-ray: {sample_fname}", fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "chart6_annotated_example.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved chart6_annotated_example.png")

# ── save CSV ──────────────────────────────────────────────
csv_path = OUT / "dentex_labels.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

# ── print summary ─────────────────────────────────────────
print("\n" + "="*55)
print("  DENTEX Validation Set — EDA Summary")
print("="*55)
print(f"  Total images     : {len(df)}")
print(f"  Total bbox anns  : {len(coco['annotations'])}")
print(f"  Avg bbox/image   : {len(coco['annotations'])/len(df):.1f}")
print()
print("  Image-level counts:")
for cls, col in zip(DISEASE_CLASSES, COLORS):
    cnt = int(df[cls].sum())
    pct = df[cls].mean()*100
    print(f"    {cls:25s}: {cnt:3d} ({pct:5.1f}%)")
print()
print("  Outputs:")
for f in sorted(OUT.glob("chart*.png")):
    print(f"    {f.name}")
print(f"    dentex_labels.csv")
print()
print("  Next → Phase 1: Train EfficientNet-B0 multi-label + Grad-CAM")
print("="*55)
