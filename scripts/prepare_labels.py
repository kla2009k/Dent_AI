"""
Phase 1 step 1 — แตก training zip + แปลง annotation เป็น multi-label CSV
รัน: python prepare_labels.py
Output: data/train_labels.csv, data/val_labels.csv
"""
import json
import zipfile
import pathlib
import pandas as pd
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "dentex_data" / "DENTEX"
OUT  = ROOT / "data"
OUT.mkdir(exist_ok=True)

# 4 disease classes (Normal = no disease above threshold, ไม่ใช่ class แยก)
DISEASES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted"]
# DENTEX category_id_3: 0=Impacted, 1=Caries, 2=Periapical Lesion, 3=Deep Caries
CID3_TO_NAME = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}


def extract_training():
    """แตก training_data.zip ถ้ายังไม่แตก"""
    zip_path = DATA / "training_data.zip"
    target = DATA / "training_data"
    if target.exists():
        print(f"Already extracted: {target}")
        return
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA)
    print("Extract done.")


def find_disease_json(base: pathlib.Path) -> pathlib.Path:
    """หาไฟล์ annotation ที่มี disease (category_id_3)"""
    for j in base.rglob("*.json"):
        if "disease" in j.name.lower() or "triple" in j.name.lower():
            return j
    # fallback: ไฟล์ json ที่มี categories_3
    for j in base.rglob("*.json"):
        try:
            with open(j) as f:
                d = json.load(f)
            if "categories_3" in d or any("category_id_3" in a for a in d.get("annotations", [])[:1]):
                return j
        except Exception:
            continue
    raise FileNotFoundError(f"No disease annotation found under {base}")


def find_xray_dir(base: pathlib.Path) -> pathlib.Path:
    """หา folder ที่มี png มากสุด"""
    best, best_n = None, 0
    for d in base.rglob("*"):
        if d.is_dir():
            n = len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg")))
            if n > best_n:
                best, best_n = d, n
    print(f"  X-ray dir: {best}  ({best_n} images)")
    return best


def build_csv(ann_path: pathlib.Path, img_dir: pathlib.Path, out_csv: pathlib.Path):
    with open(ann_path) as f:
        coco = json.load(f)
    img_diseases = defaultdict(set)
    for ann in coco["annotations"]:
        cid3 = ann.get("category_id_3")
        if cid3 is not None and cid3 in CID3_TO_NAME:
            img_diseases[ann["image_id"]].add(CID3_TO_NAME[cid3])
    rows = []
    for img in coco["images"]:
        fname = pathlib.Path(img["file_name"]).name
        fpath = img_dir / fname
        if not fpath.exists():
            cands = list(img_dir.rglob(fname))
            if not cands:
                continue
            fpath = cands[0]
        diseases = img_diseases[img["id"]]
        row = {"filepath": str(fpath.resolve()), "filename": fname}
        for d in DISEASES:
            row[d] = 1 if d in diseases else 0
        row["is_normal"] = 1 if len(diseases) == 0 else 0
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}  ({len(df)} images)")
    for d in DISEASES:
        print(f"    {d:20s}: {int(df[d].sum()):4d} ({df[d].mean()*100:.1f}%)")
    print(f"    {'Normal':20s}: {int(df['is_normal'].sum()):4d} ({df['is_normal'].mean()*100:.1f}%)")
    return df


if __name__ == "__main__":
    extract_training()

    print("\n=== TRAINING SET ===")
    train_base = DATA / "training_data"
    train_ann = find_disease_json(train_base)
    print(f"  Annotation: {train_ann}")
    # xrays folder อยู่ข้าง annotation json เสมอ — บังคับชี้ตรงนี้
    # (ห้ามใช้ find_xray_dir auto: unlabelled/xrays มี filename ซ้ำ train_NNN.png
    #  แต่เป็นคนละภาพ → จะชี้ผิด folder)
    train_imgdir = train_ann.parent / "xrays"
    assert train_imgdir.exists(), f"xrays dir not found: {train_imgdir}"
    print(f"  X-ray dir (forced): {train_imgdir}  "
          f"({len(list(train_imgdir.glob('*.png')))} images)")
    build_csv(train_ann, train_imgdir, OUT / "train_labels.csv")

    print("\n=== VALIDATION SET ===")
    val_ann = DATA / "validation_triple.json"
    val_imgdir = DATA / "validation_data" / "quadrant_enumeration_disease" / "xrays"
    build_csv(val_ann, val_imgdir, OUT / "val_labels.csv")

    print("\nDone. Next: python train_multilabel.py")
