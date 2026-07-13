"""
Build a 3-class multi-label intraoral dataset (Caries / Gingivitis / Calculus)
by combining two sources:

  - Zenodo (already prepped)  → healthy (0,0,0) + caries (1,0,0)
      reused from data/intraoral_{train,val,test}.csv (has filepath + Caries)
  - Kaggle salmansajid05      → caries / gingivitis / calculus (one-hot),
      split 80/10/10 per class

Output: data/intraoral3_{train,val,test}.csv  columns: filepath,Caries,Gingivitis,Calculus

Why combine: Kaggle has the 3 diseases but NO healthy class; Zenodo supplies
real healthy mouths (and extra caries). Each disease head still gets same-domain
negatives (other Kaggle diseases), so it must learn the lesion, not the source.
"""
import csv
import pathlib
import random

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
KAGGLE = ROOT / "intraoral_data" / "kaggle_oral"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
CLASSES = ["Caries", "Gingivitis", "Calculus"]
SEED = 42

# kaggle class folders. Caries = ALL Kaggle caries (original + augmented, ~2601)
# and forced entirely into TRAIN (see TRAIN_ONLY) — these are frontal/anterior
# close-ups, the view the model was weakest on. Test caries stays Zenodo-only so
# there's zero Kaggle-augmentation leakage into evaluation.
KAGGLE_DIRS = {
    "Caries":     KAGGLE / "Data caries" / "Data caries",
    "Gingivitis": KAGGLE / "Gingivitis" / "Gingivitis",
    "Calculus":   KAGGLE / "Calculus" / "Calculus",
}
TRAIN_ONLY = {"Caries"}   # no val/test split (Zenodo covers caries eval)
ZENODO_CSV = {"train": "intraoral_train.csv", "valid": "intraoral_val.csv", "test": "intraoral_test.csv"}
OUT_CSV = {"train": "intraoral3_train.csv", "valid": "intraoral3_val.csv", "test": "intraoral3_test.csv"}


def zenodo_rows(split):
    """(filepath, Caries) from Zenodo → 3-class row with Gingivitis=Calculus=0"""
    path = DATA / ZENODO_CSV[split]
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["filepath"], int(r["Caries"]), 0, 0))
    return rows


def kaggle_split():
    """split each Kaggle disease folder 80/10/10 → {split: [(path, c, g, k)]}"""
    rng = random.Random(SEED)
    out = {"train": [], "valid": [], "test": []}
    for cls, d in KAGGLE_DIRS.items():
        if not d.exists():
            raise SystemExit(f"missing kaggle dir: {d}")
        # note: Kaggle's caries "augmented data set" stores its 2382 images inside
        # a folder literally named "preview" (not thumbnails) — include them
        imgs = sorted(str(p.resolve()) for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)
        rng.shuffle(imgs)
        n = len(imgs)
        if cls in TRAIN_ONLY:
            parts = {"train": imgs, "valid": [], "test": []}
        else:
            n_test, n_val = int(n * 0.1), int(n * 0.1)
            parts = {"test": imgs[:n_test], "valid": imgs[n_test:n_test + n_val],
                     "train": imgs[n_test + n_val:]}
        onehot = tuple(1 if c == cls else 0 for c in CLASSES)
        for sp, lst in parts.items():
            for p in lst:
                out[sp].append((p, *onehot))
        print(f"  Kaggle {cls:11s}: {n:5d}  (train {len(parts['train'])} / val {len(parts['valid'])} / test {len(parts['test'])})")
    return out


def main():
    print("Sources: Zenodo (healthy+caries) + Kaggle salmansajid05 (3 diseases)\n")
    kag = kaggle_split()
    for split, out_name in OUT_CSV.items():
        rows = zenodo_rows(split) + kag[split]
        random.Random(SEED).shuffle(rows)
        with open(DATA / out_name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filepath"] + CLASSES)
            w.writerows(rows)
        pos = {c: sum(r[i + 1] for r in rows) for i, c in enumerate(CLASSES)}
        normal = sum(1 for r in rows if r[1] == 0 and r[2] == 0 and r[3] == 0)
        print(f"[{split:5s}] {out_name:22s} total={len(rows):5d}  "
              f"Caries={pos['Caries']} Gingivitis={pos['Gingivitis']} Calculus={pos['Calculus']} normal={normal}")
    print("\nNext: python cache_intraoral.py intraoral3  &&  python train_intraoral.py --prefix intraoral3")


if __name__ == "__main__":
    main()
