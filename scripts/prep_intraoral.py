"""
Build image-level classification CSVs from the Zenodo intraoral caries dataset
(YOLO detection labels → per-image binary "Caries" label).

Zenodo 14827784 "Annotated intraoral image dataset for dental caries detection"
Benchmarking Dataset/{train,valid,test}/{images,yolo}/
  - yolo/<stem>.txt with >=1 line  → image has caries      (Caries=1)
  - yolo/<stem>.txt empty          → reviewed, no caries    (Caries=0)
  - no <stem>.txt at all           → AMBIGUOUS (see --no-txt)

YOLO classes: 0='d' primary decay, 1='D' permanent decay → merged to one
"Caries" label (image-level: any decay present).

Usage:
  python prep_intraoral.py                 # no-txt treated as NORMAL (default)
  python prep_intraoral.py --no-txt skip   # drop no-txt images (safe, smaller)
  python prep_intraoral.py --no-txt normal # explicit: no-txt = Caries 0

Output: data/intraoral_{train,val,test}.csv  (columns: filepath, Caries)
"""
import argparse
import csv
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
EXTRACTED = ROOT / "intraoral_data" / "extracted" / "Benchmarking Dataset"
DATA = ROOT / "data"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
SPLIT_OUT = {"train": "intraoral_train.csv",
             "valid": "intraoral_val.csv",
             "test": "intraoral_test.csv"}


def label_for(stem: str, yolo_dir: pathlib.Path):
    """Return ('caries'|'normal'|'none') for an image stem."""
    txt = yolo_dir / f"{stem}.txt"
    if not txt.exists():
        return "none"
    content = txt.read_text(encoding="utf-8", errors="ignore").strip()
    return "caries" if content else "normal"


def build_split(split: str, no_txt: str):
    img_dir = EXTRACTED / split / "images"
    yolo_dir = EXTRACTED / split / "yolo"
    if not img_dir.exists():
        raise SystemExit(f"missing {img_dir} — run extraction first")

    rows = []
    n_caries = n_normal = n_dropped = 0
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lab = label_for(img.stem, yolo_dir)
        if lab == "caries":
            rows.append((str(img.resolve()), 1)); n_caries += 1
        elif lab == "normal":
            rows.append((str(img.resolve()), 0)); n_normal += 1
        else:  # no-txt
            if no_txt == "skip":
                n_dropped += 1
            else:  # 'normal'
                rows.append((str(img.resolve()), 0)); n_normal += 1
    return rows, n_caries, n_normal, n_dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-txt", choices=["normal", "skip"], default="normal",
                    help="how to treat images with no label file")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    print(f"Source: {EXTRACTED}")
    print(f"no-txt policy: {args.no_txt}\n")

    for split, out_name in SPLIT_OUT.items():
        rows, nc, nn, nd = build_split(split, args.no_txt)
        out = DATA / out_name
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "Caries"])
            w.writerows(rows)
        total = len(rows)
        pos_pct = 100 * nc / total if total else 0
        print(f"[{split:5s}] {out_name:22s} total={total:5d} "
              f"caries={nc:5d} ({pos_pct:4.1f}%) normal={nn:5d} dropped(no-txt)={nd}")

    print("\nDone. Next: python train_intraoral.py")


if __name__ == "__main__":
    main()
