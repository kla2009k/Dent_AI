"""
Pre-resize intraoral photos → 448x448 cache (fixes dataloader bottleneck).
Source photos are ~2000x2700; decoding+resizing every epoch starves the GPU.
Cache once → training reads small PNGs. Train runs at 384; 448 leaves headroom
for rotate/crop augmentation without edge loss.

Run: python cache_intraoral.py
Out: intraoral_data/cache/{train,valid,test}/*.png
     data/intraoral_{train,val,test}_cached.csv
"""
import sys
import hashlib
import pathlib
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
SIZE = 448

# prefix lets us cache other label sets, e.g. `python cache_intraoral.py intraoral3`
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "intraoral"
# per-prefix cache dir so different label sets never share files
CACHE = ROOT / "intraoral_data" / "cache" / PREFIX
SPLITS = {"train": f"{PREFIX}_train.csv",
          "valid": f"{PREFIX}_val.csv",
          "test": f"{PREFIX}_test.csv"}
OUT = {"train": f"{PREFIX}_train_cached.csv",
       "valid": f"{PREFIX}_val_cached.csv",
       "test": f"{PREFIX}_test_cached.csv"}


def _resize_one(args):
    src, dst = args
    if not dst.exists():
        try:
            Image.open(src).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(dst, "PNG")
        except Exception as e:
            return f"FAIL {src}: {e}"
    return None


def process(split, csv_name, out_name):
    df = pd.read_csv(DATA / csv_name)
    out_dir = CACHE / split
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks, new_paths = [], []
    for _, row in df.iterrows():
        src = pathlib.Path(row["filepath"])
        # hash the full source path → unique cache name (avoids cross-class/source
        # filename collisions, e.g. Kaggle "(1).jpg" existing in several folders)
        key = hashlib.md5(str(src.resolve()).encode()).hexdigest()[:16]
        dst = out_dir / (key + ".png")
        tasks.append((src, dst))
        new_paths.append(str(dst.resolve()))
    fails = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, res in enumerate(ex.map(_resize_one, tasks), 1):
            if res:
                fails += 1
                print(res)
            if i % 500 == 0:
                print(f"  {split}: {i}/{len(tasks)}")
    df["filepath"] = new_paths
    df.to_csv(DATA / out_name, index=False)
    print(f"  {split}: {len(df)} cached ({fails} failed) → {out_name}")


if __name__ == "__main__":
    print(f"Pre-resizing intraoral to {SIZE}x{SIZE} ...")
    for split, csv_name in SPLITS.items():
        process(split, csv_name, OUT[split])
    print("Done. train_intraoral.py reads *_cached.csv automatically if present.")
