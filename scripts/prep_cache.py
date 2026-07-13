"""
Pre-resize panoramic X-rays → 640x320 cache (แก้ dataloader bottleneck)
panoramic จริง ~2870x1316 → resize ทุก epoch ช้า GPU starve
cache ครั้งเดียว → train อ่านภาพเล็ก เร็วขึ้นมาก
รัน: python prep_cache.py
Output: data/cache/{train,val}/*.png + data/{train,val}_labels_cached.csv
"""
import pathlib
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
W, H = 640, 320

def process(split):
    df = pd.read_csv(DATA / f"{split}_labels.csv")
    out_dir = CACHE / split
    out_dir.mkdir(parents=True, exist_ok=True)
    new_paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
        src = row["filepath"]
        fname = pathlib.Path(src).name
        dst = out_dir / fname
        if not dst.exists():
            img = Image.open(src).convert("RGB").resize((W, H), Image.LANCZOS)
            img.save(dst, "PNG")
        new_paths.append(str(dst.resolve()))
    df["filepath"] = new_paths
    df.to_csv(DATA / f"{split}_labels_cached.csv", index=False)
    print(f"  {split}: {len(df)} images cached → {split}_labels_cached.csv")

if __name__ == "__main__":
    print(f"Pre-resizing to {W}x{H} ...")
    process("train")
    process("val")
    print("Done. train_v2 จะอ่าน *_cached.csv อัตโนมัติ")
