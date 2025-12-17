# flatten_masks.py
from pathlib import Path
import numpy as np
import re

TRAIN_DIR = Path(r"napari_pypore3d\train")

def infer_z(stem: str):
    m = re.search(r"z(\d+)", stem.lower())
    return int(m.group(1)) if m else None

for mask_path in sorted(TRAIN_DIR.glob("*.npz")):
    z = infer_z(mask_path.stem)
    if z is None:
        print(f"[skip] can't infer z from {mask_path.name}")
        continue

    data = np.load(mask_path)
    arr = data[data.files[0]]

    if arr.ndim == 2:
        print(f"[ok] already 2D: {mask_path.name} {arr.shape} {arr.dtype}")
        continue

    if arr.ndim != 3:
        print(f"[skip] weird ndim={arr.ndim}: {mask_path.name}")
        continue

    if not (0 <= z < arr.shape[0]):
        print(f"[skip] z out of range for {mask_path.name}: z={z}, Z={arr.shape[0]}")
        continue

    sl = arr[z].astype(np.uint8)  # 0..3 fits in uint8
    np.savez_compressed(mask_path, sl)  # overwrite with 2D
    print(f"[fix] {mask_path.name} -> 2D slice z={z}, saved uint8 {sl.shape}")
