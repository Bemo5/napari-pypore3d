# napari_pypore3d/check_mask_npz.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None


def load_mask(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npz":
        z = np.load(path)
        if len(z.files) == 0:
            raise ValueError(f"Empty npz: {path}")
        return z[z.files[0]]
    if suf == ".npy":
        return np.load(path)
    raise ValueError(f"Unsupported mask type: {suf} (use .npz or .npy)")


def load_image_hw(path: Path):
    if Image is None:
        return None
    im = np.asarray(Image.open(path))
    if im.ndim == 3:
        im = im[..., 0]
    return im.shape[:2]


def summarize_mask(m: np.ndarray, max_unique: int = 50) -> dict:
    m = np.asarray(m)
    out = {
        "shape": m.shape,
        "dtype": str(m.dtype),
        "ndim": m.ndim,
        "min": int(np.min(m)) if m.size else None,
        "max": int(np.max(m)) if m.size else None,
    }
    uniq = np.unique(m) if m.size else np.array([])
    out["unique_count"] = int(uniq.size)
    out["unique_preview"] = uniq[:max_unique].tolist()
    return out


def check_labels(m: np.ndarray, allowed=(0, 1, 2, 3)):
    if not np.issubdtype(m.dtype, np.integer):
        return False, f"dtype is not integer ({m.dtype})"
    uniq = np.unique(m)
    bad = [int(v) for v in uniq if int(v) not in set(allowed)]
    if bad:
        return False, f"invalid labels {bad} (allowed {list(allowed)})"
    return True, "labels OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, help="Mask file (.npz or .npy)")
    ap.add_argument("--image", default="", help="Optional paired image to check HxW match")
    ap.add_argument("--slice", type=int, default=0, help="If mask is 3D, which Z slice to inspect")
    args = ap.parse_args()

    mask_path = Path(args.mask)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    m = load_mask(mask_path)
    print(f"\n=== MASK: {mask_path} ===")
    s = summarize_mask(m)
    print("shape:", s["shape"], "dtype:", s["dtype"], "ndim:", s["ndim"])
    print("min/max:", s["min"], "/", s["max"])
    print("unique_count:", s["unique_count"])
    print("unique_preview:", s["unique_preview"])

    ok, msg = check_labels(m if m.ndim == 2 else m[args.slice])
    print("label_check:", "✅" if ok else "❌", msg)

    # If 3D, show slice info
    if m.ndim == 3:
        z = args.slice
        if not (0 <= z < m.shape[0]):
            print(f"❌ slice index {z} out of range for shape {m.shape}")
        else:
            ms = m[z]
            ss = summarize_mask(ms)
            print(f"\n--- SLICE z={z} ---")
            print("shape:", ss["shape"], "dtype:", ss["dtype"])
            print("min/max:", ss["min"], "/", ss["max"])
            print("unique_preview:", ss["unique_preview"])
            ok2, msg2 = check_labels(ms)
            print("label_check(slice):", "✅" if ok2 else "❌", msg2)

    # Optional: compare with image size
    if args.image.strip():
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"\n❌ image not found: {img_path}")
        else:
            hw = load_image_hw(img_path)
            if hw is None:
                print("\n⚠️ PIL not installed; cannot check image size. (pip install pillow)")
            else:
                H, W = hw
                if m.ndim == 2:
                    mh, mw = m.shape
                elif m.ndim == 3:
                    mh, mw = m.shape[-2], m.shape[-1]
                else:
                    mh, mw = None, None
                print(f"\nimage_hw: {(H, W)}  mask_hw: {(mh, mw)}")
                print("size_match:", "✅" if (H == mh and W == mw) else "❌")

    print("\nNOTE: If Napari shows the mask as a black IMAGE, it's usually just contrast.")
    print("      For labels 0..3, set contrast to 0..3 or load as Labels layer.\n")


if __name__ == "__main__":
    main()
