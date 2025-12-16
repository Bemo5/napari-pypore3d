# sam_vs_intensity_png_preview.py
# ---------------------------------------------------------
# Loads ONE fixed image path.
# Asks for slice index ONLY to name outputs.
# Runs:
#   (A) SAM instance segmentation -> labels uint16 + PNG overlay preview
#   (B) Intensity multi-Otsu (3 classes) -> gt uint8 + PNG overlay preview
# NO napari.
# ---------------------------------------------------------

from pathlib import Path
import numpy as np
import torch
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Optional helpers for nice coloured PNGs
from skimage.color import label2rgb
from skimage.filters import threshold_multiotsu

# ----------------------------
# Fixed input image path (ALWAYS this)
# ----------------------------
IMAGE_PATH = "napari_pypore3d/Images/image.png"
CHECKPOINT_PATH = "napari_pypore3d/models/sam_vit_b_01ec64.pth"
OUT_DIR = Path("napari_pypore3d/data")

# ----------------------------
# SAM defaults
SAM_KWARGS = dict(
    points_per_side=32,
    pred_iou_thresh=0.60,
    stability_score_thresh=0.75,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=30,
)

# Extra filtering (reduces mask spam)
IOU_MIN = 0.70
MAX_AREA_FRAC = 0.90

# Visual overlay strength
ALPHA = 0.50


def ask_slice_index_for_naming() -> int:
    s = input("Enter slice index Z for OUTPUT naming (e.g. 22): ").strip()
    if not s:
        raise SystemExit("No slice index entered.")
    try:
        z = int(s)
    except ValueError:
        raise SystemExit(f"Invalid slice index: {s}")
    if z < 0:
        raise SystemExit("Slice index must be >= 0.")
    return z


def save_overlay_png(out_path: Path, base_rgb: np.ndarray, label_img: np.ndarray, alpha: float = 0.5):
    """
    Save a coloured overlay PNG of labels on top of the RGB image.
    label2rgb gives stable colours for labels > 0.
    """
    # label2rgb expects labels int; bg_label=0 treated as transparent-ish
    overlay = label2rgb(label_img, image=base_rgb, bg_label=0, alpha=alpha)
    overlay_u8 = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(overlay_u8).save(out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    # Load RGB + grayscale
    image_rgb = np.asarray(Image.open(img_path).convert("RGB"))
    image_gray = np.asarray(Image.open(img_path).convert("L"))
    H, W = image_rgb.shape[:2]
    total_pixels = H * W
    print("Loaded image:", img_path, image_rgb.shape)

    z = ask_slice_index_for_naming()
    ztag = f"z{z:03d}"

    # ----------------------------
    # (B) Intensity segmentation (3 classes)
    # ----------------------------
    # Produces class IDs 1..3 (0 stays background only if you set it later;
    # here we label every pixel as 1..3 because it's pure intensity partition.)
    thr = threshold_multiotsu(image_gray, classes=3)
    intensity_classes = (np.digitize(image_gray, bins=thr) + 1).astype(np.uint8)  # 1..3
    out_int_npy = OUT_DIR / f"intensity_3class_{ztag}.npy"
    np.save(out_int_npy, intensity_classes)
    print("Saved intensity mask:", out_int_npy, "unique:", np.unique(intensity_classes))

    out_int_png = OUT_DIR / f"preview_intensity_3class_{ztag}.png"
    save_overlay_png(out_int_png, image_rgb, intensity_classes, alpha=ALPHA)
    print("Saved intensity overlay PNG:", out_int_png)

    # ----------------------------
    # (A) SAM instance segmentation
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH)
    sam.to(device)

    gen = SamAutomaticMaskGenerator(model=sam, **SAM_KWARGS)

    print("Generating SAM masks...")
    masks = gen.generate(image_rgb)
    print("Raw masks:", len(masks))
    if not masks:
        raise SystemExit("No masks generated.")

    # Filter
    max_area = MAX_AREA_FRAC * total_pixels
    kept = []
    for m in masks:
        if m.get("predicted_iou", 0.0) < IOU_MIN:
            continue
        if m.get("area", 0) > max_area:
            continue
        kept.append(m)
    kept = sorted(kept, key=lambda m: int(m.get("area", 0)), reverse=True)
    print("Kept masks:", len(kept))

    # Build instance label mask
    labels = np.zeros((H, W), dtype=np.uint16)
    next_id = 1
    for m in kept:
        seg = m["segmentation"]
        write = seg & (labels == 0)
        if not np.any(write):
            continue
        if next_id >= 65535:
            print("⚠️ hit uint16 label limit")
            break
        labels[write] = np.uint16(next_id)
        next_id += 1

    print("Instances written:", int(labels.max()))

    out_sam_npy = OUT_DIR / f"sam_instances_{ztag}.npy"
    np.save(out_sam_npy, labels)
    print("Saved SAM instances:", out_sam_npy)

    out_sam_png = OUT_DIR / f"preview_sam_instances_{ztag}.png"
    save_overlay_png(out_sam_png, image_rgb, labels, alpha=ALPHA)
    print("Saved SAM overlay PNG:", out_sam_png)

    print("\nDONE. Compare these two files:")
    print(" -", out_sam_png)
    print(" -", out_int_png)


if __name__ == "__main__":
    main()
