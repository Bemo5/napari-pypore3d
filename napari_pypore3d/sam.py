# sam_napari_edit_save_as_slice.py
# ---------------------------------------------------------
# Loads ONE fixed image path.
# Asks for slice index ONLY to name the output file.
# Exports ONE editable instance-label mask (.npy, uint16).
# Opens napari with automatic coloured labels (editable).
# ---------------------------------------------------------

from pathlib import Path
import numpy as np
import torch
from PIL import Image
import napari

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ----------------------------
# Fixed input image path (ALWAYS this)
# ----------------------------
IMAGE_PATH = "napari_pypore3d/Images/image.png"
CHECKPOINT_PATH = "napari_pypore3d/models/sam_vit_b_01ec64.pth"
OUT_DIR = Path("napari_pypore3d/data")

# ----------------------------
# FAST SAM defaults
SAM_KWARGS = dict(
    points_per_side=32,            # was 16 → more proposals (detect more)
    pred_iou_thresh=0.60,          # was 0.75 → keep more candidates
    stability_score_thresh=0.75,   # was 0.90 → keep less-stable masks too
    crop_n_layers=1,               # was 0 → finds smaller objects (slower)
    crop_n_points_downscale_factor=2,
    min_mask_region_area=30,       # was 80 → allow small regions
)


# Extra filtering (reduces mask spam)
IOU_MIN = 0.70 # min predicted IoU for a mask
MAX_AREA_FRAC = 0.90  # max area fraction of image for a mask


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load image normally (fixed path)
    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    image_np = np.asarray(image)
    H, W = image_np.shape[:2]
    total_pixels = H * W
    print("Loaded image:", img_path, image_np.shape)

    # 2) Ask slice index ONLY for naming output
    z = ask_slice_index_for_naming()
    ztag = f"z{z:03d}"

    # 3) SAM on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH)
    sam.to(device)

    gen = SamAutomaticMaskGenerator(model=sam, **SAM_KWARGS)

    print("Generating masks...")
    masks = gen.generate(image_np)
    print("Raw masks:", len(masks))
    if not masks:
        raise SystemExit("No masks generated.")

    # 4) Filter masks to reduce clutter
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

    # 5) Build ONE editable instance label mask (uint16)
    labels = np.zeros((H, W), dtype=np.uint16)
    next_id = 1

    for m in kept:
        seg = m["segmentation"]
        write = seg & (labels == 0)   # stable, no-overwrite
        if not np.any(write):
            continue
        if next_id >= 65535:
            print("⚠️ hit uint16 label limit")
            break
        labels[write] = np.uint16(next_id)
        next_id += 1

    print("Instances written:", int(labels.max()))

    # 6) Save with slice tag ONLY in filename
    out_npy = OUT_DIR / f"sam_labels_{ztag}.npy"
    np.save(out_npy, labels)
    print("Saved:", out_npy)

    # 7) Open in napari (napari auto-colours labels; you edit the IDs)
    v = napari.Viewer()
    v.add_image(image_np, name="Image")
    v.add_labels(labels, name=f"SAM labels {ztag} (EDIT)", opacity=0.6)
    napari.run()


if __name__ == "__main__":
    main()
