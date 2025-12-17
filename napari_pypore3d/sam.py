# sam_vs_intensity_napari_label.py
# ---------------------------------------------------------
# Opens NAPARI for fast labeling.
# Loads ONE fixed image path.
# Asks for slice index ONLY to name outputs.
# Creates:
#   (A) SAM instance labels (uint16) as a Labels layer
#   (B) Intensity multi-Otsu (3 classes) as a Labels layer
# Adds a small UI to save the *active* Labels layer to NPY + overlay PNG.
# ---------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image

import napari
from magicgui.widgets import Container, PushButton, ComboBox, Label

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from skimage.color import label2rgb
from skimage.filters import threshold_multiotsu


# ----------------------------
# Fixed paths
# ----------------------------
IMAGE_PATH = "napari_pypore3d/Images/image.png"
CHECKPOINT_PATH = "napari_pypore3d/models/sam_vit_b_01ec64.pth"
OUT_DIR = Path("napari_pypore3d/data")

# ----------------------------
# SAM defaults
# ----------------------------
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

# Overlay strength
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


def overlay_png(base_rgb: np.ndarray, label_img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Return uint8 RGB overlay image."""
    ov = label2rgb(label_img, image=base_rgb, bg_label=0, alpha=alpha)
    ov_u8 = np.clip(ov * 255.0, 0, 255).astype(np.uint8)
    return ov_u8


def build_sam_instances(image_rgb: np.ndarray) -> np.ndarray:
    H, W = image_rgb.shape[:2]
    total_pixels = H * W

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.exists():
        raise SystemExit(f"SAM checkpoint not found: {ckpt}")

    sam = sam_model_registry["vit_b"](checkpoint=str(ckpt))
    sam.to(device)
    gen = SamAutomaticMaskGenerator(model=sam, **SAM_KWARGS)

    print("Generating SAM masks...")
    masks = gen.generate(image_rgb)
    print("Raw masks:", len(masks))
    if not masks:
        return np.zeros((H, W), dtype=np.uint16)

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
    return labels


def build_intensity_3class(image_gray: np.ndarray) -> np.ndarray:
    thr = threshold_multiotsu(image_gray, classes=3)
    # 1..3 (no zeros)
    cls = (np.digitize(image_gray, bins=thr) + 1).astype(np.uint8)
    return cls


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    # Load RGB + grayscale
    image_rgb = np.asarray(Image.open(img_path).convert("RGB"))
    image_gray = np.asarray(Image.open(img_path).convert("L"))
    H, W = image_rgb.shape[:2]
    print("Loaded image:", img_path, image_rgb.shape)

    z = ask_slice_index_for_naming()
    ztag = f"z{z:03d}"

    # Build masks before opening napari (simpler + stable)
    intensity = build_intensity_3class(image_gray)
    sam_labels = build_sam_instances(image_rgb)

    # Open napari
    viewer = napari.Viewer(title=f"Labeling — {img_path.name} ({ztag})")

    # Base image
    viewer.add_image(image_gray, name=f"image_gray_{ztag}", contrast_limits=(0, 255))

    # Candidate labels as Labels layers (editable)
    L_int = viewer.add_labels(intensity, name=f"intensity_3class_{ztag}")
    L_sam = viewer.add_labels(sam_labels, name=f"sam_instances_{ztag}")

    # Make SAM active by default (usually better starting point)
    viewer.layers.selection.active = L_sam

    # ----------------------------
    # Small control panel
    # ----------------------------
    info = Label(value="Edit a Labels layer, then Save (active layer).")

    layer_choice = ComboBox(
        choices=[L_sam.name, L_int.name],
        value=L_sam.name,
        label="Set active",
    )

    btn_set_active = PushButton(text="Set Active Layer")
    btn_save_npy = PushButton(text="Save ACTIVE labels → .npy")
    btn_save_png = PushButton(text="Save overlay PNG (ACTIVE)")
    btn_save_both = PushButton(text="Save BOTH (npy + overlay)")

    def _get_active_labels_layer():
        L = viewer.layers.selection.active
        if L is None or L.__class__.__name__ != "Labels":
            raise RuntimeError("Active layer is not a Labels layer. Click a Labels layer first.")
        return L

    @btn_set_active.clicked.connect
    def _on_set_active():
        name = str(layer_choice.value)
        viewer.layers.selection.active = viewer.layers[name]

    def _save_active_npy():
        L = _get_active_labels_layer()
        arr = np.asarray(L.data)
        out = OUT_DIR / f"{L.name}.npy"
        np.save(out, arr)
        print("Saved:", out)

    def _save_active_overlay():
        L = _get_active_labels_layer()
        arr = np.asarray(L.data)
        ov = overlay_png(image_rgb, arr, alpha=ALPHA)
        out = OUT_DIR / f"preview_{L.name}.png"
        Image.fromarray(ov).save(out)
        print("Saved:", out)

    @btn_save_npy.clicked.connect
    def _on_save_npy():
        try:
            _save_active_npy()
        except Exception as e:
            print("[save npy] ERROR:", e)

    @btn_save_png.clicked.connect
    def _on_save_png():
        try:
            _save_active_overlay()
        except Exception as e:
            print("[save png] ERROR:", e)

    @btn_save_both.clicked.connect
    def _on_save_both():
        try:
            _save_active_npy()
            _save_active_overlay()
        except Exception as e:
            print("[save both] ERROR:", e)

    panel = Container(
        widgets=[info, layer_choice, btn_set_active, btn_save_npy, btn_save_png, btn_save_both],
        layout="vertical",
        labels=False,
    )
    viewer.window.add_dock_widget(panel, name="Quick Label + Save", area="right")

    napari.run()


if __name__ == "__main__":
    main()
