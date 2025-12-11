from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch  # 🔹 NEW

# ----------------------------
# Paths
# ----------------------------
IMAGE_PATH = "napari_pypore3d/Images/image.png"
CHECKPOINT_PATH = "napari_pypore3d/models/sam_vit_b_01ec64.pth"

# ----------------------------
# Load image
# ----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.asarray(image)
H, W = image_np.shape[:2]
total_pixels = H * W
print(f"Loaded image {IMAGE_PATH} with shape {image_np.shape}")

# ----------------------------
# Device (CPU / GPU)
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------------
# Load SAM model (vit_b)
# ----------------------------
print("Loading SAM model (vit_b)...")
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH)
sam.to(device)  # 🔹 Move model to GPU if available
print("SAM loaded on", device, "\n")

# ----------------------------
# LOOSE mode = default / active
# ----------------------------
# You can drop points_per_side from 64 -> 32 if still too slow
generator_kwargs = dict(
    points_per_side=64,          # denser grid -> more proposals
    pred_iou_thresh=0.70,
    stability_score_thresh=0.85,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=30,
)

IOU_MIN = 0.80          # external IoU filter
MIN_AREA = 40           # pixels
MAX_AREA_FRAC = 0.6     # ignore single huge blobs > 60% of image

print("Running SAM (LOOSE mode, active)...")
print("Generator kwargs:", generator_kwargs)
print(f"External IoU filter: >= {IOU_MIN}")
print(f"External area filter: >= {MIN_AREA} px, <= {MAX_AREA_FRAC:.2f} * image\n")

mask_generator = SamAutomaticMaskGenerator(model=sam, **generator_kwargs)

print("Generating masks...")
masks = mask_generator.generate(image_np)
print(f"Raw masks from SAM: {len(masks)}")

if not masks:
    raise SystemExit("No masks generated. Check parameters or image.")

# ----------------------------
# Filter by IoU + area
# ----------------------------
max_area = MAX_AREA_FRAC * total_pixels
filtered = []
for m in masks:
    iou = m["predicted_iou"]
    area = m["area"]
    if iou < IOU_MIN:
        continue
    if area < MIN_AREA:
        continue
    if area > max_area:
        continue
    filtered.append(m)

masks = sorted(filtered, key=lambda x: x["area"], reverse=True)
print(f"Masks after filtering: {len(masks)}")

if not masks:
    raise SystemExit("No masks survived filtering. Try lowering IOU_MIN or MIN_AREA.")

# ----------------------------
# Visualise all masks
# ----------------------------
plt.figure(figsize=(8, 8))
plt.imshow(image_np)

overlay = np.zeros((H, W, 4), dtype=float)
np.random.seed(42)

for m in masks:
    mask = m["segmentation"]
    color = np.random.random(3)
    overlay[mask] = [*color, 0.6]

plt.imshow(overlay)
plt.title(f"SAM loose mode — All detected objects (n = {len(masks)})")
plt.axis("off")
plt.tight_layout()
plt.show()

# ----------------------------
# Build labeled mask
# ----------------------------
labeled_mask = np.zeros((H, W), dtype=np.int32)
for idx, m in enumerate(masks):
    mask = m["segmentation"]
    labeled_mask[mask] = idx + 1  # labels start at 1

np.save("segmentation_labels.npy", labeled_mask)

print("\nSaved labeled mask to: segmentation_labels.npy")
print(f"Mask shape: {labeled_mask.shape}")
print(f"Unique labels (incl. background): {np.unique(labeled_mask).size}")

# ----------------------------
# Stats
# ----------------------------
areas = np.array([m["area"] for m in masks], dtype=float)
ious = np.array([m["predicted_iou"] for m in masks], dtype=float)
print("\n=== Segmentation Statistics (LOOSE, active) ===")
print(f"Total objects kept: {len(masks)}")
print(f"Area min / mean / max: {areas.min():.1f} / {areas.mean():.1f} / {areas.max():.1f}")
print(f"IoU  min / mean / max: {ious.min():.3f} / {ious.mean():.3f} / {ious.max():.3f}")

# ----------------------------
# Napari snippet (single active labels)
# ----------------------------
print("\n=== To use in napari ===")
print("import napari")
print("import numpy as np")
print("from PIL import Image")
print("")
print(f"image = np.array(Image.open('{IMAGE_PATH}'))")
print("labels = np.load('segmentation_labels.npy')")
print("")
print("viewer = napari.Viewer()")
print("viewer.add_image(image, name='Original Image')")
print("viewer.add_labels(labels, name='SAM loose (active)')")
print("napari.run()")
