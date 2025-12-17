from pathlib import Path
import numpy as np
import torch
from PIL import Image

from skimage.filters import threshold_otsu
from skimage.color import label2rgb

from monai.networks.nets import UNet
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --------------------------------------------------
# Paths
# --------------------------------------------------
IMG_PATH = Path("napari_pypore3d/Images/image.png")
SAM_CKPT = Path("napari_pypore3d/models/sam_vit_b_01ec64.pth")
OUT_DIR = Path("napari_pypore3d/data/compare_2d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.5

# --------------------------------------------------
# Load image
# --------------------------------------------------
img_rgb = np.asarray(Image.open(IMG_PATH).convert("RGB"))
img_gray = np.asarray(Image.open(IMG_PATH).convert("L"))
H, W = img_gray.shape

# --------------------------------------------------
# 1️⃣ OTSU (binary)
# --------------------------------------------------
thr = threshold_otsu(img_gray)
otsu_mask = (img_gray >= thr).astype(np.uint8)

otsu_overlay = label2rgb(otsu_mask, image=img_rgb, bg_label=0, alpha=ALPHA)
Image.fromarray((otsu_overlay * 255).astype(np.uint8)) \
     .save(OUT_DIR / "otsu_overlay.png")

print("✔ Otsu done")

# --------------------------------------------------
# 2️⃣ SAM (instances)
# --------------------------------------------------
sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
sam.to(DEVICE)

gen = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.75,
)

masks = gen.generate(img_rgb)

sam_labels = np.zeros((H, W), dtype=np.uint16)
idx = 1
for m in sorted(masks, key=lambda x: x["area"], reverse=True):
    write = m["segmentation"] & (sam_labels == 0)
    if write.any():
        sam_labels[write] = idx
        idx += 1

sam_overlay = label2rgb(sam_labels, image=img_rgb, bg_label=0, alpha=ALPHA)
Image.fromarray((sam_overlay * 255).astype(np.uint8)) \
     .save(OUT_DIR / "sam_overlay.png")

print("✔ SAM done")

# --------------------------------------------------
# 3️⃣ MONAI 2D UNet (UNTRAINED)
# --------------------------------------------------

def pad_to_multiple(img, multiple=8):
    h, w = img.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    return np.pad(img, ((0, ph), (0, pw)), mode="reflect")

# 🔹 PAD FIRST
img_gray_p = pad_to_multiple(img_gray, multiple=8)

# normalize AFTER padding
img_norm = img_gray_p.astype(np.float32)
img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

# tensor
x = torch.from_numpy(img_norm)[None, None].to(DEVICE)

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,   # binary
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
).to(DEVICE)

model.eval()
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.uint8)

# pad RGB only for overlay size consistency
img_rgb_p = pad_to_multiple(img_rgb[..., 0], multiple=8)

monai_overlay = label2rgb(pred, image=img_rgb_p, bg_label=0, alpha=ALPHA)
Image.fromarray((monai_overlay * 255).astype(np.uint8)) \
     .save(OUT_DIR / "monai_overlay.png")

print("✔ MONAI done")
