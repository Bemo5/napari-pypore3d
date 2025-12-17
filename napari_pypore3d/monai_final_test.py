# napari_pypore3d/monai_final_test.py
# Train MONAI SegResNet (2D) semantic segmentation (labels 0..3)
# - Patch training + balanced sampling + aug
# - Saves checkpoint with keys: state_dict, model_cfg, num_classes, divisible
# - Tests on test/ and shows results

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from monai.data import DataLoader
from monai.data.utils import list_data_collate
from monai.transforms import Compose
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference

# Optional SciPy for speckle cleanup
try:
    from scipy import ndimage as ndi  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

NUM_CLASSES = 4  # labels 0..3
DEFAULT_CLASS_NAMES = ["Background", "Solid", "Pores", "Holes"]
CLASS_COLOURS = ["#000000", "#D9D9D9", "#2E8B57", "#FFFF00"]
CMAP = ListedColormap(CLASS_COLOURS)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def load_image_any(path: Path) -> np.ndarray:
    """Load image as float32 in [0,1], grayscale."""
    arr = plt.imread(str(path))
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        mx = float(np.max(arr)) if arr.size else 1.0
        if mx > 1.0:
            arr = arr / (mx if mx != 0 else 1.0)

    # convert RGB/RGBA -> grayscale
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=-1)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image after grayscale, got {arr.shape} for {path}")

    return arr


def _next_multiple(v: int, k: int) -> int:
    return ((v + k - 1) // k) * k


def pad_2d_to_divisible(
    img2d: np.ndarray,
    mask2d: Optional[np.ndarray],
    k: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int]]:
    """Pad H,W to divisible by k (bottom/right). image=edge, mask=0."""
    if k is None or k <= 1:
        h, w = img2d.shape
        return img2d, mask2d, (h, w)

    h, w = img2d.shape
    H = _next_multiple(h, k)
    W = _next_multiple(w, k)
    pad_h = H - h
    pad_w = W - w
    if pad_h == 0 and pad_w == 0:
        return img2d, mask2d, (h, w)

    img_pad = np.pad(img2d, ((0, pad_h), (0, pad_w)), mode="edge")

    if mask2d is None:
        return img_pad, None, (h, w)

    mask_pad = np.pad(mask2d, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return img_pad, mask_pad, (h, w)


def crop_back_2d(arr2d: np.ndarray, orig_hw: Tuple[int, int]) -> np.ndarray:
    oh, ow = orig_hw
    return arr2d[:oh, :ow]


def _infer_z_from_stem(stem: str) -> Optional[int]:
    s = stem.lower()
    m = re.search(r"(?:^|[_\-])z(\d+)(?:$|[_\-])", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[_\-])slice[_\-]?(\d+)(?:$|[_\-])", s)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    return None


def _ensure_2d_mask(m: np.ndarray, path: Path, mask_z: Optional[int], stem_hint: str) -> np.ndarray:
    m = np.asarray(m)
    m = np.squeeze(m)

    if m.ndim == 2:
        return m

    if m.ndim == 3:
        z = mask_z if mask_z is not None else _infer_z_from_stem(stem_hint)
        if z is None:
            raise ValueError(
                f"Mask {path} is 3D {m.shape}. Provide --mask_z OR name like z349.* to infer."
            )
        if not (0 <= z < m.shape[0]):
            raise ValueError(f"--mask_z={z} out of range for {path} with shape {m.shape}")
        return m[z]

    raise ValueError(f"Mask must be 2D or 3D. Got {m.shape} for {path}")


def remap_mask_semantic_4class(m2d: np.ndarray, path: Path) -> np.ndarray:
    """Enforce labels in {0,1,2,3}. If 4 exists, remap it to 0."""
    m = np.asarray(m2d).astype(np.int64, copy=False)
    m = np.where(m == 4, 0, m)

    uniq = np.unique(m)
    allowed = {0, 1, 2, 3}
    bad = [int(v) for v in uniq if int(v) not in allowed]
    if bad:
        raise ValueError(f"Mask {path} has invalid labels {bad}. Unique={uniq}")
    return m


def load_mask_any(path: Path, mask_z: Optional[int], stem_hint: str) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npy":
        m = np.load(path)
    elif suf == ".npz":
        z = np.load(path)
        if len(z.files) == 0:
            raise ValueError(f"Empty npz mask: {path}")
        m = z[z.files[0]]
    else:
        raise ValueError(f"Unsupported mask type: {path}")

    m2d = _ensure_2d_mask(m, path, mask_z=mask_z, stem_hint=stem_hint)
    return remap_mask_semantic_4class(m2d, path)


def find_mask_for_stem(train_dir: Path, stem: str) -> Optional[Path]:
    p_npy = train_dir / f"{stem}.npy"
    if p_npy.exists():
        return p_npy
    p_npz = train_dir / f"{stem}.npz"
    if p_npz.exists():
        return p_npz
    return None


def list_train_pairs(train_dir: Path) -> List[Dict[str, Path]]:
    items: List[Dict[str, Path]] = []
    for img_path in sorted(train_dir.iterdir()):
        if not img_path.is_file() or not is_image_file(img_path):
            continue
        mask_path = find_mask_for_stem(train_dir, img_path.stem)
        if mask_path is None:
            continue
        items.append({"image": img_path, "mask": mask_path})
    return items


def list_test_images(test_dir: Path) -> List[Path]:
    return [p for p in sorted(test_dir.iterdir()) if p.is_file() and is_image_file(p)]


class SliceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        items: List[Dict[str, Path]],
        xform=None,
        mask_z: Optional[int] = None,
        divisible: int = 16,
    ):
        self.items = items
        self.xform = xform
        self.mask_z = mask_z
        self.divisible = divisible

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img_path = it["image"]
        mask_path = it["mask"]

        img = load_image_any(img_path)
        msk = load_mask_any(mask_path, mask_z=self.mask_z, stem_hint=img_path.stem)

        img, msk, _ = pad_2d_to_divisible(img, msk, k=self.divisible)

        img = img.astype(np.float32, copy=False)[None, :, :]     # (1,H,W)
        msk = msk.astype(np.int64, copy=False)[None, :, :]       # (1,H,W)

        sample = {"image": img, "mask": msk}
        if self.xform is not None:
            sample = self.xform(sample)
        return sample


def compute_class_weights_from_masks(items: List[Dict[str, Path]], mask_z: Optional[int], eps: float = 1e-6):
    counts = np.zeros((NUM_CLASSES,), dtype=np.float64)
    for it in items:
        m = load_mask_any(it["mask"], mask_z=mask_z, stem_hint=Path(it["image"]).stem)
        for c in range(NUM_CLASSES):
            counts[c] += float((m == c).sum())

    total = float(counts.sum()) + eps
    freq = counts / total

    inv = 1.0 / (freq + eps)
    inv = inv / inv.mean()
    inv = np.clip(inv, 0.25, 10.0)
    return counts, freq, inv.astype(np.float32)


def make_transforms(patch: int, num_samples: int, ratios: List[float]):
    from monai.transforms import (
        EnsureTyped,
        RandCropByLabelClassesd,
        RandFlipd,
        RandRotate90d,
        RandGaussianNoised,
        RandShiftIntensityd,
    )

    train_tf = Compose([
        EnsureTyped(keys=["image", "mask"]),
        RandCropByLabelClassesd(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(patch, patch),
            num_classes=NUM_CLASSES,
            ratios=ratios,
            num_samples=num_samples,
            allow_smaller=True,
        ),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.3),
        RandGaussianNoised(keys=["image"], std=0.01, prob=0.2),
    ])

    test_tf = Compose([EnsureTyped(keys=["image"])])
    return train_tf, test_tf


def remove_small_components(pred2d: np.ndarray, min_area: int, target_class: int, skip_classes: Tuple[int, ...]) -> np.ndarray:
    if not HAVE_SCIPY or min_area <= 0:
        return pred2d

    out = pred2d.copy()
    for c in range(NUM_CLASSES):
        if c in skip_classes:
            continue
        mask = (pred2d == c)
        if not mask.any():
            continue
        lab, n = ndi.label(mask)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        small = sizes < min_area
        small[0] = False
        out[small[lab]] = target_class
    return out


def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def make_rgba_overlay(pred2d: np.ndarray, alpha_default: float = 0.35, holes_alpha: float = 1.0) -> np.ndarray:
    h, w = pred2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    for c in [1, 2]:
        m = (pred2d == c)
        if not m.any():
            continue
        r, g, b = _hex_to_rgb01(CLASS_COLOURS[c])
        rgba[m, 0] = r
        rgba[m, 1] = g
        rgba[m, 2] = b
        rgba[m, 3] = alpha_default

    m3 = (pred2d == 3)
    if m3.any():
        r, g, b = _hex_to_rgb01(CLASS_COLOURS[3])
        rgba[m3, 0] = r
        rgba[m3, 1] = g
        rgba[m3, 2] = b
        rgba[m3, 3] = holes_alpha

    return rgba


def show_prediction_popup(img2d: np.ndarray, pred2d: np.ndarray, title: str, class_names: List[str]):
    im = img2d.astype(np.float32)
    mx = float(im.max()) if im.size else 1.0
    if mx > 1.0:
        im = im / (mx if mx != 0 else 1.0)

    fig = plt.figure(figsize=(12, 4))
    fig.suptitle(title)

    handles = [Patch(facecolor=CLASS_COLOURS[i], edgecolor="k", label=f"{i}: {class_names[i]}")
               for i in range(NUM_CLASSES)]

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(im, cmap="gray")
    ax1.set_title("Input slice")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(pred2d, interpolation="nearest", vmin=0, vmax=NUM_CLASSES - 1, cmap=CMAP)
    ax2.set_title("Predicted classes")
    ax2.axis("off")
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.05),
               ncol=2, frameon=True, fontsize=9)

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(im, cmap="gray")
    ax3.imshow(make_rgba_overlay(pred2d, alpha_default=0.35, holes_alpha=1.0), interpolation="nearest")
    ax3.set_title("Overlay")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def build_model(num_classes: int, cfg: dict) -> SegResNet:
    return SegResNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=num_classes,
        init_filters=int(cfg["init_filters"]),
        blocks_down=list(cfg["blocks_down"]),
        blocks_up=list(cfg["blocks_up"]),
        dropout_prob=float(cfg["dropout_prob"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, default="napari_pypore3d/train")
    ap.add_argument("--test_dir", type=str, default="napari_pypore3d/test")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_out", type=str, default="models/monai_seg_4class.pt")
    ap.add_argument("--show_each", action="store_true")

    ap.add_argument("--mask_z", type=int, default=None)
    ap.add_argument("--divisible", type=int, default=16)

    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--samples_per_image", type=int, default=128)

    ap.add_argument("--min_area", type=int, default=80)
    ap.add_argument("--speckle_target", type=int, default=2)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--onecycle", action="store_true")

    ap.add_argument("--ratios", type=str, default="", help='Override crop ratios "r0,r1,r2,r3"')
    ap.add_argument("--class_names", type=str, default=",".join(DEFAULT_CLASS_NAMES))

    # inference-on-test settings
    ap.add_argument("--sw_patch", type=int, default=512)
    ap.add_argument("--sw_overlap", type=float, default=0.25)

    args = ap.parse_args()
    set_determinism(seed=args.seed)

    class_names = [s.strip() for s in args.class_names.split(",")]
    if len(class_names) != NUM_CLASSES:
        class_names = DEFAULT_CLASS_NAMES

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    train_items = list_train_pairs(train_dir)
    if not train_items:
        raise RuntimeError(f"No training pairs found in {train_dir.resolve()}")

    print(f"Found {len(train_items)} training pairs.")
    for it in train_items:
        u = np.unique(load_mask_any(it["mask"], mask_z=args.mask_z, stem_hint=Path(it["image"]).stem))
        print(f"pair: {Path(it['image']).name} + {Path(it['mask']).name} | mask unique -> {u}")

    counts, freq, ce_w_np = compute_class_weights_from_masks(train_items, mask_z=args.mask_z)
    print("Class pixel counts:", counts.astype(np.int64).tolist())
    print("Class freq:", [round(float(x), 6) for x in freq.tolist()])
    print("CE weights (auto):", [round(float(x), 3) for x in ce_w_np.tolist()])

    # ---- crop ratios: either user override, or auto-but-don't-waste-on-background
    if args.ratios.strip():
        parts = [float(x) for x in args.ratios.split(",")]
        if len(parts) != NUM_CLASSES:
            raise ValueError("--ratios must have 4 comma-separated values")
        s = sum(parts)
        ratios = [p / s for p in parts]
    else:
        inv = 1.0 / (freq + 1e-6)
        inv = inv / inv.sum()
        inv = np.clip(inv, 0.05, 0.65)
        inv = inv / inv.sum()

        # push background down a bit (tiny datasets waste too many crops on bg)
        inv[0] = min(inv[0], 0.10)
        inv = inv / inv.sum()
        ratios = inv.tolist()

    print("Crop ratios used:", [round(float(x), 3) for x in ratios])

    train_tf, test_tf = make_transforms(patch=args.patch, num_samples=args.samples_per_image, ratios=ratios)

    ds_train = SliceDataset(train_items, xform=train_tf, mask_z=args.mask_z, divisible=args.divisible)
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,  # IMPORTANT for num_samples crops
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ---- MODEL CFG (must match inference)
    model_cfg = {
        "init_filters": 32,
        "blocks_down": [1, 2, 2, 4],
        "blocks_up": [1, 1, 1],
        "dropout_prob": 0.0,
    }

    model = build_model(NUM_CLASSES, model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ce_w = torch.tensor(ce_w_np, dtype=torch.float32, device=device)
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    ce_loss = nn.CrossEntropyLoss(weight=ce_w)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and torch.cuda.is_available()))

    scheduler = None
    if args.onecycle:
        steps_per_epoch = max(1, len(dl_train))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2,
            div_factor=10.0,
            final_div_factor=100.0,
        )

    best_loss = float("inf")
    best_state = None

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0

        for batch in dl_train:
            x = batch["image"].to(device)  # (B,1,H,W)
            y = batch["mask"].to(device)   # (B,1,H,W)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(args.amp and torch.cuda.is_available())):
                logits = model(x)  # (B,4,H,W)
                loss_dice = dice_loss(logits, y)
                y_ce = y.squeeze(1).long()
                loss_ce = ce_loss(logits, y_ce)
                loss = loss_dice + loss_ce

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            running += float(loss.item())

        avg = running / max(1, len(dl_train))
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch}/{args.epochs} | loss {avg:.4f} | lr {lr_now:.3e}")

        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # SAVE in a consistent format that the predictor will load correctly
    torch.save(
        {
            "state_dict": best_state,
            "model_cfg": model_cfg,
            "num_classes": NUM_CLASSES,
            "divisible": args.divisible,
        },
        str(model_out),
    )
    print(f"Saved model to: {model_out.resolve()} (best loss={best_loss:.4f})")

    # -------------------- TEST --------------------
    test_imgs = list_test_images(test_dir)
    if not test_imgs:
        print(f"No test images found in: {test_dir.resolve()}")
        return

    ckpt = torch.load(str(model_out), map_location=device)
    model = build_model(int(ckpt.get("num_classes", NUM_CLASSES)), ckpt.get("model_cfg", model_cfg)).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    print(f"Testing on {len(test_imgs)} slices from {test_dir} ...")
    with torch.no_grad():
        for p in test_imgs:
            img2d = load_image_any(p)
            img2d_pad, _, orig_hw = pad_2d_to_divisible(img2d, None, k=int(ckpt.get("divisible", args.divisible)))

            img = img2d_pad.astype(np.float32, copy=False)[None, :, :]  # (1,H,W)
            sample = test_tf({"image": img})
            x = sample["image"].unsqueeze(0).to(device)  # (1,1,H,W)

            # sliding window helps on big images
            H, W = x.shape[-2], x.shape[-1]
            if max(H, W) > args.sw_patch:
                logits = sliding_window_inference(
                    inputs=x,
                    roi_size=(args.sw_patch, args.sw_patch),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=args.sw_overlap,
                    mode="gaussian",
                )
            else:
                logits = model(x)

            pred = torch.argmax(logits, dim=1)  # (1,H,W)
            pred2d = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            pred2d = crop_back_2d(pred2d, orig_hw)

            pred2d = remove_small_components(
                pred2d,
                min_area=args.min_area,
                target_class=args.speckle_target,
                skip_classes=(0, 3),
            )

            print(p.name, "pred unique:", np.unique(pred2d))
            show_prediction_popup(img2d=img2d, pred2d=pred2d, title=f"Prediction: {p.name}", class_names=class_names)

            if not args.show_each:
                break


if __name__ == "__main__":
    main()
