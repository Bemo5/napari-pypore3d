# napari_pypore3d/monai_train_best_4class.py
# ------------------------------------------------------------
# RUN:
#   python .\napari_pypore3d\monai_train_best_4class.py
#
# Uses (relative to repo root):
#   napari_pypore3d/train  (image + mask pairs)
#   napari_pypore3d/test   (images, and optionally image+mask pairs)
#
# Saves:
#   models/monai_seg_4class_BEST.pt  (best by val dice, no background)
#   models/monai_seg_4class_LAST.pt  (last each epoch)
#
# BONUS:
# - If test/ contains pairs (png + npy/npz with same stem), it will compute test Dice.
# - It can also save predicted masks for ALL test images to: napari_pypore3d/test_preds/
#   (so you can load them in napari and correct fast).
#
# Assumptions:
# - Images: .png/.jpg/.tif etc (any size)
# - Masks : same stem as image, .npy or .npz
# - Masks are 2D and labels in {0,1,2,3}
# ------------------------------------------------------------

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from monai.data import DataLoader
from monai.data.utils import list_data_collate
from monai.transforms import Compose
from monai.utils import set_determinism
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference


# ----------------------------
# CONFIG
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = REPO_ROOT / "napari_pypore3d" / "train"
TEST_DIR = REPO_ROOT / "napari_pypore3d" / "test"

MODELS_DIR = REPO_ROOT / "models"
BEST_OUT = MODELS_DIR / "monai_seg_4class_BEST.pt"
LAST_OUT = MODELS_DIR / "monai_seg_4class_LAST.pt"

# Training hyperparams
EPOCHS = 400
BATCH = 1
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
USE_AMP = True
USE_ONECYCLE = True

SEED = 0
DIVISIBLE = 16

PATCH = 256
SAMPLES_PER_IMAGE = 32
RATIOS = [0.05, 0.30, 0.35, 0.30]  # bg,solid,pores,holes
VAL_FRAC = 0.20
EARLY_PATIENCE = 60

# Validation/test inference
SW_PATCH = 512
SW_OVERLAP = 0.25

# Test behaviour
SHOW_TEST_POPUPS = True
SHOW_EACH_TEST_IMAGE = False  # popups: False => show only first image popup
EVAL_TEST_IF_MASK_EXISTS = True  # if test has paired masks, compute dice
SAVE_TEST_PREDS = True           # save predictions for ALL test images
TEST_PREDS_DIR = REPO_ROOT / "napari_pypore3d" / "test_preds"  # output folder

# ----------------------------
# Constants
# ----------------------------
NUM_CLASSES = 4
CLASS_NAMES = ["Background", "Solid", "Pores", "Holes"]
CLASS_COLOURS = ["#000000", "#D9D9D9", "#2E8B57", "#FFFF00"]
CMAP = ListedColormap(CLASS_COLOURS)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ----------------------------
# IO helpers
# ----------------------------
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def load_image_gray01(path: Path) -> np.ndarray:
    """Load image as float32 in [0,1], shape (H,W)."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")
    if img.mode != "L":
        img = img.convert("L")
    arr = np.asarray(img, dtype=np.float32)
    mx = float(arr.max()) if arr.size else 1.0
    if mx <= 1.0:
        return arr
    if mx <= 255.0:
        return arr / 255.0
    return arr / (mx if mx != 0 else 1.0)


def load_mask_2d_0123(path: Path) -> np.ndarray:
    """Load mask npy/npz -> uint8 2D with labels in {0,1,2,3}."""
    suf = path.suffix.lower()
    if suf == ".npy":
        m = np.load(path)
    elif suf == ".npz":
        z = np.load(path)
        if len(z.files) == 0:
            raise ValueError(f"Empty npz: {path}")
        m = z[z.files[0]]
    else:
        raise ValueError(f"Unsupported mask type: {path}")

    m = np.asarray(m)
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D. Got {m.shape} in {path}")

    m = m.astype(np.int64, copy=False)
    uniq = np.unique(m)
    bad = [int(v) for v in uniq if int(v) not in {0, 1, 2, 3}]
    if bad:
        raise ValueError(f"Mask {path} has invalid labels {bad}. Unique={uniq}")

    return m.astype(np.uint8, copy=False)


def find_mask_for_stem(folder: Path, stem: str) -> Optional[Path]:
    for ext in (".npy", ".npz"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def list_pairs(folder: Path) -> List[Dict[str, Path]]:
    """Return [{'image': Path, 'mask': Path}, ...] for a folder."""
    items: List[Dict[str, Path]] = []
    for img_path in sorted(folder.iterdir()):
        if not img_path.is_file() or not is_image_file(img_path):
            continue
        mask_path = find_mask_for_stem(folder, img_path.stem)
        if mask_path is None:
            continue
        items.append({"image": img_path, "mask": mask_path})
    return items


def list_test_items(folder: Path) -> List[Dict[str, Optional[Path]]]:
    """Return [{'image': Path, 'mask': Path|None}, ...] for test folder."""
    if not folder.exists():
        return []
    out: List[Dict[str, Optional[Path]]] = []
    for img_path in sorted(folder.iterdir()):
        if not img_path.is_file() or not is_image_file(img_path):
            continue
        mask_path = find_mask_for_stem(folder, img_path.stem)
        out.append({"image": img_path, "mask": mask_path})
    return out


def _next_multiple(v: int, k: int) -> int:
    return ((v + k - 1) // k) * k


def pad_2d_to_divisible(img2d: np.ndarray, mask2d: Optional[np.ndarray], k: int):
    """Pad bottom/right so H,W divisible by k. Image=edge, mask=0."""
    if k <= 1:
        return img2d, mask2d, img2d.shape

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


# ----------------------------
# Dataset
# ----------------------------
class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Dict[str, Path]], xform=None, divisible: int = 16):
        self.items = items
        self.xform = xform
        self.divisible = divisible

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = load_image_gray01(it["image"])
        msk = load_mask_2d_0123(it["mask"])

        img, msk, _ = pad_2d_to_divisible(img, msk, k=self.divisible)

        img = img.astype(np.float32, copy=False)[None, :, :]  # (1,H,W)
        msk = msk.astype(np.int64, copy=False)[None, :, :]    # (1,H,W)

        sample = {"image": img, "mask": msk}
        if self.xform is not None:
            sample = self.xform(sample)
        return sample


# ----------------------------
# Metrics
# ----------------------------
def dice_per_class(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4, eps: float = 1e-6) -> List[float]:
    out = []
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        inter = float((p & g).sum())
        denom = float(p.sum() + g.sum()) + eps
        out.append((2.0 * inter) / denom)
    return out


# ----------------------------
# Model + training helpers
# ----------------------------
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


def compute_class_weights(items: List[Dict[str, Path]], eps: float = 1e-6) -> np.ndarray:
    counts = np.zeros((NUM_CLASSES,), dtype=np.float64)
    for it in items:
        m = load_mask_2d_0123(it["mask"])
        for c in range(NUM_CLASSES):
            counts[c] += float((m == c).sum())

    total = float(counts.sum()) + eps
    freq = counts / total

    inv = 1.0 / (freq + eps)
    inv = inv / inv.mean()
    inv = np.clip(inv, 0.25, 10.0)
    return inv.astype(np.float32)


def make_train_transforms(patch: int, samples_per_image: int, ratios: List[float]):
    from monai.transforms import (
        EnsureTyped,
        RandCropByLabelClassesd,
        RandFlipd,
        RandRotate90d,
        RandGaussianNoised,
        RandShiftIntensityd,
        RandAdjustContrastd,
    )

    return Compose([
        EnsureTyped(keys=["image", "mask"]),
        RandCropByLabelClassesd(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(patch, patch),
            num_classes=NUM_CLASSES,
            ratios=ratios,
            num_samples=samples_per_image,
            allow_smaller=True,
        ),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.35),
        RandAdjustContrastd(keys=["image"], gamma=(0.8, 1.2), prob=0.25),
        RandGaussianNoised(keys=["image"], std=0.01, prob=0.2),
    ])


def save_ckpt(path: Path, state_dict: dict, model_cfg: dict, extra: Optional[dict] = None):
    payload = {
        "state_dict": state_dict,
        "model_cfg": model_cfg,
        "num_classes": NUM_CLASSES,
        "divisible": int(DIVISIBLE),
        "seed": int(SEED),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def show_prediction_popup(img2d: np.ndarray, pred2d: np.ndarray, title: str):
    fig = plt.figure(figsize=(12, 4))
    fig.suptitle(title)

    handles = [Patch(facecolor=CLASS_COLOURS[i], edgecolor="k", label=f"{i}: {CLASS_NAMES[i]}")
               for i in range(NUM_CLASSES)]

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img2d, cmap="gray")
    ax1.set_title("Input")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(pred2d, interpolation="nearest", vmin=0, vmax=NUM_CLASSES - 1, cmap=CMAP)
    ax2.set_title("Predicted mask")
    ax2.axis("off")
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.05),
               ncol=2, frameon=True, fontsize=9)

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(img2d, cmap="gray")
    ax3.imshow(pred2d, interpolation="nearest", alpha=0.45, vmin=0, vmax=NUM_CLASSES - 1, cmap=CMAP)
    ax3.set_title("Overlay")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    set_determinism(seed=SEED)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"TRAIN_DIR not found: {TRAIN_DIR}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    items_all = list_pairs(TRAIN_DIR)
    if len(items_all) < 2:
        raise RuntimeError(f"Need at least 2 training pairs in {TRAIN_DIR} so we can keep 1 for validation.")

    # ---- split train/val
    rng = random.Random(SEED)
    items = items_all[:]
    rng.shuffle(items)

    n_val = max(1, int(round(len(items) * float(VAL_FRAC))))
    n_val = min(n_val, len(items) - 1)
    val_items = items[:n_val]
    train_items = items[n_val:]

    print(f"[paths] repo_root = {REPO_ROOT}")
    print(f"[paths] train_dir = {TRAIN_DIR} | test_dir = {TEST_DIR}")
    print(f"[data] Found {len(items_all)} pairs total -> train={len(train_items)} val={len(val_items)}")
    for it in items_all:
        u = np.unique(load_mask_2d_0123(it["mask"]))
        print(f"  pair: {it['image'].name} + {it['mask'].name} | unique={u}")

    ratios = RATIOS[:]
    s = sum(ratios)
    ratios = [r / s for r in ratios]
    print("[train] Crop ratios:", [round(float(x), 3) for x in ratios])

    train_tf = make_train_transforms(PATCH, SAMPLES_PER_IMAGE, ratios)

    ds_train = SliceDataset(train_items, xform=train_tf, divisible=DIVISIBLE)
    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model_cfg = {
        "init_filters": 32,
        "blocks_down": [1, 2, 2, 4],
        "blocks_up": [1, 1, 1],
        "dropout_prob": 0.0,
    }
    model = build_model(NUM_CLASSES, model_cfg).to(device)

    ce_w_np = compute_class_weights(train_items)
    print("[train] CE weights:", [round(float(x), 3) for x in ce_w_np.tolist()])
    ce_w = torch.tensor(ce_w_np, dtype=torch.float32, device=device)

    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    ce_loss = nn.CrossEntropyLoss(weight=ce_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    scheduler = None
    if USE_ONECYCLE:
        steps_per_epoch = max(1, len(dl_train))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LR,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2,
            div_factor=10.0,
            final_div_factor=100.0,
        )

    best_val_dice = -1.0
    best_state: Optional[dict] = None
    bad_epochs = 0

    def validate_full_images() -> Tuple[float, float, List[float]]:
        model.eval()
        dices_all: List[List[float]] = []
        dices_no_bg: List[float] = []

        with torch.no_grad():
            for it in val_items:
                img2d = load_image_gray01(it["image"])
                gt2d = load_mask_2d_0123(it["mask"])

                img_pad, gt_pad, orig_hw = pad_2d_to_divisible(img2d, gt2d, k=DIVISIBLE)
                x = torch.from_numpy(img_pad[None, None, :, :].astype(np.float32)).to(device)

                H, W = x.shape[-2], x.shape[-1]
                if max(H, W) > SW_PATCH:
                    logits = sliding_window_inference(
                        inputs=x,
                        roi_size=(SW_PATCH, SW_PATCH),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=SW_OVERLAP,
                        mode="gaussian",
                    )
                else:
                    logits = model(x)

                pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
                pred = crop_back_2d(pred, orig_hw)

                gt = crop_back_2d(gt_pad, orig_hw).astype(np.uint8)

                d = dice_per_class(pred, gt, num_classes=NUM_CLASSES)
                dices_all.append(d)
                dices_no_bg.append(float(np.mean(d[1:])))

        mean_no_bg = float(np.mean(dices_no_bg)) if dices_no_bg else 0.0
        mean_all = float(np.mean([np.mean(d) for d in dices_all])) if dices_all else 0.0
        per_class = list(np.mean(np.array(dices_all), axis=0).tolist()) if dices_all else [0.0] * NUM_CLASSES
        return mean_no_bg, mean_all, per_class

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running = 0.0

            for batch in dl_train:
                x = batch["image"].to(device)
                y = batch["mask"].to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda")):
                    logits = model(x)
                    loss_d = dice_loss(logits, y)
                    y_ce = y.squeeze(1).long()
                    loss_c = ce_loss(logits, y_ce)
                    loss = loss_d + loss_c

                scaler.scale(loss).backward()

                if GRAD_CLIP and GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(GRAD_CLIP))

                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                running += float(loss.item())

            train_loss = running / max(1, len(dl_train))
            lr_now = float(optimizer.param_groups[0]["lr"])

            mean_no_bg, mean_all, per_class = validate_full_images()

            print(
                f"epoch {epoch}/{EPOCHS} | "
                f"train_loss {train_loss:.4f} | lr {lr_now:.3e} | "
                f"val_dice(no_bg) {mean_no_bg:.4f} | val_dice(all) {mean_all:.4f}"
            )

            # Save LAST every epoch
            last_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            save_ckpt(
                LAST_OUT,
                last_state,
                model_cfg,
                extra={"epoch": epoch, "val_dice_no_bg": mean_no_bg, "val_dice_all": mean_all},
            )

            # Save BEST immediately
            if mean_no_bg > best_val_dice + 1e-5:
                best_val_dice = mean_no_bg
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                save_ckpt(
                    BEST_OUT,
                    best_state,
                    model_cfg,
                    extra={
                        "epoch": epoch,
                        "best_val_dice_no_bg": float(best_val_dice),
                        "val_dice_all": float(mean_all),
                        "val_dice_per_class": [float(x) for x in per_class],
                    },
                )
                bad_epochs = 0
                print(f"  ✅ BEST saved -> {BEST_OUT.name} (val_dice_no_bg={best_val_dice:.4f})")
            else:
                bad_epochs += 1
                if bad_epochs >= int(EARLY_PATIENCE):
                    print(f"  ⛔ Early stopping (no improvement for {EARLY_PATIENCE} epochs).")
                    break

    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl+C detected.")
        if best_state is not None:
            save_ckpt(
                BEST_OUT,
                best_state,
                model_cfg,
                extra={"best_val_dice_no_bg": float(best_val_dice)},
            )
            print(f"[interrupt] Re-saved BEST -> {BEST_OUT}")
        raise

    print(f"\nDone. BEST model: {BEST_OUT.resolve()}  (best_val_dice_no_bg={best_val_dice:.4f})")

    # ----------------------------
    # Test prediction + optional pair evaluation
    # ----------------------------
    test_items = list_test_items(TEST_DIR)
    if not test_items:
        print(f"[test] No test images found in: {TEST_DIR.resolve()}")
        return

    ckpt = torch.load(str(BEST_OUT), map_location=device)
    model2 = build_model(int(ckpt.get("num_classes", NUM_CLASSES)), ckpt.get("model_cfg", model_cfg)).to(device)
    model2.load_state_dict(ckpt["state_dict"], strict=False)
    model2.eval()

    if SAVE_TEST_PREDS:
        TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[test] Found {len(test_items)} test images. "
          f"Paired-eval={'ON' if EVAL_TEST_IF_MASK_EXISTS else 'OFF'} | "
          f"SavePreds={'ON' if SAVE_TEST_PREDS else 'OFF'}")

    all_dices: List[List[float]] = []
    no_bg_dices: List[float] = []
    n_paired = 0
    n_popup_shown = 0

    with torch.no_grad():
        for it in test_items:
            img_path = it["image"]
            mask_path = it["mask"]

            img2d = load_image_gray01(img_path)

            gt2d = None
            if EVAL_TEST_IF_MASK_EXISTS and mask_path is not None and mask_path.exists():
                gt2d = load_mask_2d_0123(mask_path)

            img_pad, gt_pad, orig_hw = pad_2d_to_divisible(
                img2d, gt2d, k=int(ckpt.get("divisible", DIVISIBLE))
            )

            x = torch.from_numpy(img_pad[None, None, :, :].astype(np.float32)).to(device)

            H, W = x.shape[-2], x.shape[-1]
            if max(H, W) > SW_PATCH:
                logits = sliding_window_inference(
                    inputs=x,
                    roi_size=(SW_PATCH, SW_PATCH),
                    sw_batch_size=1,
                    predictor=model2,
                    overlap=SW_OVERLAP,
                    mode="gaussian",
                )
            else:
                logits = model2(x)

            pred2d = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            pred2d = crop_back_2d(pred2d, orig_hw)

            print("[test]", img_path.name, "pred unique:", np.unique(pred2d))

            # Save prediction (for napari correction)
            if SAVE_TEST_PREDS:
                out_p = TEST_PREDS_DIR / f"{img_path.stem}.npz"
                np.savez_compressed(out_p, pred=pred2d.astype(np.uint8, copy=False))

            # Paired evaluation (if GT exists)
            if gt_pad is not None:
                gt = crop_back_2d(gt_pad, orig_hw).astype(np.uint8)
                d = dice_per_class(pred2d, gt, num_classes=NUM_CLASSES)
                all_dices.append(d)
                no_bg = float(np.mean(d[1:]))
                no_bg_dices.append(no_bg)
                n_paired += 1
                print(f"       dice per class: {[round(float(x), 4) for x in d]} | no_bg={no_bg:.4f}")

            # Show popups (optional)
            if SHOW_TEST_POPUPS:
                if SHOW_EACH_TEST_IMAGE or (n_popup_shown == 0):
                    show_prediction_popup(img2d, pred2d, title=f"Prediction: {img_path.name}")
                    n_popup_shown += 1
                if (not SHOW_EACH_TEST_IMAGE) and (n_popup_shown >= 1):
                    # if user wants only one popup, stop showing popups
                    pass

    # Summary if we had paired test masks
    if n_paired > 0:
        arr = np.array(all_dices, dtype=np.float32)
        mean_per_class = arr.mean(axis=0).tolist()
        mean_no_bg = float(np.mean(no_bg_dices))
        print("\n[test] SUMMARY over paired test items:")
        print("       n_paired =", n_paired)
        print("       mean dice per class:", [round(float(x), 4) for x in mean_per_class])
        print("       mean dice no_bg:", round(mean_no_bg, 4))
    else:
        print("\n[test] No paired test masks found -> predicted PNGs only (no Dice computed).")
        if SAVE_TEST_PREDS:
            print(f"[test] Saved preds to: {TEST_PREDS_DIR.resolve()}")


if __name__ == "__main__":
    main()
