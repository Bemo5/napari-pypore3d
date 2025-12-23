# napari_pypore3d/monai_segment_volume_open_napari.py
# ------------------------------------------------------------
# Segment a FULL 3D RAW volume (e.g., SC1_700x700x700.raw) using your
# 2D MONAI SegResNet checkpoint (slice-by-slice), save labels as .npy,
# then open napari in 3D to visualise.
#
# RUN (from repo root):
#   python .\napari_pypore3d\monai_segment_volume_open_napari.py
#
# Notes:
# - This is "pseudo-3D": it predicts each Z slice independently.
# - Output labels are uint8 with values {0,1,2,3}.
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from monai.networks.nets import SegResNet

# ----------------------------
# EDIT THESE
# ----------------------------
RAW_PATH  = Path(r"napari_pypore3d\data\SC1_700x700x700.raw")  # SC1 raw
RAW_DTYPE = np.uint8                                                # likely uint8 (0..255)
RAW_SHAPE = (700, 700, 700)  # (Z, Y, X)

CKPT_PATH = Path(r"models\monai_seg_4class_BEST.pt")

OUT_NPY   = Path(r"napari_pypore3d\out\SC1_pred_labels_uint8.npy")

DEVICE = "cuda"  # "cuda" or "cpu"
BATCH_SLICES = 4  # try 4-8 on RTX 3060; if OOM, set 1
USE_AMP = True

OPEN_NAPARI = True

# Label colours (RGBA 0..1) for napari Labels layer
LABEL_COLOURS = {
    0: (0.0, 0.0, 0.0, 0.0),  # background transparent
    1: (0.85, 0.85, 0.85, 1.0),  # solid
    2: (0.18, 0.55, 0.34, 1.0),  # pores (green)
    3: (1.0, 1.0, 0.0, 1.0),     # holes (yellow)
}


# ----------------------------
# Helpers
# ----------------------------
def _next_multiple(v: int, k: int) -> int:
    return ((v + k - 1) // k) * k


def pad_2d_to_divisible(img2d: np.ndarray, k: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad (H,W) -> (H',W') where H',W' divisible by k. Returns padded + original (H,W)."""
    h, w = img2d.shape
    H = _next_multiple(h, k)
    W = _next_multiple(w, k)
    pad_h = H - h
    pad_w = W - w
    if pad_h == 0 and pad_w == 0:
        return img2d, (h, w)
    img_pad = np.pad(img2d, ((0, pad_h), (0, pad_w)), mode="edge")
    return img_pad, (h, w)


def crop_back_2d(arr2d: np.ndarray, orig_hw: Tuple[int, int]) -> np.ndarray:
    oh, ow = orig_hw
    return arr2d[:oh, :ow]


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


def load_ckpt(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    num_classes = int(ckpt.get("num_classes", 4))
    model_cfg = ckpt.get(
        "model_cfg",
        {"init_filters": 32, "blocks_down": [1, 2, 2, 4], "blocks_up": [1, 1, 1], "dropout_prob": 0.0},
    )
    divisible = int(ckpt.get("divisible", 16))

    model = build_model(num_classes, model_cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, num_classes, model_cfg, divisible


# ----------------------------
# Main
# ----------------------------
def main():
    repo_root = Path(__file__).resolve().parents[1]
    raw_path = (repo_root / RAW_PATH) if not RAW_PATH.is_absolute() else RAW_PATH
    ckpt_path = (repo_root / CKPT_PATH) if not CKPT_PATH.is_absolute() else CKPT_PATH
    out_path = (repo_root / OUT_NPY) if not OUT_NPY.is_absolute() else OUT_NPY
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"RAW not found: {raw_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CKPT not found: {ckpt_path}")

    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[paths] raw={raw_path}")
    print(f"[paths] ckpt={ckpt_path}")
    print(f"[paths] out ={out_path}")
    print(f"[device] {device}")

    # Load model
    model, num_classes, model_cfg, divisible = load_ckpt(ckpt_path, device)
    print(f"[model] num_classes={num_classes} divisible={divisible} cfg={model_cfg}")

    # Memory-map the RAW volume (Z,Y,X)
    Z, Y, X = RAW_SHAPE
    vol = np.memmap(str(raw_path), dtype=RAW_DTYPE, mode="r", shape=(Z, Y, X))

    # Create output .npy as a memmap (so we don't hold all labels in RAM)
    out = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.uint8, shape=(Z, Y, X))

    # We will pad each slice to divisible (e.g. 16) -> 700 becomes 704
    dummy = np.zeros((Y, X), dtype=np.float32)
    dummy_pad, _ = pad_2d_to_divisible(dummy, k=divisible)
    Hpad, Wpad = dummy_pad.shape
    print(f"[pad] slice {Y}x{X} -> {Hpad}x{Wpad} (divisible={divisible})")

    # Inference loop
    use_amp = (USE_AMP and device.type == "cuda")
    scaler_ctx = torch.amp.autocast("cuda", enabled=use_amp)

    def run_batch(z_start: int, z_end: int):
        bsz = z_end - z_start
        # Build batch tensor (B,1,Hpad,Wpad)
        batch = np.empty((bsz, 1, Hpad, Wpad), dtype=np.float32)

        for i, z in enumerate(range(z_start, z_end)):
            sl_u8 = np.asarray(vol[z], dtype=np.float32)  # (Y,X)
            sl = sl_u8 / 255.0 if np.max(sl_u8) > 1.0 else sl_u8
            sl_pad, orig_hw = pad_2d_to_divisible(sl, k=divisible)
            batch[i, 0, :, :] = sl_pad

        x = torch.from_numpy(batch).to(device)

        with torch.no_grad():
            with scaler_ctx:
                logits = model(x)  # (B,C,Hpad,Wpad)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)  # (B,Hpad,Wpad)

        # write back cropped slices
        for i, z in enumerate(range(z_start, z_end)):
            out[z] = pred[i, :Y, :X]

    print(f"[infer] Z={Z} slices | batch={BATCH_SLICES} | amp={use_amp}")
    z = 0
    while z < Z:
        z_end = min(Z, z + max(1, int(BATCH_SLICES)))
        try:
            run_batch(z, z_end)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and device.type == "cuda":
                print(f"[oom] CUDA OOM at z={z}..{z_end-1}. Retrying with batch=1 ...")
                torch.cuda.empty_cache()
                # retry slice-by-slice
                for zz in range(z, z_end):
                    run_batch(zz, zz + 1)
            else:
                raise

        if (z_end % 25) == 0 or z_end == Z:
            # quick progress
            uniq = np.unique(out[z_end - 1])
            print(f"[infer] done {z_end}/{Z} | last slice unique={uniq}")

        z = z_end

    out.flush()
    print(f"[done] Saved labels -> {out_path}  shape={out.shape} dtype={out.dtype}")
    print(f"[done] Example uniques: z0={np.unique(out[0])} zmid={np.unique(out[Z//2])} zlast={np.unique(out[-1])}")

    # Open napari 3D
    if OPEN_NAPARI:
        import napari

        viewer = napari.Viewer(ndisplay=3)
        # napari can display memmaps fine
        viewer.add_image(vol, name="SC1_raw", colormap="gray")
        labels = viewer.add_labels(out, name="SC1_pred_labels", opacity=0.45)
        try:
            labels.color_mode = "direct"
            labels.color = LABEL_COLOURS
        except Exception as e:
            print(f"[warn] Could not set custom label colours on this napari version: {e}")

        napari.run()


if __name__ == "__main__":
    main()
