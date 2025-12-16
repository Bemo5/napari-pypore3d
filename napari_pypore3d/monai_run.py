# napari_pypore3d/monai.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference


# -----------------------------------------------------------------------------
# RAW loader (SC1)
# -----------------------------------------------------------------------------
def load_raw_volume(path: str | Path, shape_zyx: tuple[int, int, int], dtype: np.dtype) -> np.ndarray:
    """
    Load a headerless .raw volume as a NumPy array shaped (Z, Y, X).

    Parameters
    ----------
    path : str | Path
        Path to raw file.
    shape_zyx : (Z, Y, X)
        Volume shape.
    dtype : np.dtype
        Raw datatype (most likely uint8 for your SC1).

    Returns
    -------
    vol : np.ndarray
        Volume array shaped (Z, Y, X).
    """
    path = Path(path)
    data = np.fromfile(str(path), dtype=dtype)

    expected = int(np.prod(shape_zyx))
    if data.size != expected:
        raise ValueError(
            f"RAW size mismatch for {path.name}:\n"
            f"  got elements   = {data.size}\n"
            f"  expected       = {expected}\n"
            f"  shape (Z,Y,X)  = {shape_zyx}\n"
            f"  dtype          = {dtype}"
        )

    vol = data.reshape(shape_zyx)
    return vol


# -----------------------------------------------------------------------------
# Preprocess
# -----------------------------------------------------------------------------
def preprocess_to_tensor(vol_zyx: np.ndarray) -> torch.Tensor:
    """
    Convert (Z,Y,X) volume to torch tensor shaped (B,C,Z,Y,X) float32 in [0,1].
    """
    v = np.asarray(vol_zyx)

    # ensure contiguous
    if not v.flags["C_CONTIGUOUS"]:
        v = np.ascontiguousarray(v)

    v = v.astype(np.float32, copy=False)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax > vmin:
        v = (v - vmin) / (vmax - vmin)
    else:
        v = np.zeros_like(v, dtype=np.float32)

    # (1,1,Z,Y,X)
    t = torch.from_numpy(v)[None, None, ...]
    return t


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def build_unet(device: torch.device) -> torch.nn.Module:
    """
    Small-ish 3D UNet for binary segmentation (1 output channel).
    """
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,   # binary logits
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    ).to(device)
    model.eval()
    return model


def load_weights_if_provided(model: torch.nn.Module, weights_path: str | None) -> None:
    """
    Load a .pth checkpoint if provided.
    Accepts either a raw state_dict or a dict with 'state_dict'.
    """
    if not weights_path:
        return

    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Weights file not found: {p}")

    ckpt = torch.load(str(p), map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("Unsupported checkpoint format. Expected a dict/state_dict.")

    # Strip possible 'module.' prefix
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print("⚠️ Missing keys while loading weights (ok if different head):", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("⚠️ Unexpected keys while loading weights:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    print(f"✅ Loaded weights: {p.name}")


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def infer_probability(
    model: torch.nn.Module,
    x: torch.Tensor,
    roi_size: tuple[int, int, int],
    overlap: float = 0.25,
    sw_batch_size: int = 1,
) -> np.ndarray:
    """
    Sliding-window inference. Returns probability volume shaped (Z,Y,X) float32.
    """
    device = next(model.parameters()).device
    x = x.to(device)

    logits = sliding_window_inference(
        inputs=x,
        roi_size=roi_size,        # (Z,Y,X) patch
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode="gaussian",
    )

    prob = torch.sigmoid(logits)[0, 0]  # (Z,Y,X)
    return prob.detach().cpu().numpy().astype(np.float32)


def prob_to_mask(prob_zyx: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """
    Convert probability volume to uint8 mask.
    """
    return (prob_zyx >= float(thr)).astype(np.uint8)


# -----------------------------------------------------------------------------
# Optional napari preview
# -----------------------------------------------------------------------------
def open_in_napari(vol_zyx: np.ndarray, prob_zyx: np.ndarray, mask_zyx: np.ndarray) -> None:
    try:
        import napari
    except Exception as e:
        print("❌ Napari not available in this environment:", e)
        return

    v = napari.Viewer()
    v.add_image(vol_zyx, name="SC1 (raw)", contrast_limits=[float(vol_zyx.min()), float(vol_zyx.max())])
    v.add_image(prob_zyx, name="MONAI prob", opacity=0.6)
    v.add_labels(mask_zyx, name="MONAI mask")
    napari.run()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MONAI 3D inference on SC1 RAW volume.")
    parser.add_argument(
        "--raw",
        type=str,
        default=str(Path("napari_pypore3d") / "data" / "SC1_700x700x700.raw"),
        help="Path to SC1 raw file.",
    )
    parser.add_argument("--z", type=int, default=700, help="Z dimension.")
    parser.add_argument("--y", type=int, default=700, help="Y dimension.")
    parser.add_argument("--x", type=int, default=700, help="X dimension.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint8",
        choices=["uint8", "uint16", "float32"],
        help="RAW datatype (most likely uint8).",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="64,128,128",
        help="Sliding window ROI size as 'Z,Y,X' (smaller = less VRAM).",
    )
    parser.add_argument("--overlap", type=float, default=0.25, help="Sliding window overlap.")
    parser.add_argument("--thr", type=float, default=0.5, help="Threshold for binary mask.")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Optional .pth weights for the UNet. If omitted, random weights are used (output will be meaningless).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path("napari_pypore3d") / "data"),
        help="Output directory for .npy results.",
    )
    parser.add_argument(
        "--napari",
        action="store_true",
        help="Open results in napari viewer after inference.",
    )

    args = parser.parse_args()

    raw_path = Path(args.raw)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shape = (int(args.z), int(args.y), int(args.x))
    dtype = np.dtype(args.dtype)

    roi = tuple(int(v.strip()) for v in args.roi.split(","))
    if len(roi) != 3:
        raise ValueError("--roi must be like '64,128,128' (Z,Y,X)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("RAW:", raw_path)
    print("Shape (Z,Y,X):", shape, "dtype:", dtype)
    print("ROI (Z,Y,X):", roi, "overlap:", args.overlap, "thr:", args.thr)

    # Load RAW
    vol = load_raw_volume(raw_path, shape_zyx=shape, dtype=dtype)
    print("Loaded SC1:", vol.shape, vol.dtype, "min/max:", vol.min(), vol.max(), "contig:", vol.flags["C_CONTIGUOUS"])

    # Preprocess
    x = preprocess_to_tensor(vol)

    # Model
    model = build_unet(device)
    if args.weights.strip():
        load_weights_if_provided(model, args.weights.strip())
    else:
        print("⚠️ No weights provided: using RANDOM weights (segmentation output will NOT be meaningful).")

    # Inference
    prob = infer_probability(model, x, roi_size=roi, overlap=float(args.overlap), sw_batch_size=1)
    mask = prob_to_mask(prob, thr=float(args.thr))

    # Save
    prob_path = outdir / "SC1_prob.npy"
    mask_path = outdir / "SC1_mask.npy"
    np.save(prob_path, prob)
    np.save(mask_path, mask)

    print("✅ Saved:", prob_path)
    print("✅ Saved:", mask_path)
    print("Mask unique values:", np.unique(mask))

    # Optional napari preview
    if args.napari:
        open_in_napari(vol, prob, mask)


if __name__ == "__main__":
    main()
