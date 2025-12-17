# napari_pypore3d/monai_predict_open_napari.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image

from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference

# =========================
# EDIT THESE
# =========================
IMAGE_IN = r"napari_pypore3d\train\z35.png"
MODEL_PT = r"models/monai_seg_4class.pt"
OUT_NPY  = r"napari_pypore3d/out/pred_mask.npy"

DEVICE = "cuda"  # "cuda" or "cpu"
SW_PATCH = 512
SW_OVERLAP = 0.25
DIVISIBLE_OVERRIDE = -1  # -1 = use ckpt value, else e.g. 16
# =========================


def load_image_grayscale(path: Path) -> np.ndarray:
    """Load image -> float32 [0,1], 2D grayscale."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")
    if img.mode != "L":
        img = img.convert("L")

    arr = np.asarray(img, dtype=np.float32)
    mx = float(arr.max()) if arr.size else 1.0
    if mx > 1.0:
        arr = arr / 255.0 if mx <= 255.0 else (arr / mx)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale, got {arr.shape} for {path}")
    return arr


def _next_multiple(v: int, k: int) -> int:
    return ((v + k - 1) // k) * k


def pad_2d_to_divisible(img2d: np.ndarray, k: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad bottom/right so H,W divisible by k. Pad with edge values."""
    if k is None or k <= 1:
        return img2d, img2d.shape

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


def build_model(num_classes: int, model_cfg: Dict[str, Any]) -> SegResNet:
    return SegResNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=num_classes,
        init_filters=int(model_cfg.get("init_filters", 32)),
        blocks_down=list(model_cfg.get("blocks_down", [1, 2, 2, 4])),
        blocks_up=list(model_cfg.get("blocks_up", [1, 1, 1])),
        dropout_prob=float(model_cfg.get("dropout_prob", 0.0)),
    )


def load_checkpoint(model_path: Path, device: torch.device):
    ckpt = torch.load(str(model_path), map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        num_classes = int(ckpt.get("num_classes", 4))
        divisible = int(ckpt.get("divisible", 16))
        model_cfg = ckpt.get("model_cfg", {})
        state = ckpt["state_dict"]
        return state, num_classes, divisible, model_cfg

    if isinstance(ckpt, dict) and "model" in ckpt:
        num_classes = int(ckpt.get("num_classes", 4))
        divisible = int(ckpt.get("divisible", 16))
        model_cfg = ckpt.get("model_cfg", {})
        state = ckpt["model"]
        return state, num_classes, divisible, model_cfg

    raise ValueError(f"Unrecognised checkpoint format in: {model_path}")


@torch.no_grad()
def predict_mask(
    img2d: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    divisible: int,
    sw_patch: int,
    sw_overlap: float,
) -> np.ndarray:
    img_pad, orig_hw = pad_2d_to_divisible(img2d, divisible)
    x = torch.from_numpy(img_pad[None, None, :, :].astype(np.float32)).to(device)  # (1,1,H,W)

    H, W = x.shape[-2], x.shape[-1]
    if max(H, W) > sw_patch:
        logits = sliding_window_inference(
            inputs=x,
            roi_size=(sw_patch, sw_patch),
            sw_batch_size=1,
            predictor=model,
            overlap=sw_overlap,
            mode="gaussian",
        )
    else:
        logits = model(x)

    pred = torch.argmax(logits, dim=1)  # (1,H,W)
    pred2d = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)
    pred2d = crop_back_2d(pred2d, orig_hw)
    return pred2d


def open_in_napari(img2d: np.ndarray, mask2d: np.ndarray, title: str):
    import napari
    viewer = napari.Viewer(title=title)
    viewer.add_image(img2d, name="image", colormap="gray")
    viewer.add_labels(mask2d, name="mask")  # IMPORTANT: labels layer
    napari.run()


def main():
    image_path = Path(IMAGE_IN)
    model_path = Path(MODEL_PT)
    out_path = Path(OUT_NPY)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path.resolve()}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    device = torch.device("cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[predict] device={device}")

    state, num_classes, ckpt_divisible, model_cfg = load_checkpoint(model_path, device)
    divisible = ckpt_divisible if DIVISIBLE_OVERRIDE <= 0 else int(DIVISIBLE_OVERRIDE)

    model = build_model(num_classes=num_classes, model_cfg=model_cfg).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    img2d = load_image_grayscale(image_path)
    mask2d = predict_mask(
        img2d=img2d,
        model=model,
        device=device,
        divisible=divisible,
        sw_patch=int(SW_PATCH),
        sw_overlap=float(SW_OVERLAP),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mask2d)

    print(f"[predict] mask shape={mask2d.shape} | unique={np.unique(mask2d)}")
    print(f"[predict] saved npy: {out_path}")

    open_in_napari(img2d, mask2d, title=f"MONAI: {image_path.name}")


if __name__ == "__main__":
    main()
