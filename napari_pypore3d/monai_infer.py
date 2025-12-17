from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from monai.networks.nets import UNet

# Optional image reading for png/jpg/tif
try:
    import imageio.v3 as iio
except Exception:
    iio = None

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}

def load_any(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy": 
        return np.asarray(np.load(path))
    if iio is None:
        raise SystemExit("Install imageio: pip install imageio")
    return np.asarray(iio.imread(path))

def to_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if arr.ndim >= 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr

def norm01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)

def pad_to_divisible(x: np.ndarray, k: int = 8) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    # x: (H,W)
    H, W = x.shape
    Hp = (H + (k - 1)) // k * k
    Wp = (W + (k - 1)) // k * k
    pad_h = Hp - H
    pad_w = Wp - W
    # pad bottom/right only (simple)
    xpad = np.pad(x, ((0, pad_h), (0, pad_w)), mode="edge")
    return xpad, (H, W, Hp, Wp)

def main():
    model_path = Path("napari_pypore3d/models/monai_unet_otsu_2d.pth")
    inp_path = Path("napari_pypore3d/images")  # will use first file
    out_path = Path("napari_pypore3d/preds")
    out_path.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(inp_path.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not files:
        raise SystemExit(f"No images in {inp_path}")

    imgp = files[0]
    print("Infer on:", imgp)

    ckpt = torch.load(model_path, map_location="cpu")
    spatial_dims = int(ckpt.get("spatial_dims", 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = norm01(to_gray(load_any(imgp)))
    img_pad, (H, W, Hp, Wp) = pad_to_divisible(img, k=8)

    x = torch.from_numpy(img_pad[None, None, ...]).to(device)  # (1,1,H,W)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)  # (Hp,Wp)

    pred = pred[:H, :W]  # unpad
    save_npy = out_path / f"{imgp.stem}_pred.npy"
    np.save(save_npy, pred)
    print("Saved:", save_npy)

if __name__ == "__main__":
    main()
