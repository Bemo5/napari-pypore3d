from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference

from magicgui.widgets import (
    Container, Label, PushButton, FileEdit, ComboBox, SpinBox, FloatSpinBox, CheckBox
)
from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_info, show_warning, show_error


# --- your label colours (napari Labels layer expects dict: label_int -> color) ---
LABEL_COLORS: Dict[int, str] = {
    0: "#000000",  # background
    1: "#D9D9D9",  # solid
    2: "#2E8B57",  # pores
    3: "#FFFF00",  # holes
}


def _next_multiple(v: int, k: int) -> int:
    return ((v + k - 1) // k) * k


def pad_2d_to_divisible(img2d: np.ndarray, k: int) -> Tuple[np.ndarray, Tuple[int, int]]:
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


def normalize_gray01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D image, got {x.shape}")
    x = x.astype(np.float32, copy=False)
    mx = float(x.max()) if x.size else 1.0
    if mx <= 1.0:
        return x
    if mx <= 255.0:
        return x / 255.0
    return x / (mx if mx != 0 else 1.0)


def build_model(num_classes: int, cfg: dict) -> SegResNet:
    return SegResNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=num_classes,
        init_filters=int(cfg.get("init_filters", 32)),
        blocks_down=list(cfg.get("blocks_down", [1, 2, 2, 4])),
        blocks_up=list(cfg.get("blocks_up", [1, 1, 1])),
        dropout_prob=float(cfg.get("dropout_prob", 0.0)),
    )


@dataclass
class LoadedModel:
    model: torch.nn.Module
    divisible: int
    num_classes: int
    cfg: dict


class MonaiInferWidget(Container):
    def __init__(self):
        super().__init__(layout="vertical")

        self._loaded: Optional[LoadedModel] = None

        self.lbl = Label(value="MONAI SegResNet Inference")
        self.ckpt_path = FileEdit(mode="r", filter="*.pt")
        self.device = ComboBox(choices=["cuda", "cpu"], value="cuda")
        self.sw_patch = SpinBox(value=512, min=64, max=2048, step=32, label="SW patch")
        self.sw_overlap = FloatSpinBox(value=0.25, min=0.0, max=0.9, step=0.05, label="SW overlap")
        self.use_sw = CheckBox(value=True, text="Use sliding window (for big images)")
        self.batch_z = SpinBox(value=4, min=1, max=64, step=1, label="Z batch (volume)")

        self.btn_load = PushButton(text="Load model")
        self.btn_seg_slice = PushButton(text="Segment current slice (or 2D image)")
        self.btn_seg_vol = PushButton(text="Segment whole volume (3D layer)")

        self.btn_load.changed.connect(self._on_load)
        self.btn_seg_slice.changed.connect(self._on_segment_slice)
        self.btn_seg_vol.changed.connect(self._on_segment_volume)

        self.extend([
            self.lbl,
            Label(value="Checkpoint (.pt):"),
            self.ckpt_path,
            self.device,
            self.sw_patch,
            self.sw_overlap,
            self.use_sw,
            self.batch_z,
            self.btn_load,
            self.btn_seg_slice,
            self.btn_seg_vol,
        ])

    def _viewer(self):
        try:
            return current_viewer()
        except Exception:
            return None

    def _active_image_layer(self) -> Optional[NapariImage]:
        v = self._viewer()
        if v is None:
            return None
        L = v.layers.selection.active
        if isinstance(L, NapariImage):
            return L
        # fallback: first image layer
        for lay in v.layers:
            if isinstance(lay, NapariImage):
                return lay
        return None

    def _ensure_model(self) -> Optional[LoadedModel]:
        if self._loaded is None:
            show_warning("Load a model first.")
            return None
        return self._loaded

    def _set_labels_colors(self, labels_layer):
        # robust across napari versions
        try:
            if hasattr(labels_layer, "color"):
                labels_layer.color = LABEL_COLORS  # dict: int->color
        except Exception:
            pass

    def _predict_2d(self, img2d: np.ndarray, loaded: LoadedModel) -> np.ndarray:
        dev = torch.device(self.device.value if torch.cuda.is_available() else "cpu")
        m = loaded.model.to(dev).eval()

        img2d = normalize_gray01(img2d)
        img_pad, orig_hw = pad_2d_to_divisible(img2d, loaded.divisible)

        x = torch.from_numpy(img_pad[None, None, :, :].astype(np.float32)).to(dev)

        with torch.no_grad():
            use_sw = bool(self.use_sw.value)
            swp = int(self.sw_patch.value)
            ov = float(self.sw_overlap.value)

            if use_sw and max(x.shape[-2], x.shape[-1]) > swp:
                logits = sliding_window_inference(
                    inputs=x,
                    roi_size=(swp, swp),
                    sw_batch_size=1,
                    predictor=m,
                    overlap=ov,
                    mode="gaussian",
                )
            else:
                logits = m(x)

            pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        return crop_back_2d(pred, orig_hw)

    def _on_load(self):
        p = Path(str(self.ckpt_path.value))
        if not p.exists():
            show_error("Pick a valid .pt checkpoint.")
            return

        dev = torch.device(self.device.value if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(p), map_location=dev)

        num_classes = int(ckpt.get("num_classes", 4))
        divisible = int(ckpt.get("divisible", 16))
        cfg = ckpt.get("model_cfg", {"init_filters": 32, "blocks_down": [1, 2, 2, 4], "blocks_up": [1, 1, 1], "dropout_prob": 0.0})

        model = build_model(num_classes, cfg).to(dev)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()

        self._loaded = LoadedModel(model=model, divisible=divisible, num_classes=num_classes, cfg=cfg)
        show_info(f"Loaded model: classes={num_classes}, divisible={divisible}")

    def _on_segment_slice(self):
        loaded = self._ensure_model()
        if loaded is None:
            return

        v = self._viewer()
        if v is None:
            show_error("No napari viewer found.")
            return

        img_layer = self._active_image_layer()
        if img_layer is None:
            show_error("No Image layer selected.")
            return

        data = np.asarray(img_layer.data)

        # 2D image
        if data.ndim == 2:
            pred2d = self._predict_2d(data, loaded)
            name = f"{img_layer.name}_pred"
            labels = v.layers[name].data if name in v.layers else None

            if name in v.layers:
                v.layers[name].data = pred2d
                out_layer = v.layers[name]
            else:
                out_layer = v.add_labels(pred2d, name=name, opacity=0.45)

            self._set_labels_colors(out_layer)
            show_info(f"Predicted 2D: {name} unique={np.unique(pred2d)}")
            return

        # 3D image: segment current Z slice
        if data.ndim == 3:
            z = int(v.dims.current_step[0])
            z = max(0, min(z, data.shape[0] - 1))
            pred2d = self._predict_2d(data[z], loaded)

            name = f"{img_layer.name}_pred_slice"
            if name in v.layers:
                v.layers[name].data = pred2d
                out_layer = v.layers[name]
            else:
                out_layer = v.add_labels(pred2d, name=name, opacity=0.55)

            self._set_labels_colors(out_layer)
            show_info(f"Predicted Z={z}: {name} unique={np.unique(pred2d)}")
            return

        show_warning(f"Unsupported image ndim={data.ndim}")

    def _on_segment_volume(self):
        loaded = self._ensure_model()
        if loaded is None:
            return

        v = self._viewer()
        if v is None:
            show_error("No napari viewer found.")
            return

        img_layer = self._active_image_layer()
        if img_layer is None:
            show_error("No Image layer selected.")
            return

        vol = np.asarray(img_layer.data)
        if vol.ndim != 3:
            show_warning("Whole-volume segmentation requires a 3D Image layer.")
            return

        dev = torch.device(self.device.value if torch.cuda.is_available() else "cpu")
        m = loaded.model.to(dev).eval()

        Z, H, W = vol.shape
        batch_z = int(self.batch_z.value)

        # pre-pad size once (assuming consistent slice shape)
        dummy = normalize_gray01(vol[0])
        dummy_pad, orig_hw = pad_2d_to_divisible(dummy, loaded.divisible)
        Hp, Wp = dummy_pad.shape

        out = np.zeros((Z, orig_hw[0], orig_hw[1]), dtype=np.uint8)

        show_info(f"Segmenting volume: {img_layer.name} Z={Z} (batch={batch_z})")

        with torch.no_grad():
            use_amp = (dev.type == "cuda")
            for z0 in range(0, Z, batch_z):
                z1 = min(Z, z0 + batch_z)

                stack = []
                for z in range(z0, z1):
                    img2d = normalize_gray01(vol[z])
                    img_pad, _ = pad_2d_to_divisible(img2d, loaded.divisible)
                    stack.append(img_pad)

                x = np.stack(stack, axis=0).astype(np.float32)  # (B,Hp,Wp)
                x = torch.from_numpy(x[:, None, :, :]).to(dev)  # (B,1,Hp,Wp)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    # for volume we usually skip SW (Hp/Wp already reasonable); enable if you want
                    logits = m(x)
                    pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)  # (B,Hp,Wp)

                for i, z in enumerate(range(z0, z1)):
                    out[z] = crop_back_2d(pred[i], orig_hw)

                if (z1 % 50) == 0 or z1 == Z:
                    print(f"[infer] {z1}/{Z}")

        name = f"{img_layer.name}_pred_vol"
        if name in v.layers:
            v.layers[name].data = out
            lab = v.layers[name]
        else:
            lab = v.add_labels(out, name=name, opacity=0.35)

        self._set_labels_colors(lab)
        show_info(f"Done. Added labels: {name} shape={out.shape} unique={np.unique(out)}")


def make_monai_infer_widget() -> MonaiInferWidget:
    return MonaiInferWidget()
