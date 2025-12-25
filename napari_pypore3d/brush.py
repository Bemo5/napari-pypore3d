# napari_pypore3d/brush.py — Brush widget for mask painting
from __future__ import annotations

import os
import json
import zipfile
import datetime
from typing import Optional, List, Tuple, Dict

import numpy as np
from magicgui.widgets import (
    Container, ComboBox, PushButton, Label, SpinBox
)

try:
    from magicgui.widgets import TextEdit
except Exception:
    TextEdit = None

from napari import current_viewer
from napari.layers import Image as NapariImage, Labels as NapariLabels
from napari.utils.notifications import show_info, show_warning, show_error

from qtpy.QtWidgets import QFileDialog
from PIL import Image, ImageDraw


# --------------------------------------------------------------------- helpers

def _get_image_layers() -> List[NapariImage]:
    v = current_viewer()
    if v is None:
        return []
    return [L for L in v.layers if isinstance(L, NapariImage)]


def _pick_layer_by_name(name: Optional[str]) -> Optional[NapariImage]:
    v = current_viewer()
    if v is None:
        return None

    imgs = [L for L in v.layers if isinstance(L, NapariImage)]
    if not imgs:
        return None

    if not name or name == "<active image>":
        lyr = v.layers.selection.active
        if isinstance(lyr, NapariImage):
            return lyr
        return imgs[0]

    for L in imgs:
        if L.name == name:
            return L

    lyr = v.layers.selection.active
    return lyr if isinstance(lyr, NapariImage) else imgs[0]


def _find_labels_for_image(image_layer: NapariImage) -> Optional[NapariLabels]:
    v = current_viewer()
    if v is None:
        return None

    labels_layers = [L for L in v.layers if isinstance(L, NapariLabels)]
    if not labels_layers:
        return None

    for L in labels_layers:
        try:
            if np.shape(L.data) == np.shape(image_layer.data):
                return L
        except Exception:
            pass

    return labels_layers[0]


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr[:] = 0
    return (arr * 255).clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------- classes
# 0 = background
# 1 = solid
# 2 = porous
# 3 = holes

CLASS_TO_LABEL: Dict[str, int] = {
    "Background": 0,
    "Solid": 1,
    "Porous": 2,
    "Holes": 3,
}

CLASS_COLORS = {
    "Background": "#000000",  # unused (transparent)
    "Solid": "#00ff00",
    "Porous": "#0000ff",
    "Holes": "#ff0000",
}

# --------------------------------------------------------------------- state

class BrushState:
    def __init__(self):
        self.brush_on = False
        self.prev_layer = None
        self.prev_mode: Optional[str] = None
        self.current_label: int = 1
        self.current_class: str = "Solid"


# --------------------------------------------------------------------- mask helpers

def _find_mask_layer() -> Optional[NapariLabels]:
    v = current_viewer()
    if v is None:
        return None
    for L in v.layers:
        if isinstance(L, NapariLabels) and L.name == "mask":
            return L
    return None


def _hex_to_rgba(hex_color: str) -> Tuple[float, float, float, float]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r, g, b, 1.0)


def _apply_mask_colourmap(lab: NapariLabels, label: int, class_name: str) -> None:
    cmap = dict(getattr(lab, "color", {}) or {})
# Always keep background transparent
    cmap[0] = (0, 0, 0, 0)

# Only add color if NOT background
    if label != 0:
        cmap[label] = _hex_to_rgba(CLASS_COLORS[class_name])

    lab.color = cmap
    lab.selected_label = label

    meta = dict(getattr(lab, "metadata", {}) or {})
    classes = dict(meta.get("classes", {}) or {})
    classes[str(label)] = {
        "class": class_name,
        "color": CLASS_COLORS[class_name],
    }
    meta["classes"] = classes
    lab.metadata = meta


def _ensure_mask_layer(img: Optional[NapariImage]) -> Optional[NapariLabels]:
    v = current_viewer()
    if v is None:
        return None

    lab = _find_mask_layer()
    if lab is not None:
        return lab

    if img is None:
        return None

    data = np.zeros_like(img.data, dtype=np.uint16)
    lab = v.add_labels(data, name="mask")
    lab.opacity = 0.7
    lab.blending = "translucent"
    return lab


# --------------------------------------------------------------------- export slice ZIP (logic unchanged)

def _export_slice_to_zip(layer: NapariImage) -> None:
    v = current_viewer()
    if v is None:
        show_warning("No viewer found.")
        return

    img_data = np.asarray(layer.data)
    labels_layer = _find_labels_for_image(layer)
    if labels_layer is None:
        show_warning("No matching Labels layer found.")
        return

    lbl_data = np.asarray(labels_layer.data)

    # -------------------- pick CURRENT slice --------------------
    if img_data.ndim == 3:
        try:
            slider_axes = [ax for ax in range(img_data.ndim)
                           if ax not in v.dims.displayed]
            z_axis = slider_axes[0] if slider_axes else 0
            z_index = int(v.dims.current_step[z_axis])
        except Exception:
            z_axis = 0
            z_index = 0

        z_index = max(0, min(z_index, img_data.shape[z_axis] - 1))

        slicer = [slice(None)] * img_data.ndim
        slicer[z_axis] = z_index

        img_slice = img_data[tuple(slicer)]
        mask_slice = lbl_data[tuple(slicer)]
    else:
        z_index = 0
        img_slice = img_data
        mask_slice = lbl_data

    # -------------------- file dialog --------------------
    zip_path, _ = QFileDialog.getSaveFileName(
        None,
        "Save slice ZIP",
        "",
        "Zip files (*.zip)",
    )
    if not zip_path:
        return
    if not zip_path.lower().endswith(".zip"):
        zip_path += ".zip"

    # -------------------- prepare data --------------------
    img_u8 = _normalize_to_uint8(img_slice)
    mask_ids = np.asarray(mask_slice, dtype=np.uint16)

    H, W = img_u8.shape
    img_rgb = np.stack([img_u8] * 3, axis=-1)
    mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    legend_dict = {}

    for lbl in np.unique(mask_ids):
        if lbl == 0:
            continue
        info = labels_layer.metadata.get("classes", {}).get(str(lbl), {})
        hex_color = info.get("color", "#ffffff").lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        mask_rgb[mask_ids == lbl] = (r, g, b)
        legend_dict[int(lbl)] = info

    # -------------------- overlay --------------------
    overlay_rgb = img_rgb.copy()
    m = mask_ids != 0
    overlay_rgb[m] = (
        0.6 * overlay_rgb[m].astype(np.float32)
        + 0.4 * mask_rgb[m].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    # -------------------- filenames --------------------
    base = f"z{z_index}"

    image_png = f"image_{base}.png"
    mask_png = f"mask_{base}.png"
    overlay_png = f"overlay_{base}.png"
    mask_npz = f"mask_{base}.npz"

    # -------------------- write temp files --------------------
    Image.fromarray(img_u8).save(image_png)
    Image.fromarray(mask_rgb).save(mask_png)
    Image.fromarray(overlay_rgb).save(overlay_png)
    np.savez_compressed(mask_npz, mask=mask_ids)

    # -------------------- zip --------------------
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(image_png)
            zf.write(mask_png)
            zf.write(overlay_png)
            zf.write(mask_npz)
            zf.writestr("legend.json", json.dumps(legend_dict, indent=2))
    finally:
        for f in (image_png, mask_png, overlay_png, mask_npz):
            try:
                os.remove(f)
            except Exception:
                pass

    msg = f"Slice z={z_index} exported.\nSaved to:\n{zip_path}"
    show_info(msg)

    # also write to console if available
    try:
        v = current_viewer()
        if v:
            pass
    except Exception:
        pass
    return zip_path



# --------------------------------------------------------------------- widget

def brush_widget() -> Container:
    brush_state = BrushState()

    header_label = Label(
        value="Brush: select class, adjust brush size, then paint.",
    )

    layer_combo = ComboBox(
        label="Layer",
        choices=["<active image>"],
        value="<active image>",
    )

    class_combo = ComboBox(
        label="Class",
        choices=list(CLASS_TO_LABEL.keys()),
        value="Solid",
    )

    brush_size_spin = SpinBox(
        value=5,
        min=1,
        max=100,
        step=1,
        label="Brush size",
    )

    btn_brush = PushButton(text="Brush")
    btn_export = PushButton(text="Export slice (ZIP)")

    console = TextEdit(value="") if TextEdit else Label(value="")

    def _append(msg: str):
        try:
            console.value += msg + "\n"
        except Exception:
            pass

    def _refresh_layers(*_):
        names = ["<active image>"] + [L.name for L in _get_image_layers()]
        layer_combo.choices = names

    def _update_label(*_):
        name = class_combo.value
        brush_state.current_label = CLASS_TO_LABEL[name]
        brush_state.current_class = name
        lab = _find_mask_layer()
        if lab:
            _apply_mask_colourmap(lab, brush_state.current_label, name)
            lab.brush_size = int(brush_size_spin.value)

    def _toggle_brush(*_):
        v = current_viewer()
        if v is None:
            return

        if not brush_state.brush_on:
            img = _pick_layer_by_name(layer_combo.value)
            lab = _ensure_mask_layer(img)
            if lab is None:
                return

            brush_state.prev_layer = v.layers.selection.active
            brush_state.prev_mode = getattr(brush_state.prev_layer, "mode", None)

            v.layers.selection.active = lab
            lab.mode = "paint"
            _update_label()

            brush_state.brush_on = True
            btn_brush.text = "Brush (on)"
            _append("Brush ON")
        else:
            brush_state.brush_on = False
            btn_brush.text = "Brush"
            if brush_state.prev_layer in v.layers:
                v.layers.selection.active = brush_state.prev_layer
                if brush_state.prev_mode:
                    brush_state.prev_layer.mode = brush_state.prev_mode
            _append("Brush OFF")

    btn_brush.changed.connect(_toggle_brush)
    def _run_export(*_):
        layer = _pick_layer_by_name(layer_combo.value)
        if layer is None:
            _append("Export failed: no image layer.")
            return

        _append(f"Exporting slice…")
        zip_path = _export_slice_to_zip(layer)
        if zip_path:
            _append(f"Saved to: {zip_path}")

    btn_export.changed.connect(_run_export)
    brush_size_spin.changed.connect(lambda *_: _update_label())
    class_combo.changed.connect(_update_label)

    v = current_viewer()
    if v:
        v.layers.events.inserted.connect(_refresh_layers)
        v.layers.events.removed.connect(_refresh_layers)

    _refresh_layers()
    _update_label()

    return Container(
        widgets=[
            header_label,
            layer_combo,
            class_combo,
            brush_size_spin,
            btn_brush,
            btn_export,
            console,
        ],
        layout="vertical",
        labels=False,
    )
