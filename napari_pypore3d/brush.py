# napari_pypore3d/brush.py — Brush widget for mask painting
from __future__ import annotations

import os
import json
import zipfile
import datetime
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
from magicgui.widgets import (
    Container, ComboBox, PushButton, Label, SpinBox, CheckBox
)

try:
    from magicgui.widgets import TextEdit
except Exception:
    TextEdit = None

from napari import current_viewer
from napari.layers import Image as NapariImage, Labels as NapariLabels
from napari.utils.notifications import show_info, show_warning, show_error

from qtpy.QtWidgets import QWidget, QVBoxLayout, QFrame, QFileDialog
from PIL import Image, ImageDraw


# --------------------------------------------------------------------- helpers

def _get_image_layers() -> List[NapariImage]:
    """Return all Image layers in the current viewer."""
    v = current_viewer()
    if v is None:
        return []
    return [L for L in v.layers if isinstance(L, NapariImage)]


def _pick_layer_by_name(name: Optional[str]) -> Optional[NapariImage]:
    """
    Return layer by name, or:
      - if name is '<active image>' → active Image,
      - if no active Image → first Image in viewer.
    """
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
    if isinstance(lyr, NapariImage):
        return lyr
    return imgs[0]


def _find_labels_for_image(image_layer: NapariImage) -> Optional[NapariLabels]:
    """
    Try to find a Labels layer that matches the given Image layer.

    Strategy:
      1) Look for Labels layers with identical data.shape.
      2) If none, return the first Labels layer (if any).
    """
    v = current_viewer()
    if v is None:
        return None

    labels_layers = [L for L in v.layers if isinstance(L, NapariLabels)]
    if not labels_layers:
        return None

    # Prefer exact shape match
    for L in labels_layers:
        try:
            if np.shape(L.data) == np.shape(image_layer.data):
                return L
        except Exception:
            continue

    # Fallback: first labels layer
    return labels_layers[0]


def _normalize_to_uint8(slice_arr: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to uint8 [0, 255] for export."""
    arr = np.asarray(slice_arr, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


# label indices per class
CLASS_TO_LABEL: Dict[str, int] = {
    "Air": 1,
    "Bone": 2,
    "Pores": 3,
    "Other": 4,
}

# Color palette for classes (without color picker)
CLASS_COLORS = {
    "Air": "#ff0000",      # Red
    "Bone": "#00ff00",     # Green
    "Pores": "#0000ff",    # Blue
    "Other": "#ffff00",    # Yellow
}


def _find_mask_layer() -> Optional[NapariLabels]:
    """Find the 'mask' labels layer in the current viewer."""
    v = current_viewer()
    if v is None:
        return None
    for L in v.layers:
        if isinstance(L, NapariLabels) and L.name == "mask":
            return L
    return None


def _rgba_to_hex(rgba: Tuple[float, float, float, float]) -> str:
    """Convert RGBA tuple (0-1 range) to hex color string."""
    r = int(np.clip(rgba[0] * 255, 0, 255))
    g = int(np.clip(rgba[1] * 255, 0, 255))
    b = int(np.clip(rgba[2] * 255, 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgba(hex_color: str) -> Tuple[float, float, float, float]:
    """Convert hex color string to RGBA tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b, 1.0)
    return (1.0, 0.4, 0.0, 1.0)  # default orange


# --------------------------------------------------------------------- brush state

class BrushState:
    """Manage brush mode state (on/off, previous layer, etc)."""
    
    def __init__(self):
        self.brush_on = False
        self.prev_layer = None
        self.prev_mode: Optional[str] = None
        self.current_label: int = 1
        self.current_class: str = "Air"


# --------------------------------------------------------------------- mask helpers

def _apply_mask_colourmap(
    lab: NapariLabels,
    label: int,
    class_name: str,
) -> None:
    """Apply colormap to mask layer with transparent background."""
    hex_color = CLASS_COLORS.get(class_name, "#ff0000")
    rgba = _hex_to_rgba(hex_color)

    try:
        cmap = dict(getattr(lab, "color", {}) or {})
    except Exception:
        cmap = {}
    cmap[0] = (0.0, 0.0, 0.0, 0.0)
    cmap[label] = rgba
    try:
        lab.color = cmap
    except Exception:
        pass

    try:
        lab.selected_label = label
    except Exception:
        pass

    try:
        meta = dict(getattr(lab, "metadata", {}) or {})
        classes = dict(meta.get("classes", {}) or {})
        classes[str(label)] = {
            "class": class_name,
            "color": hex_color,
        }
        meta["classes"] = classes
        lab.metadata = meta
    except Exception:
        pass


def _ensure_mask_layer(img: Optional[NapariImage]) -> Optional[NapariLabels]:
    """Return 'mask' labels layer, create if missing."""
    v = current_viewer()
    if v is None:
        show_warning("No viewer available for mask.")
        return None

    lab = _find_mask_layer()
    if lab is None:
        if img is None:
            show_warning("Need an image to size the mask.")
            return None

        try:
            data = np.zeros_like(img.data, dtype=np.uint16)
        except Exception:
            show_warning("Could not create mask layer (bad image data).")
            return None

        lab = v.add_labels(data, name="mask")
        try:
            lab.opacity = 0.7
            lab.blending = "translucent"
        except Exception:
            pass

    return lab


# --------------------------------------------------------------------- SAM export to ZIP

def _export_sam_to_zip(layer: NapariImage) -> None:
    """
    Export the *current* slice in SAM-style to a ZIP file:
    
    ZIP contents:
      image.png        -> current CT slice (no mask, grayscale)
      mask.png         -> colored mask for that slice
      overlay.png      -> CT + mask overlay
      mask_legend.png  -> legend image
      legend.json      -> label -> colour mapping
      info.txt         -> basic information about the export

    - Uses the given image layer as CT source.
    - Finds a matching Labels layer.
    - Uses the *current viewer slice*.
    """
    v = current_viewer()
    if v is None:
        show_warning("No viewer found.")
        return

    img_data = np.asarray(layer.data)
    if img_data.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {img_data.shape}")

    labels_layer = _find_labels_for_image(layer)
    if labels_layer is None:
        show_warning("No matching Labels layer found for this image.")
        return

    lbl_data = np.asarray(labels_layer.data)
    if lbl_data.shape != img_data.shape:
        raise ValueError(
            f"Image and labels shapes differ: {img_data.shape} vs {lbl_data.shape}"
        )

    # -------------------- pick CURRENT viewer slice --------------------
    if img_data.ndim == 3:
        try:
            slider_axes = [ax for ax in range(img_data.ndim)
                           if ax not in v.dims.displayed]
            if slider_axes:
                z_axis = slider_axes[0]
            else:
                z_axis = 0
            z_index = int(v.dims.current_step[z_axis])
        except Exception:
            z_axis = 0
            z_index = 0

        z_index = max(0, min(z_index, img_data.shape[z_axis] - 1))

        slicer = [slice(None)] * img_data.ndim
        slicer[z_axis] = z_index
        img_slice = img_data[tuple(slicer)]
        mask_slice = lbl_data[tuple(slicer)]

        slice_info = f"Slice z={z_index} (axis={z_axis})"
    else:
        img_slice = img_data
        mask_slice = lbl_data
        slice_info = "2D image"

    # -------------------- ask for ZIP file path -------------------------
    start_dir = ""
    try:
        start_dir = getattr(getattr(v, "window", None), "workspace_dir", "") or ""
    except Exception:
        pass

    zip_path, _ = QFileDialog.getSaveFileName(
        None,
        "Save SAM slice as ZIP",
        start_dir,
        "Zip files (*.zip)",
    )
    if not zip_path:
        show_info("SAM export cancelled.")
        return
    
    if not zip_path.lower().endswith(".zip"):
        zip_path = zip_path + ".zip"

    base_no_ext = os.path.splitext(zip_path)[0]
    
    # Temporary files
    temp_files = []
    
    # -------------------- prepare data ---------------------------------
    img_u8 = _normalize_to_uint8(img_slice)
    mask_ids = np.asarray(mask_slice, dtype=np.int32)

    H, W = img_u8.shape
    img_rgb = np.stack([img_u8] * 3, axis=-1)

    # Labels > 0
    labels = np.unique(mask_ids)
    labels = labels[labels != 0]

    # Use class colors from palette
    legend_dict = {}
    mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for idx, lbl in enumerate(labels):
        # Find class name for this label from metadata
        class_name = f"class_{int(lbl)}"
        hex_color = "#ff0000"  # default red
        
        # Try to get class info from labels layer metadata
        try:
            if labels_layer.metadata and "classes" in labels_layer.metadata:
                class_info = labels_layer.metadata["classes"].get(str(lbl), {})
                class_name = class_info.get("class", class_name)
                hex_color = class_info.get("color", hex_color)
        except Exception:
            pass
        
        # Convert hex to RGB
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            mask_rgb[mask_ids == lbl] = (r, g, b)
        else:
            # Fallback to palette based on index
            palette_idx = idx % 4
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            r, g, b = colors[palette_idx]
            mask_rgb[mask_ids == lbl] = (r, g, b)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
        
        legend_dict[int(lbl)] = {
            "class": class_name,
            "color": f"#{hex_color}" if not hex_color.startswith("#") else hex_color,
        }

    # -------------------- overlay image --------------------------------
    overlay_rgb = img_rgb.copy()
    mask_any = mask_ids != 0
    if np.any(mask_any):
        a_img = 0.6
        a_msk = 0.4
        overlay_rgb[mask_any] = (
            a_img * overlay_rgb[mask_any].astype(np.float32)
            + a_msk * mask_rgb[mask_any].astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

    # -------------------- legend image ---------------------------------
    if len(legend_dict) > 0:
        row_h = 24
        width = 180
        height = 8 + row_h * len(legend_dict)
        legend_img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(legend_img)

        for i, (lbl, info) in enumerate(legend_dict.items()):
            y0 = 4 + i * row_h
            y1 = y0 + 18
            col_hex = info["color"]
            rr = int(col_hex[1:3], 16)
            gg = int(col_hex[3:5], 16)
            bb = int(col_hex[5:7], 16)
            draw.rectangle([8, y0, 32, y1], fill=(rr, gg, bb))
            draw.text((40, y0 + 2), f"label {lbl}: {info['class']}", fill=(255, 255, 255))
    else:
        legend_img = Image.new("RGB", (120, 32), (0, 0, 0))
        draw = ImageDraw.Draw(legend_img)
        draw.text((8, 8), "no labels", fill=(255, 255, 255))

    # -------------------- write temporary files ----------------------------------
    image_path = base_no_ext + "_image.png"
    mask_path = base_no_ext + "_mask.png"
    overlay_path = base_no_ext + "_overlay.png"
    legend_png_path = base_no_ext + "_mask_legend.png"
    legend_json_path = base_no_ext + "_legend.json"
    info_txt_path = base_no_ext + "_info.txt"
    
    temp_files = [image_path, mask_path, overlay_path, legend_png_path, legend_json_path, info_txt_path]

    Image.fromarray(img_u8).save(image_path)
    Image.fromarray(mask_rgb).save(mask_path)
    Image.fromarray(overlay_rgb).save(overlay_path)
    legend_img.save(legend_png_path)

    try:
        with open(legend_json_path, "w", encoding="utf-8") as f:
            json.dump(legend_dict, f, indent=2)
    except Exception as e:
        show_warning(f"Could not save legend.json: {e!r}")

    # Create info.txt
    try:
        with open(info_txt_path, "w", encoding="utf-8") as f:
            f.write(f"SAM Slice Export Information\n")
            f.write(f"============================\n\n")
            f.write(f"Source Image: {layer.name}\n")
            f.write(f"Image Shape: {img_data.shape}\n")
            f.write(f"{slice_info}\n")
            f.write(f"Export Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"ZIP Contents:\n")
            f.write(f"- image.png: Grayscale CT slice\n")
            f.write(f"- mask.png: Colored mask (labels > 0)\n")
            f.write(f"- overlay.png: CT + mask overlay\n")
            f.write(f"- mask_legend.png: Color legend\n")
            f.write(f"- legend.json: Label to color/class mapping\n")
            f.write(f"- info.txt: This file\n\n")
            f.write(f"Labels found: {len(labels)}\n")
            for lbl, info in legend_dict.items():
                f.write(f"  Label {lbl}: {info['class']} ({info['color']})\n")
    except Exception as e:
        show_warning(f"Could not save info.txt: {e!r}")

    # -------------------- pack everything into ZIP ---------------------
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Add all files to zip
            for file_path, arcname in [
                (image_path, "image.png"),
                (mask_path, "mask.png"),
                (overlay_path, "overlay.png"),
                (legend_png_path, "mask_legend.png"),
                (legend_json_path, "legend.json"),
                (info_txt_path, "info.txt"),
            ]:
                if os.path.exists(file_path):
                    zf.write(file_path, arcname=arcname)
        
        show_info(
            f"SAM slice exported to ZIP:\n{zip_path}\n\n"
            f"Contains: image.png, mask.png, overlay.png, mask_legend.png, legend.json, info.txt"
        )
        
    except Exception as e:
        show_error(f"Creating ZIP failed: {e!r}")
        return
    finally:
        # Clean up temporary files
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass


# --------------------------------------------------------------------- widget

def brush_widget() -> Container:
    """
    Brush / Labels page with multi-class masking:

    - Layer dropdown for image selection (auto-refreshes when layers change)
    - Class selector: Air / Bone / Pores / Other (mapped to labels 1–4)
    - Brush size control (updates immediately when changed)
    - Brush toggle (switches to paint mode)
    - Export SAM slice to ZIP (all files in one archive)
    """
    brush_state = BrushState()

    header_label = Label(
        label="",
        value="Brush / Labels: select image, pick class, adjust brush size, then paint.",
    )

    layer_combo = ComboBox(
        label="Layer",
        choices=["<active image>"],
        value="<active image>",
    )

    class_combo = ComboBox(
        label="Class",
        choices=list(CLASS_TO_LABEL.keys()),
        value="Air",
    )

    brush_size_spin = SpinBox(
        value=5,
        min=1,
        max=100,
        step=1,
        label="Brush size",
    )

    btn_brush = PushButton(text="Brush")
    btn_export_sam = PushButton(text="Export SAM slice (ZIP)")

    console_label = Label(label="", value="Output console:")

    if TextEdit is not None:
        console = TextEdit(label="", value="")
        try:
            console.native.setReadOnly(True)
            console.native.setMinimumHeight(160)
        except Exception:
            pass
    else:
        console = Label(label="", value="")

    def _append_console(msg: str) -> None:
        val = getattr(console, "value", "")
        new_val = (val + "\n" + msg) if val else msg
        try:
            console.value = new_val
        except Exception:
            pass

    def _refresh_layers(*_):
        """Update the layer dropdown with current image layers in viewer."""
        layers = _get_image_layers()
        names = ["<active image>"] + [L.name for L in layers]
        current = layer_combo.value if layer_combo.value in names else "<active image>"
        layer_combo.choices = names
        layer_combo.value = current

    def _update_current_label(*_):
        """Update the current label when class changes."""
        name = str(class_combo.value or "Air")
        brush_state.current_label = CLASS_TO_LABEL.get(name, 1)
        brush_state.current_class = name
        
        lab = _find_mask_layer()
        if lab is not None:
            _apply_mask_colourmap(lab, brush_state.current_label, name)
            # Update brush size when class changes
            _update_brush_size()

    def _update_brush_size(*_):
        """Update brush size on the mask layer."""
        lab = _find_mask_layer()
        if lab is not None:
            try:
                brush_size_val = int(brush_size_spin.value or 5)
                lab.brush_size = brush_size_val
            except Exception:
                pass

    def _toggle_brush(*_):
        """Toggle brush mode on/off."""
        v = current_viewer()
        if v is None:
            show_warning("No viewer available.")
            return

        # turn ON
        if not brush_state.brush_on:
            img = _pick_layer_by_name(layer_combo.value)
            lab = _ensure_mask_layer(img)
            if lab is None:
                return

            try:
                active = v.layers.selection.active
            except Exception:
                active = None

            brush_state.prev_layer = active
            brush_state.prev_mode = getattr(active, "mode", None) if active else None

            try:
                v.layers.selection.active = lab
                lab.mode = "paint"
                class_name = str(class_combo.value or "Air")
                _apply_mask_colourmap(lab, brush_state.current_label, class_name)
                # Set initial brush size
                _update_brush_size()
            except Exception:
                show_warning("Could not switch to brush mode.")
                return

            brush_state.brush_on = True
            btn_brush.text = "Brush (on)"
            _append_console("→ Brush ON - paint on 'mask' layer")
            show_info("Brush ON — paint in the viewer, click again to turn off.")
            return

        # turn OFF
        brush_state.brush_on = False
        btn_brush.text = "Brush"

        prev_layer = brush_state.prev_layer
        prev_mode = brush_state.prev_mode

        try:
            layers = list(v.layers)
        except Exception:
            layers = []

        if prev_layer is not None and prev_layer in layers:
            try:
                v.layers.selection.active = prev_layer
                if prev_mode is not None and hasattr(prev_layer, "mode"):
                    prev_layer.mode = prev_mode
                _append_console("← Brush OFF - returned to previous layer")
                show_info("Brush OFF.")
                return
            except Exception:
                pass

        try:
            active = v.layers.selection.active
            if hasattr(active, "mode"):
                active.mode = "pan_zoom"
        except Exception:
            pass

        _append_console("← Brush OFF")
        show_info("Brush OFF.")

    def _run_export_sam(*_):
        """Export current slice to SAM-compatible ZIP."""
        # Auto-refresh layers before export
        _refresh_layers()
        name = layer_combo.value
        layer = _pick_layer_by_name(name)
        if layer is None:
            msg = "No target Image layer."
            show_warning(msg)
            _append_console(f"[warn] {msg}")
            return

        _append_console(f"→ Exporting SAM slice for '{layer.name}' to ZIP...")
        try:
            _export_sam_to_zip(layer)
            _append_console("[ok] SAM export finished (ZIP created).")
        except Exception as e:
            msg = f"SAM export failed: {e!r}"
            show_error(msg)
            _append_console(f"[error] {msg}")

    # Connect signals
    btn_brush.changed.connect(_toggle_brush)
    btn_export_sam.changed.connect(_run_export_sam)
    
    # Connect brush size updates
    brush_size_spin.changed.connect(_update_brush_size)
    
    # Connect class changes
    class_combo.changed.connect(_update_current_label)
    
    # Auto-refresh layers when viewer layers change
    v = current_viewer()
    if v is not None:
        try:
            # Auto-refresh when layers are added/removed
            v.layers.events.inserted.connect(_refresh_layers)
            v.layers.events.removed.connect(_refresh_layers)
            # Auto-refresh when active layer changes
            v.layers.selection.events.active.connect(_refresh_layers)
        except Exception:
            pass

    # Initial setup
    _refresh_layers()
    _update_current_label()

    # Create main container
    root = Container(
        widgets=[
            header_label,
            layer_combo,
            class_combo,
            brush_size_spin,
            btn_brush,
            btn_export_sam,
            console_label,
        ],
        layout="vertical",
        labels=False,
    )

    try:
        root.append(console)
    except Exception:
        pass

    try:
        lay = root.native.layout()
        if lay is not None:
            lay.setSpacing(6)
            lay.setContentsMargins(8, 8, 8, 8)
    except Exception:
        pass

    # Set minimum widths for better layout
    def _minw(w, px: int) -> None:
        try:
            w.native.setMinimumWidth(px)
        except Exception:
            pass

    for w in (layer_combo, class_combo, brush_size_spin,
              btn_brush, btn_export_sam):
        _minw(w, 180)

    return root