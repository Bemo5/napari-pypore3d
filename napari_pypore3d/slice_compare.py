# napari_pypore3d/slice_compare.py — SIMPLE ORTHOGONAL VIEWER (XY / XZ / YZ)
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QPushButton,
)
from napari.layers import Image as NapariImage, Points as NapariPoints
from napari import current_viewer
from napari.utils.notifications import show_info, show_warning

# ---------------------------------------------------------------------#
# Optional helpers from .helpers                                      #
# ---------------------------------------------------------------------#

try:
    from .helpers import last_zyx, active_image, iter_images, ensure_caption
except Exception:  # pragma: no cover

    def last_zyx(a: np.ndarray) -> Tuple[int, int, int]:
        a = np.asarray(a)
        if a.ndim >= 3:
            z, y, x = a.shape[-3], a.shape[-2], a.shape[-1]
            return int(z), int(y), int(x)
        if a.ndim == 2:
            y, x = a.shape[-2], a.shape[-1]
            return 1, int(y), int(x)
        return 1, 1, 1

    def active_image():
        v = current_viewer()
        if not v:
            return None, None
        L = v.layers.selection.active
        return (L, v) if isinstance(L, NapariImage) else (None, v)

    def iter_images(v):
        return [L for L in (v.layers if v else []) if isinstance(L, NapariImage)]

    def ensure_caption(_L, *_a, **_k):
        return


# ---------------- basic image helpers ----------------------------------------


def _images(v) -> List[NapariImage]:
    """Return true 3D volume images only (ignore 2D slice-compare layers)."""
    if not v:
        return []
    out: List[NapariImage] = []
    for L in v.layers:
        if not isinstance(L, NapariImage):
            continue
        md = getattr(L, "metadata", {}) or {}
        if md.get("_slice_compare", False):
            continue
        if np.asarray(L.data).ndim < 3:
            continue
        out.append(L)
    return out


def _safe_contrast(L: NapariImage):
    try:
        if hasattr(L, "reset_contrast_limits"):
            L.reset_contrast_limits()
        else:
            a = np.asarray(L.data, dtype=np.float32)
            if a.size == 0:
                return
            step = max(1, a.size // 256_000)
            flat = a.ravel()[::step]
            lo, hi = np.nanpercentile(flat, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
            L.contrast_limits = (float(lo), float(hi))
        L.visible = True
        L.opacity = 1.0
        try:
            L.colormap = "gray"
        except Exception:
            pass
    except Exception:
        pass


def _full_source(L: NapariImage) -> np.ndarray:
    """Get full 3D array; no boolean ops on arrays."""
    md = getattr(L, "metadata", {}) or {}
    full = md.get("_orig_full", None)
    if full is None:
        full = md.get("_orig_data", None)
    if full is None:
        full = L.data
    return np.asarray(full)


def _short_name(name: str) -> str:
    """
    Shorten parent layer name for slice layers.

    - 'SC1_700x700x700 [700×700×700 uint8]' → 'SC1_700x700x700'
    - If there's no ' [', returns the name as-is.
    """
    base = name.split(" [", 1)[0].strip()
    return base or name


def _toggle_center_points(v, parent_name: str, visible: bool):
    """Show / hide the '[center] ...' Points layer that belongs to parent_name."""
    if not v or not parent_name:
        return
    short = _short_name(parent_name)
    for lay in getattr(v, "layers", []):
        if isinstance(lay, NapariPoints):
            nm = (lay.name or "").strip()
            # Typical name: "[center] SC1_700x700x700 [700×700×700 uint8]"
            if nm.startswith("[center]") and short in nm:
                try:
                    lay.visible = visible
                except Exception:
                    pass


# ---------------- slicing helpers --------------------------------------------


def _slice_xy(a: np.ndarray, z_idx: int) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim < 3:
        return a
    z_axis = a.ndim - 3
    z_idx = int(np.clip(int(z_idx), 0, a.shape[z_axis] - 1))
    idx = [slice(None)] * a.ndim
    idx[z_axis] = z_idx
    sl = a[tuple(idx)]
    sl = np.squeeze(sl)
    if sl.ndim == 1:
        sl = sl[np.newaxis, :]
    return sl


def _slice_xz(a: np.ndarray, y_idx: int) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim < 3:
        return a
    z_axis = a.ndim - 3
    y_axis = a.ndim - 2
    x_axis = a.ndim - 1
    y_idx = int(np.clip(int(y_idx), 0, a.shape[y_axis] - 1))
    a_zyx = np.moveaxis(a, (z_axis, y_axis, x_axis), (-3, -2, -1))
    sl = a_zyx[..., :, y_idx, :]
    sl = np.squeeze(sl)
    if sl.ndim == 1:
        sl = sl[np.newaxis, :]
    return sl


def _slice_yz(a: np.ndarray, x_idx: int) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim < 3:
        return a
    z_axis = a.ndim - 3
    y_axis = a.ndim - 2
    x_axis = a.ndim - 1
    x_idx = int(np.clip(int(x_idx), 0, a.shape[x_axis] - 1))
    a_zyx = np.moveaxis(a, (z_axis, y_axis, x_axis), (-3, -2, -1))
    sl = a_zyx[..., :, :, x_idx]
    sl = np.squeeze(sl)
    if sl.ndim >= 2:
        sl = np.swapaxes(sl, -2, -1)
    if sl.ndim == 1:
        sl = sl[np.newaxis, :]
    return sl


# ---------------- controller --------------------------------------------------


@dataclass
class SliceCompareController:
    combo: QComboBox
    z_spin: QSpinBox
    y_spin: QSpinBox
    x_spin: QSpinBox
    make_btn: QPushButton
    clear_btn: QPushButton
    target_label: QLabel
    _busy: bool = False

    # layout backup (for restoring 3D lighting / grid)
    _layout_saved: bool = False
    _prev_ndisplay: int = 2
    _prev_grid_enabled: bool = False
    _prev_grid_stride: int = 1
    _prev_grid_shape: Tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------#
    # Target resolution                                                 #
    # ------------------------------------------------------------------#

    def _target_layer(self) -> Optional[NapariImage]:
        """Resolve the chosen 3D volume."""
        v = current_viewer()
        if not v:
            self._update_target_label(None)
            return None

        text = self.combo.currentText().strip()

        # 'Active (auto)' → active 3D non-slice image, else first 3D image
        if text == "" or text == "Active (auto)":
            L_active = v.layers.selection.active
            if isinstance(L_active, NapariImage):
                md = getattr(L_active, "metadata", {}) or {}
                if not md.get("_slice_compare", False) and np.asarray(
                    L_active.data
                ).ndim >= 3:
                    self._update_target_label(L_active)
                    return L_active
            imgs = _images(v)
            L = imgs[0] if imgs else None
            self._update_target_label(L)
            return L

        # named image
        for L in _images(v):
            if L.name == text:
                self._update_target_label(L)
                return L

        imgs = _images(v)
        L = imgs[0] if imgs else None
        self._update_target_label(L)
        return L

    def _update_target_label(self, L: Optional[NapariImage]):
        """Update 'Current target: ...' label."""
        if L is None:
            self.target_label.setText("Current target: (none)")
        else:
            self.target_label.setText(f"Current target: {_short_name(L.name)}")

    # ------------------------------------------------------------------#

    def _sync_limits(self):
        """Update spinbox ranges; always allow manual input."""
        L = self._target_layer()
        if not L:
            for sb in (self.z_spin, self.y_spin, self.x_spin):
                sb.setRange(0, 2_000_000)
            return
        a = _full_source(L)
        if a.ndim < 3:
            for sb in (self.z_spin, self.y_spin, self.x_spin):
                sb.setRange(0, 2_000_000)
            return
        z, y, x = last_zyx(a)
        sizes = [max(1, int(z)), max(1, int(y)), max(1, int(x))]
        for sb, size in zip((self.z_spin, self.y_spin, self.x_spin), sizes):
            sb.setRange(0, size - 1)
            # clamp current value so if user spammed 412124412 it becomes volume_max
            cur = min(max(int(sb.value()), 0), size - 1)
            sb.setValue(cur)

    def on_layers_changed(self):
        if self._busy:
            return
        v = current_viewer()
        names = [L.name for L in _images(v)] if v else []
        cur = self.combo.currentText() or "Active (auto)"
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("Active (auto)")
        for n in names:
            self.combo.addItem(n)
        idx = self.combo.findText(cur)
        self.combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo.blockSignals(False)
        self._sync_limits()

    # ------------------------------------------------------------------#
    # Layout save/restore (kept inside controller, not on Viewer)       #
    # ------------------------------------------------------------------#

    def _save_viewer_layout(self, v):
        if self._layout_saved:
            return
        try:
            self._prev_ndisplay = int(getattr(v.dims, "ndisplay", 2))
            self._prev_grid_enabled = bool(getattr(v.grid, "enabled", False))
            self._prev_grid_stride = int(getattr(v.grid, "stride", 1))
            shape = getattr(v.grid, "shape", (0, 0))
            self._prev_grid_shape = tuple(shape) if shape is not None else (0, 0)
            self._layout_saved = True
        except Exception:
            self._layout_saved = False

    def _restore_viewer_layout(self, v):
        if not self._layout_saved:
            return
        try:
            v.dims.ndisplay = self._prev_ndisplay
            v.grid.enabled = self._prev_grid_enabled
            v.grid.stride = self._prev_grid_stride
            if self._prev_grid_shape != (0, 0):
                v.grid.shape = self._prev_grid_shape
        except Exception:
            pass
        self._layout_saved = False

    # ------------------------------------------------------------------#
    # Highlight active XY/XZ/YZ by opacity                              #
    # ------------------------------------------------------------------#

    def _update_highlight(self):
        v = current_viewer()
        if not v:
            return
        active = v.layers.selection.active
        if not isinstance(active, NapariImage):
            return
        md_active = getattr(active, "metadata", {}) or {}
        if not md_active.get("_slice_compare", False):
            return
        parent_name = md_active.get("_slice_parent")
        if not parent_name:
            return

        for lay in v.layers:
            if not isinstance(lay, NapariImage):
                continue
            md = getattr(lay, "metadata", {}) or {}
            if not (md.get("_slice_compare", False) and md.get("_slice_parent") == parent_name):
                continue
            try:
                lay.opacity = 1.0 if lay is active else 0.6
            except Exception:
                pass

    # ------------------------------------------------------------------#

    def _ensure_plane(self, L: NapariImage, plane: str, index: int) -> Optional[NapariImage]:
        v = current_viewer()
        if not v:
            return None
        a = _full_source(L)
        if a.ndim < 3:
            show_warning(f"Image '{L.name}' is not 3D.")
            return None

        if plane == "XY":
            data2d = _slice_xy(a, index)
        elif plane == "XZ":
            data2d = _slice_xz(a, index)
        elif plane == "YZ":
            data2d = _slice_yz(a, index)
        else:
            return None

        # Search all images (including existing slice_compare) for this plane
        existing: Optional[NapariImage] = None
        v_all_imgs = [L2 for L2 in v.layers if isinstance(L2, NapariImage)]
        for lay in v_all_imgs:
            md = getattr(lay, "metadata", {}) or {}
            if (
                md.get("_slice_compare", False)
                and md.get("_slice_parent") == L.name
                and md.get("_slice_plane") == plane
            ):
                existing = lay
                break

        parent_label = _short_name(L.name)
        layer_name = f"{parent_label} [{plane} @ {int(index)}]"
        meta = {
            "_slice_compare": True,
            "_slice_parent": L.name,
            "_slice_plane": plane,
            "_slice_index": int(index),
        }

        if existing is None:
            child = v.add_image(data2d, name=layer_name, metadata=meta)
        else:
            existing.data = data2d
            existing.name = layer_name
            existing.metadata.update(meta)
            child = existing

        _safe_contrast(child)
        try:
            ensure_caption(child)
        except Exception:
            pass
        return child

    def make_views(self):
        if self._busy:
            return
        self._busy = True
        try:
            self._sync_limits()
            L = self._target_layer()
            if not L:
                show_warning("Pick a 3D image first.")
                return
            a = _full_source(L)
            if a.ndim < 3:
                show_warning(f"Image '{L.name}' is not 3D.")
                return

            v = current_viewer()
            if not v:
                return

            # Save current layout so 3D lighting etc. can be restored later
            self._save_viewer_layout(v)

            z, y, x = last_zyx(a)
            z = max(1, int(z))
            y = max(1, int(y))
            x = max(1, int(x))

            # clamp typed values to [0, size-1] even if user spammed 412124412
            zi = int(np.clip(int(self.z_spin.value()), 0, z - 1))
            yi = int(np.clip(int(self.y_spin.value()), 0, y - 1))
            xi = int(np.clip(int(self.x_spin.value()), 0, x - 1))
            self.z_spin.setValue(zi)
            self.y_spin.setValue(yi)
            self.x_spin.setValue(xi)

            # create / update the three slice planes
            lay_xy = self._ensure_plane(L, "XY", zi)
            lay_xz = self._ensure_plane(L, "XZ", yi)
            lay_yz = self._ensure_plane(L, "YZ", xi)

            if lay_xy or lay_xz or lay_yz:
                # hide the big 3D parent so only the 3 slices are visible
                try:
                    L.visible = False
                except Exception:
                    pass

                _toggle_center_points(v, L.name, False)

                # Force 2D tiled view: 2x2 grid, stride=1 → no overlap
                try:
                    v.dims.ndisplay = 2
                    v.grid.enabled = True
                    v.grid.stride = 1
                    v.grid.shape = (2, 2)
                except Exception:
                    pass

                # Select the three planes
                try:
                    sel = v.layers.selection
                    sel.clear()
                    for lay in (lay_xy, lay_xz, lay_yz):
                        if lay is not None:
                            sel.add(lay)
                    if lay_xy is not None:
                        v.layers.selection.active = lay_xy
                except Exception:
                    pass

                try:
                    v.reset_view()
                except Exception:
                    pass

                # apply highlight (opacity) after selection
                self._update_highlight()

            show_info(
                f"Orthogonal views for '{L.name}' "
                f"(XY z={zi}, XZ y={yi}, YZ x={xi}) updated."
            )
        finally:
            self._busy = False

    def clear_views(self):
        if self._busy:
            return
        self._busy = True
        try:
            v = current_viewer()
            if not v:
                return
            L = self._target_layer()
            if not L:
                return

            # remove all orthogonal slice layers for this parent
            to_remove: List[NapariImage] = []
            for lay in [L2 for L2 in v.layers if isinstance(L2, NapariImage)]:
                md = getattr(lay, "metadata", {}) or {}
                if md.get("_slice_compare", False) and md.get("_slice_parent") == L.name:
                    to_remove.append(lay)
            for lay in to_remove:
                try:
                    v.layers.remove(lay)
                except Exception:
                    pass

            # show parent 3D image again
            try:
                L.visible = True
                _safe_contrast(L)
            except Exception:
                pass

            _toggle_center_points(v, L.name, True)

            # restore viewer layout (ndisplay, grid) so 3D lighting works again
            self._restore_viewer_layout(v)

            if to_remove:
                show_info(f"Removed {len(to_remove)} orthogonal views for '{L.name}'.")
        finally:
            self._busy = False


# ---------------- panel factory ----------------------------------------------


def make_slice_compare_panel() -> tuple[SliceCompareController, QWidget]:
    panel = QWidget()
    vbox = QVBoxLayout(panel)
    vbox.setContentsMargins(14, 14, 14, 14)
    vbox.setSpacing(10)

    title = QLabel("Orthogonal slice viewer (XY / XZ / YZ)")
    title.setStyleSheet("font-weight:600;")
    vbox.addWidget(title)

    # Target image row
    row_t = QHBoxLayout()
    row_t.addWidget(QLabel("Target image:"))
    combo = QComboBox()
    combo.addItem("Active (auto)")
    combo.setMinimumWidth(200)
    row_t.addWidget(combo)
    row_t.addStretch(1)
    vbox.addLayout(row_t)

    # Current target label
    target_label = QLabel("Current target: (none)")
    target_label.setStyleSheet("color:#aaaaaa; font-size:11px;")
    vbox.addWidget(target_label)

    def _row(label: str, spin: QSpinBox) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(spin)
        row.addStretch(1)
        return row

    z_spin = QSpinBox()
    y_spin = QSpinBox()
    x_spin = QSpinBox()
    # Allow manual input even before we know Z/Y/X
    for sb in (z_spin, y_spin, x_spin):
        sb.setRange(0, 2_000_000)

    vbox.addLayout(_row("Z (XY plane):", z_spin))
    vbox.addLayout(_row("Y (for XZ):", y_spin))
    vbox.addLayout(_row("X (for YZ):", x_spin))

    make_btn = QPushButton("Make / update orthogonal views")
    clear_btn = QPushButton("Clear orthogonal views")
    for b in (make_btn, clear_btn):
        b.setMinimumHeight(32)
    vbox.addWidget(make_btn)
    vbox.addWidget(clear_btn)
    vbox.addStretch(1)

    ctrl = SliceCompareController(
        combo=combo,
        z_spin=z_spin,
        y_spin=y_spin,
        x_spin=x_spin,
        make_btn=make_btn,
        clear_btn=clear_btn,
        target_label=target_label,
    )

    make_btn.clicked.connect(ctrl.make_views)
    clear_btn.clicked.connect(ctrl.clear_views)

    def _on_combo_changed(_text: str):
        if ctrl._busy:
            return
        v = current_viewer()
        if not v:
            return
        name = combo.currentText().strip()
        if name and name != "Active (auto)":
            for L in _images(v):
                if L.name == name:
                    try:
                        v.layers.selection.active = L
                    except Exception:
                        pass
                    break
        ctrl._sync_limits()

    combo.currentTextChanged.connect(_on_combo_changed)

    try:
        v = current_viewer()
        if v:

            def _struct_change(*_args, **_kwargs):
                ctrl.on_layers_changed()

            def _data_change(*_args, **_kwargs):
                ctrl._sync_limits()

            def _active_change(*_args, **_kwargs):
                ctrl._update_highlight()
                ctrl._sync_limits()

            v.layers.events.inserted.connect(_struct_change)
            v.layers.events.removed.connect(_struct_change)
            v.layers.events.reordered.connect(_struct_change)
            v.layers.selection.events.active.connect(_struct_change)
            v.layers.selection.events.active.connect(_active_change)
            v.layers.events.changed.connect(_data_change)
    except Exception:
        pass

    ctrl.on_layers_changed()
    return ctrl, panel
