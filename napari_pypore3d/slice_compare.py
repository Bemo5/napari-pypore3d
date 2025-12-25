# napari_pypore3d/slice_compare.py — SIMPLE ORTHOGONAL VIEWER (XY / XZ / YZ)
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QPushButton,
    QFormLayout,
    QFrame,
    QGroupBox,
)
from napari.layers import Image as NapariImage, Points as NapariPoints
from napari import current_viewer
from napari.utils.notifications import show_info, show_warning

# ---------------------------------------------------------------------#
# Optional helpers from .helpers                                        #
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

# ---------------------------------------------------------------------#
# Session Recorder integration (optional)                               #
# ---------------------------------------------------------------------#
try:
    from .session_recorder import get_recorder, register_handler, Step
except Exception:
    get_recorder = None
    register_handler = None
    Step = None

_REC = get_recorder() if callable(get_recorder) else None


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


def _safe_contrast(
    L: NapariImage,
    parent: Optional[NapariImage] = None,
    *,
    keep_opacity: bool = False,
):
    """
    Apply 'safe' contrast limits.

    - If a parent is given, copy its contrast + colormap.
    - Otherwise, approximate robust contrast from the layer's own data.
    """
    try:
        if parent is not None:
            try:
                if hasattr(parent, "contrast_limits"):
                    L.contrast_limits = tuple(parent.contrast_limits)
            except Exception:
                pass
            try:
                L.colormap = getattr(parent, "colormap", "gray")
            except Exception:
                pass
        else:
            if hasattr(L, "reset_contrast_limits"):
                L.reset_contrast_limits()
            else:
                a = np.asarray(L.data, dtype=np.float32)
                if a.size > 0:
                    step = max(1, a.size // 256_000)
                    flat = a.ravel()[::step]
                    lo, hi = np.nanpercentile(flat, [0.5, 99.5])
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        lo = float(np.nanmin(flat))
                        hi = float(np.nanmax(flat))
                    if hi == lo:
                        hi = lo + 1.0
                    L.contrast_limits = (float(lo), float(hi))

        L.visible = True
        if not keep_opacity:
            L.opacity = getattr(L, "opacity", 1.0)
        try:
            if not hasattr(L, "colormap"):
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


# ---------------- core apply/clear (shared by UI + replay) --------------------
def _ensure_plane(v, parent: NapariImage, plane: str, index: int) -> Optional[NapariImage]:
    a = _full_source(parent)
    if a.ndim < 3:
        show_warning(f"Image '{parent.name}' is not 3D.")
        return None

    if plane == "XY":
        data2d = _slice_xy(a, index)
    elif plane == "XZ":
        data2d = _slice_xz(a, index)
    elif plane == "YZ":
        data2d = _slice_yz(a, index)
    else:
        return None

    existing: Optional[NapariImage] = None
    for lay in [L2 for L2 in v.layers if isinstance(L2, NapariImage)]:
        md = getattr(lay, "metadata", {}) or {}
        if (
            md.get("_slice_compare", False)
            and md.get("_slice_parent") == parent.name
            and md.get("_slice_plane") == plane
        ):
            existing = lay
            break

    parent_label = _short_name(parent.name)
    layer_name = f"{parent_label} [{plane} @ {int(index)}]"
    meta = {
        "_slice_compare": True,
        "_slice_parent": parent.name,
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

    _safe_contrast(child, parent=parent)

    try:
        ensure_caption(child)
    except Exception:
        pass

    return child


def _apply_slice_compare_views(
    v,
    parent: NapariImage,
    zi: int,
    yi: int,
    xi: int,
    *,
    saved_layout: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Apply/update XY/XZ/YZ layers and force 2D tiled grid.
    Returns saved layout if we captured it.
    """
    a = _full_source(parent)
    if a.ndim < 3:
        show_warning(f"Image '{parent.name}' is not 3D.")
        return saved_layout

    if saved_layout is None:
        try:
            saved_layout = {
                "ndisplay": int(getattr(v.dims, "ndisplay", 2)),
                "grid_enabled": bool(getattr(v.grid, "enabled", False)),
                "grid_stride": int(getattr(v.grid, "stride", 1)),
                "grid_shape": tuple(getattr(v.grid, "shape", (0, 0)) or (0, 0)),
            }
        except Exception:
            saved_layout = None

    z, y, x = last_zyx(a)
    z = max(1, int(z))
    y = max(1, int(y))
    x = max(1, int(x))

    zi = int(np.clip(int(zi), 0, z - 1))
    yi = int(np.clip(int(yi), 0, y - 1))
    xi = int(np.clip(int(xi), 0, x - 1))

    lay_xy = _ensure_plane(v, parent, "XY", zi)
    lay_xz = _ensure_plane(v, parent, "XZ", yi)
    lay_yz = _ensure_plane(v, parent, "YZ", xi)

    if lay_xy or lay_xz or lay_yz:
        try:
            parent.visible = True
            parent.opacity = 0.8
        except Exception:
            pass

        _toggle_center_points(v, parent.name, False)

        try:
            v.dims.ndisplay = 2
            v.grid.enabled = True
            v.grid.stride = 1
            v.grid.shape = (2, 2)
        except Exception:
            pass

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

    return saved_layout


def _clear_slice_compare_views(v, parent: NapariImage, saved_layout: Optional[Dict[str, Any]]):
    to_remove: List[NapariImage] = []
    for lay in [L2 for L2 in v.layers if isinstance(L2, NapariImage)]:
        md = getattr(lay, "metadata", {}) or {}
        if md.get("_slice_compare", False) and md.get("_slice_parent") == parent.name:
            to_remove.append(lay)

    for lay in to_remove:
        try:
            v.layers.remove(lay)
        except Exception:
            pass

    try:
        parent.visible = True
        _safe_contrast(parent)
    except Exception:
        pass

    _toggle_center_points(v, parent.name, True)

    if saved_layout:
        try:
            v.dims.ndisplay = int(saved_layout.get("ndisplay", 2))
            v.grid.enabled = bool(saved_layout.get("grid_enabled", False))
            v.grid.stride = int(saved_layout.get("grid_stride", 1))
            shp = saved_layout.get("grid_shape", (0, 0))
            if shp and tuple(shp) != (0, 0):
                v.grid.shape = tuple(shp)
        except Exception:
            pass

    if to_remove:
        show_info(f"Removed {len(to_remove)} orthogonal views for '{parent.name}'.")


def _find_parent_by_name(v, name: str) -> Optional[NapariImage]:
    if not v or not name:
        return None
    for L in _images(v):
        if L.name == name:
            return L
    return None


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

    _layout_saved: Optional[Dict[str, Any]] = None

    def _target_layer(self) -> Optional[NapariImage]:
        v = current_viewer()
        if not v:
            self._update_target_label(None)
            return None

        text = self.combo.currentText().strip()

        if text == "" or text == "Active (auto)":
            L_active = v.layers.selection.active
            if isinstance(L_active, NapariImage):
                md = getattr(L_active, "metadata", {}) or {}
                if not md.get("_slice_compare", False) and np.asarray(L_active.data).ndim >= 3:
                    self._update_target_label(L_active)
                    return L_active
            imgs = _images(v)
            L = imgs[0] if imgs else None
            self._update_target_label(L)
            return L

        for L in _images(v):
            if L.name == text:
                self._update_target_label(L)
                return L

        imgs = _images(v)
        L = imgs[0] if imgs else None
        self._update_target_label(L)
        return L

    def _update_target_label(self, L: Optional[NapariImage]):
        if L is None:
            self.target_label.setText("Current target: (none)")
        else:
            self.target_label.setText(f"Current target: {_short_name(L.name)}")

    def _sync_limits(self):
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
            max_idx = size - 1
            sb.setRange(0, max_idx)

            text = sb.text().strip()
            try:
                raw = int(text)
            except Exception:
                raw = int(sb.value())

            raw = max(0, min(raw, max_idx))
            sb.setValue(raw)

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

    def make_views(self):
        if self._busy:
            return
        self._busy = True
        try:
            self._sync_limits()
            parent = self._target_layer()
            if not parent:
                show_warning("Pick a 3D image first.")
                return

            v = current_viewer()
            if not v:
                return

            zi = int(self.z_spin.value())
            yi = int(self.y_spin.value())
            xi = int(self.x_spin.value())

            self._layout_saved = _apply_slice_compare_views(
                v, parent, zi, yi, xi, saved_layout=self._layout_saved
            )
            self._update_highlight()

            show_info(
                f"Orthogonal views for '{parent.name}' "
                f"(XY z={int(self.z_spin.value())}, XZ y={int(self.y_spin.value())}, YZ x={int(self.x_spin.value())}) updated."
            )

            # ✅ record (correct API)
            if _REC is not None:
                try:
                    _REC.add_step(
                        "slice_compare_make",
                        target="__ACTIVE__",
                        params={"z": int(self.z_spin.value()), "y": int(self.y_spin.value()), "x": int(self.x_spin.value())},
                    )

                except Exception:
                    pass
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
            parent = self._target_layer()
            if not parent:
                return

            _clear_slice_compare_views(v, parent, self._layout_saved)
            self._layout_saved = None

            # ✅ record (correct API)
            if _REC is not None:
                try:
                    _REC.add_step("slice_compare_clear", target="__ACTIVE__", params={})
                except Exception:
                    pass
        finally:
            self._busy = False


# ---------------- replay handlers --------------------------------------------
def _resolve_parent_for_replay(v, target: str) -> Optional[NapariImage]:
    """
    Replay target resolver:
    - "__ACTIVE__" / "" -> active 3D image (or first 3D image)
    - explicit name -> that layer, else fallback to active/first 3D image
    """
    t = (target or "").strip()

    # active token
    if t in ("", "__ACTIVE__", "__ANY__", "__ANY_IMAGE__", "__ANY_IMAGE__"):
        active = v.layers.selection.active if v else None
        if isinstance(active, NapariImage):
            md = getattr(active, "metadata", {}) or {}
            if (not md.get("_slice_compare", False)) and np.asarray(active.data).ndim >= 3:
                return active
        imgs = _images(v)
        return imgs[0] if imgs else None

    # by name
    parent = _find_parent_by_name(v, t)
    if parent is not None:
        return parent

    # fallback
    active = v.layers.selection.active if v else None
    if isinstance(active, NapariImage):
        md = getattr(active, "metadata", {}) or {}
        if (not md.get("_slice_compare", False)) and np.asarray(active.data).ndim >= 3:
            return active
    imgs = _images(v)
    return imgs[0] if imgs else None


def _handle_slice_compare_make(v, step):
    target = getattr(step, "target", "") or ""
    params = getattr(step, "params", {}) or {}
    zi = int(params.get("z", 0))
    yi = int(params.get("y", 0))
    xi = int(params.get("x", 0))

    parent = _resolve_parent_for_replay(v, target)
    if parent is None:
        show_warning("SliceCompare replay: no 3D image found to apply the recipe to.")
        return

    if target and target not in ("__ACTIVE__", "__ANY__", "__ANY_IMAGE__", "__ANY_IMAGE__") and parent.name != target:
        show_warning(f"SliceCompare replay: target '{target}' not found — using '{parent.name}' instead.")

    _apply_slice_compare_views(v, parent, zi, yi, xi, saved_layout=None)


def _handle_slice_compare_clear(v, step):
    target = getattr(step, "target", "") or ""
    parent = _resolve_parent_for_replay(v, target)
    if parent is None:
        show_warning("SliceCompare replay: no 3D image found to apply the recipe to.")
        return

    if target and target not in ("__ACTIVE__", "__ANY__", "__ANY_IMAGE__", "__ANY_IMAGE__") and parent.name != target:
        show_warning(f"SliceCompare replay: target '{target}' not found — using '{parent.name}' instead.")

    _clear_slice_compare_views(v, parent, saved_layout=None)


# ✅ REGISTER handlers (THIS is what your error is complaining about)
if callable(register_handler):
    try:
        register_handler("slice_compare_make", _handle_slice_compare_make)
        register_handler("slice_compare_clear", _handle_slice_compare_clear)
    except Exception:
        pass

# ---------------- panel factory ----------------------------------------------
def make_slice_compare_panel() -> tuple[SliceCompareController, QWidget]:
    panel = QWidget()
    root = QVBoxLayout(panel)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(12)

    title = QLabel("Slice Compare")
    title.setStyleSheet("font-weight:700; font-size:14px;")
    root.addWidget(title)

    subtitle = QLabel("Create orthogonal planes (XY / XZ / YZ) and tile them in a 2×2 grid.")
    subtitle.setStyleSheet("color:#9aa0a6; font-size:11px;")
    subtitle.setWordWrap(True)
    root.addWidget(subtitle)

    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Sunken)
    root.addWidget(sep)

    # ---- Group: Target ----
    gb_target = QGroupBox("Target")
    gb_target.setStyleSheet("QGroupBox{font-weight:600;} QGroupBox::title{padding:0 6px;}")
    lay_target = QVBoxLayout(gb_target)
    lay_target.setContentsMargins(10, 10, 10, 10)
    lay_target.setSpacing(10)

    row_t = QHBoxLayout()
    row_t.setSpacing(10)
    row_t.addWidget(QLabel("Target image:"))
    combo = QComboBox()
    combo.addItem("Active (auto)")
    combo.setMinimumWidth(240)
    row_t.addWidget(combo)
    row_t.addStretch(1)
    lay_target.addLayout(row_t)

    target_label = QLabel("Current target: (none)")
    target_label.setStyleSheet("color:#9aa0a6; font-size:11px;")
    lay_target.addWidget(target_label)

    root.addWidget(gb_target)

    # ---- Group: Indices ----
    gb_idx = QGroupBox("Indices")
    gb_idx.setStyleSheet("QGroupBox{font-weight:600;} QGroupBox::title{padding:0 6px;}")
    form = QFormLayout(gb_idx)
    form.setContentsMargins(10, 10, 10, 10)
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(10)

    z_spin = QSpinBox()
    y_spin = QSpinBox()
    x_spin = QSpinBox()
    for sb in (z_spin, y_spin, x_spin):
        sb.setRange(0, 2_000_000)
        sb.setMinimumWidth(120)

    form.addRow("Z (XY plane)", z_spin)
    form.addRow("Y (XZ plane)", y_spin)
    form.addRow("X (YZ plane)", x_spin)

    root.addWidget(gb_idx)

    # ---- Group: Actions ----
    gb_actions = QGroupBox("Actions")
    gb_actions.setStyleSheet("QGroupBox{font-weight:600;} QGroupBox::title{padding:0 6px;}")
    lay_actions = QVBoxLayout(gb_actions)
    lay_actions.setContentsMargins(10, 10, 10, 10)
    lay_actions.setSpacing(10)

    make_btn = QPushButton("Make / update orthogonal views")
    clear_btn = QPushButton("Clear orthogonal views")
    for b in (make_btn, clear_btn):
        b.setMinimumHeight(34)

    lay_actions.addWidget(make_btn)
    lay_actions.addWidget(clear_btn)

    hint = QLabel("Tip: click one of the created planes to highlight it (opacity).")
    hint.setStyleSheet("color:#9aa0a6; font-size:11px;")
    hint.setWordWrap(True)
    lay_actions.addWidget(hint)

    root.addWidget(gb_actions)
    root.addStretch(1)

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
