# napari_pypore3d/crop.py — r46 (manual crop, this vs ALL, 3D-safe)
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from magicgui.widgets import (
    CheckBox,
    RangeSlider,
    PushButton,
    Container,
    Label,
    ComboBox,
)
from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_info

# ---- fallbacks if helpers missing -------------------------------------------
try:
    from .helpers import active_image, ensure_caption
except Exception:
    def active_image():
        v = current_viewer()
        if not v:
            return None, None
        L = v.layers.selection.active
        return (L, v) if isinstance(L, NapariImage) else (None, v)

    def ensure_caption(_L: NapariImage, *_a, **_k):
        return


# ---------------------------------------------------------------------------

def _images(v) -> List[NapariImage]:
    return [l for l in (v.layers if v else []) if isinstance(l, NapariImage)]


def _safe_contrast(L: NapariImage):
    try:
        if hasattr(L, "reset_contrast_limits"):
            L.reset_contrast_limits()
        else:
            a = np.asarray(L.data, dtype=np.float32)
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


def _shape_zyx(a: np.ndarray):
    """Assume napari style: last 3 dims are Z, Y, X."""
    a = np.asarray(a)
    if a.ndim == 2:
        y, x = a.shape
        return 1, int(y), int(x)
    if a.ndim >= 3:
        z, y, x = a.shape[-3], a.shape[-2], a.shape[-1]
        return int(z), int(y), int(x)
    return 1, 1, 1


def _slice_zyx(a: np.ndarray, zs, ze, ys, ye, xs, xe):
    """Slice last 3 dims as (Z,Y,X)."""
    a = np.asarray(a)
    if a.ndim == 2:
        return a[ys:ye, xs:xe]
    if a.ndim >= 3:
        pre = (slice(None),) * (a.ndim - 3)
        return a[pre + (slice(zs, ze), slice(ys, ye), slice(xs, xe))]
    return a


# ---------------------------------------------------------------------------

@dataclass
class CropController:
    target_pick: ComboBox
    z_range: RangeSlider
    y_range: RangeSlider
    x_range: RangeSlider
    apply_btn: PushButton
    reset_btn: PushButton
    apply_all: CheckBox

    _busy: bool = False

    # -------- basic helpers ----------
    def _viewer(self):
        return current_viewer()

    def _target_layer(self) -> Optional[NapariImage]:
        v = self._viewer()
        if not v:
            return None
        pick = self.target_pick.value
        if pick == "Active (auto)" or not isinstance(pick, str):
            L, _ = active_image()
            return L
        for L in _images(v):
            if L.name == pick:
                return L
        L, _ = active_image()
        return L

    def _ensure_full(self, L: NapariImage):
        """
        _orig_full: first full volume for this layer (napari orientation).
        """
        md = L.metadata
        if "_orig_full" not in md:
            # Prefer loader-provided full if present
            full = md.get("_orig_full", None)
            if full is None:
                full = md.get("_orig_data", None)
            if full is None:
                full = np.asarray(L.data)
            md["_orig_full"] = np.asarray(full)

    def _full_array(self, L: NapariImage) -> np.ndarray:
        md = L.metadata
        if "_orig_full" in md:
            return np.asarray(md["_orig_full"])
        full = md.get("_orig_data", L.data)
        return np.asarray(full)

    # -------- slider logic -----------
    def _init_slider_full(self, sl: RangeSlider, size: int):
        size = int(max(1, size))
        sl.min = 0
        sl.max = max(0, size - 1)
        sl.value = (0, sl.max)

    def _sync_sliders_to_layer(self, L: Optional[NapariImage]):
        if not L:
            for sl in (self.z_range, self.y_range, self.x_range):
                sl.min = 0
                sl.max = 0
                sl.value = (0, 0)
            return

        self._ensure_full(L)
        a_full = self._full_array(L)
        sz, sy, sx = _shape_zyx(a_full)

        self._init_slider_full(self.z_range, sz)
        self._init_slider_full(self.y_range, sy)
        self._init_slider_full(self.x_range, sx)

        if sz <= 1:
            self.z_range.min = 0
            self.z_range.max = 0
            self.z_range.value = (0, 0)

    # -------- core operations --------
    def _crop_layer(self, L: NapariImage):
        self._ensure_full(L)
        a_full = self._full_array(L)

        zs_idx, ze_idx = map(int, self.z_range.value)
        ys_idx, ye_idx = map(int, self.y_range.value)
        xs_idx, xe_idx = map(int, self.x_range.value)

        sz, sy, sx = _shape_zyx(a_full)

        def _clamp(lo, hi, size):
            if size <= 1:
                return 0, 0
            lo = max(0, min(int(lo), size - 1))
            hi = max(0, min(int(hi), size - 1))
            if hi < lo:
                hi = lo
            return lo, hi

        zs_idx, ze_idx = _clamp(zs_idx, ze_idx, sz)
        ys_idx, ye_idx = _clamp(ys_idx, ye_idx, sy)
        xs_idx, xe_idx = _clamp(xs_idx, xe_idx, sx)

        zs, ze = zs_idx, ze_idx + 1
        ys, ye = ys_idx, ye_idx + 1
        xs, xe = xs_idx, xe_idx + 1

        L.data = _slice_zyx(a_full, zs, ze, ys, ye, xs, xe)
        _safe_contrast(L)
        try:
            ensure_caption(L, position="bottom")
        except Exception:
            pass

    def _reset_layer(self, L: NapariImage):
        self._ensure_full(L)
        full = self._full_array(L)
        L.data = full
        _safe_contrast(L)
        try:
            ensure_caption(L, position="bottom")
        except Exception:
            pass

    # -------- UI actions -------------
    def apply(self, *_):
        if self._busy:
            return
        self._busy = True
        try:
            v = self._viewer()
            if not v:
                return

            if bool(self.apply_all.value):
                imgs = _images(v)
                if not imgs:
                    show_info("No image layers to crop.")
                    return
                for L in imgs:
                    self._crop_layer(L)
                show_info("Crop applied to all images.")
            else:
                L = self._target_layer()
                if not L:
                    show_info("Pick an image layer first.")
                    return
                self._crop_layer(L)
                show_info(f"Cropped '{L.name}'.")
        finally:
            self._busy = False

    def reset(self, *_):
        if self._busy:
            return
        self._busy = True
        try:
            v = self._viewer()
            if not v:
                return

            if bool(self.apply_all.value):
                imgs = _images(v)
                if not imgs:
                    return
                for L in imgs:
                    self._reset_layer(L)
                # resync sliders from active layer
                self._sync_sliders_to_layer(self._target_layer())
                show_info("Reset crop on all images.")
            else:
                L = self._target_layer()
                if not L:
                    return
                self._reset_layer(L)
                self._sync_sliders_to_layer(L)
                show_info(f"Reset crop on '{L.name}'.")
        finally:
            self._busy = False

    def on_layers_changed(self, *_):
        if self._busy:
            return
        self._busy = True
        try:
            v = self._viewer()
            names = [L.name for L in _images(v)] if v else []
            old_value = (
                self.target_pick.value
                if isinstance(self.target_pick.value, str)
                else "Active (auto)"
            )
            choices = ["Active (auto)"] + names

            # update combo choices
            try:
                self.target_pick.choices = choices
            except Exception:
                try:
                    self.target_pick.choices = [(c, c) for c in choices]
                except Exception:
                    pass

            if old_value in choices:
                self.target_pick.value = old_value
            else:
                L, _ = active_image()
                self.target_pick.value = (
                    L.name if (L and L.name in names) else "Active (auto)"
                )

            self._sync_sliders_to_layer(self._target_layer())
        finally:
            self._busy = False

    def connect(self):
        self.apply_btn.changed.connect(self.apply)
        self.reset_btn.changed.connect(self.reset)


# ---------------------------------------------------------------------------

def make_crop_panel() -> tuple[CropController, object]:
    target_pick = ComboBox(
        name="Target image",
        choices=["Active (auto)"],
        value="Active (auto)",
    )
    z_range = RangeSlider(name="Z", min=0, max=0, value=(0, 0))
    y_range = RangeSlider(name="Y", min=0, max=0, value=(0, 0))
    x_range = RangeSlider(name="X", min=0, max=0, value=(0, 0))
    apply_all = CheckBox(text="Apply to ALL images", value=False)
    apply_btn = PushButton(text="Apply crop")
    reset_btn = PushButton(text="Reset")

    ctrl = CropController(
        target_pick=target_pick,
        z_range=z_range,
        y_range=y_range,
        x_range=x_range,
        apply_btn=apply_btn,
        reset_btn=reset_btn,
        apply_all=apply_all,
    )
    ctrl.connect()

    panel = Container(
        widgets=[
            Label(value="Crop (FULL volume, napari Z/Y/X)"),
            target_pick,
            z_range,
            y_range,
            x_range,
            apply_all,
            apply_btn,
            reset_btn,
        ],
        layout="vertical",
        labels=False,
    )

    ctrl.on_layers_changed()

    # update sliders when layers change
    try:
        v = current_viewer()
        if v:
            v.layers.events.inserted.connect(ctrl.on_layers_changed)
            v.layers.events.removed.connect(ctrl.on_layers_changed)
            v.layers.events.reordered.connect(ctrl.on_layers_changed)
            v.layers.selection.events.active.connect(ctrl.on_layers_changed)
    except Exception:
        pass

    return ctrl, panel.native
