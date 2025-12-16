# napari_pypore3d/crop.py — r103 (fast: debounced live preview + 3D-safe)
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

from magicgui.widgets import ComboBox, RangeSlider, PushButton, CheckBox, Label, Container
from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_info, show_warning, show_error

from qtpy.QtCore import QTimer


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _viewer():
    try:
        return current_viewer()
    except Exception:
        return None


def _images(v) -> List[NapariImage]:
    """Return all Image layers (2D or 3D)."""
    if v is None:
        return []
    return [l for l in v.layers if isinstance(l, NapariImage)]


def _safe_contrast(L: NapariImage):
    """Make sure the layer is visible with sane contrast."""
    try:
        if hasattr(L, "reset_contrast_limits"):
            L.reset_contrast_limits()
        else:
            a = np.asarray(L.data)
            if a.size:
                step = max(1, a.size // 256_000)
                samp = a.ravel()[::step].astype(float, copy=False)
                lo = float(np.nanpercentile(samp, 0.5))
                hi = float(np.nanpercentile(samp, 99.5))
                if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
                    lo = float(np.nanmin(samp))
                    hi = float(np.nanmax(samp))
                    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
                        hi = lo + 1.0
                L.contrast_limits = (lo, hi)
        L.visible = True
        L.opacity = 1.0
        try:
            L.colormap = "gray"
        except Exception:
            pass
    except Exception:
        pass


def _shape_zyx(a: np.ndarray) -> Tuple[int, int, int]:
    a = np.asarray(a)
    if a.ndim == 2:
        y, x = a.shape
        return 1, int(y), int(x)
    if a.ndim >= 3:
        z, y, x = a.shape[-3], a.shape[-2], a.shape[-1]
        return int(z), int(y), int(x)
    return 1, 1, 1


def _slice_zyx(a, z0, z1, y0, y1, x0, x1):
    """Slice by Z/Y/X, preserving any leading dims."""
    a = np.asarray(a)
    if a.ndim == 2:
        return a[y0:y1, x0:x1]
    pre = (slice(None),) * (a.ndim - 3)
    return a[pre + (slice(z0, z1), slice(y0, y1), slice(x0, x1))]


def _as_int_pair(v) -> Tuple[int, int]:
    """RangeSlider may return floats; force to safe ints."""
    a, b = v
    a = int(round(float(a)))
    b = int(round(float(b)))
    if a > b:
        a, b = b, a
    return a, b


def _clamp_pair(a: int, b: int, lo: int, hi: int) -> Tuple[int, int]:
    a = max(lo, min(hi, a))
    b = max(lo, min(hi, b))
    if a > b:
        a = b
    return a, b


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

@dataclass
class CropCtrl:
    pick: ComboBox
    rz: RangeSlider
    ry: RangeSlider
    rx: RangeSlider
    live_preview: CheckBox
    apply_all: CheckBox
    b_apply: PushButton
    b_reset: PushButton

    busy: bool = False
    _timer: Optional[QTimer] = None
    _warned_3d: bool = False

    def _get_viewer(self):
        return _viewer()

    def _is_3d(self) -> bool:
        v = self._get_viewer()
        try:
            return bool(v and int(getattr(v.dims, "ndisplay", 2)) == 3)
        except Exception:
            return False

    def _get_target(self) -> Optional[NapariImage]:
        v = self._get_viewer()
        if v is None:
            return None

        name = self.pick.value

        if name == "Active (auto)":
            L = v.layers.selection.active
            return L if isinstance(L, NapariImage) else None

        for L in _images(v):
            if L.name == name:
                return L

        L = v.layers.selection.active
        return L if isinstance(L, NapariImage) else None

    # ---------- full-volume storage ----------

    def _ensure_full(self, L: NapariImage):
        md = L.metadata
        if "_orig_full" not in md:
            md["_orig_full"] = np.asarray(L.data)

    def _full(self, L: NapariImage) -> np.ndarray:
        full = L.metadata.get("_orig_full")
        return full if isinstance(full, np.ndarray) else np.asarray(L.data)

    # ---------- slider sync ----------

    def _sync_sliders_to_layer(self, L: Optional[NapariImage]):
        if L is None:
            return
        self._ensure_full(L)
        a = self._full(L)
        z, y, x = _shape_zyx(a)

        self.rz.min = 0
        self.rz.max = max(0, z - 1)
        self.rz.value = (0, self.rz.max)

        self.ry.min = 0
        self.ry.max = max(0, y - 1)
        self.ry.value = (0, self.ry.max)

        self.rx.min = 0
        self.rx.max = max(0, x - 1)
        self.rx.value = (0, self.rx.max)

    # ---------- core ops ----------

    def _crop_single(self, L: NapariImage):
        self._ensure_full(L)
        full = self._full(L)
        z, y, x = _shape_zyx(full)

        z0, z1 = _as_int_pair(self.rz.value)
        y0, y1 = _as_int_pair(self.ry.value)
        x0, x1 = _as_int_pair(self.rx.value)

        z0, z1 = _clamp_pair(z0, z1, 0, max(0, z - 1))
        y0, y1 = _clamp_pair(y0, y1, 0, max(0, y - 1))
        x0, x1 = _clamp_pair(x0, x1, 0, max(0, x - 1))

        L.data = _slice_zyx(full, z0, z1 + 1, y0, y1 + 1, x0, x1 + 1)
        _safe_contrast(L)

    def _reset_single(self, L: NapariImage):
        self._ensure_full(L)
        L.data = self._full(L)
        _safe_contrast(L)

    # ---------- debounced live preview ----------

    def _ensure_timer(self):
        if self._timer is None:
            self._timer = QTimer()
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._do_live_crop)

    def _do_live_crop(self):
        if self.busy:
            return
        self.busy = True
        try:
            # Live preview is 2D-only
            if self._is_3d():
                return
            L = self._get_target()
            if L is not None:
                self._crop_single(L)
        finally:
            self.busy = False

    # ---------- UI callbacks ----------

    def apply(self, *_):
        if self.busy:
            return
        self.busy = True
        try:
            v = self._get_viewer()
            if v is None:
                show_warning("No napari viewer found.")
                return

            if bool(self.apply_all.value):
                imgs = _images(v)
                if not imgs:
                    show_warning("No image layers to crop.")
                    return
                for L in imgs:
                    self._crop_single(L)
                show_info("Crop applied to ALL images.")
            else:
                L = self._get_target()
                if not L:
                    show_warning("No image selected.")
                    return
                self._crop_single(L)
                show_info(f"Cropped '{L.name}'.")
        except Exception as e:
            show_error(f"Crop failed: {e!r}")
        finally:
            self.busy = False

    def reset(self, *_):
        if self.busy:
            return
        self.busy = True
        try:
            v = self._get_viewer()
            if v is None:
                return

            if bool(self.apply_all.value):
                for L in _images(v):
                    self._reset_single(L)
                show_info("Reset ALL images.")
                self._sync_sliders_to_layer(self._get_target())
            else:
                L = self._get_target()
                if not L:
                    return
                self._reset_single(L)
                self._sync_sliders_to_layer(L)
                show_info(f"Reset '{L.name}'.")
        except Exception as e:
            show_error(f"Reset failed: {e!r}")
        finally:
            self.busy = False

    def live_crop(self, *_):
        """Debounced live preview (2D only)."""
        try:
            if not bool(self.live_preview.value):
                return

            if self._is_3d():
                # auto-disable; 3D live updates rebuild rendering and kill UI
                if not self._warned_3d:
                    show_warning("Live preview disabled in 3D (too slow). Use 'Apply crop'.")
                    self._warned_3d = True
                try:
                    self.live_preview.value = False
                except Exception:
                    pass
                return

            self._warned_3d = False
            self._ensure_timer()
            # debounce: restart timer while dragging
            self._timer.start(150)
        except Exception:
            pass

    def on_layers_changed(self, *_):
        if self.busy:
            return
        self.busy = True
        try:
            v = self._get_viewer()
            if v is None:
                return

            names = [L.name for L in _images(v)]
            old = self.pick.value if isinstance(self.pick.value, str) else "Active (auto)"

            choices = ["Active (auto)"] + names
            try:
                self.pick.choices = choices
            except Exception:
                self.pick.choices = [(c, c) for c in choices]

            self.pick.value = old if old in choices else "Active (auto)"
            self._sync_sliders_to_layer(self._get_target())
        finally:
            self.busy = False

    def connect(self):
        """Wire widget callbacks using Qt-native signals (reliable)."""
        # Buttons
        try:
            self.b_apply.native.clicked.connect(lambda *_: self.apply())
        except Exception:
            self.b_apply.changed.connect(self.apply)

        try:
            self.b_reset.native.clicked.connect(lambda *_: self.reset())
        except Exception:
            self.b_reset.changed.connect(self.reset)

        # Sliders (debounced)
        for rs in (self.rz, self.ry, self.rx):
            try:
                rs.native.valueChanged.connect(lambda *_: self.live_crop())
            except Exception:
                rs.changed.connect(self.live_crop)

        # Picker
        try:
            self.pick.native.currentTextChanged.connect(lambda *_: self.on_layers_changed())
        except Exception:
            self.pick.changed.connect(self.on_layers_changed)

        # Live preview toggle (if user turns it on while already in 3D, disable)
        try:
            self.live_preview.changed.connect(self.live_crop)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_crop_panel():
    pick = ComboBox(name="Target image", choices=["Active (auto)"], value="Active (auto)")
    rz = RangeSlider(name="Z", min=0, max=0, value=(0, 0))
    ry = RangeSlider(name="Y", min=0, max=0, value=(0, 0))
    rx = RangeSlider(name="X", min=0, max=0, value=(0, 0))

    live_preview = CheckBox(text="Live preview (2D only)", value=False)
    apply_all = CheckBox(text="Apply to ALL images", value=False)
    b_apply = PushButton(text="Apply crop")
    b_reset = PushButton(text="Reset")

    ctrl = CropCtrl(
        pick=pick,
        rz=rz,
        ry=ry,
        rx=rx,
        live_preview=live_preview,
        apply_all=apply_all,
        b_apply=b_apply,
        b_reset=b_reset,
    )
    ctrl.connect()

    panel = Container(
        widgets=[
            Label(value="Crop (FULL volume — Z / Y / X)"),
            pick, rz, ry, rx,
            live_preview,
            apply_all, b_apply, b_reset,
        ],
        layout="vertical",
        labels=False,
    )

    try:
        ctrl.on_layers_changed()
    except Exception:
        pass

    return ctrl, panel.native
