# napari_pypore3d/_widget.py — r28 ULTRA
# ---------------------------------------------------------------------------
# A consolidated, production-ready napari dock widget for:
#   • Loading RAW/BIN volumes (robust shape inference; endianness; memmap/RAM)
#   • Multi-load with unique layer names; batch add (files/folder)
#   • Auto grid tiling, stride, column count, and contrast sync
#   • Cropping via [start:end) sliders, apply-to-active or apply-to-all, and reset
#   • “Match sizes” crop across layers to common Z×Y×X
#   • Quick Plot: histogram (+optional 1–99% clip) and one-click Otsu → mask
#   • Layer inspector: dtype, shape, min/max/mean/std, voxel size metadata
#   • Export tools: save (cropped or full) as .npy / .tif (if tifffile) / .raw
#   • Lightweight persistence of select UI settings (JSON in user home)
#   • Optional simple transforms (flip/rotate 90°/transpose), non-destructive
#   • Hard width clamp for the entire dock to guarantee NO horizontal scrollbar
#
# Notes:
#   - Dependencies beyond napari/magicgui/numpy are optional (tifffile, matplotlib).
#   - The code is organized into small helpers + lightweight controllers.
#   - All actions provide clear user feedback via napari notifications.
#
# Drop this file into napari_pypore3d/_widget.py in your plugin.
# ---------------------------------------------------------------------------

from __future__ import annotations  # MUST be first (after the header comment)
# near the top
import pathlib
from pathlib import Path  # you can keep this for your own uses

# ===== Standard library
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Literal,
    Tuple,
    Optional,
    Sequence,
    List,
    Dict,
    Any,
    Iterable,
)
import json
import os
import re
import sys
import math
import traceback

# ===== Third-party
import numpy as np

from magicgui import magicgui
from magicgui.widgets import (
    Container,
    PushButton,
    ComboBox,
    Label,
    RangeSlider,
    LineEdit,
    SpinBox,
    CheckBox,
)
from magicgui.widgets import Label
from qtpy.QtWidgets import QFrame

def _mgui_hline():
    """A magicgui-safe horizontal line for use inside Container(widgets=[...])."""
    L = Label(value="")
    try:
        L.native.setFrameShape(QFrame.HLine)
        L.native.setFrameShadow(QFrame.Sunken)
        L.native.setText("")  # just in case
    except Exception:
        pass
    return L


from napari import current_viewer
from napari.layers import Image as NapariImage, Labels
from napari.utils.notifications import show_error, show_warning, show_info

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QScrollArea,
    QSizePolicy,
    QFileDialog,
    QFrame,
)
from qtpy.QtCore import Qt

# ---- Optional libraries
# matplotlib (for Quick Plot)
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    FigureCanvas = None  # type: ignore
    Figure = None  # type: ignore

# tifffile (for TIFF export)
try:
    import tifffile as _tifffile  # pragma: no cover
    HAVE_TIFF = True
except Exception:
    HAVE_TIFF = False
    _tifffile = None  # type: ignore

# ----------------------------------------------------------------------------
# Global metadata & settings
# ----------------------------------------------------------------------------

PLUGIN_BUILD = "napari-pypore3d r28 ULTRA"

# Settings file for a few persistent options
SETTINGS_FILE = Path.home() / ".napari_pypore3d_settings.json"

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def _debug(msg: str) -> None:
    """Debug print when NPP3D_DEBUG=1."""
    if os.environ.get("NPP3D_DEBUG", "0").strip() == "1":
        print(f"[npp3d] {msg}")

def _iter_images(v) -> List[NapariImage]:
    """Return all Image layers in a viewer (safe on None)."""
    return [l for l in (v.layers if v else []) if isinstance(l, NapariImage)]

def _iter_labels(v) -> List[Labels]:
    return [l for l in (v.layers if v else []) if isinstance(l, Labels)]

def _active_image() -> tuple[Optional[NapariImage], Optional[object]]:
    """Return (active_image, viewer) tuple. Active may be None."""
    v = current_viewer()
    if v is None:
        return None, None
    layer = v.layers.selection.active
    return (layer, v) if isinstance(layer, NapariImage) else (None, v)

def _unique_layer_name(base: str) -> str:
    """Return a unique layer name in the current viewer by appending (2), (3), ..."""
    v = current_viewer()
    if v is None:
        return base
    names = {l.name for l in v.layers}
    if base not in names:
        return base
    i = 2
    while f"{base} ({i})" in names:
        i += 1
    return f"{base} ({i})"

def _last_zyx_shape(a: np.ndarray) -> Tuple[int, int, int]:
    """Return Z, Y, X from the last 3 dims of an array.
    2D → 1×Y×X ; ≥3D → last three dims; otherwise 1×1×1."""
    if a.ndim == 2:
        sy, sx = map(int, a.shape[-2:])
        return 1, sy, sx
    if a.ndim >= 3:
        z, y, x = map(int, a.shape[-3:])
        return z, y, x
    return 1, 1, 1

def _slice_last_zyx(
    a: np.ndarray,
    zs: int, ze: int,
    ys: int, ye: int,
    xs: int, xe: int,
) -> np.ndarray:
    """Slice the last three dimensions of an array using [start:end)."""
    if a.ndim == 2:
        return a[ys:ye, xs:xe]
    prefix = (slice(None),) * (a.ndim - 3)
    return a[prefix + (slice(zs, ze), slice(ys, ye), slice(xs, xe))]

def _best_shape_from_name(name: str) -> Optional[Tuple[int, int, int]]:
    """Pick the largest Z×Y×X triple found in a filename (e.g., '700x700x700')."""
    triples = re.findall(r"(\d+)[xX](\d+)[xX](\d+)", name)
    if not triples:
        return None
    best = max(((int(a), int(b), int(c)) for a, b, c in triples),
               key=lambda t: t[0] * t[1] * t[2])
    return best

def _ensure_labels_layer(base_layer: NapariImage, name: Optional[str] = None) -> Labels:
    """Return a labels layer (creating if needed) for a given image layer."""
    v = current_viewer()
    if v is None:
        raise RuntimeError("No active viewer")
    name = name or f"mask:{base_layer.name}"
    for L in v.layers:
        if isinstance(L, Labels) and L.name == name:
            return L
    lbl = v.add_labels(
        np.zeros_like(np.asarray(base_layer.data), dtype=np.uint8),
        name=name,
        opacity=0.55,
        color={0: (0, 0, 0, 0), 1: (1, 0, 0, 0.7)},
        blending="translucent",
    )
    return lbl

def _cap_width(maxw: int, *widgets):
    """Clamp widget widths and prevent horizontal growth."""
    for w in widgets:
        try:
            w.native.setMaximumWidth(maxw)
            w.native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        except Exception:
            try:
                w.setMaximumWidth(maxw)
                w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            except Exception:
                pass

def _numpy_dtype_from_string(s: str) -> np.dtype:
    mapping = {
        "uint8": np.uint8,
        "int8": np.int8,
        "uint16": np.uint16,
        "int16": np.int16,
        "uint32": np.uint32,
        "int32": np.int32,
        "float32": np.float32,
        "float64": np.float64,
    }
    if s not in mapping:
        raise ValueError(f"Unsupported dtype '{s}'")
    return np.dtype(mapping[s])

def _natural_sort_key(s: str) -> List[Any]:
    """Key for natural sorting of paths/strings with numbers."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _likely_big_endian(a: np.ndarray) -> bool:
    """Heuristic: sometimes values look wildly wrong; try byteswap sample and compare spread."""
    try:
        if a.dtype.itemsize <= 1:
            return False
        sample = a.ravel()[:: max(1, a.size // 10000 + 1)].astype(a.dtype, copy=False)
        if sample.size < 8:
            return False
        s1 = float(np.nanstd(sample))
        s2 = float(np.nanstd(sample.byteswap().newbyteorder()))
        return s2 > 8 * s1
    except Exception:
        return False

def _with_optional_voxel_size(layer: NapariImage, voxel_size: Optional[Tuple[float, float, float]]) -> None:
    """Store voxel size (ZYX order) in layer.metadata for later export."""
    if voxel_size:
        try:
            layer.metadata["_voxel_size"] = tuple(float(x) for x in voxel_size)
        except Exception:
            pass

def _export_raw(path: Path, arr: np.ndarray, dtype: np.dtype, byte_order: Literal["little","big"]) -> None:
    """Export numpy array to raw binary stream with given dtype & endianness."""
    a = np.asarray(arr)
    if a.dtype != dtype:
        a = a.astype(dtype, copy=False)
    if dtype.itemsize > 1:
        a = a.byteswap().newbyteorder(">" if byte_order == "big" else "<")
    with open(path, "wb") as f:
        f.write(a.tobytes(order="C"))

# ----------------------------------------------------------------------------
# Settings persistence
# ----------------------------------------------------------------------------

@dataclass
class AppSettings:
    """Optional persistence for a subset of UI choices."""
    dock_max_width: int = 360
    grid_cols: int = 2
    grid_stride: int = 2
    grid_enable: bool = True
    last_dir: str = ""
    default_dtype: str = "auto"
    default_byte_order: str = "little"
    default_bins: int = 256
    clip_1_99: bool = True
    prefer_memmap: bool = True

    @classmethod
    def load(cls) -> "AppSettings":
        try:
            if SETTINGS_FILE.exists():
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                filtered = {k: data.get(k) for k in cls.__dataclass_fields__.keys()}
                return cls(**filtered)
        except Exception:
            _debug("Failed to load settings; using defaults.")
        return cls()

    def save(self) -> None:
        try:
            SETTINGS_FILE.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")
        except Exception:
            _debug("Failed to save settings.")

# ----------------------------------------------------------------------------
# RAW Loader widget (magicgui)
# ----------------------------------------------------------------------------

_DTYPE_CHOICES = [
    "auto", "uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"
]

@magicgui(
    call_button="Load",
    layout="vertical",
    path={
        "widget_type": "FileEdit",
        "mode": "r",
        "filter": "*.raw;*.RAW;*.bin;*.BIN;*",
        "label": "file",
    },
    dtype={"choices": _DTYPE_CHOICES, "label": "dtype"},
    shape_z={"min": 1, "max": 10_000_000, "label": "Z"},
    shape_y={"min": 1, "max": 10_000_000, "label": "Y"},
    shape_x={"min": 1, "max": 10_000_000, "label": "X"},
    byte_order={"choices": ["little", "big"], "label": "byte order"},
    memmap={"label": "memory-map file"},
    copy_to_ram={"label": "copy to RAM after load"},
    voxel_z={"label": "voxel Z", "min": 0.0, "max": 1e9, "step": 0.0},
    voxel_y={"label": "voxel Y", "min": 0.0, "max": 1e9, "step": 0.0},
    voxel_x={"label": "voxel X", "min": 0.0, "max": 1e9, "step": 0.0},
)
def _raw_loader_widget(
    path: "pathlib.Path" = pathlib.Path(""),
    shape_z: int = 700,
    shape_y: int = 700,
    shape_x: int = 700,
    dtype: Literal[
        "auto","uint8","int8","uint16","int16","uint32","int32","float32","float64"
    ] = "auto",
    byte_order: Literal["little","big"] = "little",
    memmap: bool = True,
    copy_to_ram: bool = False,
    voxel_z: float = 0.0,
    voxel_y: float = 0.0,
    voxel_x: float = 0.0,
):
    """Robust RAW/BIN volume loader with shape/dtype inference and endianness."""
    p = Path(path)
    if not p or str(p).strip() == "" or not p.exists() or p.is_dir():
        raise FileNotFoundError("Select a RAW/BIN file first.")
    try:
        fsize = p.stat().st_size
    except Exception as e:
        raise FileNotFoundError(f"Cannot stat file: {e}")
    if fsize <= 0:
        raise ValueError("File is empty.")

    # Infer shape/dtype
    triple = _best_shape_from_name(p.name)
    chosen_shape: Tuple[int,int,int] | None = None
    np_dtype: np.dtype | None = None

    _cands = [
        np.uint8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.float64, np.int8
    ]

    if triple:
        z,y,x = triple
        nvox = z*y*x
        fits = [dt for dt in _cands if fsize == nvox * np.dtype(dt).itemsize]
        if not fits:
            raise ValueError(
                f"Name suggests {(z,y,x)} but size {fsize} B doesn't match any of "
                "u8/u16/i16/u32/i32/f32/f64/i8."
            )
        np_dtype = np.dtype(fits[0]) if dtype == "auto" else _numpy_dtype_from_string(dtype)
        chosen_shape = (z,y,x)
    else:
        Y, X = int(shape_y), int(shape_x)
        if Y <= 0 or X <= 0:
            raise ValueError("Y and X must be > 0.")
        candidates: List[Tuple[Tuple[int,int,int], np.dtype]] = []
        for dt in _cands:
            bpv = np.dtype(dt).itemsize
            if fsize % bpv:
                continue
            nvox = fsize // bpv
            if nvox % (Y*X) == 0:
                Z = nvox // (Y*X)
                if Z > 0:
                    candidates.append(((int(Z),Y,X), np.dtype(dt)))
        if not candidates:
            raise ValueError(
                f"Cannot infer Z for Y={Y}, X={X}. Size={fsize} B incompatible with "
                "u8/u16/i16/u32/i32/f32/f64/i8."
            )
        pref = {np.dtype(dt):i for i,dt in enumerate(_cands)}
        candidates.sort(key=lambda t: pref[t[1]])
        chosen_shape, infer_dt = candidates[0]
        np_dtype = infer_dt if dtype == "auto" else _numpy_dtype_from_string(dtype)

    # Expected sanity
    expected = int(np.prod(chosen_shape)) * int(np_dtype.itemsize)
    if expected != fsize:
        show_warning(
            f"Expected {expected} B for shape {chosen_shape} / {np_dtype}, file is {fsize} B. "
            "Loading anyway (may be padded/truncated)."
        )
        # Adjust shape to actual file size so memmap/ndarray won't error
        voxels = fsize // int(np_dtype.itemsize)
        YX = int(chosen_shape[1]) * int(chosen_shape[2])
        if voxels < YX:
            raise ValueError(
                "File is smaller than a single Y×X plane for the chosen dtype. "
                "Adjust Y/X/dtype or select a different file."
            )
        Z = voxels // YX
        chosen_shape = (int(Z), int(chosen_shape[1]), int(chosen_shape[2]))
        expected = int(np.prod(chosen_shape)) * int(np_dtype.itemsize)

    # Load data (memmap or RAM)
    if memmap:
        vol = np.memmap(p, dtype=np_dtype, mode="r", shape=chosen_shape, order="C")
    else:
        with open(p, "rb") as fh:
            buf = fh.read()
        vol = np.ndarray(chosen_shape, dtype=np_dtype, buffer=buf)

    # Endianness
    if np_dtype.itemsize > 1:
        desired = "<" if byte_order == "little" else ">"
        if vol.dtype.byteorder not in ("=", desired):
            vol = vol.byteswap().newbyteorder(desired)

    # Heuristic warning
    try:
        if np_dtype.itemsize > 1 and _likely_big_endian(np.asarray(vol)):
            show_warning("Values look odd; try toggling byte order.")
    except Exception:
        pass

    # Optional copy to RAM (even if memmap)
    if copy_to_ram:
        vol = np.asarray(vol).copy(order="C")

    v = current_viewer()
    if v is None:
        raise RuntimeError("No active viewer.")
    base_name = f"{p.stem} {chosen_shape[0]}x{chosen_shape[1]}x{chosen_shape[2]}"
    layer_name = _unique_layer_name(base_name)
    layer = v.add_image(vol, name=layer_name)
    layer.metadata["_orig_data"] = layer.data  # preserve for reset/crop
    layer.metadata["_file_path"] = str(p)
    _with_optional_voxel_size(layer, (
        voxel_z if voxel_z>0 else None,
        voxel_y if voxel_y>0 else None,
        voxel_x if voxel_x>0 else None
    ) if (voxel_z>0 or voxel_y>0 or voxel_x>0) else None)
    if layer.ndim >= 3:
        z,y,x = chosen_shape
        v.dims.current_step = (z//2, y//2, x//2) if layer.data.ndim==3 else tuple(s//2 for s in layer.data.shape[-3:])
    return layer

# ----------------------------------------------------------------------------
# Controllers
# ----------------------------------------------------------------------------

@dataclass
class GridManager:
    """Manage napari grid layout for image comparison."""
    enable: CheckBox
    cols: SpinBox
    stride: SpinBox
    apply_btn: PushButton
    sync_contrast: CheckBox

    def apply(self) -> None:
        v = current_viewer()
        if not v:
            return
        imgs = _iter_images(v)
        if bool(self.enable.value) and len(imgs) >= 2:
            v.grid.enabled = True
            v.grid.shape = (-1, max(1, int(self.cols.value)))
            # guard for napari versions without 'stride'
            stride_val = max(1, int(self.stride.value))
            try:
                v.grid.stride = stride_val
            except Exception:
                _debug("viewer.grid has no 'stride' attribute in this napari version")
            for l in imgs:
                l.visible = True
            if bool(self.sync_contrast.value) and imgs:
                cl = tuple(map(float, imgs[0].contrast_limits))
                for l in imgs:
                    l.contrast_limits = cl
        else:
            v.grid.enabled = False
            v.grid.shape = (-1,-1)

    def auto_apply_on_layers(self) -> None:
        v = current_viewer()
        if v and len(_iter_images(v)) > 1 and bool(self.enable.value):
            self.apply()

    def connect(self) -> None:
        self.apply_btn.clicked.connect(self.apply)
        self.enable.changed.connect(self.apply)
        self.cols.changed.connect(self.apply)
        self.stride.changed.connect(self.apply)
        self.sync_contrast.changed.connect(self.apply)

@dataclass
class CropController:
    """Simple crop controller using [start:end) sliders on Z/Y/X."""
    enable: CheckBox
    z_range: RangeSlider
    y_range: RangeSlider
    x_range: RangeSlider
    reset_btn: PushButton
    apply_all_chk: CheckBox

    def _ensure_orig(self, layer: NapariImage) -> None:
        if "_orig_data" not in layer.metadata:
            layer.metadata["_orig_data"] = layer.data

    def _sync_limits_to_active(self) -> None:
        layer, _ = _active_image()
        if not layer:
            return
        data = np.asarray(layer.metadata.get("_orig_data", layer.data))
        sz, sy, sx = _last_zyx_shape(data)
        for sl, mx in ((self.z_range, sz), (self.y_range, sy), (self.x_range, sx)):
            sl.max = int(mx)
            lo, hi = sl.value
            if hi > mx:
                sl.value = (lo, int(mx))

    def _apply_to_layer(self, layer: NapariImage) -> None:
        self._ensure_orig(layer)
        orig = layer.metadata["_orig_data"]
        a = np.asarray(orig)
        if not self.enable.value:
            layer.data = orig
            return
        zs, ze = map(int, self.z_range.value)
        ys, ye = map(int, self.y_range.value)
        xs, xe = map(int, self.x_range.value)
        if ze <= zs: ze = zs + 1
        if ye <= ys: ye = ys + 1
        if xe <= xs: xe = xs + 1
        sz, sy, sx = _last_zyx_shape(a)
        zs = max(0, min(zs, sz-1))
        ze = max(1, min(ze, sz))
        ys = max(0, min(ys, sy-1))
        ye = max(1, min(ye, sy))
        xs = max(0, min(xs, sx-1))
        xe = max(1, min(xe, sx))
        layer.data = _slice_last_zyx(a, zs, ze, ys, ye, xs, xe)

    def apply(self) -> None:
        v = current_viewer()
        if not v:
            return
        if bool(self.apply_all_chk.value):
            for L in _iter_images(v):
                self._apply_to_layer(L)
            show_info("Crop applied to all image layers.")
        else:
            L, _ = _active_image()
            if L is None:
                show_warning("Pick an image layer.")
                return
            self._apply_to_layer(L)

    def reset(self) -> None:
        v = current_viewer()
        if not v:
            return
        if bool(self.apply_all_chk.value):
            cnt = 0
            for L in _iter_images(v):
                if "_orig_data" in L.metadata:
                    L.data = L.metadata["_orig_data"]
                    cnt += 1
            if cnt:
                self.on_layer_change()
                self.enable.value = False
                show_info("Reset crop on all images.")
        else:
            L, _ = _active_image()
            if not L:
                return
            self._ensure_orig(L)
            L.data = L.metadata["_orig_data"]
            sz, sy, sx = _last_zyx_shape(np.asarray(L.data))
            self.z_range.value = (0, sz)
            self.y_range.value = (0, sy)
            self.x_range.value = (0, sx)
            self.enable.value = False
            show_info(f"Reset crop on '{L.name}'.")

    def connect(self) -> None:
        self.enable.changed.connect(self.apply)
        self.z_range.changed.connect(self.apply)
        self.y_range.changed.connect(self.apply)
        self.x_range.changed.connect(self.apply)
        self.reset_btn.clicked.connect(self.reset)

    def on_layer_change(self) -> None:
        self._sync_limits_to_active()

@dataclass
class QuickPlot:
    """Histogram + one-click Otsu → mask."""
    pick_layer: ComboBox
    bins: SpinBox
    clip: CheckBox
    plot_btn: PushButton
    otsu_btn: PushButton
    canvas: Optional[FigureCanvas]
    figure: Optional[Figure]
    invert_chk: CheckBox
    mask_name_edit: LineEdit

    def refresh_layer_choices(self) -> None:
        v = current_viewer()
        names = [l.name for l in _iter_images(v)] if v else []
        self.pick_layer.choices = names
        if names and (self.pick_layer.value not in names):
            self.pick_layer.value = names[0]

    def _get_layer_by_name(self, name: Optional[str]) -> Optional[NapariImage]:
        v = current_viewer()
        if not v or not name:
            return None
        for l in _iter_images(v):
            if l.name == name:
                return l
        return None

    def _plot_hist(self) -> None:
        if not (HAVE_MPL and self.figure and self.canvas):
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        L = self._get_layer_by_name(self.pick_layer.value)
        if L is None:
            self.refresh_layer_choices()
            self.canvas.draw()
            return
        arr = np.asarray(L.data).ravel().astype(float)
        if arr.size == 0:
            self.canvas.draw()
            return
        if bool(self.clip.value):
            lo, hi = np.percentile(arr, (1, 99))
            arr = arr[(arr >= lo) & (arr <= hi)]
        nb = int(self.bins.value)
        hist, edges = np.histogram(arr, bins=nb)
        ax.bar(edges[:-1], hist, width=np.diff(edges), align="edge")
        ax.set_xlabel("intensity")
        ax.set_ylabel("count")
        ax.set_title("Histogram")
        ax.grid(True, alpha=0.2)
        self.canvas.draw()

    def _otsu_to_mask(self) -> None:
        L = self._get_layer_by_name(self.pick_layer.value)
        if L is None:
            show_warning("Select a layer.")
            return
        a = np.asarray(L.data).astype(float)
        v = a.ravel()
        if v.size == 0:
            show_warning("Empty array.")
            return
        if bool(self.clip.value):
            lo, hi = np.percentile(v, (1, 99))
            v = v[(v >= lo) & (v <= hi)]
        nb = int(self.bins.value)
        h, e = np.histogram(v, bins=nb)
        p = h.astype(float)
        s = p.sum() or 1.0
        p /= s
        w = np.cumsum(p)
        mu = np.cumsum(p * e[:-1])
        mu_t = mu[-1]
        sb2 = (mu_t * w - mu) ** 2 / (w * (1 - w) + 1e-12)
        thr = float(e[int(np.nanargmax(sb2))])
        if bool(self.invert_chk.value):
            m = (np.asarray(L.data) < thr).astype(np.uint8)
        else:
            m = (np.asarray(L.data) >= thr).astype(np.uint8)
        name = self.mask_name_edit.value.strip() or f"mask:{L.name}"
        lbl = _ensure_labels_layer(L, name=name)
        lbl.data = m
        show_info(f"Otsu threshold ≈ {thr:.6g} — mask updated ({name}).")

    def connect(self) -> None:
        if HAVE_MPL:
            self.plot_btn.clicked.connect(self._plot_hist)
        self.otsu_btn.clicked.connect(self._otsu_to_mask)

    def on_layers_changed(self) -> None:
        self.refresh_layer_choices()

# ----------------------------------------------------------------------------
# Batch loaders
# ----------------------------------------------------------------------------

def _add_many_files(settings: AppSettings) -> None:
    v = current_viewer()
    if not v:
        return
    start_dir = settings.last_dir if settings.last_dir and Path(settings.last_dir).exists() else ""
    paths, _ = QFileDialog.getOpenFileNames(
        None,
        "Add RAW files",
        start_dir,
        "RAW/BIN (*.raw *.RAW *.bin *.BIN);;All (*)",
    )
    if not paths:
        return
    paths = sorted(paths, key=_natural_sort_key)
    settings.last_dir = str(Path(paths[0]).parent)
    for p in paths:
        try:
            _raw_loader_widget(
                path=Path(p),
                dtype=settings.default_dtype,  # persist user preference
                byte_order=settings.default_byte_order,
                memmap=settings.prefer_memmap,
                copy_to_ram=False,
            )
        except Exception as e:
            show_warning(str(e))
    settings.save()

def _add_from_folder(settings: AppSettings) -> None:
    v = current_viewer()
    if not v:
        return
    start_dir = settings.last_dir if settings.last_dir and Path(settings.last_dir).exists() else ""
    folder = QFileDialog.getExistingDirectory(None, "Pick folder with RAW/BIN", start_dir)
    if not folder:
        return
    exts = {".raw", ".RAW", ".bin", ".BIN"}
    files = [str(Path(folder) / f) for f in sorted(os.listdir(folder), key=_natural_sort_key)
             if Path(f).suffix in exts]
    if not files:
        show_warning("No RAW/BIN files in folder.")
        return
    settings.last_dir = folder
    for p in files:
        try:
            _raw_loader_widget(
                path=Path(p),
                dtype=settings.default_dtype,
                byte_order=settings.default_byte_order,
                memmap=settings.prefer_memmap,
                copy_to_ram=False,
            )
        except Exception as e:
            show_warning(str(e))
    settings.save()

# ----------------------------------------------------------------------------
# Info & Export helpers
# ----------------------------------------------------------------------------

def _array_stats(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a)
    if a.size == 0:
        return dict(min=np.nan, max=np.nan, mean=np.nan, std=np.nan)
    return dict(
        min=float(np.nanmin(a)),
        max=float(np.nanmax(a)),
        mean=float(np.nanmean(a)),
        std=float(np.nanstd(a)),
    )

def _get_voxel_size(layer: NapariImage) -> Optional[Tuple[float,float,float]]:
    vs = layer.metadata.get("_voxel_size")
    if isinstance(vs, (tuple, list)) and len(vs) == 3:
        try:
            return (float(vs[0]), float(vs[1]), float(vs[2]))
        except Exception:
            return None
    return None

def _export_as_npy(path: Path, layer: NapariImage) -> None:
    a = np.asarray(layer.data)
    np.save(path, a)

def _export_as_tif(path: Path, layer: NapariImage) -> None:
    if not HAVE_TIFF:
        raise RuntimeError("tifffile is not installed. Install it to export TIFF.")
    a = np.asarray(layer.data)
    vs = _get_voxel_size(layer)
    metadata = {}
    if vs:
        metadata["spacing"] = vs[0]
        metadata["unit"] = "unknown"
        metadata["axes"] = "ZYX" if a.ndim>=3 else "YX"
    _tifffile.imwrite(str(path), a, metadata=metadata)

def _export_as_raw(path: Path, layer: NapariImage,
                   dtype: str, byte_order: Literal["little","big"]) -> None:
    a = np.asarray(layer.data)
    dt = _numpy_dtype_from_string(dtype)
    _export_raw(path, a, dt, byte_order)

# ----------------------------------------------------------------------------
# Main dock assembly
# ----------------------------------------------------------------------------

def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line

def raw_loader_widget():
    """Create and return the full dock widget (wrapped in a scroll area)."""
    settings = AppSettings.load()
    show_info(PLUGIN_BUILD)

    # ---- Load & Compare controls ----
    gui = _raw_loader_widget

    grid_enable = CheckBox(text="Enable grid", value=bool(settings.grid_enable))
    grid_cols = SpinBox(label="Cols", min=1, max=16, value=int(settings.grid_cols))
    grid_stride = SpinBox(label="Stride", min=1, max=16, value=int(settings.grid_stride))
    sync_contrast = CheckBox(text="Sync contrast", value=True)
    btn_apply_grid = PushButton(text="Apply grid")

    btn_add_many = PushButton(text="Add files…")
    btn_add_folder = PushButton(text="Add from folder…")

    btn_match_sizes = PushButton(text="Match sizes")
    btn_rename_active = PushButton(text="Rename active")
    name_edit = LineEdit(value="")
    try:
        name_edit.native.setPlaceholderText("New layer name")
    except Exception:
        pass
    # ---- Crop (simple) ----
    crop_enable = CheckBox(text="crop [start:end)", value=False)
    z_range = RangeSlider(label="Z", min=0, max=1, value=(0,1), step=1)
    y_range = RangeSlider(label="Y", min=0, max=1, value=(0,1), step=1)
    x_range = RangeSlider(label="X", min=0, max=1, value=(0,1), step=1)
    chk_apply_all = CheckBox(text="Apply to all images", value=False)
    btn_reset_crop = PushButton(text="Reset crop")

    # ---- Tools: simple transforms ----
    btn_flip_z = PushButton(text="Flip Z")
    btn_flip_y = PushButton(text="Flip Y")
    btn_flip_x = PushButton(text="Flip X")
    btn_rot90_yx = PushButton(text="Rotate 90° (Y↔X)")
    btn_transpose_last = PushButton(text="Transpose last 2 dims")

    # ---- Quick Plot ----
    plots_widget = QWidget()
    pv = QVBoxLayout(plots_widget)
    pv.setContentsMargins(4, 4, 4, 4)
    pv.setSpacing(6)

    if HAVE_MPL:
        pick_layer = ComboBox(label="layer", choices=[], value=None, nullable=True)
        bins = SpinBox(label="bins", min=8, max=8192, value=int(settings.default_bins))
        clip = CheckBox(text="clip 1–99%", value=bool(settings.clip_1_99))
        btn_plot = PushButton(text="Plot")
        btn_otsu_to_mask = PushButton(text="Apply Otsu → mask (red)")
        invert_chk = CheckBox(text="invert", value=False)
        mask_name_edit = LineEdit(value="")
        try:
            mask_name_edit.native.setPlaceholderText("mask layer name (optional)")
        except Exception:
            pass
        fig = Figure(figsize=(4.6, 3.1), tight_layout=True)
        canvas = FigureCanvas(fig)
        pv.addWidget(pick_layer.native)
        pv.addWidget(
            Container(widgets=[bins, clip, invert_chk, btn_plot], layout="horizontal", labels=False).native
        )
        pv.addWidget(mask_name_edit.native)
        pv.addWidget(btn_otsu_to_mask.native)
        pv.addWidget(canvas)
        quick_plot = QuickPlot(
            pick_layer=pick_layer, bins=bins, clip=clip,
            plot_btn=btn_plot, otsu_btn=btn_otsu_to_mask,
            canvas=canvas, figure=fig, invert_chk=invert_chk,
            mask_name_edit=mask_name_edit,
        )
        quick_plot.connect()
    else:
        pv.addWidget(Label(value="Install matplotlib in the current venv for Quick Plot.").native)
        quick_plot = None  # type: ignore

    # ---- Info & Export tab ----
    info_widget = QWidget()
    iv = QVBoxLayout(info_widget)
    iv.setContentsMargins(6,6,6,6)
    iv.setSpacing(6)

    lbl_active_name = Label(value="—")
    lbl_dtype = Label(value="—")
    lbl_shape = Label(value="—")
    lbl_stats = Label(value="—")
    lbl_voxel = Label(value="—")

    iv.addWidget(Label(value="Active layer:").native)
    iv.addWidget(lbl_active_name.native)
    iv.addWidget(_hline())

    iv.addWidget(Label(value="dtype / shape:").native)
    iv.addWidget(lbl_dtype.native)
    iv.addWidget(lbl_shape.native)
    iv.addWidget(_hline())

    iv.addWidget(Label(value="min / max / mean / std:").native)
    iv.addWidget(lbl_stats.native)
    iv.addWidget(_hline())

    iv.addWidget(Label(value="voxel size (Z,Y,X):").native)
    iv.addWidget(lbl_voxel.native)
    iv.addWidget(_hline())

    # Export controls
    exp_folder_btn = PushButton(text="Export…")
    exp_format = ComboBox(label="format", choices=[".npy", ".tif", ".raw"], value=".npy")
    exp_use_orig = CheckBox(text="use original (ignore crop)", value=False)
    exp_dtype = ComboBox(label="raw dtype", choices=[d for d in _DTYPE_CHOICES if d != "auto"], value="uint16")
    exp_endian = ComboBox(label="raw endianness", choices=["little","big"], value="little")
    iv.addWidget(
        Container(widgets=[exp_format, exp_dtype, exp_endian], layout="horizontal", labels=True).native
    )
    iv.addWidget(
        Container(widgets=[exp_use_orig, exp_folder_btn], layout="horizontal", labels=False).native
    )

    # ---- Assemble tabs ----
    load_box = Container(
        widgets=[
            Label(value="Load RAW"),
            gui,
            _mgui_hline(),
            Label(value="Compare"),
            Container(
                widgets=[grid_enable, grid_cols, grid_stride, sync_contrast, btn_apply_grid],
                layout="horizontal",
                labels=False,
            ),
            Container(
                widgets=[btn_add_many, btn_add_folder],
                layout="horizontal",
                labels=False,
            ),
            _mgui_hline(),
            Container(
                widgets=[btn_match_sizes, Label(value="|"), name_edit, btn_rename_active],
                layout="horizontal",
                labels=False,
            ),
        ],
        layout="vertical",
        labels=False,
    )

    crop_box = Container(
        widgets=[
            crop_enable,
            z_range, y_range, x_range,
            Container(widgets=[chk_apply_all, btn_reset_crop], layout="horizontal", labels=False),
            _mgui_hline(),
            Label(value="Transforms (applied to current data view – non-destructive):"),
            Container(widgets=[btn_flip_z, btn_flip_y, btn_flip_x], layout="horizontal", labels=False),
            Container(widgets=[btn_rot90_yx, btn_transpose_last], layout="horizontal", labels=False),
        ],
        layout="vertical",
        labels=False,
    )

    tabs = QTabWidget()
    tabs.addTab(load_box.native, "Load")
    tabs.addTab(crop_box.native, "Crop & Tools")
    tabs.addTab(plots_widget, "Quick Plot")
    tabs.addTab(info_widget, "Info & Export")

    panel = QWidget()
    pl = QVBoxLayout(panel)
    pl.setContentsMargins(8, 8, 8, 8)
    pl.setSpacing(6)
    pl.addWidget(tabs)

    scroll = QScrollArea()
    scroll.setWidget(panel)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    wrapper = QWidget()
    wl = QVBoxLayout(wrapper)
    wl.setContentsMargins(0, 0, 0, 0)
    wl.addWidget(scroll)
    wrapper.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    # Clamp width so controls never force a horizontal scrollbar
    MAXW = int(settings.dock_max_width)

    def _apply_caps():
        _cap_width(
            MAXW,
            gui,
            grid_enable, grid_cols, grid_stride, btn_apply_grid, sync_contrast,
            btn_add_many, btn_add_folder, btn_match_sizes, btn_rename_active, name_edit,
            crop_enable, z_range, y_range, x_range, btn_reset_crop, chk_apply_all,
            btn_flip_z, btn_flip_y, btn_flip_x, btn_rot90_yx, btn_transpose_last,
            exp_format, exp_dtype, exp_endian, exp_use_orig, exp_folder_btn,
        )
        if HAVE_MPL and quick_plot:
            _cap_width(MAXW,
                       quick_plot.pick_layer, quick_plot.bins, quick_plot.clip,
                       quick_plot.plot_btn, quick_plot.otsu_btn,
                       quick_plot.invert_chk, quick_plot.mask_name_edit)
        wrapper.setMinimumWidth(MAXW + 8)
        wrapper.setMaximumWidth(MAXW + 14)

    _apply_caps()

    # ---- Connect controllers ----
    grid = GridManager(grid_enable, grid_cols, grid_stride, btn_apply_grid, sync_contrast)
    grid.connect()

    crop = CropController(crop_enable, z_range, y_range, x_range, btn_reset_crop, chk_apply_all)
    crop.connect()

    # ---- Match sizes logic ----
    def _match_sizes():
        v = current_viewer()
        if not v:
            return
        imgs = _iter_images(v)
        if len(imgs) < 2:
            show_warning("Load 2+ image layers first.")
            return
        Z = min(_last_zyx_shape(np.asarray(l.data))[0] for l in imgs)
        Y = min(_last_zyx_shape(np.asarray(l.data))[1] for l in imgs)
        X = min(_last_zyx_shape(np.asarray(l.data))[2] for l in imgs)
        for l in imgs:
            a = np.asarray(l.data)
            l.data = _slice_last_zyx(a, 0, Z, 0, Y, 0, X)
        show_info(f"Cropped all to {Z}×{Y}×{X}.")
        if bool(sync_contrast.value) and imgs:
            cl = tuple(map(float, imgs[0].contrast_limits))
            for l in imgs:
                l.contrast_limits = cl
            show_info("Contrast synced.")

    btn_match_sizes.clicked.connect(_match_sizes)

    # ---- Rename active ----
    def _rename_active():
        L, _ = _active_image()
        new = name_edit.value.strip()
        if not L:
            show_warning("Pick an image layer first.")
            return
        if not new:
            show_warning("Enter a new name.")
            return
        v = current_viewer()
        if v:
            if any(l.name == new for l in v.layers):
                show_warning(f"A layer named '{new}' already exists.")
                return
        L.name = new
        show_info(f"Renamed layer to '{new}'.")

    btn_rename_active.clicked.connect(_rename_active)

    # ---- Add files / folder ----
    btn_add_many.clicked.connect(lambda: _add_many_files(settings))
    btn_add_folder.clicked.connect(lambda: _add_from_folder(settings))

    # ---- Layer events ----
    if current_viewer():
        v = current_viewer()
        v.layers.events.inserted.connect(lambda *_: (grid.auto_apply_on_layers(), quick_plot and quick_plot.on_layers_changed(), _refresh_info()))
        v.layers.events.removed.connect(lambda *_: (grid.auto_apply_on_layers(), quick_plot and quick_plot.on_layers_changed(), _refresh_info()))
        v.layers.selection.events.changed.connect(lambda *_: (crop.on_layer_change(), _refresh_info()))

    # ---- Simple transform buttons ----
    def _transform_active(fn, desc: str):
        L, _ = _active_image()
        if not L:
            show_warning("Pick an image layer.")
            return
        a = np.asarray(L.data)
        try:
            out = fn(a)
        except Exception as e:
            show_error(f"{desc} failed: {e}")
            return
        L.data = out
        show_info(f"{desc} applied.")
        crop.on_layer_change()
        _refresh_info()

    btn_flip_z.clicked.connect(lambda: _transform_active(
        lambda a: a[::-1,...] if a.ndim>=3 else a,
        "Flip Z",
    ))
    btn_flip_y.clicked.connect(lambda: _transform_active(
        lambda a: a[..., ::-1, :] if a.ndim>=2 else a,
        "Flip Y",
    ))
    btn_flip_x.clicked.connect(lambda: _transform_active(
        lambda a: a[..., ::-1] if a.ndim>=1 else a,
        "Flip X",
    ))
    btn_rot90_yx.clicked.connect(lambda: _transform_active(
        lambda a: np.rot90(a, k=1, axes=(-2,-1)) if a.ndim>=2 else a,
        "Rotate 90° (Y↔X)",
    ))
    btn_transpose_last.clicked.connect(lambda: _transform_active(
        lambda a: np.transpose(a, axes=(*range(a.ndim-2), a.ndim-1, a.ndim-2)) if a.ndim>=2 else a,
        "Transpose last 2 dims",
    ))

    # ---- Info/Inspector & Export wiring ----
    def _refresh_info():
        L, _ = _active_image()
        if not L:
            lbl_active_name.value = "—"
            lbl_dtype.value = "—"
            lbl_shape.value = "—"
            lbl_stats.value = "—"
            lbl_voxel.value = "—"
            return
        a = np.asarray(L.data)
        lbl_active_name.value = f"{L.name}"
        lbl_dtype.value = f"{a.dtype}"
        lbl_shape.value = f"{tuple(a.shape)}"
        st = _array_stats(a)
        lbl_stats.value = f"min={st['min']:.6g}  max={st['max']:.6g}  mean={st['mean']:.6g}  std={st['std']:.6g}"
        vs = _get_voxel_size(L)
        lbl_voxel.value = f"{vs}" if vs else "—"

    def _export_dialog():
        v = current_viewer()
        if not v:
            return
        L, _ = _active_image()
        if not L:
            show_warning("Pick an image layer to export.")
            return
        fmt = exp_format.value
        start_dir = settings.last_dir if settings.last_dir and Path(settings.last_dir).exists() else ""
        filters = {
            ".npy": "NumPy (*.npy)",
            ".tif": "TIFF (*.tif *.tiff)",
            ".raw": "RAW (*.raw *.bin)",
        }
        path_str, _ = QFileDialog.getSaveFileName(
            None, "Export layer", start_dir, filters.get(fmt, "All (*)")
        )
        if not path_str:
            return
        path = Path(path_str)
        # ensure suffix
        if path.suffix.lower() not in {fmt, ".tiff"}:
            if fmt == ".tif" and path.suffix.lower() == ".tiff":
                pass
            else:
                path = path.with_suffix(fmt)

        a = np.asarray(L.metadata.get("_orig_data", L.data)) if bool(exp_use_orig.value) else np.asarray(L.data)
        try:
            if fmt == ".npy":
                np.save(path, a)
            elif fmt == ".tif":
                if not HAVE_TIFF:
                    raise RuntimeError("tifffile is not installed.")
                vs = _get_voxel_size(L)
                metadata = {}
                if vs:
                    metadata["spacing"] = vs[0]
                    metadata["unit"] = "unknown"
                    metadata["axes"] = "ZYX" if a.ndim>=3 else "YX"
                _tifffile.imwrite(str(path), a, metadata=metadata)  # type: ignore
            elif fmt == ".raw":
                _export_raw(path, a, _numpy_dtype_from_string(exp_dtype.value), exp_endian.value)  # type: ignore[arg-type]
            else:
                raise RuntimeError(f"Unsupported export format '{fmt}'.")
        except Exception as e:
            show_error(f"Export failed: {e}")
            return
        settings.last_dir = str(path.parent)
        settings.save()
        show_info(f"Exported to: {path}")

    exp_folder_btn.clicked.connect(_export_dialog)

    # ---- Persist a few settings when dock closes ----
    def _save_settings():
        settings.grid_enable = bool(grid_enable.value)
        settings.grid_cols = int(grid_cols.value)
        settings.grid_stride = int(grid_stride.value)
        if HAVE_MPL and quick_plot:
            settings.default_bins = int(quick_plot.bins.value)
            settings.clip_1_99 = bool(quick_plot.clip.value)
        try:
            settings.default_dtype = gui.dtype.value
            settings.default_byte_order = gui.byte_order.value
            settings.prefer_memmap = bool(gui.memmap.value)
        except Exception:
            pass
        settings.save()

    try:
        wrapper.destroyed.connect(lambda *_: _save_settings())  # type: ignore[attr-defined]
    except Exception:
        pass

    # ---- Initial state ----
    grid.apply()
    crop.on_layer_change()
    if quick_plot:
        quick_plot.on_layers_changed()
    _refresh_info()

    return wrapper

# ----------------------------------------------------------------------------
# napari plugin entry point
# ----------------------------------------------------------------------------

def napari_experimental_provide_dock_widget():
    """napari discovers this to add our dock widget."""
    return [raw_loader_widget]
