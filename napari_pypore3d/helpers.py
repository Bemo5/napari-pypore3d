# napari_pypore3d/helpers.py
# ---------------------------------------------------------------------
# Shared utilities used across the plugin (no UI/tab code here).
# - Debug + small UI helpers
# - Persisted AppSettings (JSON in user home)
# - Napari layer helpers (images/labels/active/unique names)
# - Array shape/slicing helpers (last Z,Y,X)
# - Filename-based shape hints (AxBxC, N^3)
# - Sorting helpers (natural sort)
# - Voxel metadata helpers
# - Basic stats, RAW little-endian exporter
# - SAFE PyPore3D function wiring (median/mean 3D uint8)

from __future__ import annotations
import ctypes
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pypore3d import p3dFilt as _P3F
import numpy as np
from magicgui.widgets import Label
from napari import current_viewer
from napari.layers import Image as NapariImage, Labels
from qtpy.QtWidgets import QFrame

# ---------------------------------------------------------------------
# Debug + tiny UI helpers
# ---------------------------------------------------------------------
def debug(msg: str) -> None:
    """Print debug logs when NPP3D_DEBUG=1 is set in the environment."""
    if os.environ.get("NPP3D_DEBUG", "0").strip() == "1":
        print(f"[napari-pypore3d] {msg}")


def mgui_hline() -> Label:
    """A thin horizontal line separator usable inside magicgui Containers."""
    L = Label(value="")
    try:
        L.native.setFrameShape(QFrame.HLine)
        L.native.setFrameShadow(QFrame.Sunken)
    except Exception:
        # In headless or non-Qt contexts, just return the label quietly.
        pass
    return L


def _as_qwidget(obj):
    """Return a QWidget from magicgui/our objects or None."""
    if obj is None:
        return None
    # magicgui widgets
    if hasattr(obj, "native"):
        return obj.native
    # our objects that expose .as_qwidget()
    if hasattr(obj, "as_qwidget"):
        try:
            return obj.as_qwidget()
        except Exception:
            return None
    # already a QWidget?
    return obj


def cap_width(max_width: int, **widgets) -> None:
    """
    Set setMaximumWidth(max_width) on any passed widgets.

    Accepts raw QWidgets, magicgui widgets (with .native),
    or our objects exposing .as_qwidget().

    Usage:
        cap_width(860, tabs=tabs, wrapper=wrapper, info_widget=info_widget)
    """
    mw = int(max(1, int(max_width)))
    for _name, obj in widgets.items():
        w = _as_qwidget(obj)
        if w is None:
            continue
        try:
            w.setMaximumWidth(mw)
        except Exception:
            # some objects may not be QWidget; ignore quietly
            pass


# ---------------------------------------------------------------------
# Persisted settings
# ---------------------------------------------------------------------
SETTINGS_FILE = pathlib.Path.home() / ".napari_pypore3d_settings.json"


@dataclass
class AppSettings:
    dock_max_width: int = 860
    last_dir: str = ""
    default_dtype: str = "auto"
    default_bins: int = 256
    clip_1_99: bool = True
    prefer_memmap: bool = True

    @classmethod
    def load(cls) -> "AppSettings":
        try:
            if SETTINGS_FILE.exists():
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                use = {k: data.get(k) for k in cls.__dataclass_fields__.keys()}
                obj = cls(**use)
            else:
                obj = cls()
        except Exception:
            obj = cls()

        if obj.default_dtype not in [
            "auto", "uint8", "int8", "uint16", "int16",
            "uint32", "int32", "float32", "float64",
        ]:
            obj.default_dtype = "auto"

        if not isinstance(obj.default_bins, int) or obj.default_bins < 8:
            obj.default_bins = 256
        return obj

    def save(self) -> None:
        try:
            SETTINGS_FILE.write_text(
                json.dumps(self.__dict__, indent=2),
                encoding="utf-8",
            )
        except Exception:
            debug("settings save failed")


# ---------------------------------------------------------------------
# Napari layer helpers
# ---------------------------------------------------------------------
def iter_images(v) -> List[NapariImage]:
    return [l for l in (v.layers if v else []) if isinstance(l, NapariImage)]


def iter_labels(v) -> List[Labels]:
    return [l for l in (v.layers if v else []) if isinstance(l, Labels)]


def active_image() -> Tuple[Optional[NapariImage], Any]:
    v = current_viewer()
    if v is None:
        return None, None
    L = v.layers.selection.active
    return (L, v) if isinstance(L, NapariImage) else (None, v)


def unique_layer_name(base: str) -> str:
    v = current_viewer()
    if not v:
        return base
    names = {l.name for l in v.layers}
    if base not in names:
        return base
    i = 2
    while f"{base} ({i})" in names:
        i += 1
    return f"{base} ({i})"


def with_voxel(layer: NapariImage, vs: Optional[Tuple[float, float, float]]) -> None:
    """Attach voxel size (z,y,x) to layer.metadata['_voxel_size'] if provided."""
    if vs:
        try:
            layer.metadata["_voxel_size"] = tuple(float(x) for x in vs)
        except Exception:
            pass


def voxel(layer: NapariImage) -> Optional[Tuple[float, float, float]]:
    """Read voxel size (z,y,x) from metadata if present."""
    vs = layer.metadata.get("_voxel_size")
    if isinstance(vs, (tuple, list)) and len(vs) == 3:
        try:
            return (float(vs[0]), float(vs[1]), float(vs[2]))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------
# Array shape & slicing helpers (operate on the last Z,Y,X axes)
# ---------------------------------------------------------------------
def last_zyx(a: np.ndarray) -> Tuple[int, int, int]:
    if a.ndim == 2:
        y, x = map(int, a.shape[-2:])
        return (1, y, x)
    if a.ndim >= 3:
        z, y, x = map(int, a.shape[-3:])
        return (z, y, x)
    return (1, 1, 1)


def slice_last_zyx(
    a: np.ndarray,
    zs: int, ze: int,
    ys: int, ye: int,
    xs: int, xe: int,
) -> np.ndarray:
    if a.ndim == 2:
        return a[ys:ye, xs:xe]
    prefix = (slice(None),) * (a.ndim - 3)
    return a[prefix + (slice(zs, ze), slice(ys, ye), slice(xs, xe))]


def shape_last_zyx_of(arr: np.ndarray) -> Tuple[int, int, int]:
    if arr.ndim == 2:
        H, W = map(int, arr.shape[-2:])
        return (1, H, W)
    if arr.ndim >= 3:
        Z, Y, X = map(int, arr.shape[-3:])
        return (Z, Y, X)
    return (1, 1, 1)


def current_min_zyx(v) -> Optional[Tuple[int, int, int]]:
    imgs = iter_images(v) if v else []
    if not imgs:
        return None
    sizes = []
    for L in imgs:
        arr = np.asarray(L.metadata.get("_orig_full", L.data))
        sizes.append(last_zyx(arr))
    return (
        min(s[0] for s in sizes),
        min(s[1] for s in sizes),
        min(s[2] for s in sizes),
    )


def center_crop_indices(
    full: Tuple[int, int, int],
    target: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    fz, fy, fx = full
    tz, ty, tx = target
    z0 = max(0, (fz - tz) // 2)
    y0 = max(0, (fy - ty) // 2)
    x0 = max(0, (fx - tx) // 2)
    return slice(z0, z0 + tz), slice(y0, y0 + ty), slice(x0, x0 + tx)


# ---------------------------------------------------------------------
# Filename hints & sorting
# ---------------------------------------------------------------------
def best_trip_from_name(name: str) -> Optional[Tuple[int, int, int]]:
    """Return (Z,Y,X) if the filename contains AxBxC or N^3 / 'cubed' hints."""
    m = re.findall(r"(\d+)[xX](\d+)[xX](\d+)", name)
    if m:
        return max(((int(a), int(b), int(c)) for a, b, c in m),
                   key=lambda t: t[0] * t[1] * t[2])
    m2 = re.search(r"(\d+)\s*(?:\^?\s*3|³|cube(?:d)?)", name, flags=re.IGNORECASE)
    if m2:
        n = int(m2.group(1))
        return (n, n, n)
    return None


def natural_sort_key(s: str) -> List[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ---------------------------------------------------------------------
# Stats + RAW exporter
# ---------------------------------------------------------------------
def array_stats(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a)
    if a.size == 0:
        return dict(min=np.nan, max=np.nan, mean=np.nan, std=np.nan)
    return dict(
        min=float(np.nanmin(a)),
        max=float(np.nanmax(a)),
        mean=float(np.nanmean(a)),
        std=float(np.nanstd(a)),
    )


def export_raw_little(path: pathlib.Path, arr: np.ndarray, dtype: np.dtype) -> None:
    """Write array to RAW with little-endian order (no header)."""
    a = np.asarray(arr)
    if a.dtype != dtype:
        a = a.astype(dtype, copy=False)
    if dtype.itemsize > 1:
        a = a.byteswap().newbyteorder("<")
    with open(path, "wb") as f:
        f.write(a.tobytes(order="C"))


def _caption_text(img: NapariImage) -> str:
    """Build a short HUD line for the active image."""
    import numpy as _np
    a = _np.asarray(img.data)
    shape = tuple(a.shape)
    dtype = getattr(a.dtype, "name", str(a.dtype))
    ztxt = ""
    try:
        v = current_viewer()
        if a.ndim >= 3:
            ztxt = f" | z={int(v.dims.current_step[0])}"
    except Exception:
        pass
    try:
        lo, hi = (
            img.contrast_limits if img.contrast_limits is not None else (None, None)
        )
        cl = f" | CL=[{int(lo)},{int(hi)}]" if (lo is not None and hi is not None) else ""
    except Exception:
        cl = ""
    return f"{img.name}{ztxt} | shape={shape} | dtype={dtype}{cl}"


def ensure_caption(img: NapariImage, position: str = "bottom") -> None:
    """
    Ensure a single safe HUD is visible on the viewer (no extra layers).
    `position` kept for compatibility; napari's text_overlay anchors automatically.
    """
    v = current_viewer()
    if not v:
        return
    try:
        v.text_overlay.visible = True
        v.text_overlay.color = "white"
        v.text_overlay.border = "black"
        v.text_overlay.font_size = 12
        v.text_overlay.text = _caption_text(img)
    except Exception:
        pass


# ---------------------------------------------------------------------
# PyPore3D function wiring (SAFE subset only)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# PyPore3D function wiring (SAFE subset only)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# PyPore3D function wiring (SAFE subset only)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# PyPore3D function wiring (SAFE subset only)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# PyPore3D function wiring (SAFE subset only)
# ---------------------------------------------------------------------
try:  # optional: pypore3d might not be installed
    from pypore3d import _p3dFilt as _P3F  # type: ignore
except Exception:  # optional dependency
    _P3F = None  # type: ignore

# List shown in Plot Lab "p3d" dropdown.
# (nice_label, internal_key)
P3D_FUNCTION_CHOICES: List[Tuple[str, str]] = []
if _P3F is not None:
    P3D_FUNCTION_CHOICES = [
        ("Median 3D (uint8, k=3)", "p3dMedianFilter3D_8"),
        ("Mean 3D (uint8, k=3)", "p3dMeanFilter3D_8"),
    ]
else:
    # single dummy option when PyPore3D truly isn’t importable
    P3D_FUNCTION_CHOICES = [("— PyPore3D not available —", "")]

# lazy-loaded ctypes handle to the same .pyd/.dll that SWIG uses
_P3D_LIB: Optional[ctypes.CDLL] = None
_U8P = ctypes.POINTER(ctypes.c_ubyte)


def _ensure_p3d_lib() -> ctypes.CDLL:
    """Load the underlying PyPore3D C library via ctypes (once)."""
    global _P3D_LIB
    if _P3D_LIB is not None:
        return _P3D_LIB

    if _P3F is None:
        raise RuntimeError("PyPore3D not installed (_p3dFilt import failed).")

    # _P3F.__file__ points to the compiled extension (.pyd/.so)
    lib_path = getattr(_P3F, "__file__", None)
    if not lib_path:
        raise RuntimeError("Cannot locate PyPore3D binary path from _p3dFilt.")

    debug(f"Loading PyPore3D C library via ctypes: {lib_path}")
    lib = ctypes.CDLL(lib_path)

    # set up function signatures ONCE
    # void p3dMedianFilter3D_8(unsigned char *in, unsigned char *out,
    #                          unsigned int Z, unsigned int Y, unsigned int X,
    #                          unsigned char kZ, unsigned char kY, unsigned char kX);
    lib.p3dMedianFilter3D_8.argtypes = [
        _U8P, _U8P,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte,
    ]
    lib.p3dMedianFilter3D_8.restype = None

    # same for mean filter
    lib.p3dMeanFilter3D_8.argtypes = [
        _U8P, _U8P,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte,
    ]
    lib.p3dMeanFilter3D_8.restype = None

    _P3D_LIB = lib
    return lib


def apply_p3d_function(data: np.ndarray, key: str) -> np.ndarray:
    """
    Apply a *safe, pre-wired* PyPore3D function to `data` and return a new array.

    Currently we only expose:
      - p3dMedianFilter3D_8
      - p3dMeanFilter3D_8

    The function:
      • Works on the last Z,Y,X axes (3D volume)
      • Casts to uint8 as required by the C code (unsigned char*)
      • Uses kernel size 3×3×3
      • Returns a fresh numpy array
    """
    if not key:
        raise RuntimeError("No PyPore3D function selected.")

    lib = _ensure_p3d_lib()

    arr = np.asarray(data)
    if arr.ndim < 3:
        raise ValueError(f"PyPore3D: need at least 3D data, got shape {arr.shape!r}")

    # operate on the last Z,Y,X block; preserve any leading dims as-is
    z, y, x = last_zyx(arr)
    lead_shape = arr.shape[:-3]

    vol = arr.reshape((-1, z, y, x)) if lead_shape else arr.reshape(1, z, y, x)

    # currently we only support single-volume calls; take the first block
    vol0 = np.ascontiguousarray(vol[0], dtype=np.uint8)

    k = 3  # kernel size 3×3×3

    # NumPy arrays that own their buffers
    src_arr = vol0
    dst_arr = np.empty_like(src_arr, dtype=np.uint8)

    # get real C pointers (unsigned char *)
    src_ptr = src_arr.ctypes.data_as(_U8P)
    dst_ptr = dst_arr.ctypes.data_as(_U8P)

    if key == "p3dMedianFilter3D_8":
        lib.p3dMedianFilter3D_8(
            src_ptr, dst_ptr,
            ctypes.c_uint(z), ctypes.c_uint(y), ctypes.c_uint(x),
            ctypes.c_ubyte(k), ctypes.c_ubyte(k), ctypes.c_ubyte(k),
        )
    elif key == "p3dMeanFilter3D_8":
        lib.p3dMeanFilter3D_8(
            src_ptr, dst_ptr,
            ctypes.c_uint(z), ctypes.c_uint(y), ctypes.c_uint(x),
            ctypes.c_ubyte(k), ctypes.c_ubyte(k), ctypes.c_ubyte(k),
        )
    else:
        raise NotImplementedError(
            f"apply_p3d_function not implemented for key={key!r}. "
            "This key is not in the safe subset yet."
        )

    # dst_arr now holds the filtered (Z, Y, X) volume
    out = dst_arr.reshape((z, y, x))

    if lead_shape:
        # broadcast the single filtered volume over any leading dims
        out = np.broadcast_to(out, lead_shape + (z, y, x)).copy()

    return out






# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    # debug/ui
    "debug", "mgui_hline", "cap_width",
    # settings
    "AppSettings", "SETTINGS_FILE",
    # napari helpers
    "iter_images", "iter_labels", "active_image", "unique_layer_name",
    "with_voxel", "voxel",
    # array helpers
    "last_zyx", "slice_last_zyx", "shape_last_zyx_of",
    "current_min_zyx", "center_crop_indices",
    # filenames/sort
    "best_trip_from_name", "natural_sort_key",
    # stats/io
    "array_stats", "export_raw_little",
    # HUD
    "ensure_caption",
    # PyPore3D
    "P3D_FUNCTION_CHOICES", "apply_p3d_function",
]
