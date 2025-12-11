# napari_pypore3d/functions.py — r7
# ----------------------------------
# Function page:
# - Layer selector: <active image> or any image layer by name
# - Normal (NumPy) functions dropdown + Run button
# - PyPore3D functions dropdown + Run button
# - Output mode: new layer vs overwrite target
# - Console showing success / warnings / errors
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, List
import numpy as np
from magicgui.widgets import Container, ComboBox, PushButton, Label
try:
    from magicgui.widgets import TextEdit
except Exception:
    TextEdit = None  # fallback later

from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_info, show_warning, show_error
import ctypes


try:
    # IMPORTANT: use the *Python wrapper* module, not the low-level C one
    from pypore3d import p3dFiltPy as p3d  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    p3d = None


# <<< NEW: p3dBlob (C helpers like invert_vol) ------------------------
try:
    from pypore3d import p3dBlob as p3d_blob  # type: ignore
except Exception:  # pragma: no cover
    p3d_blob = None


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


def _add_result_layer(data: np.ndarray, base_name: str, suffix: str) -> None:
    """Add a new image layer with a derived name."""
    v = current_viewer()
    if v is None:
        return
    safe_suffix = suffix.replace(" ", "_")
    name = f"{base_name}_{safe_suffix}"
    v.add_image(data, name=name)
# --------------------------------------------------------------------- function entries



@dataclass
class FunctionEntry:
    label: str
    runner: Callable[[NapariImage], Optional[np.ndarray]]
    tooltip: str = ""


# -------------------------------------------------------- normal functions ----

def _fn_normal_normalize_01(layer: NapariImage) -> np.ndarray:
    """Normalize intensities to [0, 1] float32."""
    arr = np.asarray(layer.data, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx > mn:
        out = (arr - mn) / (mx - mn)
    else:
        out = np.zeros_like(arr, dtype=np.float32)
    return out


def _fn_normal_normalize_uint8(layer: NapariImage) -> np.ndarray:
    """Normalize intensities to uint8 [0, 255]."""
    arr = np.asarray(layer.data, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def _fn_normal_invert(layer: NapariImage) -> np.ndarray:
    """
    Simple contrast-preserving invert.

    For integer types: max - (x - min)
    For float: max + min - x
    """
    arr = np.asarray(layer.data)
    if np.issubdtype(arr.dtype, np.floating):
        mn = float(arr.min())
        mx = float(arr.max())
        return (mx + mn) - arr
    else:
        mn = int(arr.min())
        mx = int(arr.max())
        tmp = (mx - (arr.astype(np.int64) - mn)).astype(arr.dtype)
        return tmp


def _fn_normal_clip_1_99(layer: NapariImage) -> np.ndarray:
    """Clip intensities to [1%, 99%] percentiles."""
    arr = np.asarray(layer.data, dtype=np.float32)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if hi <= lo:
        return arr
    return np.clip(arr, lo, hi)


def _fn_normal_mean_threshold(layer: NapariImage) -> np.ndarray:
    """Binary threshold at mean value (output uint8 0/255)."""
    arr = np.asarray(layer.data, dtype=np.float32)
    thr = float(arr.mean())
    out = np.zeros_like(arr, dtype=np.uint8)
    out[arr >= thr] = 255
    return out


def _fn_normal_cast_float32(layer: NapariImage) -> np.ndarray:
    """Cast data to float32 (no scaling)."""
    return np.asarray(layer.data, dtype=np.float32)


def _fn_normal_cast_uint8(layer: NapariImage) -> np.ndarray:
    """Cast via clipping to uint8 (no normalization)."""
    arr = np.asarray(layer.data, dtype=np.float32)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


NORMAL_FUNCTIONS: Dict[str, FunctionEntry] = {
    "Normalize [0, 1] (float32)": FunctionEntry(
        label="Normalize [0, 1] (float32)",
        runner=_fn_normal_normalize_01,
        tooltip="Scale current image to floats in [0, 1].",
    ),
    "Normalize [0, 255] (uint8)": FunctionEntry(
        label="Normalize [0, 255] (uint8)",
        runner=_fn_normal_normalize_uint8,
        tooltip="Scale image then cast to uint8.",
    ),
    "Invert intensities": FunctionEntry(
        label="Invert intensities",
        runner=_fn_normal_invert,
        tooltip="Contrast-preserving inversion of the image.",
    ),
    "Clip 1–99 percentiles": FunctionEntry(
        label="Clip 1–99 percentiles",
        runner=_fn_normal_clip_1_99,
        tooltip="Clip values to the central [1%, 99%] range.",
    ),
    "Mean threshold (binary uint8)": FunctionEntry(
        label="Mean threshold (binary uint8)",
        runner=_fn_normal_mean_threshold,
        tooltip="Threshold at mean; output is 0 / 255 uint8 mask.",
    ),
    "Cast to float32": FunctionEntry(
        label="Cast to float32",
        runner=_fn_normal_cast_float32,
        tooltip="Just change dtype to float32.",
    ),
    "Cast to uint8 (clipped)": FunctionEntry(
        label="Cast to uint8 (clipped)",
        runner=_fn_normal_cast_uint8,
        tooltip="Clip range to [0, 255] and cast to uint8.",
    ),
}


# ----------------------------------------------------------- PyPore3D funcs ---

def _p3d_check_import() -> None:
    if p3d is None:
        raise RuntimeError(
            "PyPore3D (p3dFilt) could not be imported. "
            "Check installation / PYTHONPATH."
        )

# <<< NEW: blob import check ----------------------------------------------
def _p3d_blob_check_import() -> None:
    if p3d_blob is None:
        raise RuntimeError(
            "PyPore3D (p3dBlob) could not be imported. "
            "Check installation / PYTHONPATH."
        )

def _fn_p3d_median_uint8(layer, radius: int):
    from pypore3d import p3dFiltPy as F

    fn = getattr(F, "py_p3dMedianFilter8", None)
    if fn is None:
        raise RuntimeError("py_p3dMedianFilter8 not found!")

    arr = np.asarray(layer.data)
    if arr.ndim != 3:
        raise ValueError("Need a 3D volume")

    data = np.ascontiguousarray(arr, dtype=np.uint8)

    z, y, x = data.shape

    # CALL USING KEYWORD ARGUMENTS
    out = fn(data, x, y, z, width=0, radius=radius)

    return np.asarray(out, dtype=np.uint8)

def _fn_p3d_median_uint8_r1(layer: NapariImage) -> Optional[np.ndarray]:
    _p3d_check_import()

    fn = getattr(p3d, "py_p3dMedianFilter8", None)
    if fn is None:
        raise RuntimeError("py_p3dMedianFilter8 not found in p3dFiltPy.")

    arr = np.asarray(layer.data)
    if arr.ndim != 3:
        raise ValueError(f"Need 3D data, got {arr.shape}")

    data = np.ascontiguousarray(arr, dtype=np.uint8)

    z, y, x = data.shape

    # Correct signature: (image_data, dimx, dimy, dimz, width, radius)
    out = fn(data, x, y, z, 0, 1)

    return np.asarray(out, dtype=np.uint8)

def _fn_p3d_median_uint8_r2(layer: NapariImage) -> Optional[np.ndarray]:
    _p3d_check_import()

    fn = getattr(p3d, "py_p3dMedianFilter8", None)
    if fn is None:
        raise RuntimeError("py_p3dMedianFilter8 not found in p3dFiltPy.")

    arr = np.asarray(layer.data)
    if arr.ndim != 3:
        raise ValueError(f"Need 3D data, got {arr.shape}")

    data = np.ascontiguousarray(arr, dtype=np.uint8)

    z, y, x = data.shape

    out = fn(data, x, y, z, 0, 2)

    return np.asarray(out, dtype=np.uint8)







def _fn_p3d_import_test(layer: NapariImage) -> Optional[np.ndarray]:
    """
    Just check that PyPore3D imports; no array changes.
    """
    _p3d_check_import()
    show_info("PyPore3D (p3dFilt) import OK. This is a no-op test.")
    return None


# <<< NEW: BLOB HELPERS (invert via C) -------------------------------------

def _fn_p3d_blob_invert_uint8(layer: NapariImage) -> Optional[np.ndarray]:
    """
    Windows PyPore3D invert_vol (uint8).
    Signature confirmed via inspect:
        invert_vol(in_im, dimx, dimy, dimz)
    """
    _p3d_blob_check_import()

    fn = getattr(p3d_blob, "invert_vol", None)
    if fn is None:
        raise RuntimeError("p3dBlob.invert_vol not found")

    arr = np.asarray(layer.data)
    if arr.ndim != 3:
        raise ValueError("invert_vol expects a 3D volume")

    # MUST BE uint8 + C-contiguous → fixes unsigned char * error
    data = np.ascontiguousarray(arr, dtype=np.uint8)

    # dims
    z, y, x = data.shape  # (z,y,x)
    fn(data, x, y, z)      # exact signature for your build

    return data


def _fn_p3d_blob_invert_uint16(layer: NapariImage) -> Optional[np.ndarray]:
    _p3d_blob_check_import()

    fn = getattr(p3d_blob, "invert_vol_16", None)
    if fn is None:
        raise RuntimeError("p3dBlob.invert_vol_16 not found.")

    arr = np.asarray(layer.data)
    if arr.ndim != 3:
        raise ValueError("invert_vol_16 expects a 3D volume.")

    data = np.ascontiguousarray(arr, dtype=np.uint16)

    z, y, x = data.shape
    fn(data, x, y, z)

    return data



P3D_FUNCTIONS: Dict[str, FunctionEntry] = {
    "PyPore3D import test (no-op)": FunctionEntry(
        label="PyPore3D import test (no-op)",
        runner=_fn_p3d_import_test,
        tooltip="Check that PyPore3D (p3dFilt) imports correctly.",
    ),
    "Median 3D (uint8, r=1)": FunctionEntry(
        label="Median 3D (uint8, r=1)",
        runner=_fn_p3d_median_uint8_r1,
        tooltip="Run p3dMedianFilter3D_8 with radius=1.",
    ),
    "Median 3D (uint8, r=2)": FunctionEntry(
        label="Median 3D (uint8, r=2)",
        runner=_fn_p3d_median_uint8_r2,
        tooltip="Run p3dMedianFilter3D_8 with radius=2.",
    ),
    # <<< NEW: BLOB ENTRIES ------------------------------------------------
    "Blob invert (uint8, C helper)": FunctionEntry(
        label="Blob invert (uint8, C helper)",
        runner=_fn_p3d_blob_invert_uint8,
        tooltip="Use p3dBlob.invert_vol on a 3D uint8 volume (in-place on a copy).",
    ),
    "Blob invert (uint16, C helper)": FunctionEntry(
        label="Blob invert (uint16, C helper)",
        runner=_fn_p3d_blob_invert_uint16,
        tooltip="Use p3dBlob.invert_vol_16 on a 3D uint16 volume (in-place on a copy).",
    ),
}


# --------------------------------------------------------------- main widget --

def functions_widget() -> Container:
    """
    Function page: layer selector + two function dropdowns + output mode + console.

    - Layer dropdown: <active image> plus all image layer names.
    - Normal functions: pure NumPy operations.
    - PyPore3D functions: call into pypore3d (if installed).
    - Output mode: new layer vs overwrite target layer.
    """

    # ----------------- labels -----------------
    header_label = Label(
        label="",
        value="Target layer: <active image> or pick by name.",
    )

    # ----------------- layer selector -----------------
    layer_combo = ComboBox(
        label="Layer",
        choices=["<active image>"],
        value="<active image>",
    )

    refresh_layers_btn = PushButton(text="Refresh layers")

    def _refresh_layers(*_):
        layers = _get_image_layers()
        names = ["<active image>"] + [L.name for L in layers]
        current = layer_combo.value if layer_combo.value in names else "<active image>"
        layer_combo.choices = names
        layer_combo.value = current

    refresh_layers_btn.changed.connect(_refresh_layers)
    _refresh_layers()  # initial fill

    # ----------------- function selectors -----------------
    normal_label = Label(label="", value="Normal functions (NumPy):")

    normal_combo = ComboBox(
        label="",
        choices=list(NORMAL_FUNCTIONS.keys()),
        value=list(NORMAL_FUNCTIONS.keys())[0] if NORMAL_FUNCTIONS else None,
    )

    run_normal_btn = PushButton(text="Run normal")

    p3d_label = Label(label="", value="PyPore3D functions:")

    p3d_combo = ComboBox(
        label="",
        choices=list(P3D_FUNCTIONS.keys()),
        value=list(P3D_FUNCTIONS.keys())[0] if P3D_FUNCTIONS else None,
    )

    run_p3d_btn = PushButton(text="Run PyPore3D")

    # ----------------- output mode -----------------
    output_label = Label(label="", value="Output mode:")
    output_combo = ComboBox(
        label="",
        choices=["new layer", "overwrite target"],
        value="new layer",
    )

    console_label = Label(label="", value="Output console:")

    # ----------------- console widget -----------------
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
            try:
                console.value = new_val
            except Exception:
                pass

    # ----------------- button callbacks -----------------

    def _current_overwrite_flag() -> bool:
        mode = str(output_combo.value or "new layer").lower()
        return mode.startswith("overwrite")

    def _reset_contrast(layer: NapariImage, data: np.ndarray) -> None:
        """Try to reset contrast limits after overwriting data."""
        try:
            # napari 0.5+ has this
            layer.reset_contrast_limits()
            return
        except Exception:
            pass

        # fallback: set from data
        try:
            lo = float(np.nanmin(data))
            hi = float(np.nanmax(data))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                layer.contrast_limits = (lo, hi)
        except Exception:
            pass

    def _run_normal(evt=None) -> None:
        _refresh_layers()

        name = layer_combo.value
        layer = _pick_layer_by_name(name)
        if layer is None:
            imgs = _get_image_layers()
            msg = "No target Image layer. Load an image first."
            if imgs:
                msg += " (Unexpected: images exist but could not pick one.)"
            show_warning(msg)
            _append_console(f"[warn] {msg}")
            return

        key = normal_combo.value
        if not key:
            msg = "No normal function selected."
            show_warning(msg)
            _append_console(f"[warn] {msg}")
            return

        entry = NORMAL_FUNCTIONS.get(key)
        if entry is None:
            msg = f"Normal function '{key}' not found."
            show_error(msg)
            _append_console(f"[error] {msg}")
            return

        overwrite = _current_overwrite_flag()

        _append_console(f"→ Running NORMAL: {entry.label} on '{layer.name}' ({'overwrite' if overwrite else 'new layer'})...")
        try:
            result = entry.runner(layer)
        except Exception as e:  # noqa: BLE001
            msg = f"Normal function '{entry.label}' failed: {e!r}"
            show_error(msg)
            _append_console(f"[error] {msg}")
            return

        if isinstance(result, np.ndarray):
            if overwrite:
                layer.data = result
                _reset_contrast(layer, result)
                msg_ok = f"Normal function '{entry.label}' finished. Target layer overwritten."
            else:
                _add_result_layer(result, layer.name, entry.label)
                msg_ok = f"Normal function '{entry.label}' finished. Result layer added."
            show_info(msg_ok)
            _append_console(f"[ok] {msg_ok}")
        else:
            msg_ok = f"Normal function '{entry.label}' finished (no output layer)."
            show_info(msg_ok)
            _append_console(f"[ok] {msg_ok}")

    def _run_p3d(evt=None) -> None:
        _refresh_layers()

        name = layer_combo.value
        layer = _pick_layer_by_name(name)
        if layer is None:
            imgs = _get_image_layers()
            msg = "No target Image layer. Load an image first."
            if imgs:
                msg += " (Unexpected: images exist but could not pick one.)"
            show_warning(msg)
            _append_console(f"[warn] {msg}")
            return

        key = p3d_combo.value
        if not key:
            msg = "No PyPore3D function selected."
            show_warning(msg)
            _append_console(f"[warn] {msg}")
            return

        entry = P3D_FUNCTIONS.get(key)
        if entry is None:
            msg = f"PyPore3D function '{key}' not found."
            show_error(msg)
            _append_console(f"[error] {msg}")
            return

        overwrite = _current_overwrite_flag()

        _append_console(f"→ Running P3D: {entry.label} on '{layer.name}' ({'overwrite' if overwrite else 'new layer'})...")
        try:
            result = entry.runner(layer)
        except Exception as e:  # noqa: BLE001
            msg = f"PyPore3D function '{entry.label}' failed: {e!r}"
            show_error(msg)
            _append_console(f"[error] {msg}")
            return

        if isinstance(result, np.ndarray):
            if overwrite:
                layer.data = result
                _reset_contrast(layer, result)
                msg_ok = f"PyPore3D function '{entry.label}' finished. Target layer overwritten."
            else:
                _add_result_layer(result, layer.name, entry.label)
                msg_ok = f"PyPore3D function '{entry.label}' finished. Result layer added."
            show_info(msg_ok)
            _append_console(f"[ok] {msg_ok}")
        else:
            msg_ok = f"PyPore3D function '{entry.label}' finished (no output layer)."
            show_info(msg_ok)
            _append_console(f"[ok] {msg_ok}")

    run_normal_btn.changed.connect(_run_normal)
    run_p3d_btn.changed.connect(_run_p3d)

    # ----------------- assemble container -----------------
    root = Container(
        widgets=[
            header_label,
            layer_combo,
            refresh_layers_btn,
            normal_label,
            normal_combo,
            run_normal_btn,
            p3d_label,
            p3d_combo,
            run_p3d_btn,
            output_label,
            output_combo,
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
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setSpacing(8)
    except Exception:
        pass

    return root
