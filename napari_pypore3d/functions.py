# napari_pypore3d/functions.py — r15 (SAFE MODE)
# - ONLY uint8 pipeline
# - NO crashy/slow ops (no NLM/bilateral/aniso/CLAHE/hist-eq)
# - Hard guards against huge volumes
# - Uses SciPy ndimage for robust N-D filtering

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional, List, Tuple

import numpy as np

from magicgui.widgets import Container, ComboBox, PushButton, Label
try:
    from magicgui.widgets import LineEdit
except Exception:
    LineEdit = None  # type: ignore

try:
    from magicgui.widgets import TextEdit
except Exception:
    TextEdit = None  # type: ignore

from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_info, show_warning, show_error

# Optional SciPy (recommended)
try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

# Optional skimage (only for otsu + clear_border; both are typically safe)
try:
    from skimage.filters import threshold_otsu
except Exception:  # pragma: no cover
    threshold_otsu = None

try:
    from skimage.segmentation import clear_border
except Exception:  # pragma: no cover
    clear_border = None


# --------------------------------------------------------------------- types --

@dataclass
class FunctionEntry:
    label: str
    runner: Callable[[NapariImage], Optional[np.ndarray]]
    tooltip: str = ""


# ------------------------------------------------------------------- helpers --

def _v():
    return current_viewer()

def _get_image_layers() -> List[NapariImage]:
    v = _v()
    if v is None:
        return []
    return [L for L in v.layers if isinstance(L, NapariImage)]

def _pick_layer_by_name(name: Optional[str]) -> Optional[NapariImage]:
    v = _v()
    if v is None:
        return None

    imgs = [L for L in v.layers if isinstance(L, NapariImage)]
    if not imgs:
        return None

    if not name or name == "<active image>":
        lyr = v.layers.selection.active
        return lyr if isinstance(lyr, NapariImage) else imgs[0]

    for L in imgs:
        if L.name == name:
            return L

    lyr = v.layers.selection.active
    return lyr if isinstance(lyr, NapariImage) else imgs[0]

def _clone_visuals(src: NapariImage, dst: NapariImage) -> None:
    props = [
        "colormap", "contrast_limits", "gamma", "opacity", "blending",
        "interpolation", "rendering", "visible", "scale", "translate",
        "rotate", "shear", "units"
    ]
    for p in props:
        try:
            setattr(dst, p, getattr(src, p))
        except Exception:
            pass
    try:
        dst.visible = True
    except Exception:
        pass

def _add_result_layer(data: np.ndarray, src: NapariImage, suffix: str) -> NapariImage:
    v = _v()
    if v is None:
        raise RuntimeError("No active napari viewer.")
    safe_suffix = str(suffix).replace(" ", "_").replace("/", "_")
    name = f"{src.name}_{safe_suffix}"
    new_layer = v.add_image(data, name=name)
    _clone_visuals(src, new_layer)
    return new_layer

def _reset_contrast(layer: NapariImage, data: np.ndarray) -> None:
    try:
        layer.reset_contrast_limits()
        return
    except Exception:
        pass
    try:
        lo = float(np.nanmin(data))
        hi = float(np.nanmax(data))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            layer.contrast_limits = (lo, hi)
    except Exception:
        pass

def _require_scipy() -> None:
    if ndi is None:
        raise RuntimeError("SciPy is required (scipy.ndimage not available).")

def _require_skimage_otsu() -> None:
    if threshold_otsu is None:
        raise RuntimeError("scikit-image is required (skimage.filters.threshold_otsu not available).")

def _require_skimage_clear_border() -> None:
    if clear_border is None:
        raise RuntimeError("scikit-image is required (skimage.segmentation.clear_border not available).")

def _ensure_2d_or_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image/volume, got shape {arr.shape!r}")
    return np.ascontiguousarray(arr)

def _kernel_size(arr: np.ndarray, radius: int) -> Tuple[int, ...]:
    k = 2 * int(radius) + 1
    return (k,) * arr.ndim

def _guard_voxels(arr: np.ndarray, max_voxels: int) -> None:
    vox = int(np.prod(arr.shape))
    if vox > int(max_voxels):
        raise RuntimeError(
            f"Refused: volume too large ({vox:,} voxels). "
            f"Safety limit is {max_voxels:,} voxels for this operation."
        )

def _to_uint8_fast(src: np.ndarray) -> np.ndarray:
    """
    Always create a uint8 working copy:
    - float: min-max to 0..255
    - uint16/int: min-max to 0..255
    - uint8: pass-through (contiguous copy)
    """
    a = np.asarray(src)
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)

    af = np.asarray(a, dtype=np.float32)
    mn = float(np.nanmin(af))
    mx = float(np.nanmax(af))
    if not (np.isfinite(mn) and np.isfinite(mx)) or mx <= mn:
        return np.zeros_like(af, dtype=np.uint8)

    out = (af - mn) / (mx - mn)
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)

def _invert_u8(a: np.ndarray) -> np.ndarray:
    return (255 - a).astype(np.uint8, copy=False)

def _clip_percentiles_u8(a_u8: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo = float(np.percentile(a_u8, p_low))
    hi = float(np.percentile(a_u8, p_high))
    if hi <= lo:
        return a_u8.copy()
    out = np.clip(a_u8.astype(np.float32), lo, hi)
    # rescale back to 0..255 for consistent uint8 output
    out = (out - lo) / (hi - lo)
    return (out * 255.0).clip(0, 255).astype(np.uint8)

def _gamma_u8(a_u8: np.ndarray, gamma: float) -> np.ndarray:
    # LUT = fastest + stable
    g = float(gamma)
    if g <= 0:
        g = 1.0
    lut = (np.power(np.arange(256, dtype=np.float32) / 255.0, g) * 255.0).clip(0, 255).astype(np.uint8)
    return lut[a_u8]

def _log_u8(a_u8: np.ndarray) -> np.ndarray:
    # LUT log1p
    x = np.arange(256, dtype=np.float32)
    lut = (np.log1p(x) / np.log1p(255.0) * 255.0).clip(0, 255).astype(np.uint8)
    return lut[a_u8]

def _sqrt_u8(a_u8: np.ndarray) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    lut = (np.sqrt(x) / np.sqrt(255.0) * 255.0).clip(0, 255).astype(np.uint8)
    return lut[a_u8]


# ============================================================
# SAFE intensity ops (uint8 only)
# ============================================================

def _fn_u8_normalise(layer: NapariImage) -> np.ndarray:
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    return _to_uint8_fast(src)

def _fn_u8_invert(layer: NapariImage) -> np.ndarray:
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    a = _to_uint8_fast(src)
    return _invert_u8(a)

def _fn_u8_clip_1_99(layer: NapariImage) -> np.ndarray:
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    a = _to_uint8_fast(src)
    return _clip_percentiles_u8(a, 1.0, 99.0)

def _fn_u8_clip_5_95(layer: NapariImage) -> np.ndarray:
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    a = _to_uint8_fast(src)
    return _clip_percentiles_u8(a, 5.0, 95.0)

def _fn_u8_gamma_05(layer: NapariImage) -> np.ndarray:
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    a = _to_uint8_fast(src)
    return _gamma_u8(a, 0.5)

def _fn_u8_gamma_20(layer: NapariImage) -> np.ndarray:
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    a = _to_uint8_fast(src)
    return _gamma_u8(a, 2.0)

def _fn_u8_log(layer: NapariImage) -> np.ndarray:
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    a = _to_uint8_fast(src)
    return _log_u8(a)

def _fn_u8_sqrt(layer: NapariImage) -> np.ndarray:
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    a = _to_uint8_fast(src)
    return _sqrt_u8(a)


# ============================================================
# SAFE smoothing filters (uint8 only, guarded)
# ============================================================

# These are the only filters that should be allowed in a viewer tab.
# Guard values are conservative so you don't get "napari hangs".
_MAX_VOXELS_FILTER = 20_000_000  # ~20M voxels; adjust later if you want

def _fn_u8_median_r1(layer: NapariImage) -> np.ndarray:
    _require_scipy()
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    _guard_voxels(src, _MAX_VOXELS_FILTER)
    a = _to_uint8_fast(src)
    out = ndi.median_filter(a, size=_kernel_size(a, 1))
    return np.asarray(out, dtype=np.uint8)

def _fn_u8_median_r2(layer: NapariImage) -> np.ndarray:
    _require_scipy()
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    _guard_voxels(src, _MAX_VOXELS_FILTER)
    a = _to_uint8_fast(src)
    out = ndi.median_filter(a, size=_kernel_size(a, 2))
    return np.asarray(out, dtype=np.uint8)

def _fn_u8_mean_r1(layer: NapariImage) -> np.ndarray:
    _require_scipy()
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    _guard_voxels(src, _MAX_VOXELS_FILTER)
    a = _to_uint8_fast(src).astype(np.float32)
    out = ndi.uniform_filter(a, size=_kernel_size(a, 1))
    return np.rint(out).clip(0, 255).astype(np.uint8)

def _fn_u8_mean_r2(layer: NapariImage) -> np.ndarray:
    _require_scipy()
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    _guard_voxels(src, _MAX_VOXELS_FILTER)
    a = _to_uint8_fast(src).astype(np.float32)
    out = ndi.uniform_filter(a, size=_kernel_size(a, 2))
    return np.rint(out).clip(0, 255).astype(np.uint8)

def _fn_u8_gaussian_s1(layer: NapariImage) -> np.ndarray:
    _require_scipy()
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    _guard_voxels(src, _MAX_VOXELS_FILTER)
    a = _to_uint8_fast(src).astype(np.float32)
    out = ndi.gaussian_filter(a, sigma=1.0)
    return np.rint(out).clip(0, 255).astype(np.uint8)

def _fn_u8_gaussian_s2(layer: NapariImage) -> np.ndarray:
    _require_scipy()
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    _guard_voxels(src, _MAX_VOXELS_FILTER)
    a = _to_uint8_fast(src).astype(np.float32)
    out = ndi.gaussian_filter(a, sigma=2.0)
    return np.rint(out).clip(0, 255).astype(np.uint8)


# ============================================================
# SAFE segmentation (uint8 only)
# ============================================================

_MAX_VOXELS_SEG = 50_000_000  # segmentation is usually OK a bit higher

def _fn_u8_mean_threshold(layer: NapariImage) -> np.ndarray:
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    _guard_voxels(src, _MAX_VOXELS_SEG)
    a = _to_uint8_fast(src)
    thr = int(np.mean(a))
    out = np.zeros_like(a, dtype=np.uint8)
    out[a >= thr] = 255
    return out

def _fn_u8_otsu(layer: NapariImage) -> np.ndarray:
    _require_skimage_otsu()
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    _guard_voxels(src, _MAX_VOXELS_SEG)
    a = _to_uint8_fast(src)
    thr = int(threshold_otsu(a))
    out = np.zeros_like(a, dtype=np.uint8)
    out[a >= thr] = 255
    return out

def _fn_u8_clear_border(layer: NapariImage) -> np.ndarray:
    _require_skimage_clear_border()
    src = _ensure_2d_or_3d(np.asarray(layer.data))
    _guard_voxels(src, _MAX_VOXELS_SEG)
    a = _to_uint8_fast(src)
    # clear_border expects a binary-ish image; do it on nonzero
    mask = a > 0
    out = clear_border(mask)
    return (out.astype(np.uint8) * 255)


# ------------------------------------------------------- function registry ---

NORMAL_FUNCTIONS: Dict[str, FunctionEntry] = {
    # 1) Intensity (safe)
    "Convert to uint8 (0..255) [SAFE]": FunctionEntry("Convert to uint8 (0..255) [SAFE]", _fn_u8_normalise),
    "Invert (uint8) [SAFE]": FunctionEntry("Invert (uint8) [SAFE]", _fn_u8_invert),
    "Clip 1–99% + rescale (uint8) [SAFE]": FunctionEntry("Clip 1–99% + rescale (uint8) [SAFE]", _fn_u8_clip_1_99),
    "Clip 5–95% + rescale (uint8) [SAFE]": FunctionEntry("Clip 5–95% + rescale (uint8) [SAFE]", _fn_u8_clip_5_95),
    "Gamma (γ=0.5) (uint8 LUT) [SAFE]": FunctionEntry("Gamma (γ=0.5) (uint8 LUT) [SAFE]", _fn_u8_gamma_05),
    "Gamma (γ=2.0) (uint8 LUT) [SAFE]": FunctionEntry("Gamma (γ=2.0) (uint8 LUT) [SAFE]", _fn_u8_gamma_20),
    "Log (uint8 LUT) [SAFE]": FunctionEntry("Log (uint8 LUT) [SAFE]", _fn_u8_log),
    "Sqrt (uint8 LUT) [SAFE]": FunctionEntry("Sqrt (uint8 LUT) [SAFE]", _fn_u8_sqrt),

    # 2) Filters (safe, guarded)
    "Median filter (r=1) (uint8) [SAFE]": FunctionEntry("Median filter (r=1) (uint8) [SAFE]", _fn_u8_median_r1),
    "Median filter (r=2) (uint8) [SAFE]": FunctionEntry("Median filter (r=2) (uint8) [SAFE]", _fn_u8_median_r2),
    "Mean / box filter (r=1) (uint8) [SAFE]": FunctionEntry("Mean / box filter (r=1) (uint8) [SAFE]", _fn_u8_mean_r1),
    "Mean / box filter (r=2) (uint8) [SAFE]": FunctionEntry("Mean / box filter (r=2) (uint8) [SAFE]", _fn_u8_mean_r2),
    "Gaussian blur (σ=1) (uint8) [SAFE]": FunctionEntry("Gaussian blur (σ=1) (uint8) [SAFE]", _fn_u8_gaussian_s1),
    "Gaussian blur (σ=2) (uint8) [SAFE]": FunctionEntry("Gaussian blur (σ=2) (uint8) [SAFE]", _fn_u8_gaussian_s2),

    # 3) Thresholding (safe, guarded)
    "Mean threshold → binary (0/255) [SAFE]": FunctionEntry("Mean threshold → binary (0/255) [SAFE]", _fn_u8_mean_threshold),
    "Otsu threshold → binary (0/255) [SAFE]": FunctionEntry("Otsu threshold → binary (0/255) [SAFE]", _fn_u8_otsu),
    "Clear border (binary) [SAFE]": FunctionEntry("Clear border (binary) [SAFE]", _fn_u8_clear_border),
}

ALL_KEYS = list(NORMAL_FUNCTIONS.keys())


# --------------------------------------------------------------- main widget --

def functions_widget() -> Container:
    header = Label(label="", value="Target layer: <active image> or pick by name.")

    layer_combo = ComboBox(label="Layer", choices=["<active image>"], value="<active image>")

    def refresh_layers(*_):
        layers = _get_image_layers()
        names = ["<active image>"] + [L.name for L in layers]
        current = layer_combo.value if layer_combo.value in names else "<active image>"
        layer_combo.choices = names
        layer_combo.value = current

    refresh_layers()

    search_label = Label(label="", value="Search (filters the list):")

    if LineEdit is not None:
        search_box = LineEdit(label="", value="")
        try:
            search_box.native.setPlaceholderText("type e.g. uint8 / median / gaussian / otsu ...")
        except Exception:
            pass
    elif TextEdit is not None:
        search_box = TextEdit(label="", value="")
        try:
            search_box.native.setMaximumHeight(32)
        except Exception:
            pass
    else:
        search_box = None

    count_label = Label(label="", value=f"Showing {len(ALL_KEYS)} functions")

    func_label = Label(label="", value="Functions:")
    func_combo = ComboBox(label="", choices=ALL_KEYS, value=ALL_KEYS[0] if ALL_KEYS else None)

    output_label = Label(label="", value="Output mode:")
    output_combo = ComboBox(label="", choices=["new layer", "overwrite target"], value="new layer")

    run_btn = PushButton(text="Run selected")

    console_title = Label(label="", value="Activity Log:")

    if TextEdit is not None:
        console = TextEdit(label="", value="")
        try:
            console.native.setReadOnly(True)
            console.native.setMinimumHeight(180)
            console.native.setStyleSheet(
                "background-color:#000000;color:#FFFFFF;font-family:Consolas,monospace;font-size:10pt;"
            )
        except Exception:
            pass
    else:
        console = Label(label="", value="(Console unavailable: magicgui TextEdit missing)")

    def ts() -> str:
        return datetime.now().strftime("[%H:%M:%S]")

    def log(level: str, msg: str) -> None:
        line = f"{ts()} {level:<7} {msg}"
        try:
            native = getattr(console, "native", None)
            if native is not None and hasattr(native, "appendPlainText"):
                native.appendPlainText(line)
                return
            if native is not None and hasattr(native, "append"):
                native.append(line)
                return
        except Exception:
            pass

        try:
            old = getattr(console, "value", "") or ""
            console.value = (old + "\n" + line) if old else line
        except Exception:
            pass

    def is_overwrite() -> bool:
        return str(output_combo.value or "").lower().startswith("overwrite")

    def apply_filtered_list(query: str) -> None:
        q = (query or "").strip().lower()
        if not q:
            filtered = ALL_KEYS
        else:
            filtered = [k for k in ALL_KEYS if q in k.lower()]
        if not filtered:
            filtered = ["(no matches)"]
        cur = func_combo.value
        func_combo.choices = filtered
        if cur in filtered:
            func_combo.value = cur
        else:
            func_combo.value = filtered[0]
        count_label.value = "Showing 0 functions" if filtered == ["(no matches)"] else f"Showing {len(filtered)} functions"

    if search_box is not None:
        def _on_search_change(evt=None):
            apply_filtered_list(getattr(search_box, "value", "") or "")
        try:
            search_box.changed.connect(_on_search_change)
        except Exception:
            pass

    def run_selected(evt=None) -> None:
        refresh_layers()

        key = func_combo.value
        if not key or key == "(no matches)":
            show_warning("No function selected.")
            log("WARN", "No function selected.")
            return

        entry = NORMAL_FUNCTIONS.get(key)
        if entry is None:
            show_error(f"Function not found: {key}")
            log("ERROR", f"Function not found: {key}")
            return

        layer = _pick_layer_by_name(layer_combo.value)
        if layer is None:
            show_warning("No Image layer found. Load/select an image first.")
            log("WARN", "No Image layer found.")
            return

        overwrite = is_overwrite()
        log("INFO", f"Running: {entry.label} on '{layer.name}' ({'overwrite' if overwrite else 'new layer'})")

        try:
            result = entry.runner(layer)
        except Exception as e:  # noqa: BLE001
            show_error(f"Failed: {entry.label}\n{e!r}")
            log("ERROR", f"{entry.label} failed: {e!r}")
            return

        if not isinstance(result, np.ndarray):
            show_info(f"Done: {entry.label} (no output layer).")
            log("OK", f"Done: {entry.label} (no output layer).")
            return

        if overwrite:
            try:
                layer.data = result
                _reset_contrast(layer, result)
            except Exception as e:
                show_error(f"Overwrite failed: {e!r}")
                log("ERROR", f"Overwrite failed: {e!r}")
                return
            show_info(f"Done: {entry.label} (overwritten).")
            log("OK", f"Done: {entry.label} (overwritten).")
        else:
            new_layer = _add_result_layer(result, layer, entry.label)
            try:
                layer.visible = False
            except Exception:
                pass
            show_info(f"Done: {entry.label} → '{new_layer.name}' (previous hidden).")
            log("OK", f"Done: {entry.label} → '{new_layer.name}' (previous hidden).")

    try:
        run_btn.clicked.connect(run_selected)
    except Exception:
        try:
            run_btn.changed.connect(run_selected)
        except Exception:
            pass

    widgets: List[object] = [header, layer_combo, search_label]
    if search_box is not None:
        widgets.append(search_box)
    widgets.extend([count_label, func_label, func_combo, output_label, output_combo, run_btn, console_title])

    root = Container(widgets=widgets, layout="vertical", labels=False)
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

    log("INFO", "Functions tab ready (SAFE MODE: uint8 only).")
    return root
