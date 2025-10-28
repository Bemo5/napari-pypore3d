# DO NOT use: from __future__ import annotations  (causes Path ForwardRef issue with magicgui)

from pathlib import Path
from typing import Literal, Optional, Tuple, Any, Dict
import importlib
import inspect
import json
import ast
import numpy as np

from magicgui import magicgui
from magicgui.widgets import (
    Container, PushButton, ComboBox, Label, Slider, FloatSlider, CheckBox,
    RangeSlider, LineEdit, FloatSpinBox, SpinBox
)
from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_error, show_warning, show_info

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QScrollArea, QSizePolicy,
    QPlainTextEdit, QHBoxLayout
)
from qtpy.QtCore import Qt

PLUGIN_BUILD = "napari-pypore3d r16 (console + function browser)"

# ---------- helpers ----------
def _infer_dtype_from_size(file_size: int, shape: Tuple[int, int, int]) -> np.dtype:
    nvox = int(np.prod(shape))
    if nvox <= 0:
        raise ValueError("Invalid shape; the product must be > 0.")
    if file_size % nvox != 0:
        raise ValueError(
            f"File size {file_size} B is not divisible by voxel count {nvox}. "
            "Check shape or dtype."
        )
    bpv = file_size // nvox
    return {1: np.uint8, 2: np.uint16, 4: np.float32}.get(bpv, np.uint8)


def _active_image():
    v = current_viewer()
    if v is None:
        return None, None
    layer = v.layers.selection.active
    return (layer, v) if isinstance(layer, NapariImage) else (None, v)


def _last_zyx_shape(a: np.ndarray) -> Tuple[int, int, int]:
    if a.ndim == 2:
        sy, sx = map(int, a.shape[-2:])
        return 1, sy, sx
    if a.ndim >= 3:
        z, y, x = map(int, a.shape[-3:])
        return z, y, x
    return 1, 1, 1


def _slice_last_zyx(a: np.ndarray, zs: int, ze: int, ys: int, ye: int, xs: int, xe: int) -> np.ndarray:
    if a.ndim == 2:
        return a[ys:ye, xs:xe]
    prefix = (slice(None),) * (a.ndim - 3)
    return a[prefix + (slice(zs, ze), slice(ys, ye), slice(xs, xe))]


def _safe_eval_tuple_list(text: str) -> Tuple[Any, ...]:
    """
    Parse args string into a tuple. Accepts Python tuple/list literal or comma-separated values.
    """
    text = (text or "").strip()
    if not text:
        return tuple()
    try:
        node = ast.literal_eval(text)
        if isinstance(node, (list, tuple)):
            return tuple(node)
        return (node,)
    except Exception:
        return tuple(s.strip() for s in text.split(","))


def _safe_eval_kwargs(text: str) -> Dict[str, Any]:
    """
    Parse kwargs string into a dict. Accepts JSON or Python dict literal or k=v pairs.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        node = ast.literal_eval(text)
        if isinstance(node, dict):
            return node
    except Exception:
        pass
    out: Dict[str, Any] = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k].strip() if k else None
            out[k.strip()] = ast.literal_eval(v.strip())
    return out


# ---------- RAW loader ----------
@magicgui(
    call_button="Load RAW",
    layout="vertical",
    path={"widget_type": "FileEdit", "mode": "r", "filter": "*.raw;*.RAW;*.bin", "label": "RAW file"},
    dtype={"choices": ["auto", "uint8", "uint16", "float32"], "label": "dtype"},
    shape_z={"min": 1, "max": 1_000_000},
    shape_y={"min": 1, "max": 1_000_000},
    shape_x={"min": 1, "max": 1_000_000},
)
def _raw_loader_widget(
    path: Path = Path("."),
    shape_z: int = 700,
    shape_y: int = 700,
    shape_x: int = 700,
    dtype: Literal["auto", "uint8", "uint16", "float32"] = "auto",
):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    shape = (int(shape_z), int(shape_y), int(shape_x))
    file_size = p.stat().st_size
    np_dtype = _infer_dtype_from_size(file_size, shape) if dtype == "auto" else np.dtype(dtype)
    itemsize = int(np.dtype(np_dtype).itemsize)
    expected = int(np.prod(shape)) * itemsize
    if expected != file_size:
        raise ValueError(
            f"Shape {shape} with dtype {np_dtype} expects {expected} B, but file is {file_size} B. "
            "Adjust shape or dtype."
        )

    vol = np.memmap(p, dtype=np_dtype, mode="r", shape=shape)
    v = current_viewer()
    layer = v.add_image(vol, name=p.name)
    v.dims.current_step = tuple(s // 2 for s in shape)
    return layer


# ---------- Main widget ----------
def raw_loader_widget():
    gui = _raw_loader_widget
    show_info(f"{PLUGIN_BUILD} loaded")

    # ===== View tab =====
    render_mode = ComboBox(label="render", choices=["mip", "attenuated_mip", "translucent", "additive", "iso"], value="mip")
    colormap = ComboBox(label="cmap", choices=["gray", "magma", "inferno", "plasma", "viridis", "turbo", "cyan", "red", "green", "blue"], value="gray")
    blending = ComboBox(label="blend", choices=["translucent", "additive", "opaque", "minimum"], value="translucent")
    opacity = FloatSlider(label="opacity", min=0.0, max=1.0, value=1.0, step=0.01)

    gamma = FloatSlider(label="gamma", min=0.05, max=3.0, value=1.0, step=0.01)
    iso_thr = FloatSlider(label="iso thr", min=0.0, max=1.0, value=0.5, step=0.005)
    attenuation = FloatSlider(label="atten", min=0.0, max=1.0, value=0.08, step=0.005)
    downsample = Slider(label="downsample", min=1, max=6, value=1)
    perspective = CheckBox(text="perspective", value=False)
    clipping = CheckBox(text="clipping plane", value=False)

    btn_auto_cl = PushButton(text="Auto contrast (1–99%)")
    view_preset = ComboBox(
        label="preset",
        choices=["Neutral (MIP)", "Bone (ISO high)", "Soft tissue (att MIP)", "Glow (additive)", "Translucent"],
        value="Neutral (MIP)",
    )
    btn_apply_preset = PushButton(text="Apply")
    btn_toggle = PushButton(text="2D/3D")
    btn_rot90 = PushButton(text="Rotate 90°")
    btn_reset = PushButton(text="Reset")

    view_tab = Container(
        widgets=[
            render_mode, colormap, blending, opacity,
            gamma, iso_thr, attenuation, downsample,
            Container(widgets=[perspective, clipping, btn_auto_cl], layout="horizontal"),
            Container(widgets=[view_preset, btn_apply_preset], layout="horizontal"),
            Container(widgets=[btn_toggle, btn_rot90, btn_reset], layout="horizontal"),
        ],
        layout="vertical",
        labels=False,
    )

    # ===== Crop tab =====
    crop_enable = CheckBox(text="enable crop [start:end)", value=False)
    z_range = RangeSlider(label="Z", min=0, max=1, value=(0, 1), step=1)
    y_range = RangeSlider(label="Y", min=0, max=1, value=(0, 1), step=1)
    x_range = RangeSlider(label="X", min=0, max=1, value=(0, 1), step=1)

    z0, z1 = SpinBox(label="Z start", min=0, max=1, value=0), SpinBox(label="Z end", min=1, max=1, value=1)
    y0, y1 = SpinBox(label="Y start", min=0, max=1, value=0), SpinBox(label="Y end", min=1, max=1, value=1)
    x0, x1 = SpinBox(label="X start", min=0, max=1, value=0), SpinBox(label="X end", min=1, max=1, value=1)

    btn_reset_crop = PushButton(text="Reset crop")

    crop_tab = Container(
        widgets=[
            crop_enable,
            Label(value="Exclusive end [start:end) — type exact numbers"),
            Label(value="Z"), z_range, Container(widgets=[z0, z1], layout="horizontal"),
            Label(value="Y"), y_range, Container(widgets=[y0, y1], layout="horizontal"),
            Label(value="X"), x_range, Container(widgets=[x0, x1], layout="horizontal"),
            btn_reset_crop,
        ],
        layout="vertical",
        labels=False,
    )

    # ===== Effects tab (SciPy always; p3d optional) =====
    run_on_crop = CheckBox(text="run on current crop", value=True)
    func_path = LineEdit(label="callable", value="scipy.ndimage.gaussian_filter")
    func_param = FloatSpinBox(label="param", min=0.0, max=100.0, value=1.5, step=0.5)
    btn_run_func = PushButton(text="Run → replace layer")

    # SciPy quick dropdown
    effect_choice = ComboBox(
        label="Filters (fallback)",
        choices=["Gaussian (SciPy)", "Median (SciPy)"],
        value="Gaussian (SciPy)",
    )
    btn_run_selected = PushButton(text="Run selected → replace layer")

    # ===== p3d function browser + console (Qt-only panel) =====
    console = QPlainTextEdit()
    console.setReadOnly(True)
    console.setMaximumHeight(220)

    def log(msg: str):
        console.appendPlainText(msg)

    # try importing any p3d module names
    p3d_mod_names = ("pypore3d._p3dFilt", "pypore3d.p3dFilt", "PyPore3D._p3dFilt", "PyPore3D.p3dFilt")
    p3d_module = None
    for name in p3d_mod_names:
        try:
            p3d_module = importlib.import_module(name)
            log(f"[p3d] loaded: {name}")
            show_info(f"PyPore3D loaded: {name}")
            break
        except Exception as e:
            log(f"[p3d] import failed {name}: {e}")

    p3d_fn = ComboBox(label="p3d function", choices=[], value=None, nullable=True)
    p3d_args = LineEdit(label="args (tuple/list or comma-sep)", value="")
    p3d_kwargs = LineEdit(label="kwargs (JSON/dict or k=v,...) ", value="")
    btn_list = PushButton(text="List functions")
    btn_inspect = PushButton(text="Inspect")
    btn_run_p3d = PushButton(text="Run p3d function")

    def refresh_p3d_functions():
        funcs = []
        if p3d_module is not None:
            for k in sorted(dir(p3d_module)):
                if k.startswith("_"):
                    continue
                obj = getattr(p3d_module, k, None)
                if callable(obj):
                    funcs.append(k)
        p3d_fn.choices = funcs
        if funcs:
            p3d_fn.value = funcs[0]
        log(f"[p3d] {len(funcs)} callable(s) found")

    def do_list():
        refresh_p3d_functions()

    def do_inspect():
        name = p3d_fn.value
        if not (p3d_module and name):
            log("[p3d] no function selected")
            return
        f = getattr(p3d_module, name, None)
        if f is None:
            log(f"[p3d] missing: {name}")
            return
        try:
            sig = inspect.signature(f)
        except Exception:
            sig = "(signature unavailable)"
        doc = inspect.getdoc(f) or "(no docstring)"
        log(f"=== {name} {sig}\n{doc}\n")

    def _active_image_and_viewer():
        layer, v = _active_image()
        if not layer:
            vv = current_viewer()
            if vv and vv.layers:
                lay = vv.layers[-1]
                if isinstance(lay, NapariImage):
                    return lay, vv
        return layer, v

    def _ensure_orig_data():
        layer, _ = _active_image_and_viewer()
        if not layer:
            return None
        if "_orig_data" not in layer.metadata:
            layer.metadata["_orig_data"] = layer.data
        return layer

    def _get_source_array():
        layer = _ensure_orig_data()
        if not layer:
            return None, False
        src = layer.data if (run_on_crop.value and crop_enable.value) else layer.metadata["_orig_data"]
        a = np.asarray(src)
        squeezed = False
        if a.ndim == 2:
            a = a[np.newaxis, ...]
            squeezed = True
        return a, squeezed

    # Clean NumPy 3D-call path (best path): relies on SWIG numpy.i typemaps
    def _run_p3d_numpy(choice: str, sigma_or_k: float):
        layer = _ensure_orig_data()
        if not layer:
            return
        src = layer.data if (run_on_crop.value and crop_enable.value) else layer.metadata["_orig_data"]
        vol = np.asarray(src)
        squeeze_back = False
        if vol.ndim == 2:
            vol = vol[np.newaxis, ...]
            squeeze_back = True
        vol8 = np.ascontiguousarray(vol, dtype=np.uint8)
        out8 = np.empty_like(vol8)
        Z, Y, X = map(int, vol8.shape[-3:])

        if p3d_module is None:
            show_warning("PyPore3D extension not found, skipping.")
            return False

        try:
            if choice == "Gaussian":
                k = int(max(3, 2 * int(round(sigma_or_k)) + 1))
                err = p3d_module.p3dGaussianFilter3D_8(vol8, out8, X, Y, Z, k, float(sigma_or_k), None, None)
            elif choice == "Median":
                k = max(3, (int(sigma_or_k) | 1))
                err = p3d_module.p3dMedianFilter3D_8(vol8, out8, X, Y, Z, k, None, None)
            else:
                return False
        except Exception as e:
            show_warning(f"p3d NumPy path failed: {e}")
            return False

        if err != 0:
            show_error(f"pypore3d returned error code {err}")
            return True

        layer.data = out8[0] if squeeze_back else out8
        show_info(f"pypore3d {choice} applied (NumPy 3D path).")
        return True

    def do_run_p3d():
        # Use function browser manual runner
        name = p3d_fn.value
        if not (p3d_module and name):
            log("[p3d] no function selected")
            return
        f = getattr(p3d_module, name, None)
        if not callable(f):
            log(f"[p3d] not callable: {name}")
            return

        a, squeezed = _get_source_array()
        if a is None:
            return

        args = list(_safe_eval_tuple_list(p3d_args.value))
        kwargs = _safe_eval_kwargs(p3d_kwargs.value)

        out_buffer = None
        if not args and ("3D_8" in name or name.endswith("_wrap") or name.endswith("_buf")):
            vol8 = np.ascontiguousarray(a, dtype=np.uint8)
            out8 = np.empty_like(vol8)
            Z, Y, X = map(int, vol8.shape[-3:])
            args = [vol8, out8, X, Y, Z]  # rely on numpy.i mapping directly
            if "Gaussian" in name:
                k = int(max(3, 2 * int(round(float(func_param.value))) + 1))
                args += [k, float(func_param.value), None, None]
            elif "Median" in name:
                k = max(3, (int(func_param.value) | 1))
                args += [k, None, None]
            out_buffer = out8

        try:
            log(f"[p3d] calling {name}(*{args}, **{kwargs})")
            ret = f(*args, **kwargs)
            log(f"[p3d] returned: {ret!r}")
            if out_buffer is not None and (ret == 0 or ret is None):
                layer = _ensure_orig_data()
                if layer:
                    layer.data = out_buffer[0] if squeezed else out_buffer
                    show_info("p3d output applied.")
        except Exception as e:
            log(f"[p3d] EXCEPTION: {e}")

    # ===== Effects (magicgui column) =====
    effects_col = Container(
        widgets=[
            run_on_crop, func_path, func_param, btn_run_func,
            Label(value="Filters:"),
            effect_choice, btn_run_selected,
            Label(value="pypore3d console & function browser:"),
        ],
        layout="vertical",
        labels=False,
    )

    # ===== Effects tab as pure Qt (NO mixing) =====
    effects_qt = QWidget()
    eff_layout = QVBoxLayout(effects_qt)
    eff_layout.setContentsMargins(8, 8, 8, 8)
    eff_layout.setSpacing(6)

    # Add the magicgui column via .native
    eff_layout.addWidget(effects_col.native)

    # Build the p3d browser panel (Qt) and add it
    p3d_panel = QWidget()
    p3d_layout = QVBoxLayout(p3d_panel)
    # top controls row (Qt)
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.addWidget(btn_list.native)
    row_layout.addWidget(btn_inspect.native)
    row_layout.addWidget(btn_run_p3d.native)
    # stack the magicgui inputs (via .native) and the console
    p3d_layout.addWidget(p3d_fn.native)
    p3d_layout.addWidget(p3d_args.native)
    p3d_layout.addWidget(p3d_kwargs.native)
    p3d_layout.addWidget(row)
    p3d_layout.addWidget(console)
    eff_layout.addWidget(p3d_panel)

    # ===== Build Qt UI =====
    header_native = gui.native
    tabs = QTabWidget()
    tabs.addTab(view_tab.native, "View")
    tabs.addTab(crop_tab.native, "Crop")
    tabs.addTab(effects_qt, "Effects")  # <- pure Qt tab

    panel = QWidget()
    panel_layout = QVBoxLayout(panel)
    panel_layout.setContentsMargins(8, 8, 8, 8)
    panel_layout.setSpacing(6)
    panel_layout.addWidget(header_native)
    panel_layout.addWidget(tabs)

    panel.setMinimumWidth(330)
    panel.setMaximumWidth(480)

    scroll = QScrollArea()
    scroll.setWidget(panel)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    wrapper = QWidget()
    wrap_layout = QVBoxLayout(wrapper)
    wrap_layout.setContentsMargins(0, 0, 0, 0)
    wrap_layout.addWidget(scroll)
    wrapper.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    wrapper.setMinimumWidth(330)
    wrapper.setMaximumWidth(500)

    # ===== Logic / wiring =====
    def _set_limits(sz: int, sy: int, sx: int):
        for sl, mx in ((z_range, sz), (y_range, sy), (x_range, sx)):
            sl.max = int(mx)
            lo, hi = sl.value
            if hi > mx:
                sl.value = (lo, int(mx))
        z0.max, y0.max, x0.max = max(0, sz - 1), max(0, sy - 1), max(0, sx - 1)
        z1.min = y1.min = x1.min = 1
        z1.max, y1.max, x1.max = max(1, sz), max(1, sy), max(1, sx)

    def _sync_ranges_to_shape():
        layer, _ = _active_image()
        if not layer:
            return
        data = np.asarray(layer.metadata.get("_orig_data", layer.data))
        sz, sy, sx = _last_zyx_shape(data)
        _set_limits(sz, sy, sx)
        z_range.value = (0, sz); y_range.value = (0, sy); x_range.value = (0, sx)
        z0.value, z1.value = 0, sz; y0.value, y1.value = 0, sy; x0.value, x1.value = 0, sx

    def _apply_when_3d(_=None):
        layer, v = _active_image()
        if not layer or v.dims.ndisplay != 3:
            return
        layer.rendering = render_mode.value
        layer.iso_threshold = iso_thr.value
        layer.attenuation = attenuation.value

    def _apply_gamma(_=None):
        layer, _ = _active_image()
        if layer:
            layer.gamma = gamma.value

    def _apply_colormap(_=None):
        layer, _ = _active_image()
        if layer:
            layer.colormap = colormap.value

    def _apply_blending(_=None):
        layer, _ = _active_image()
        if layer:
            layer.blending = blending.value
            layer.opacity = opacity.value

    def _apply_opacity(_=None):
        layer, _ = _active_image()
        if layer:
            layer.opacity = opacity.value

    def _apply_downsample(_=None):
        layer, v = _active_image()
        if layer:
            step = int(downsample.value)
            d = v.dims
            new_step = list(d.step)
            for ax in range(d.ndim):
                new_step[ax] = step if ax in d.order[-3:] else 1
            d.step = tuple(new_step)

    def _toggle_ndisplay():
        layer, v = _active_image()
        if not layer:
            return
        v.dims.ndisplay = 3 if v.dims.ndisplay == 2 else 2
        _apply_when_3d()

    def _reset_view():
        _, v = _active_image()
        if v:
            v.reset_view()

    def _rotate_90():
        layer, v = _active_image()
        if layer:
            a = list(v.camera.angles)
            a[2] = (a[2] + 90) % 360
            v.camera.angles = tuple(a)

    def _toggle_perspective(_=None):
        _, v = _active_image()
        if v:
            v.camera.perspective = 45.0 if perspective.value else 0.0

    def _toggle_clipping(_=None):
        layer, _ = _active_image()
        if not layer:
            return
        if clipping.value:
            plane = dict(position=(0, 0, 0), normal=(0, 0, -1), enabled=True)
            layer.experimental_clipping_planes = [plane]
        else:
            layer.experimental_clipping_planes = []

    def _auto_contrast():
        layer = _ensure_orig_data()
        if not layer:
            return
        data = layer.metadata["_orig_data"]
        lo, hi = np.percentile(np.asarray(data), (1, 99))
        try:
            layer.contrast_limits = (float(lo), float(hi))
        except Exception:
            pass

    def _apply_preset():
        name = view_preset.value
        render_mode.value = "mip"; gamma.value = 1.0; iso_thr.value = 0.5; attenuation.value = 0.08
        blending.value = "translucent"; opacity.value = 1.0; colormap.value = "gray"
        if name == "Bone (ISO high)":
            render_mode.value = "iso"; iso_thr.value = 0.7; gamma.value = 0.9; colormap.value = "magma"
        elif name == "Soft tissue (att MIP)":
            render_mode.value = "attenuated_mip"; attenuation.value = 0.12; gamma.value = 0.85; colormap.value = "inferno"
        elif name == "Glow (additive)":
            render_mode.value = "additive"; gamma.value = 1.2; blending.value = "additive"; colormap.value = "plasma"
        elif name == "Translucent":
            render_mode.value = "translucent"; gamma.value = 1.0; colormap.value = "viridis"
        _apply_when_3d(); _apply_gamma(); _apply_colormap(); _apply_blending()

    # ----- crop -----
    def _apply_crop(_=None):
        layer = _ensure_orig_data()
        if not layer:
            return
        orig = layer.metadata["_orig_data"]
        a = np.asarray(orig)
        if not crop_enable.value:
            layer.data = orig
            return

        zs, ze = map(int, z_range.value)
        ys, ye = map(int, y_range.value)
        xs, xe = map(int, x_range.value)
        if ze <= zs: ze = zs + 1
        if ye <= ys: ye = ys + 1
        if xe <= xs: xe = xs + 1

        sz, sy, sx = _last_zyx_shape(a)
        zs = max(0, min(zs, sz - 1)); ze = max(1, min(ze, sz))
        ys = max(0, min(ys, sy - 1)); ye = max(1, min(ye, sy))
        xs = max(0, min(xs, sx - 1)); xe = max(1, min(xe, sx))

        z0.value, z1.value = zs, ze
        y0.value, y1.value = ys, ye
        x0.value, x1.value = xs, xe

        layer.data = _slice_last_zyx(a, zs, ze, ys, ye, xs, xe)

    def _reset_crop():
        layer = _ensure_orig_data()
        if not layer:
            return
        data = np.asarray(layer.metadata["_orig_data"])
        sz, sy, sx = _last_zyx_shape(data)
        _set_limits(sz, sy, sx)
        z_range.value = (0, sz); y_range.value = (0, sy); x_range.value = (0, sx)
        z0.value, z1.value = 0, sz; y0.value, y1.value = 0, sy; x0.value, x1.value = 0, sx
        crop_enable.value = False
        layer.data = layer.metadata["_orig_data"]

    def _from_slider_to_spin(_=None):
        (zs, ze), (ys, ye), (xs, xe) = z_range.value, y_range.value, x_range.value
        z0.value, z1.value = int(zs), int(ze)
        y0.value, y1.value = int(ys), int(ye)
        x0.value, x1.value = int(xs), int(xe)

    def _from_spin_to_slider(_=None):
        zs, ze = int(z0.value), int(z1.value)
        ys, ye = int(y0.value), int(y1.value)
        xs, xe = int(x0.value), int(x1.value)
        if ze <= zs: ze = zs + 1
        if ye <= ys: ye = ys + 1
        if xe <= xs: xe = xs + 1
        z_range.value = (zs, ze); y_range.value = (ys, ye); x_range.value = (xs, xe)
        _apply_crop()

    def _on_loaded(layer):
        if isinstance(layer, NapariImage):
            layer.metadata["_orig_data"] = layer.data
            v = current_viewer()
            if v:
                v.layers.selection.active = layer
        _sync_ranges_to_shape()
        _auto_contrast()
        _apply_preset()
        _apply_crop()
        _apply_blending()

    gui.called.connect(_on_loaded)

    def _on_selection_change(_=None):
        _sync_ranges_to_shape()
    if current_viewer():
        current_viewer().layers.selection.events.changed.connect(_on_selection_change)

    # ----- generic callable (e.g., SciPy) -----
    def _run_callable():
        layer = _ensure_orig_data()
        if not layer:
            return
        src = layer.data if (run_on_crop.value and crop_enable.value) else layer.metadata["_orig_data"]
        path = func_path.value.strip()
        if "." not in path:
            show_error("Provide a full path like 'module.submodule.function'")
            return
        mod_name, func_name = path.rsplit(".", 1)
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, func_name)
        except Exception as e:
            show_error(f"Import failed: {e}")
            return
        try:
            param = float(func_param.value)
            out = fn(src, param) if param != 0.0 else fn(src)
        except Exception as e:
            show_error(f"Function call failed: {e}")
            return
        layer.data = out
        show_info(f"Applied {path}")

    # ----- SciPy quick filters -----
    def _run_selected():
        layer = _ensure_orig_data()
        if not layer:
            return
        src = layer.data if (run_on_crop.value and crop_enable.value) else layer.metadata["_orig_data"]
        vol = np.asarray(src)
        squeeze = False
        if vol.ndim == 2:
            vol = vol[np.newaxis, ...]
            squeeze = True

        try:
            from scipy.ndimage import gaussian_filter, median_filter
        except Exception:
            show_error("SciPy not available. Install with: pip install scipy")
            return

        if "Gaussian" in effect_choice.value:
            out = gaussian_filter(vol, sigma=float(func_param.value)).astype(vol.dtype, copy=False)
            show_info("SciPy Gaussian applied.")
        else:
            k = max(3, (int(func_param.value) | 1))
            out = median_filter(vol, size=k).astype(vol.dtype, copy=False)
            show_info("SciPy Median applied.")
        layer.data = out[0] if squeeze else out

    # ----- wiring -----
    btn_apply_preset.clicked.connect(_apply_preset)
    btn_auto_cl.clicked.connect(_auto_contrast)

    render_mode.changed.connect(_apply_when_3d)
    iso_thr.changed.connect(_apply_when_3d)
    attenuation.changed.connect(_apply_when_3d)

    gamma.changed.connect(_apply_gamma)
    colormap.changed.connect(_apply_colormap)
    blending.changed.connect(_apply_blending)
    opacity.changed.connect(_apply_opacity)

    downsample.changed.connect(_apply_downsample)
    btn_toggle.clicked.connect(_toggle_ndisplay)
    btn_rot90.clicked.connect(_rotate_90)
    btn_reset.clicked.connect(_reset_view)
    perspective.changed.connect(_toggle_perspective)
    clipping.changed.connect(_toggle_clipping)

    crop_enable.changed.connect(_apply_crop)
    z_range.changed.connect(_from_slider_to_spin); z_range.changed.connect(_apply_crop)
    y_range.changed.connect(_from_slider_to_spin); y_range.changed.connect(_apply_crop)
    x_range.changed.connect(_from_slider_to_spin); x_range.changed.connect(_apply_crop)
    for sb in (z0, z1, y0, y1, x0, x1):
        sb.changed.connect(_from_spin_to_slider)
    btn_reset_crop.clicked.connect(_reset_crop)

    btn_run_func.clicked.connect(_run_callable)
    btn_run_selected.clicked.connect(_run_selected)

    btn_list.clicked.connect(refresh_p3d_functions)
    btn_inspect.clicked.connect(do_inspect)
    btn_run_p3d.clicked.connect(do_run_p3d)

    # populate function list if we imported p3d
    refresh_p3d_functions()

    return wrapper


def napari_experimental_provide_dock_widget():
    return [raw_loader_widget]
