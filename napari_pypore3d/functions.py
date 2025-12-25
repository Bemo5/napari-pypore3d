# napari_pypore3d/functions.py — r20
# Scientist Mode + session recipe + sane placement + correct preview scaling
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Callable, Any, Tuple
import os
import tempfile
from uuid import uuid4

import numpy as np

from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels as NapariLabels
from napari.utils.notifications import show_info, show_warning, show_error

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QSpinBox, QPushButton, QPlainTextEdit,
    QGroupBox, QFrame, QSizePolicy, QCheckBox
)

# Session recipe
from .session_recorder import RECORDER, register_handler, Step
from .session_recorder import resolve_target_layer

# napari worker to avoid UI freeze
try:
    from napari.qt.threading import thread_worker
except Exception:
    thread_worker = None  # type: ignore

# ---------------------------------------------------------------------
# PyPore3D imports (FILT + BLOB)
# ---------------------------------------------------------------------
_HAVE_FILT = False
_HAVE_BLOB = False

try:
    from pypore3d.p3dFiltPy import (
        py_p3dReadRaw8,
        py_p3dWriteRaw8,
        py_p3dMedianFilter8,
        py_p3dMeanFilter8,
        py_p3dGaussianFilter8,
        py_p3dAutoThresholding8,
        py_p3dClearBorderFilter8,
        py_printErrorMessage as _p3d_err_filt,
    )
    _HAVE_FILT = True
except Exception:
    py_p3dReadRaw8 = py_p3dWriteRaw8 = None  # type: ignore
    py_p3dMedianFilter8 = py_p3dMeanFilter8 = py_p3dGaussianFilter8 = None  # type: ignore
    py_p3dAutoThresholding8 = py_p3dClearBorderFilter8 = None  # type: ignore
    _p3d_err_filt = None  # type: ignore

try:
    from pypore3d.p3dBlobPy import (
        py_p3dMinVolumeFilter3D,
        py_p3dBlobLabeling,
        py_printErrorMessage as _p3d_err_blob,
    )
    _HAVE_BLOB = True
except Exception:
    py_p3dMinVolumeFilter3D = py_p3dBlobLabeling = None  # type: ignore
    _p3d_err_blob = None  # type: ignore

# ---------------------------------------------------------------------
# Settings (tuned for big volumes)
# ---------------------------------------------------------------------
_PREVIEW_TARGET_VOXELS = 30_000_000
_FULL_SAFE_VOXELS = 180_000_000
_FULL_ABS_MAX_VOXELS = 700_000_000

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _v():
    try:
        return current_viewer()
    except Exception:
        return None

def _now() -> str:
    return datetime.now().strftime("[%H:%M:%S]")

def _ensure_2d_or_3d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape={a.shape!r}")
    return np.ascontiguousarray(a)

def _voxels(a: np.ndarray) -> int:
    return int(np.prod(a.shape))

def _to_uint8_fast(src: np.ndarray) -> np.ndarray:
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

def _napari_to_p3d_u8(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int], bool]:
    a = np.ascontiguousarray(arr)
    if a.ndim == 2:
        u8 = _to_uint8_fast(a).T
        u8 = u8[:, :, None]
        x, y, z = u8.shape
        return np.ascontiguousarray(u8), (x, y, z), False
    u8 = _to_uint8_fast(a)
    u8 = np.transpose(u8, (2, 1, 0))
    x, y, z = u8.shape
    return np.ascontiguousarray(u8), (x, y, z), True

def _p3d_to_napari_u8(p3d_u8: np.ndarray, was_3d: bool) -> np.ndarray:
    if not was_3d:
        return np.ascontiguousarray(p3d_u8[:, :, 0].T)
    return np.ascontiguousarray(np.transpose(p3d_u8, (2, 1, 0)))

def _tmp_paths() -> tuple[Path, Path]:
    tmp_dir = Path(tempfile.gettempdir()) / "napari_pypore3d_p3dtmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}_{uuid4().hex}"
    return (tmp_dir / f"in_{tag}.raw", tmp_dir / f"out_{tag}.raw")

def _cleanup(*paths: Path) -> None:
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

def _p3d_call3d(fn, vol, x: int, y: int, z: int):
    try:
        return fn(vol, x, y, dimz=z)
    except TypeError:
        return fn(vol, x, y, z)

def _p3d_call3d_minvol(fn, vol, x: int, y: int, z: int, minvol: int):
    mv = int(minvol)
    last = None
    for args in [
        (x, y, z, mv),
        (x, y, z),
        (x, y, mv),
    ]:
        try:
            return fn(vol, *args)
        except Exception as e:
            last = e
    raise RuntimeError(f"MinVolumeFilter3D failed: {last!r}")

def _reset_contrast(layer: NapariImage) -> None:
    try:
        layer.reset_contrast_limits()
    except Exception:
        pass

def _downsample_stride_for_target(voxels: int, target: int) -> int:
    if voxels <= target:
        return 1
    s = int(np.ceil((voxels / float(target)) ** (1.0 / 3.0)))
    return max(2, s)

def _downsample(a: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return a
    if a.ndim == 2:
        return a[::stride, ::stride]
    return a[::stride, ::stride, ::stride]

def _inherit_transform_and_fix_stride(src: Any, dst: Any, stride: int) -> None:
    try:
        dst.translate = src.translate
        dst.scale = tuple(np.array(src.scale) * stride)
        dst.rotate = src.rotate
        dst.shear = src.shear
    except Exception:
        pass

def _is_maskish(op: str) -> bool:
    return any(k in op.lower() for k in ["autothreshold", "minvol", "clearborder", "bloblabeling"])

def _to_labels_array(out_u8: np.ndarray) -> np.ndarray:
    if out_u8.dtype == np.uint8 and out_u8.max() == 255:
        return (out_u8 > 0).astype(np.uint16)
    return out_u8.astype(np.uint16, copy=False)

# ---------------------------------------------------------------------
# Core runners (uint8 bridge)
# ---------------------------------------------------------------------
def _run_filt_u8(arr: np.ndarray, op_name: str, fn) -> np.ndarray:
    src = _ensure_2d_or_3d(arr)
    p3d_u8, (x, y, z), was_3d = _napari_to_p3d_u8(src)
    in_path, out_path = _tmp_paths()
    p3d_u8.ravel().tofile(in_path)

    try:
        v = py_p3dReadRaw8(str(in_path), x, y, dimz=z)
        v2 = _p3d_call3d(fn, v, x, y, z)
        py_p3dWriteRaw8(v2, str(out_path), x, y, dimz=z)
    except Exception as e:
        if _p3d_err_filt:
            _p3d_err_filt()
        raise RuntimeError(f"{op_name} failed: {e!r}")

    out = np.fromfile(out_path, dtype=np.uint8).reshape((x, y, z))
    _cleanup(in_path, out_path)
    return _p3d_to_napari_u8(out, was_3d)

def _run_minvol_u8(arr: np.ndarray, minvol: int) -> np.ndarray:
    src = _ensure_2d_or_3d(arr)
    p3d_u8, (x, y, z), was_3d = _napari_to_p3d_u8(src)
    in_path, out_path = _tmp_paths()
    p3d_u8.ravel().tofile(in_path)

    v = py_p3dReadRaw8(str(in_path), x, y, dimz=z)
    v2 = _p3d_call3d_minvol(py_p3dMinVolumeFilter3D, v, x, y, z, minvol)
    py_p3dWriteRaw8(v2, str(out_path), x, y, dimz=z)

    out = np.fromfile(out_path, dtype=np.uint8).reshape((x, y, z))
    _cleanup(in_path, out_path)
    return _p3d_to_napari_u8(out, was_3d)

def _run_blob_label_u8(arr: np.ndarray) -> np.ndarray:
    src = _ensure_2d_or_3d(arr)
    p3d_u8, (x, y, z), was_3d = _napari_to_p3d_u8(src)
    in_path, out_path = _tmp_paths()
    p3d_u8.ravel().tofile(in_path)

    v = py_p3dReadRaw8(str(in_path), x, y, dimz=z)
    v2 = _p3d_call3d(py_p3dBlobLabeling, v, x, y, z)
    py_p3dWriteRaw8(v2, str(out_path), x, y, dimz=z)

    out = np.fromfile(out_path, dtype=np.uint8).reshape((x, y, z))
    _cleanup(in_path, out_path)
    return _p3d_to_napari_u8(out, was_3d)

# ---------------------------------------------------------------------
# UI (NO PRESETS)
# ---------------------------------------------------------------------
@dataclass
class _Job:
    op: str
    fn: Callable[[], np.ndarray]
    mode: str
    stride: int
    src_name: str
    output: str
    minvol: int = 0

def functions_widget() -> QWidget:
    root = QWidget()
    outer = QVBoxLayout(root)

    title = QLabel("PyPore3D Workflow (Scientist Mode)")
    title.setStyleSheet("font-size:12pt; font-weight:600;")
    outer.addWidget(title)

    # Controls
    grid = QGridLayout()
    cmb_mode = QComboBox()
    cmb_mode.addItems(["Preview", "Full"])
    cmb_out = QComboBox()
    cmb_out.addItems(["new layer", "overwrite"])
    spn_minvol = QSpinBox()
    spn_minvol.setRange(0, 50_000_000)
    spn_minvol.setValue(50)

    grid.addWidget(QLabel("Mode"), 0, 0)
    grid.addWidget(cmb_mode, 0, 1)
    grid.addWidget(QLabel("Output"), 1, 0)
    grid.addWidget(cmb_out, 1, 1)
    grid.addWidget(QLabel("Min volume"), 2, 0)
    grid.addWidget(spn_minvol, 2, 1)
    outer.addLayout(grid)

    outer.addWidget(QFrame(frameShape=QFrame.HLine))

    def btn(text): 
        b = QPushButton(text)
        b.setMinimumHeight(32)
        return b

    b_med = btn("Median")
    b_mean = btn("Mean")
    b_gauss = btn("Gaussian")
    b_thr = btn("AutoThreshold")
    b_min = btn("MinVolume")
    b_clr = btn("ClearBorder")
    b_lab = btn("BlobLabeling")

    for b in (b_med, b_mean, b_gauss, b_thr, b_min, b_clr, b_lab):
        outer.addWidget(b)

    def run(op, fn, minvol=0):
        v = _v()
        if not v or not isinstance(v.layers.selection.active, NapariImage):
            show_warning("Select an Image layer")
            return

        L = v.layers.selection.active
        arr = np.asarray(L.data)
        vox = _voxels(arr)
        preview = cmb_mode.currentIndex() == 0
        stride = _downsample_stride_for_target(vox, _PREVIEW_TARGET_VOXELS) if preview else 1
        inp = _downsample(arr, stride)

        out = fn(inp)
        if cmb_out.currentIndex() == 1:
            L.data = out
        else:
            if _is_maskish(op):
                labels_2d = _to_labels_array(out)
                src = np.asarray(L.data)
                if src.ndim == 3 and labels_2d.ndim == 2:
                    z = int(v.dims.current_step[0])  # active Z slice
                    labels_3d = np.zeros(src.shape, dtype=labels_2d.dtype)
                    labels_3d[z, :, :] = labels_2d
                    lab = v.add_labels(labels_3d, name=f"{L.name} | {op}")
                else:
                    lab = v.add_labels(labels_2d, name=f"{L.name} | {op}")
                _inherit_transform_and_fix_stride(L, lab, stride)
            else:
                img = v.add_image(out, name=f"{L.name} | {op}")
                _inherit_transform_and_fix_stride(L, img, stride)

        RECORDER.add_step(
            op="p3d_run",
            target="__ACTIVE__",
            params=dict(
                algo=op,
                mode="preview" if preview else "full",
                stride=stride,
                output="overwrite" if cmb_out.currentIndex() else "new",
                minvol=minvol,
                # optional but useful for stable naming across volumes
                result_layer=f"| {op}",
            ),
        )


    b_med.clicked.connect(lambda: run("Median", lambda a: _run_filt_u8(a, "Median", py_p3dMedianFilter8)))
    b_mean.clicked.connect(lambda: run("Mean", lambda a: _run_filt_u8(a, "Mean", py_p3dMeanFilter8)))
    b_gauss.clicked.connect(lambda: run("Gaussian", lambda a: _run_filt_u8(a, "Gaussian", py_p3dGaussianFilter8)))
    b_thr.clicked.connect(lambda: run("AutoThreshold", lambda a: _run_filt_u8(a, "AutoThreshold", py_p3dAutoThresholding8)))
    b_clr.clicked.connect(lambda: run("ClearBorder", lambda a: _run_filt_u8(a, "ClearBorder", py_p3dClearBorderFilter8)))
    b_min.clicked.connect(lambda: run("MinVolume", lambda a: _run_minvol_u8(a, spn_minvol.value()), spn_minvol.value()))
    b_lab.clicked.connect(lambda: run("BlobLabeling", lambda a: _run_blob_label_u8(a)))

    return root

# ---------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------
def _replay_p3d_run(viewer, step: Step) -> None:
    # ✅ Resolve target in a robust way:
    # - supports "__ACTIVE__"
    # - falls back to active image if old name isn't found
    L = resolve_target_layer(viewer, str(getattr(step, "target", "") or ""))

    p = step.params or {}
    arr = np.asarray(L.data)

    stride = int(p.get("stride", 1))
    mode = str(p.get("mode", "full"))
    inp = _downsample(arr, stride) if mode == "preview" else arr

    algo = str(p.get("algo", ""))
    if algo == "Median":
        out = _run_filt_u8(inp, algo, py_p3dMedianFilter8)
    elif algo == "Mean":
        out = _run_filt_u8(inp, algo, py_p3dMeanFilter8)
    elif algo == "Gaussian":
        out = _run_filt_u8(inp, algo, py_p3dGaussianFilter8)
    elif algo == "AutoThreshold":
        out = _run_filt_u8(inp, algo, py_p3dAutoThresholding8)
    elif algo == "ClearBorder":
        out = _run_filt_u8(inp, algo, py_p3dClearBorderFilter8)
    elif algo == "MinVolume":
        out = _run_minvol_u8(inp, int(p.get("minvol", 0)))
    elif algo == "BlobLabeling":
        out = _run_blob_label_u8(inp)
    else:
        raise RuntimeError(f"Unknown op {algo!r}")

    # ✅ Respect recorded output mode
    output = str(p.get("output", "new"))
    if output == "overwrite":
        L.data = out
        try:
            _reset_contrast(L)
        except Exception:
            pass
        show_info(f"Replayed p3d_run overwrite: '{L.name}' | {algo}")
        return

    # ✅ New layer naming:
    # If recipe stored a stale name, rebuild using current target name
    result_layer = str(p.get("result_layer", "") or "")
    if result_layer:
        # if it looks like "<old> | Algo", replace prefix with current
        if " | " in result_layer:
            suffix = result_layer.split(" | ", 1)[1]
            name = f"{L.name} | {suffix}"
        else:
            # allow user-specified name
            name = result_layer
    else:
        name = f"{L.name} | {algo}"

    viewer.add_image(out, name=name)
    show_info(f"Replayed p3d_run new layer: '{name}'")


register_handler("p3d_run", _replay_p3d_run)
