# napari_pypore3d/functions.py — r19 FIXED
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
    u8 = np.transpose(u8, (2, 1, 0))  # (Z,Y,X)->(X,Y,Z)
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
    try:
        return fn(vol, x, y, z, mv)
    except TypeError as e:
        last = e
    try:
        return fn(vol, x, y, z, minvol=mv)
    except TypeError as e:
        last = e
    try:
        return fn(vol, x, y, dimz=z, minvol=mv)
    except TypeError as e:
        last = e
    try:
        return fn(vol, x, y, mv, dimz=z)
    except TypeError as e:
        last = e
    raise TypeError(f"MinVolumeFilter3D call failed. Last error: {last!r}")

def _reset_contrast(layer: NapariImage) -> None:
    try:
        layer.reset_contrast_limits()
    except Exception:
        pass

def _short_name(src_name: str, op: str, mode: str, extra: str = "") -> str:
    base = src_name
    if len(base) > 22:
        base = base[:19] + "…"
    tag = "PREV" if mode.lower().startswith("preview") else "FULL"
    s = f"{base} | {op} [{tag}]"
    if extra:
        s += f" {extra}"
    return s

def _downsample_stride_for_target(voxels: int, target: int) -> int:
    if voxels <= target:
        return 1
    s = int(np.ceil((voxels / float(target)) ** (1.0 / 3.0)))
    return max(2, s)

def _downsample(a: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return a
    if a.ndim == 2:
        return np.ascontiguousarray(a[::stride, ::stride])
    return np.ascontiguousarray(a[::stride, ::stride, ::stride])

def _inherit_transform_and_fix_stride(src: Any, dst: Any, stride: int) -> None:
    """
    Copy spatial metadata AND fix preview downsample scaling:
      - If we downsample by stride, each pixel represents stride*src.scale
      - So dst.scale = src.scale * stride  (per axis)
    """
    # Copy translate first (origin)
    try:
        if hasattr(src, "translate") and hasattr(dst, "translate"):
            dst.translate = src.translate
    except Exception:
        pass

    # Copy scale then multiply by stride
    try:
        if hasattr(src, "scale") and hasattr(dst, "scale"):
            s = np.array(getattr(src, "scale", (1, 1, 1)), dtype=float)
            if s.size >= 1:
                s = s * float(stride)
            dst.scale = tuple(s.tolist())
    except Exception:
        # fallback: do nothing
        pass

    # Copy rotate/shear if present
    for attr in ("rotate", "shear"):
        try:
            if hasattr(src, attr) and hasattr(dst, attr):
                setattr(dst, attr, getattr(src, attr))
        except Exception:
            pass

def _is_maskish(op: str) -> bool:
    op = op.lower()
    return any(k in op for k in ["autothreshold", "minvol", "clearborder", "presetfastmask", "presetcleanlabel", "bloblabeling"])

def _to_labels_array(out_u8: np.ndarray) -> np.ndarray:
    """
    Convert uint8-ish output to labels:
      - threshold outputs are typically 0/255 (or 0/1)
      - blob labeling returns labeled ints (but stored in uint8)
    """
    a = np.asarray(out_u8)
    # make 0/255 -> 0/1
    if a.dtype == np.uint8:
        # fast check on a small sample
        flat = a.ravel()
        if flat.size:
            samp = flat[:: max(1, flat.size // 4096)]
            mx = int(samp.max())
            if mx == 255:
                a = (a > 0).astype(np.uint16)
            else:
                a = a.astype(np.uint16, copy=False)
        else:
            a = a.astype(np.uint16, copy=False)
    else:
        a = a.astype(np.uint16, copy=False)
    return np.ascontiguousarray(a)

# ---------------------------------------------------------------------
# Core runners (uint8 bridge)
# ---------------------------------------------------------------------
def _require_filt():
    if not _HAVE_FILT:
        raise RuntimeError("PyPore3D FILT wrappers missing (pypore3d.p3dFiltPy import failed).")

def _require_blob():
    if not _HAVE_BLOB:
        raise RuntimeError("PyPore3D BLOB wrappers missing (pypore3d.p3dBlobPy import failed).")

def _run_filt_u8(arr: np.ndarray, op_name: str, fn) -> np.ndarray:
    _require_filt()
    src = _ensure_2d_or_3d(arr)
    p3d_u8, (x, y, z), was_3d = _napari_to_p3d_u8(src)
    in_path, out_path = _tmp_paths()
    p3d_u8.ravel(order="C").tofile(in_path)

    try:
        v = py_p3dReadRaw8(str(in_path), x, y, dimz=z)
        v2 = _p3d_call3d(fn, v, x, y, z)
        py_p3dWriteRaw8(v2, str(out_path), x, y, dimz=z)
    except Exception as e:
        try:
            if _p3d_err_filt:
                _p3d_err_filt()
        except Exception:
            pass
        _cleanup(in_path, out_path)
        raise RuntimeError(f"PyPore3D failed ({op_name}): {e!r}") from e

    out = np.fromfile(out_path, dtype=np.uint8)
    _cleanup(in_path, out_path)
    if out.size != x * y * z:
        raise RuntimeError(f"Output size mismatch: got {out.size}, expected {x*y*z}")

    out_p3d = out.reshape((x, y, z), order="C")
    return _p3d_to_napari_u8(out_p3d, was_3d)

def _run_minvol_u8(arr: np.ndarray, minvol: int) -> np.ndarray:
    _require_filt()
    _require_blob()

    src = _ensure_2d_or_3d(arr)
    u8_src = _to_uint8_fast(src)

    p3d_u8, (x, y, z), was_3d = _napari_to_p3d_u8(u8_src)
    in_path, out_path = _tmp_paths()
    p3d_u8.ravel(order="C").tofile(in_path)

    try:
        v = py_p3dReadRaw8(str(in_path), x, y, dimz=z)
        v2 = _p3d_call3d_minvol(py_p3dMinVolumeFilter3D, v, x, y, z, minvol=minvol)
        py_p3dWriteRaw8(v2, str(out_path), x, y, dimz=z)
    except Exception as e:
        try:
            if _p3d_err_blob:
                _p3d_err_blob()
        except Exception:
            pass
        _cleanup(in_path, out_path)
        raise RuntimeError(f"PyPore3D failed (MinVolumeFilter3D): {e!r}") from e

    out = np.fromfile(out_path, dtype=np.uint8)
    _cleanup(in_path, out_path)
    if out.size != x * y * z:
        raise RuntimeError(f"Output size mismatch: got {out.size}, expected {x*y*z}")

    out_p3d = out.reshape((x, y, z), order="C")
    return _p3d_to_napari_u8(out_p3d, was_3d)

def _run_blob_label_u8(arr: np.ndarray) -> np.ndarray:
    _require_filt()
    _require_blob()

    src = _ensure_2d_or_3d(arr)
    u8_src = _to_uint8_fast(src)

    p3d_u8, (x, y, z), was_3d = _napari_to_p3d_u8(u8_src)
    in_path, out_path = _tmp_paths()
    p3d_u8.ravel(order="C").tofile(in_path)

    try:
        v = py_p3dReadRaw8(str(in_path), x, y, dimz=z)
        v2 = _p3d_call3d(py_p3dBlobLabeling, v, x, y, z)
        py_p3dWriteRaw8(v2, str(out_path), x, y, dimz=z)
    except Exception as e:
        try:
            if _p3d_err_blob:
                _p3d_err_blob()
        except Exception:
            pass
        _cleanup(in_path, out_path)
        raise RuntimeError(f"PyPore3D failed (BlobLabeling): {e!r}") from e

    out = np.fromfile(out_path, dtype=np.uint8)
    _cleanup(in_path, out_path)
    if out.size != x * y * z:
        raise RuntimeError(f"Output size mismatch: got {out.size}, expected {x*y*z}")

    out_p3d = out.reshape((x, y, z), order="C")
    return _p3d_to_napari_u8(out_p3d, was_3d)

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
@dataclass
class _Job:
    op: str
    fn: Callable[[], np.ndarray]
    mode: str
    stride: int
    src_name: str
    output: str  # "new" or "overwrite"
    minvol: int = 0
    allow_huge: bool = False
    preset: str = ""

def functions_widget() -> QWidget:
    root = QWidget()
    root.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    outer = QVBoxLayout(root)
    outer.setContentsMargins(10, 10, 10, 10)
    outer.setSpacing(10)

    title = QLabel("PyPore3D Workflow (Scientist Mode)")
    title.setStyleSheet("font-size: 12pt; font-weight: 600;")
    outer.addWidget(title)

    subtitle = QLabel("Preview = auto-downsample for huge volumes. Full = run on full resolution (guarded).")
    subtitle.setStyleSheet("opacity: 0.85;")
    subtitle.setWordWrap(True)
    outer.addWidget(subtitle)

    # ---------- top controls ----------
    top = QGridLayout()
    top.setHorizontalSpacing(10)
    top.setVerticalSpacing(6)

    lbl_layer = QLabel("Target layer")
    cmb_layer = QComboBox()
    cmb_layer.addItem("<active image>")

    lbl_mode = QLabel("Run mode")
    cmb_mode = QComboBox()
    cmb_mode.addItems(["Preview (auto-downsample)", "Full (original resolution)"])

    lbl_out = QLabel("Output")
    cmb_out = QComboBox()
    cmb_out.addItems(["new layer", "overwrite target"])

    lbl_minvol = QLabel("Min volume (voxels)")
    spn_minvol = QSpinBox()
    spn_minvol.setRange(0, 50_000_000)
    spn_minvol.setValue(50)
    spn_minvol.setSingleStep(10)

    chk_allow_huge = QCheckBox("Allow huge (unsafe)")
    chk_allow_huge.setToolTip("If enabled, Full mode can run up to a very high limit. Might be slow.")

    top.addWidget(lbl_layer, 0, 0)
    top.addWidget(cmb_layer, 0, 1)
    top.addWidget(lbl_mode, 1, 0)
    top.addWidget(cmb_mode, 1, 1)
    top.addWidget(lbl_out, 2, 0)
    top.addWidget(cmb_out, 2, 1)
    top.addWidget(lbl_minvol, 3, 0)
    top.addWidget(spn_minvol, 3, 1)
    top.addWidget(chk_allow_huge, 4, 0, 1, 2)

    outer.addLayout(top)

    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    outer.addWidget(line)

    # ---------- groups ----------
    def group(title_text: str) -> tuple[QGroupBox, QVBoxLayout]:
        g = QGroupBox(title_text)
        g.setStyleSheet("QGroupBox { font-weight: 600; }")
        lay = QVBoxLayout(g)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        return g, lay

    def big_button(text: str, tip: str = "") -> QPushButton:
        b = QPushButton(text)
        b.setMinimumHeight(34)
        b.setToolTip(tip)
        b.setStyleSheet("QPushButton { font-size: 10pt; padding: 6px 10px; }")
        return b

    g_pre, lay_pre = group("Preprocess")
    g_mask, lay_mask = group("Mask")
    g_clean, lay_clean = group("Clean")
    g_lab, lay_lab = group("Label")
    g_presets, lay_presets = group("Presets")

    btn_median = big_button("Median filter (8-bit)", "py_p3dMedianFilter8 (uint8)")
    btn_mean = big_button("Mean filter (8-bit)", "py_p3dMeanFilter8 (uint8)")
    btn_gauss = big_button("Gaussian filter (8-bit)", "py_p3dGaussianFilter8 (uint8)")
    lay_pre.addWidget(btn_median)
    lay_pre.addWidget(btn_mean)
    lay_pre.addWidget(btn_gauss)

    btn_thresh = big_button("Make mask (AutoThreshold)", "py_p3dAutoThresholding8 (uint8)")
    lay_mask.addWidget(btn_thresh)

    btn_minvol = big_button("Remove small blobs (MinVolume)", "py_p3dMinVolumeFilter3D (uses Min volume)")
    btn_clear = big_button("Clear border", "py_p3dClearBorderFilter8 (mask-ish)")
    lay_clean.addWidget(btn_minvol)
    lay_clean.addWidget(btn_clear)

    btn_label = big_button("Label blobs (BlobLabeling)", "py_p3dBlobLabeling → labels (uint8)")
    lay_lab.addWidget(btn_label)

    btn_preset_fast = big_button("Fast mask preset: threshold → minvol → clear", "One-click quick mask pipeline")
    btn_preset_label = big_button("Clean + label preset: minvol → clear → label", "Assumes input is mask-ish")
    lay_presets.addWidget(btn_preset_fast)
    lay_presets.addWidget(btn_preset_label)

    # Two-column layout
    cols = QHBoxLayout()
    cols.setSpacing(10)
    left = QVBoxLayout()
    right = QVBoxLayout()
    left.setSpacing(10)
    right.setSpacing(10)
    left.addWidget(g_pre)
    left.addWidget(g_mask)
    right.addWidget(g_clean)
    right.addWidget(g_lab)
    cols.addLayout(left, 1)
    cols.addLayout(right, 1)
    outer.addLayout(cols)
    outer.addWidget(g_presets)

    # ---------- log ----------
    log_title = QLabel("Log")
    log_title.setStyleSheet("font-weight: 600;")
    outer.addWidget(log_title)

    log = QPlainTextEdit()
    log.setReadOnly(True)
    log.setMinimumHeight(160)
    log.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
    outer.addWidget(log)

    def _log(level: str, msg: str) -> None:
        log.appendPlainText(f"{_now()} {level:<6} {msg}")

    def refresh_layers() -> None:
        v = _v()
        names = ["<active image>"]
        if v is not None:
            for L in v.layers:
                if isinstance(L, NapariImage):
                    names.append(L.name)
        cur = cmb_layer.currentText()
        cmb_layer.blockSignals(True)
        cmb_layer.clear()
        cmb_layer.addItems(names)
        if cur in names:
            cmb_layer.setCurrentText(cur)
        else:
            cmb_layer.setCurrentIndex(0)
        cmb_layer.blockSignals(False)

    def pick_layer() -> Optional[NapariImage]:
        v = _v()
        if v is None:
            return None
        name = cmb_layer.currentText()

        if name == "<active image>":
            lyr = getattr(v.layers.selection, "active", None)
            if isinstance(lyr, NapariImage):
                return lyr
            for L in reversed(list(v.layers)):
                if isinstance(L, NapariImage):
                    return L
            return None

        for L in v.layers:
            if isinstance(L, NapariImage) and L.name == name:
                return L
        return None

    def is_preview() -> bool:
        return cmb_mode.currentText().lower().startswith("preview")

    def overwrite() -> bool:
        return cmb_out.currentText().lower().startswith("overwrite")

    def guard_volume(vox: int) -> None:
        if is_preview():
            return
        if vox > _FULL_ABS_MAX_VOXELS:
            raise RuntimeError(
                f"Refused: volume too large ({vox:,} voxels). Absolute max is {_FULL_ABS_MAX_VOXELS:,}."
            )
        if (not chk_allow_huge.isChecked()) and vox > _FULL_SAFE_VOXELS:
            raise RuntimeError(
                f"Refused (Full mode): {vox:,} voxels exceeds safe limit {_FULL_SAFE_VOXELS:,}. "
                f"Use Preview or enable 'Allow huge (unsafe)'."
            )

    def make_input(layer: NapariImage) -> tuple[np.ndarray, int]:
        arr = _ensure_2d_or_3d(np.asarray(layer.data))
        vox = _voxels(arr)
        guard_volume(vox)

        if is_preview():
            s = _downsample_stride_for_target(vox, _PREVIEW_TARGET_VOXELS)
            return _downsample(arr, s), s

        return arr, 1

    def _insert_above(viewer, src_layer, new_layer) -> None:
        """Ensure result is ABOVE the source in layer list."""
        try:
            idx = list(viewer.layers).index(src_layer)
            viewer.layers.move(viewer.layers.index(new_layer), idx + 1)
        except Exception:
            pass

    def _apply_output(layer: NapariImage, out_arr: np.ndarray, job: _Job) -> str:
        v = _v()
        if v is None:
            return layer.name

        if job.output == "overwrite":
            layer.data = out_arr
            _reset_contrast(layer)
            try:
                layer.metadata = dict(layer.metadata) if isinstance(layer.metadata, dict) else {}
                layer.metadata["p3d_last_op"] = {
                    "op": job.op, "mode": job.mode, "stride": job.stride,
                    "minvol": job.minvol, "preset": job.preset,
                }
            except Exception:
                pass
            try:
                v.layers.selection.active = layer
            except Exception:
                pass
            show_info(f"Done: {job.op} (overwritten)")
            _log("OK", f"Done: {job.op} (overwrite)")
            return layer.name

        extra = f"(s={job.stride})" if job.stride > 1 else ""
        new_name = _short_name(job.src_name, job.op, job.mode, extra=extra)

        # ✅ IMPORTANT: mask-ish outputs become Labels overlay, not Image
        if _is_maskish(job.op):
            labels = _to_labels_array(out_arr)
            new_layer = v.add_labels(labels, name=new_name)
            # overlay feel
            try:
                new_layer.opacity = 0.70
                new_layer.contour = 1
            except Exception:
                pass
        else:
            new_layer = v.add_image(out_arr, name=new_name)

        # ✅ Fix tiny/misplaced preview: copy transforms AND scale*stride
        _inherit_transform_and_fix_stride(layer, new_layer, job.stride)

        # Make sure it is ABOVE the source in layer list
        _insert_above(v, layer, new_layer)

        # Keep source visible (so you don't get "black screen")
        try:
            layer.visible = True
        except Exception:
            pass

        # Metadata
        try:
            new_layer.metadata = dict(new_layer.metadata) if isinstance(new_layer.metadata, dict) else {}
            new_layer.metadata["p3d_op"] = {
                "op": job.op,
                "mode": job.mode,
                "stride": job.stride,
                "source": job.src_name,
                "minvol": job.minvol,
                "preset": job.preset,
            }
            new_layer.metadata["_pypore3d_result"] = True
        except Exception:
            pass

        try:
            v.layers.selection.active = new_layer
        except Exception:
            pass

        show_info(f"Done: {job.op} → {new_name}")
        _log("OK", f"Done: {job.op} → '{new_name}'")
        return new_layer.name

    def _record_step(op_name: str, src_layer: NapariImage, result_layer_name: str, params: Dict[str, Any], notes: str = ""):
        p = dict(params)
        p["result_layer"] = str(result_layer_name)
        RECORDER.add_step(op=op_name, target=src_layer.name, params=p, notes=notes)

    def run_job(op: str, runner: Callable[[np.ndarray], np.ndarray], *, minvol: int = 0) -> None:
        refresh_layers()
        layer = pick_layer()
        if layer is None:
            show_warning("No Image layer selected.")
            _log("WARN", "No Image layer selected.")
            return

        try:
            in_arr, stride = make_input(layer)
        except Exception as e:
            show_error(str(e))
            _log("ERROR", str(e))
            return

        mode = "preview" if is_preview() else "full"
        out_mode = "overwrite" if overwrite() else "new"
        allow_huge = bool(chk_allow_huge.isChecked())
        vox = _voxels(np.asarray(layer.data))
        _log("INFO", f"Run {op} on '{layer.name}' | mode={mode} | voxels={vox:,} | stride={stride} | out={out_mode}")

        job = _Job(
            op=op,
            fn=lambda: runner(in_arr),
            mode=mode,
            stride=stride,
            src_name=layer.name,
            output=out_mode,
            minvol=int(minvol),
            allow_huge=allow_huge,
        )

        def _do() -> np.ndarray:
            return job.fn()

        def _on_done(out_arr: np.ndarray) -> None:
            try:
                result_name = _apply_output(layer, out_arr, job)
                _record_step(
                    "p3d_run",
                    layer,
                    result_name,
                    params={
                        "algo": job.op,
                        "mode": job.mode,
                        "stride": int(job.stride),
                        "output": job.output,
                        "allow_huge": bool(job.allow_huge),
                        "minvol": int(job.minvol),
                    },
                )
            except Exception as e:
                show_error(f"Apply output failed: {e!r}")
                _log("ERROR", f"Apply output failed: {e!r}")

        def _on_err(err: Any) -> None:
            show_error(f"{op} failed: {err!r}")
            _log("ERROR", f"{op} failed: {err!r}")

        if thread_worker is None:
            try:
                _on_done(_do())
            except Exception as e:
                _on_err(e)
            return

        worker = thread_worker(_do)()
        worker.returned.connect(_on_done)
        worker.errored.connect(lambda e: _on_err(e))  # type: ignore
        worker.start()

    # ---- ops
    def op_median():
        run_job("Median", lambda a: _run_filt_u8(a, "MedianFilter8", py_p3dMedianFilter8))

    def op_mean():
        run_job("Mean", lambda a: _run_filt_u8(a, "MeanFilter8", py_p3dMeanFilter8))

    def op_gauss():
        run_job("Gaussian", lambda a: _run_filt_u8(a, "GaussianFilter8", py_p3dGaussianFilter8))

    def op_thresh():
        run_job("AutoThreshold", lambda a: _run_filt_u8(a, "AutoThresholding8", py_p3dAutoThresholding8))

    def op_clear():
        run_job("ClearBorder", lambda a: _run_filt_u8(a, "ClearBorderFilter8", py_p3dClearBorderFilter8))

    def op_minvol():
        mv = int(spn_minvol.value())
        run_job(f"MinVol({mv})", lambda a: _run_minvol_u8(a, mv), minvol=mv)

    def op_label():
        run_job("BlobLabeling", lambda a: _run_blob_label_u8(a))

    # ---- presets
    def preset_fast_mask():
        refresh_layers()
        layer = pick_layer()
        if layer is None:
            show_warning("No Image layer selected.")
            _log("WARN", "No Image layer selected.")
            return

        mv = int(spn_minvol.value())
        mode = "preview" if is_preview() else "full"
        out_mode = "overwrite" if overwrite() else "new"
        allow_huge = bool(chk_allow_huge.isChecked())

        try:
            in_arr, stride = make_input(layer)
        except Exception as e:
            show_error(str(e))
            _log("ERROR", str(e))
            return

        _log("INFO", f"Preset: fast mask | mode={mode} | stride={stride} | minvol={mv} | out={out_mode}")

        job = _Job(
            op="PresetFastMask",
            fn=lambda: in_arr,
            mode=mode,
            stride=stride,
            src_name=layer.name,
            output=out_mode,
            minvol=mv,
            allow_huge=allow_huge,
            preset="fast_mask",
        )

        def _pipeline() -> np.ndarray:
            a1 = _run_filt_u8(in_arr, "AutoThresholding8", py_p3dAutoThresholding8)
            a2 = _run_minvol_u8(a1, mv)
            a3 = _run_filt_u8(a2, "ClearBorderFilter8", py_p3dClearBorderFilter8)
            return a3

        def _on_done(out: np.ndarray) -> None:
            try:
                result_name = _apply_output(layer, out, job)
                _record_step(
                    "p3d_preset_fast_mask",
                    layer,
                    result_name,
                    params={
                        "mode": mode,
                        "stride": int(stride),
                        "output": out_mode,
                        "allow_huge": bool(allow_huge),
                        "minvol": int(mv),
                    },
                )
            except Exception as e:
                show_error(f"Preset apply failed: {e!r}")
                _log("ERROR", f"Preset apply failed: {e!r}")

        def _on_err(err: Any) -> None:
            show_error(f"Preset failed: {err!r}")
            _log("ERROR", f"Preset failed: {err!r}")

        if thread_worker is None:
            try:
                _on_done(_pipeline())
            except Exception as e:
                _on_err(e)
            return

        worker = thread_worker(_pipeline)()
        worker.returned.connect(_on_done)
        worker.errored.connect(lambda e: _on_err(e))  # type: ignore
        worker.start()

    def preset_clean_label():
        refresh_layers()
        layer = pick_layer()
        if layer is None:
            show_warning("No Image layer selected.")
            _log("WARN", "No Image layer selected.")
            return

        mv = int(spn_minvol.value())
        mode = "preview" if is_preview() else "full"
        out_mode = "overwrite" if overwrite() else "new"
        allow_huge = bool(chk_allow_huge.isChecked())

        try:
            in_arr, stride = make_input(layer)
        except Exception as e:
            show_error(str(e))
            _log("ERROR", str(e))
            return

        _log("INFO", f"Preset: clean+label | mode={mode} | stride={stride} | minvol={mv} | out={out_mode}")

        job = _Job(
            op="PresetCleanLabel",
            fn=lambda: in_arr,
            mode=mode,
            stride=stride,
            src_name=layer.name,
            output=out_mode,
            minvol=mv,
            allow_huge=allow_huge,
            preset="clean_label",
        )

        def _pipeline() -> np.ndarray:
            a1 = _run_minvol_u8(in_arr, mv)
            a2 = _run_filt_u8(a1, "ClearBorderFilter8", py_p3dClearBorderFilter8)
            a3 = _run_blob_label_u8(a2)
            return a3

        def _on_done(out: np.ndarray) -> None:
            try:
                result_name = _apply_output(layer, out, job)
                _record_step(
                    "p3d_preset_clean_label",
                    layer,
                    result_name,
                    params={
                        "mode": mode,
                        "stride": int(stride),
                        "output": out_mode,
                        "allow_huge": bool(allow_huge),
                        "minvol": int(mv),
                    },
                )
            except Exception as e:
                show_error(f"Preset apply failed: {e!r}")
                _log("ERROR", f"Preset apply failed: {e!r}")

        def _on_err(err: Any) -> None:
            show_error(f"Preset failed: {err!r}")
            _log("ERROR", f"Preset failed: {err!r}")

        if thread_worker is None:
            try:
                _on_done(_pipeline())
            except Exception as e:
                _on_err(e)
            return

        worker = thread_worker(_pipeline)()
        worker.returned.connect(_on_done)
        worker.errored.connect(lambda e: _on_err(e))  # type: ignore
        worker.start()

    # wire buttons
    btn_median.clicked.connect(op_median)
    btn_mean.clicked.connect(op_mean)
    btn_gauss.clicked.connect(op_gauss)
    btn_thresh.clicked.connect(op_thresh)
    btn_clear.clicked.connect(op_clear)
    btn_minvol.clicked.connect(op_minvol)
    btn_label.clicked.connect(op_label)
    btn_preset_fast.clicked.connect(preset_fast_mask)
    btn_preset_label.clicked.connect(preset_clean_label)

    refresh_layers()
    if _HAVE_FILT and _HAVE_BLOB:
        _log("INFO", "Ready: PyPore3D FILT + BLOB available.")
    elif _HAVE_FILT:
        _log("WARN", "FILT available, but BLOB missing (p3dBlobPy import failed).")
    else:
        _log("ERROR", "PyPore3D FILT missing (p3dFiltPy import failed).")

    return root

# ---------------------------------------------------------------------
# Replay handlers (recipe reproduction)
# ---------------------------------------------------------------------
def _find_image(viewer, name: str) -> Optional[NapariImage]:
    for L in reversed(list(viewer.layers)):
        if isinstance(L, NapariImage) and L.name == name:
            return L
    return None

def _replay_apply_output(viewer, src: NapariImage, out_arr: np.ndarray, output: str, result_layer_name: str, stride: int, op: str) -> None:
    if output == "overwrite":
        src.data = out_arr
        try:
            src.reset_contrast_limits()
        except Exception:
            pass
        try:
            viewer.layers.selection.active = src
        except Exception:
            pass
        return

    name = str(result_layer_name) if result_layer_name else f"{src.name}_p3d"

    if _is_maskish(op):
        labels = _to_labels_array(out_arr)
        newL = viewer.add_labels(labels, name=name)
        try:
            newL.opacity = 0.70
            newL.contour = 1
        except Exception:
            pass
    else:
        newL = viewer.add_image(out_arr, name=name)
        try:
            newL.reset_contrast_limits()
        except Exception:
            pass

    _inherit_transform_and_fix_stride(src, newL, stride)

    # place above src
    try:
        idx = list(viewer.layers).index(src)
        viewer.layers.move(viewer.layers.index(newL), idx + 1)
    except Exception:
        pass

    try:
        viewer.layers.selection.active = newL
    except Exception:
        pass

def _replay_p3d_run(viewer, step: Step) -> None:
    L = _find_image(viewer, step.target)
    if L is None:
        raise RuntimeError(f"p3d_run: target layer not found: {step.target}")

    p = step.params or {}
    algo = str(p.get("algo", ""))
    mode = str(p.get("mode", "full"))
    stride = int(p.get("stride", 1) or 1)
    output = str(p.get("output", "new"))
    minvol = int(p.get("minvol", 0) or 0)
    result_name = str(p.get("result_layer") or "")

    arr = _ensure_2d_or_3d(np.asarray(L.data))
    in_arr = _downsample(arr, stride) if (mode == "preview" and stride > 1) else arr

    if algo.startswith("Median"):
        out = _run_filt_u8(in_arr, "MedianFilter8", py_p3dMedianFilter8)
    elif algo.startswith("Mean"):
        out = _run_filt_u8(in_arr, "MeanFilter8", py_p3dMeanFilter8)
    elif algo.startswith("Gaussian"):
        out = _run_filt_u8(in_arr, "GaussianFilter8", py_p3dGaussianFilter8)
    elif algo.startswith("AutoThreshold"):
        out = _run_filt_u8(in_arr, "AutoThresholding8", py_p3dAutoThresholding8)
    elif algo.startswith("ClearBorder"):
        out = _run_filt_u8(in_arr, "ClearBorderFilter8", py_p3dClearBorderFilter8)
    elif algo.startswith("MinVol"):
        out = _run_minvol_u8(in_arr, minvol)
    elif algo.startswith("BlobLabeling"):
        out = _run_blob_label_u8(in_arr)
    else:
        raise RuntimeError(f"p3d_run: unknown algo '{algo}'")

    _replay_apply_output(viewer, L, out, output, result_name, stride=stride, op=algo)

def _replay_preset_fast_mask(viewer, step: Step) -> None:
    L = _find_image(viewer, step.target)
    if L is None:
        raise RuntimeError(f"p3d_preset_fast_mask: target layer not found: {step.target}")

    p = step.params or {}
    mode = str(p.get("mode", "full"))
    stride = int(p.get("stride", 1) or 1)
    output = str(p.get("output", "new"))
    mv = int(p.get("minvol", 50) or 50)
    result_name = str(p.get("result_layer") or "")

    arr = _ensure_2d_or_3d(np.asarray(L.data))
    in_arr = _downsample(arr, stride) if (mode == "preview" and stride > 1) else arr

    a1 = _run_filt_u8(in_arr, "AutoThresholding8", py_p3dAutoThresholding8)
    a2 = _run_minvol_u8(a1, mv)
    a3 = _run_filt_u8(a2, "ClearBorderFilter8", py_p3dClearBorderFilter8)

    _replay_apply_output(viewer, L, a3, output, result_name, stride=stride, op="PresetFastMask")

def _replay_preset_clean_label(viewer, step: Step) -> None:
    L = _find_image(viewer, step.target)
    if L is None:
        raise RuntimeError(f"p3d_preset_clean_label: target layer not found: {step.target}")

    p = step.params or {}
    mode = str(p.get("mode", "full"))
    stride = int(p.get("stride", 1) or 1)
    output = str(p.get("output", "new"))
    mv = int(p.get("minvol", 50) or 50)
    result_name = str(p.get("result_layer") or "")

    arr = _ensure_2d_or_3d(np.asarray(L.data))
    in_arr = _downsample(arr, stride) if (mode == "preview" and stride > 1) else arr

    a1 = _run_minvol_u8(in_arr, mv)
    a2 = _run_filt_u8(a1, "ClearBorderFilter8", py_p3dClearBorderFilter8)
    a3 = _run_blob_label_u8(a2)

    _replay_apply_output(viewer, L, a3, output, result_name, stride=stride, op="PresetCleanLabel")

register_handler("p3d_run", _replay_p3d_run)
register_handler("p3d_preset_fast_mask", _replay_preset_fast_mask)
register_handler("p3d_preset_clean_label", _replay_preset_clean_label)
