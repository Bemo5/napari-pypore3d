# napari_pypore3d/info_export.py — r35
# ---------------------------------------------------------------------
# Info panel (shows active image details) + Export tools:
#  • Save as .npy
#  • Save as .tif (if tifffile installed)
#  • Save as .raw (little-endian; pick dtype)
#
# Public API:
#   build_info_export_panel(settings) -> (QWidget, Callable[[], None])
#     - QWidget: panel you can add to a tab
#     - Callable: refresh() to re-read active image & update fields
#
from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict
import pathlib
import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFileDialog, QComboBox, QSizePolicy
)
from qtpy.QtCore import Qt
from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_info, show_warning, show_error

# helpers from the package
from .helpers import (
    active_image,
    array_stats,
    voxel,
    export_raw_little,
)

# optional tifffile
try:
    import tifffile as _tifffile
    HAVE_TIFF = True
except Exception:
    HAVE_TIFF = False
    _tifffile = None  # type: ignore


_DTYPE_CHOICES = [
    "uint8", "int8", "uint16", "int16",
    "uint32", "int32", "float32", "float64",
]
_DTYPE_MAP = {
    "uint8":  np.uint8,
    "int8":   np.int8,
    "uint16": np.uint16,
    "int16":  np.int16,
    "uint32": np.uint32,
    "int32":  np.int32,
    "float32": np.float32,
    "float64": np.float64,
}


def _fmt_tuple(t) -> str:
    try:
        return str(tuple(int(x) for x in t))
    except Exception:
        return str(tuple(t))


def _layer_summary(L: NapariImage) -> Dict[str, str]:
    a = np.asarray(L.data)
    st = array_stats(a)
    vs = voxel(L)
    path = L.metadata.get("_file_path", "")
    cl_txt = ""
    try:
        lo, hi = L.contrast_limits if L.contrast_limits is not None else (None, None)
        if lo is not None and hi is not None:
            cl_txt = f"[{float(lo):.6g}, {float(hi):.6g}]"
    except Exception:
        cl_txt = ""
    return {
        "name": L.name,
        "shape": _fmt_tuple(a.shape),
        "dtype": getattr(a.dtype, "name", str(a.dtype)),
        "min": f"{st['min']:.6g}",
        "max": f"{st['max']:.6g}",
        "mean": f"{st['mean']:.6g}",
        "std": f"{st['std']:.6g}",
        "voxel": _fmt_tuple(vs) if vs else "—",
        "contrast": cl_txt or "—",
        "path": str(path) if path else "—",
    }


def build_info_export_panel(_settings) -> tuple[QWidget, Callable[[], None]]:
    """
    Create the Info / Export tab and a refresh() function.
    """
    # --- UI skeleton
    root = QWidget()
    root.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    vbox = QVBoxLayout(root)

    grid = QGridLayout()
    grid.setVerticalSpacing(4)
    labels_left = [
        "Layer", "Shape", "Dtype", "Min", "Max", "Mean", "Std",
        "Voxel size (z,y,x)", "Contrast limits", "Source path"
    ]
    value_labels: Dict[str, QLabel] = {}
    for i, key in enumerate(labels_left):
        grid.addWidget(QLabel(key + ":"), i, 0, Qt.AlignRight)
        lab = QLabel("—")
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        value_labels[key] = lab
        grid.addWidget(lab, i, 1)
    vbox.addLayout(grid)

    # --- export row (.npy / .tif / .raw)
    row = QHBoxLayout()
    btn_npy = QPushButton("Save as .npy")
    btn_tif = QPushButton("Save as .tif")
    if not HAVE_TIFF:
        btn_tif.setEnabled(False)
        btn_tif.setToolTip("Install 'tifffile' to enable TIFF export.")
    btn_raw = QPushButton("Save as .raw")
    raw_dtype = QComboBox()
    raw_dtype.addItems(_DTYPE_CHOICES)
    raw_dtype.setCurrentText("uint16")

    row.addWidget(btn_npy)
    row.addWidget(btn_tif)
    row.addWidget(btn_raw)
    row.addWidget(QLabel("dtype:"))
    row.addWidget(raw_dtype)
    row.addStretch(1)
    vbox.addLayout(row)

    # --------------------------------------------------------
    # behavior
    # --------------------------------------------------------
    def _get_active() -> Optional[NapariImage]:
        L, _ = active_image()
        return L

    def refresh() -> None:
        L = _get_active()
        if not L:
            for lab in value_labels.values():
                lab.setText("—")
            return
        info = _layer_summary(L)
        value_labels["Layer"].setText(info["name"])
        value_labels["Shape"].setText(info["shape"])
        value_labels["Dtype"].setText(info["dtype"])
        value_labels["Min"].setText(info["min"])
        value_labels["Max"].setText(info["max"])
        value_labels["Mean"].setText(info["mean"])
        value_labels["Std"].setText(info["std"])
        value_labels["Voxel size (z,y,x)"].setText(info["voxel"])
        value_labels["Contrast limits"].setText(info["contrast"])
        value_labels["Source path"].setText(info["path"])

    def _pick_save_path(suffix: str, title: str) -> Optional[pathlib.Path]:
        default_dir = ""
        L = _get_active()
        if L:
            p = L.metadata.get("_file_path")
            if p:
                default_dir = str(pathlib.Path(p).parent)
        fn, _ = QFileDialog.getSaveFileName(
            None,
            title,
            str(pathlib.Path(default_dir) / f"{value_labels['Layer'].text()}{suffix}"),
            f"{suffix[1:].upper()} files (*{suffix});;All files (*)",
        )
        return pathlib.Path(fn) if fn else None

    def _on_save_npy():
        L = _get_active()
        if not L:
            show_warning("No active image.")
            return
        path = _pick_save_path(".npy", "Save array as .npy")
        if not path:
            return
        try:
            np.save(path, np.asarray(L.data))
            show_info(f"Saved {path.name}")
        except Exception as e:
            show_error(f"Save failed: {e}")

    def _on_save_tif():
        if not HAVE_TIFF:
            return
        L = _get_active()
        if not L:
            show_warning("No active image.")
            return
        path = _pick_save_path(".tif", "Save array as .tif")
        if not path:
            return
        try:
            arr = np.asarray(L.data)
            _tifffile.imwrite(str(path), arr, dtype=arr.dtype, bigtiff=True)
            show_info(f"Saved {path.name}")
        except Exception as e:
            show_error(f"TIFF save failed: {e}")

    def _on_save_raw():
        L = _get_active()
        if not L:
            show_warning("No active image.")
            return
        path = _pick_save_path(".raw", "Save array as .raw (little-endian)")
        if not path:
            return
        try:
            dt = _DTYPE_MAP[raw_dtype.currentText()]
            export_raw_little(path, np.asarray(L.data), np.dtype(dt))
            show_info(f"Saved {path.name} (dtype={np.dtype(dt).name})")
        except Exception as e:
            show_error(f"RAW save failed: {e}")

    btn_npy.clicked.connect(_on_save_npy)
    btn_tif.clicked.connect(_on_save_tif)
    btn_raw.clicked.connect(_on_save_raw)

    # initial paint
    refresh()
    return root, refresh
