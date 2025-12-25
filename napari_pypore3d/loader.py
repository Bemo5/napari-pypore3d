# napari_pypore3d/loader.py
# ---------------------------------------------------------------------
# RAW loader + bulk add helpers.
# Uses helpers.py for shared utilities (shape hints, cropping, names, etc.).
#####----------------------DEPRICATED--------------------------#####
#####----------------------DEPRICATED--------------------------#####
#####----------------------DEPRICATED--------------------------#####
#####----------------------DEPRICATED--------------------------#####
#####----------------------DEPRICATED--------------------------#####
#####----------------------DEPRICATED--------------------------#####
#####----------------------DEPRICATED--------------------------#####

from __future__ import annotations
import pathlib
import os
from typing import Optional, Tuple, List

import numpy as np
from magicgui import magicgui
from magicgui.widgets import Label
from qtpy.QtWidgets import QFileDialog, QFrame
from qtpy.QtCore import QTimer
from qtpy.QtCore import QItemSelectionModel

from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_warning, show_info

from .helpers import (
    AppSettings,            # persisted settings
    best_trip_from_name,    # filename AxBxC / N^3 hints
    current_min_zyx,        # global min Z,Y,X across images
    center_crop_indices,    # center-crop to target
    unique_layer_name,      # avoid name clashes
    last_zyx,               # shape helper
    natural_sort_key,       # sort file names naturally
)

# ---------------------------------------------------------------------
# tiny MGUI hline (local, so loader.py works standalone during imports)
# ---------------------------------------------------------------------
def _mgui_hline() -> Label:
    L = Label(value="")
    try:
        L.native.setFrameShape(QFrame.HLine)
        L.native.setFrameShadow(QFrame.Sunken)
    except Exception:
        pass
    return L


# ---------------------------------------------------------------------
# ACTIVE LAYER SELECTION (robust: model + Qt layer list highlight)
# ---------------------------------------------------------------------
def _unwrap(obj):
    return getattr(obj, "__wrapped__", obj)


def _find_qt_layer_list_view(v):
    """
    Try hard to locate the Qt view that backs the Layers panel list,
    so we can visually highlight the selected row.
    """
    try:
        win = getattr(v, "window", None)
        if win is None:
            return None

        qt_viewer = getattr(win, "qt_viewer", None)
        if qt_viewer is None:
            qt_viewer = getattr(win, "_qt_viewer", None)
        if qt_viewer is None:
            return None

        qt_layers = getattr(qt_viewer, "layers", None)
        if qt_layers is None:
            qt_layers = getattr(qt_viewer, "_layers", None)
        if qt_layers is None:
            return None

        candidates = [
            getattr(qt_layers, "list_view", None),
            getattr(qt_layers, "_list_view", None),
            getattr(qt_layers, "view", None),
            getattr(qt_layers, "_view", None),
            qt_layers,  # sometimes it *is* the view-ish object
        ]

        for w in candidates:
            if w is None:
                continue
            if hasattr(w, "model") and hasattr(w, "setCurrentIndex"):
                return w

        return None
    except Exception:
        return None


def _try_set_model_active(v, layer) -> None:
    # napari data-model selection
    try:
        v.layers.selection.select_only(layer)
    except Exception:
        try:
            v.layers.selection.clear()
            v.layers.selection.add(layer)
        except Exception:
            pass

    # some versions expose selection.active
    try:
        v.layers.selection.active = layer
    except Exception:
        pass

    # some versions expose viewer.layers.active
    try:
        v.layers.active = layer
    except Exception:
        pass


def _try_set_qt_highlight(v, layer) -> None:
    """
    Force the GUI Layers list to highlight the row.
    """
    view = _find_qt_layer_list_view(v)
    if view is None:
        return

    try:
        row = v.layers.index(layer)
    except Exception:
        return

    try:
        model = view.model()
        idx = model.index(row, 0)
        if not idx.isValid():
            return

        view.setCurrentIndex(idx)

        sm = view.selectionModel()
        if sm is not None:
            sm.select(idx, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
    except Exception:
        pass


def _force_select_layer(viewer, layer) -> None:
    """
    Make the given layer active/selected and keep trying until it "sticks".
    This beats other callbacks that steal selection right after add_image().
    """
    v = _unwrap(viewer)
    target = _unwrap(layer)

    # keep timers alive (avoid GC)
    try:
        timers = getattr(v, "_pypore3d_select_timers", None)
        if timers is None:
            timers = []
            setattr(v, "_pypore3d_select_timers", timers)
    except Exception:
        timers = []

    tries = {"n": 0}
    timer = QTimer()
    timer.setSingleShot(False)

    def _tick():
        tries["n"] += 1

        # find the actual instance in v.layers (proxy-safe)
        actual = None
        try:
            if target in v.layers:
                actual = target
        except Exception:
            actual = None

        if actual is None:
            try:
                actual = v.layers[target.name]
            except Exception:
                actual = None

        if actual is None:
            try:
                actual = v.layers[-1]
            except Exception:
                timer.stop()
                return

        _try_set_model_active(v, actual)
        _try_set_qt_highlight(v, actual)

        # stop early if it's active now
        try:
            if getattr(v.layers.selection, "active", None) is actual:
                timer.stop()
                return
        except Exception:
            pass

        # hard stop after ~2s
        if tries["n"] >= 40:
            timer.stop()

    timer.timeout.connect(_tick)
    timer.start(50)
    QTimer.singleShot(0, _tick)

    try:
        timers.append(timer)
    except Exception:
        pass


# ---------------------------------------------------------------------
# dtype choices for the combo
_DTYPE_CHOICES = [
    "auto", "uint8", "int8", "uint16", "int16",
    "uint32", "int32", "float32", "float64",
]


def _infer_shape_and_dtype(
    p: pathlib.Path,
    shape_y: int,
    shape_x: int,
) -> Tuple[Tuple[int, int, int], np.dtype]:
    """
    Infer (Z,Y,X) and dtype from file size and either:
      - AxBxC / N^3 hints in filename, or
      - provided Y/X fields, or
      - perfect cube fallback.
    Returns (shape, dtype).
    Raises ValueError with a user-friendly message when ambiguous.
    """
    fsize = p.stat().st_size
    if fsize <= 0:
        raise ValueError("File is empty.")

    cand_dtypes = [
        np.uint8, np.uint16, np.int16,
        np.uint32, np.int32, np.float32, np.float64, np.int8,
    ]
    order = {np.dtype(t): i for i, t in enumerate(cand_dtypes)}

    def pref_index(dt: np.dtype) -> int:
        return order.get(np.dtype(dt), 999)

    triple = best_trip_from_name(p.name)
    chosen_shape: Optional[Tuple[int, int, int]] = None
    inferred_dtype: Optional[np.dtype] = None

    if triple:
        zN, yN, xN = map(int, triple)
        planes = yN * xN
        viable: List[Tuple[Tuple[int, int, int], np.dtype]] = []
        for dt in cand_dtypes:
            item = np.dtype(dt).itemsize
            if fsize < item * planes:
                continue
            Z = (fsize // item) // planes
            if Z > 0:
                viable.append(((int(Z), yN, xN), np.dtype(dt)))
        if not viable:
            raise ValueError(
                f"Size {fsize}B incompatible with any dtype for name hint {triple}."
            )
        viable.sort(key=lambda t: pref_index(t[1]))
        chosen_shape, inferred_dtype = viable[0]
        if chosen_shape[0] != zN:
            show_warning(
                f"Adjusted Z from name {zN}→{chosen_shape[0]} (ignored header/padding)."
            )
    else:
        Y, X = int(shape_y), int(shape_x)
        if Y <= 0 or X <= 0:
            raise ValueError("Y/X must be > 0 (or encode shape in filename).")

        cands: List[Tuple[Tuple[int, int, int], np.dtype]] = []
        for dt in cand_dtypes:
            item = np.dtype(dt).itemsize
            if fsize % item:
                continue
            vox = fsize // item
            if vox % (Y * X) == 0:
                Z = vox // (Y * X)
                if Z > 0:
                    cands.append(((int(Z), Y, X), np.dtype(dt)))
        if not cands:
            cube_cands: List[Tuple[Tuple[int, int, int], np.dtype]] = []
            for dt in cand_dtypes:
                item = np.dtype(dt).itemsize
                if fsize % item:
                    continue
                vox = fsize // item
                n = int(round(vox ** (1 / 3)))
                if n > 0 and n * n * n == vox:
                    cube_cands.append(((n, n, n), np.dtype(dt)))
            if cube_cands:
                cube_cands.sort(key=lambda t: pref_index(t[1]))
                chosen_shape, inferred_dtype = cube_cands[0]
            else:
                show_warning(
                    "Cannot infer Z for given Y/X; also not a perfect cube for known dtypes."
                )
                raise ValueError("Rename file with AxBxC or set Y/X correctly.")
        else:
            cands.sort(key=lambda t: pref_index(t[1]))
            chosen_shape, inferred_dtype = cands[0]

    return chosen_shape, inferred_dtype  # type: ignore[return-value]


# ---------------------------------------------------------------------
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
)
def raw_loader_widget(
    path: pathlib.Path = pathlib.Path(""),
    shape_z: int = 700,   # kept for UI symmetry; we infer Z
    shape_y: int = 700,
    shape_x: int = 700,
    dtype: str = "auto",
):
    """
    RAW loader (little-endian). Supports filename hints (AxBxC, N^3).
    Aligns the displayed view to the smallest existing volume (center-crop),
    but preserves the FULL array in layer.metadata['_orig_full'] for tools/crop.
    """
    p = pathlib.Path(path).expanduser().resolve() if path else None
    if not p or not p.exists() or p.is_dir():
        raise FileNotFoundError("Select a valid RAW file.")

    chosen_shape, inferred_dtype = _infer_shape_and_dtype(
        p, shape_y=shape_y, shape_x=shape_x
    )

    # resolve dtype choice
    if dtype == "auto":
        np_dtype = inferred_dtype
    else:
        np_dtype = np.dtype(
            {
                "uint8": np.uint8,
                "int8": np.int8,
                "uint16": np.uint16,
                "int16": np.int16,
                "uint32": np.uint32,
                "int32": np.int32,
                "float32": np.float32,
                "float64": np.float64,
            }[dtype]
        )

    # validate byte size once more for chosen dtype
    fsize = p.stat().st_size
    exp_bytes = int(np.prod(chosen_shape)) * int(np_dtype.itemsize)
    if exp_bytes != fsize:
        vox = fsize // int(np_dtype.itemsize)
        YX = int(chosen_shape[1]) * int(chosen_shape[2])
        if vox < YX:
            raise ValueError("File smaller than one plane for chosen dtype/Y/X.")
        Z = vox // YX
        if Z != int(chosen_shape[0]):
            show_warning(f"Adjusted Z {chosen_shape[0]}→{Z} to match size.")
        chosen_shape = (int(Z), chosen_shape[1], chosen_shape[2])

    # memmap little-endian
    vol = np.memmap(
        p,
        dtype=np_dtype.newbyteorder("<"),
        mode="r",
        shape=chosen_shape,
        order="C",
    )

    v = current_viewer()
    if not v:
        raise RuntimeError("No active viewer.")

    # FULL vs. view-only aligned slice
    full_vol = vol
    show_vol = full_vol
    target = current_min_zyx(v)
    if target:
        fz, fy, fx = chosen_shape
        tz, ty, tx = target
        if (fz, fy, fx) != (tz, ty, tx):
            nz, ny, nx = min(fz, tz), min(fy, ty), min(fx, tx)
            zsl, ysl, xsl = center_crop_indices((fz, fy, fx), (nz, ny, nx))
            show_vol = full_vol[zsl, ysl, xsl]

    lname = unique_layer_name(p.stem)
    L: NapariImage = v.add_image(show_vol, name=lname)

    # store both versions for downstream tools
    L.metadata["_orig_full"] = full_vol
    L.metadata["_orig_data"] = show_vol
    L.metadata["_file_path"] = str(p)

    if L.ndim >= 3:
        z, y, x = last_zyx(np.asarray(L.data))
        try:
            v.dims.current_step = (z // 2, y // 2, x // 2)
        except Exception:
            pass

    # ✅ guarantee this is the active layer
    _force_select_layer(v, L)

    show_info(f"Loaded {p.name} → {L.name} (shape={tuple(L.data.shape)}, dtype={L.data.dtype})")
    return L


# ---------------------------------------------------------------------
# Bulk add helpers (wired from the main widget)
def add_many_files(settings: AppSettings) -> None:
    v = current_viewer()
    if not v:
        return
    start_dir = (settings.last_dir if settings.last_dir and pathlib.Path(settings.last_dir).exists() else "")
    paths, _ = QFileDialog.getOpenFileNames(
        None,
        "Add RAW files",
        start_dir,
        "RAW (*.raw *.RAW *.bin *.BIN);;All (*)",
    )
    if not paths:
        return
    paths = sorted(paths, key=natural_sort_key)
    settings.last_dir = str(pathlib.Path(paths[0]).parent)
    for p in paths:
        try:
            raw_loader_widget(path=pathlib.Path(p), dtype=settings.default_dtype)
        except Exception as e:
            show_warning(str(e))
    settings.save()


def add_from_folder(settings: AppSettings) -> None:
    v = current_viewer()
    if not v:
        return
    start_dir = (settings.last_dir if settings.last_dir and pathlib.Path(settings.last_dir).exists() else "")
    folder = QFileDialog.getExistingDirectory(None, "Pick folder with RAW/BIN", start_dir)
    if not folder:
        return
    exts = {".raw", ".RAW", ".bin", ".BIN"}
    files = [
        str(pathlib.Path(folder) / f)
        for f in sorted(os.listdir(folder), key=natural_sort_key)
        if pathlib.Path(f).suffix in exts
    ]
    if not files:
        show_warning("No RAW files in folder.")
        return
    settings.last_dir = folder
    for p in files:
        try:
            raw_loader_widget(path=pathlib.Path(p), dtype=settings.default_dtype)
        except Exception as e:
            show_warning(str(e))
    settings.save()


# ---------------------------------------------------------------------
__all__ = [
    "raw_loader_widget",
    "add_many_files",
    "add_from_folder",
]
