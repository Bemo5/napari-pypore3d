# napari_pypore3d/info_export.py — r38
from __future__ import annotations

from typing import Callable, Optional, Dict
import pathlib
import numpy as np
import json

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QComboBox, QSizePolicy,
    QListWidget, QListWidgetItem, QFrame
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_info, show_warning, show_error

from .helpers import export_raw_little
from .session_recorder import SessionRecorder

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


def _get_or_make_recorder(settings) -> SessionRecorder:
    """
    Store the recorder in _settings so every tab can log steps into the same stack.
    Supports dict-like or attribute-like settings.
    """
    if settings is None:
        # fallback global instance
        if not hasattr(_get_or_make_recorder, "_global"):
            _get_or_make_recorder._global = SessionRecorder()  # type: ignore
        return _get_or_make_recorder._global  # type: ignore

    # dict-like
    if isinstance(settings, dict):
        rec = settings.get("session_recorder")
        if isinstance(rec, SessionRecorder):
            return rec
        rec = SessionRecorder()
        settings["session_recorder"] = rec
        return rec

    # attribute-like
    rec = getattr(settings, "session_recorder", None)
    if isinstance(rec, SessionRecorder):
        return rec
    rec = SessionRecorder()
    try:
        setattr(settings, "session_recorder", rec)
    except Exception:
        pass
    return rec


def build_info_export_panel(_settings) -> tuple[QWidget, Callable[[], None]]:
    root = QWidget()
    root.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    vbox = QVBoxLayout(root)

    rec = _get_or_make_recorder(_settings)

    # ---------------------------
    # Session / Recipe section
    # ---------------------------
    title = QLabel("Session Recipe (reproducible stack)")
    f = QFont()
    f.setPointSize(11)
    f.setBold(True)
    title.setFont(f)
    vbox.addWidget(title)

    hint = QLabel(
        "This records *plugin actions* (buttons you press). "
        "Save as JSON, replay later, or copy a replay script."
    )
    hint.setWordWrap(True)
    hint.setStyleSheet("opacity: 0.85;")
    vbox.addWidget(hint)

    steps_list = QListWidget()
    steps_list.setSelectionMode(QListWidget.SingleSelection)
    vbox.addWidget(steps_list, stretch=1)

    row_recipe = QHBoxLayout()
    btn_copy = QPushButton("Copy replay script")
    btn_save_recipe = QPushButton("Save recipe (.json)")
    btn_load_recipe = QPushButton("Load recipe (.json)")
    btn_replay = QPushButton("Replay recipe")
    btn_clear = QPushButton("Clear stack")
    row_recipe.addWidget(btn_copy)
    row_recipe.addWidget(btn_save_recipe)
    row_recipe.addWidget(btn_load_recipe)
    row_recipe.addWidget(btn_replay)
    row_recipe.addWidget(btn_clear)
    row_recipe.addStretch(1)
    vbox.addLayout(row_recipe)

    # optional snapshot: guarantees exact state but can be huge
    row_snap = QHBoxLayout()
    btn_snapshot = QPushButton("Save snapshot (.npz) (exact but large)")
    row_snap.addWidget(btn_snapshot)
    row_snap.addStretch(1)
    vbox.addLayout(row_snap)

    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    vbox.addWidget(line)

    # ---------------------------
    # Export section (keep yours)
    # ---------------------------
    export_title = QLabel("Export active Image layer")
    export_title.setFont(f)
    vbox.addWidget(export_title)

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
    # helpers
    # --------------------------------------------------------
    def _find_image_layer() -> Optional[NapariImage]:
        try:
            viewer = current_viewer()
        except Exception:
            return None
        if viewer is None:
            return None

        lyr = getattr(viewer, "active_layer", None)
        if isinstance(lyr, NapariImage):
            return lyr

        for L in reversed(list(viewer.layers)):
            if isinstance(L, NapariImage):
                return L
        return None

    def _pick_save_path(suffix: str, title: str, default_name: str = "output") -> Optional[pathlib.Path]:
        default_dir = ""
        L = _find_image_layer()
        if L:
            p = L.metadata.get("_file_path")
            if p:
                default_dir = str(pathlib.Path(p).parent)
            default_name = L.name

        fn, _ = QFileDialog.getSaveFileName(
            None,
            title,
            str(pathlib.Path(default_dir) / f"{default_name}{suffix}"),
            f"{suffix[1:].upper()} files (*{suffix});;All files (*)",
        )
        return pathlib.Path(fn) if fn else None

    # --------------------------------------------------------
    # refresh UI
    # --------------------------------------------------------
    def refresh() -> None:
        steps_list.clear()
        steps = rec.steps()
        if not steps:
            steps_list.addItem(QListWidgetItem("— no recorded steps yet —"))
            return
        for s in steps:
            txt = f"{s.idx:03d}) {s.op} | target: {s.target}"
            if s.params:
                txt += f" | {json.dumps(s.params, ensure_ascii=False)}"
            if s.notes:
                txt += f" | note: {s.notes}"
            steps_list.addItem(QListWidgetItem(txt))

    rec.changed.connect(refresh)
    refresh()

    # --------------------------------------------------------
    # recipe buttons
    # --------------------------------------------------------
    def _on_save_recipe():
        path = _pick_save_path(".json", "Save recipe as .json", default_name="recipe")
        if not path:
            return
        try:
            path.write_text(rec.dumps_json_text(indent=2), encoding="utf-8")
            show_info(f"Saved recipe: {path.name}")
        except Exception as e:
            show_error(f"Save recipe failed: {e}")

    def _on_load_recipe():
        fn, _ = QFileDialog.getOpenFileName(
            None,
            "Load recipe (.json)",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not fn:
            return
        try:
            text = pathlib.Path(fn).read_text(encoding="utf-8")
            rec.loads_json_text(text)
            show_info(f"Loaded recipe: {pathlib.Path(fn).name}")
        except Exception as e:
            show_error(f"Load recipe failed: {e}")

    def _on_copy_script():
        path_hint = "recipe.json"
        try:
            script = rec.make_replay_script(recipe_path=path_hint)
            from qtpy.QtWidgets import QApplication
            QApplication.clipboard().setText(script)
            show_info("Copied replay script to clipboard (expects recipe.json in cwd).")
        except Exception as e:
            show_error(f"Copy failed: {e}")

    def _on_replay():
        try:
            viewer = current_viewer()
        except Exception:
            viewer = None
        if viewer is None:
            show_warning("Replay: no active napari viewer.")
            return
        try:
            rec.replay(viewer, strict=True)
            show_info("Replay done ✅")
        except Exception as e:
            show_error(f"Replay failed: {e}")

    def _on_clear():
        rec.clear()
        show_info("Cleared recipe stack.")

    btn_save_recipe.clicked.connect(_on_save_recipe)
    btn_load_recipe.clicked.connect(_on_load_recipe)
    btn_copy.clicked.connect(_on_copy_script)
    btn_replay.clicked.connect(_on_replay)
    btn_clear.clicked.connect(_on_clear)

    # --------------------------------------------------------
    # snapshot export (exact reproducibility, big files)
    # --------------------------------------------------------
    def _on_snapshot():
        try:
            viewer = current_viewer()
        except Exception:
            viewer = None
        if viewer is None:
            show_warning("Snapshot: no active napari viewer.")
            return

        path = _pick_save_path(".npz", "Save snapshot as .npz", default_name="session_snapshot")
        if not path:
            return

        try:
            imgs = [L for L in viewer.layers if isinstance(L, NapariImage)]
            if not imgs:
                show_warning("Snapshot: no Image layers to save.")
                return

            arrays = {}
            meta = []
            for i, L in enumerate(imgs):
                key = f"img_{i:03d}"
                arrays[key] = np.asarray(L.data)
                meta.append({
                    "key": key,
                    "name": L.name,
                    "dtype": str(np.asarray(L.data).dtype),
                    "shape": tuple(np.asarray(L.data).shape),
                    "metadata": dict(L.metadata) if isinstance(L.metadata, dict) else {},
                })

            # Save arrays + metadata + recipe
            np.savez_compressed(
                path,
                **arrays,
                __meta__=np.array([json.dumps(meta, ensure_ascii=False)]),
                __recipe__=np.array([rec.dumps_json_text(indent=2)]),
            )
            show_info(f"Saved snapshot: {path.name}")
        except Exception as e:
            show_error(f"Snapshot save failed: {e}")

    btn_snapshot.clicked.connect(_on_snapshot)

    # --------------------------------------------------------
    # Export behavior (same as before)
    # --------------------------------------------------------
    def _on_save_npy():
        L = _find_image_layer()
        if not L:
            show_warning("Export: no Image layer to save.")
            return
        path = _pick_save_path(".npy", "Save array as .npy", default_name=L.name)
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
        L = _find_image_layer()
        if not L:
            show_warning("Export: no Image layer to save.")
            return
        path = _pick_save_path(".tif", "Save array as .tif", default_name=L.name)
        if not path:
            return
        try:
            arr = np.asarray(L.data)
            _tifffile.imwrite(str(path), arr, dtype=arr.dtype, bigtiff=True)
            show_info(f"Saved {path.name}")
        except Exception as e:
            show_error(f"TIFF save failed: {e}")

    def _on_save_raw():
        L = _find_image_layer()
        if not L:
            show_warning("Export: no Image layer to save.")
            return
        path = _pick_save_path(".raw", "Save array as .raw (little-endian)", default_name=L.name)
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

    return root, refresh
