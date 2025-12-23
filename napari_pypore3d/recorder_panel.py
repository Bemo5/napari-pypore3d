# napari_pypore3d/recorder_panel.py
from __future__ import annotations

import json

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QFileDialog, QCheckBox
)
from qtpy.QtGui import QGuiApplication

from napari import current_viewer
from napari.utils.notifications import show_info, show_warning, show_error

from .session_recorder import get_recorder


def make_session_recorder_panel() -> QWidget:
    """
    UI to view/export/replay the global SessionRecorder.
    """
    rec = get_recorder()

    root = QWidget()
    lay = QVBoxLayout(root)
    lay.setContentsMargins(14, 14, 14, 14)
    lay.setSpacing(10)

    title = QLabel("Session Recorder (recipe)")
    title.setStyleSheet("font-weight:600; font-size:14px;")
    lay.addWidget(title)

    steps_list = QListWidget()
    lay.addWidget(steps_list)

    # Row 1
    r1 = QHBoxLayout()
    btn_clear = QPushButton("Clear")
    btn_save = QPushButton("Save recipe…")
    btn_load = QPushButton("Load recipe…")
    r1.addWidget(btn_clear)
    r1.addWidget(btn_save)
    r1.addWidget(btn_load)
    r1.addStretch(1)
    lay.addLayout(r1)

    # Row 2
    r2 = QHBoxLayout()
    strict = QCheckBox("Strict replay")
    strict.setChecked(True)
    btn_replay = QPushButton("Replay now")
    btn_copy_json = QPushButton("Copy JSON")
    btn_copy_script = QPushButton("Copy replay script")
    r2.addWidget(strict)
    r2.addWidget(btn_replay)
    r2.addStretch(1)
    r2.addWidget(btn_copy_json)
    r2.addWidget(btn_copy_script)
    lay.addLayout(r2)

    hint = QLabel(
        "Tabs should record only meaningful actions (Apply/Run/Export), not live previews.\n"
        "Replay expects layers to exist by name."
    )
    hint.setWordWrap(True)
    lay.addWidget(hint)

    def _refresh():
        steps_list.clear()
        for s in rec.steps():
            params = s.params or {}
            params_short = json.dumps(params, ensure_ascii=False)
            if len(params_short) > 140:
                params_short = params_short[:137] + "..."
            txt = f"{s.idx:03d} | {s.time} | {s.op} | {s.target} | {params_short}"
            it = QListWidgetItem(txt)
            it.setToolTip(json.dumps(s.to_dict(), indent=2, ensure_ascii=False))
            steps_list.addItem(it)

    def _clear():
        rec.clear()
        show_info("Session recipe cleared.")

    def _save():
        path, _ = QFileDialog.getSaveFileName(
            root, "Save recipe JSON", "recipe.json", "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(rec.dumps_json_text(indent=2))
            show_info(f"Saved recipe: {path}")
        except Exception as e:
            show_error(f"Save failed: {e!r}")

    def _load():
        path, _ = QFileDialog.getOpenFileName(
            root, "Load recipe JSON", "", "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec.loads_json_text(f.read())
            show_info(f"Loaded recipe: {path}")
        except Exception as e:
            show_error(f"Load failed: {e!r}")

    def _replay():
        v = current_viewer()
        if v is None:
            show_warning("No napari viewer found.")
            return
        try:
            rec.replay(v, strict=bool(strict.isChecked()))
            show_info("✅ Replay done.")
        except Exception as e:
            show_error(f"Replay failed: {e!r}")

    def _copy_json():
        try:
            QGuiApplication.clipboard().setText(rec.dumps_json_text(indent=2))
            show_info("Copied recipe JSON to clipboard.")
        except Exception as e:
            show_error(f"Copy failed: {e!r}")

    def _copy_script():
        try:
            QGuiApplication.clipboard().setText(rec.make_replay_script("recipe.json"))
            show_info("Copied replay script to clipboard.")
        except Exception as e:
            show_error(f"Copy failed: {e!r}")

    # wire
    btn_clear.clicked.connect(_clear)
    btn_save.clicked.connect(_save)
    btn_load.clicked.connect(_load)
    btn_replay.clicked.connect(_replay)
    btn_copy_json.clicked.connect(_copy_json)
    btn_copy_script.clicked.connect(_copy_script)

    # auto refresh on changes
    try:
        rec.changed.connect(_refresh)
    except Exception:
        pass

    _refresh()
    return root
