# napari_pypore3d/recorder_panel.py
from __future__ import annotations

from typing import Optional
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QFrame
)
from napari.utils.notifications import show_info, show_warning, show_error

from .session_recorder import get_recorder, reset_global_recorder


def make_session_recorder_panel() -> QWidget:
    recorder = get_recorder()

    root = QWidget()
    outer = QVBoxLayout(root)
    outer.setContentsMargins(10, 10, 10, 10)
    outer.setSpacing(8)

    title = QLabel("Session Recorder")
    title.setStyleSheet("font-size: 12pt; font-weight: 600;")
    outer.addWidget(title)

    sub = QLabel("Logs replayable actions (Functions, presets, etc.). Save / load / replay.")
    sub.setWordWrap(True)
    sub.setStyleSheet("opacity: 0.85;")
    outer.addWidget(sub)

    row = QHBoxLayout()
    row.setSpacing(8)

    btn_replay = QPushButton("Replay")
    btn_save = QPushButton("Save JSON…")
    btn_load = QPushButton("Load JSON…")
    btn_clear = QPushButton("Clear")
    btn_reset = QPushButton("Reset (global)")

    for b in (btn_replay, btn_save, btn_load, btn_clear, btn_reset):
        b.setMinimumHeight(30)

    row.addWidget(btn_replay)
    row.addWidget(btn_save)
    row.addWidget(btn_load)
    row.addWidget(btn_clear)
    row.addWidget(btn_reset)
    row.addStretch(1)
    outer.addLayout(row)

    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    outer.addWidget(line)

    steps_list = QListWidget()
    steps_list.setMinimumHeight(220)
    outer.addWidget(steps_list)

    def refresh_list() -> None:
        steps_list.clear()
        if not recorder.steps:
            QListWidgetItem("(empty) Run a function and it will appear here.", steps_list)
            return

        for i, s in enumerate(recorder.steps, start=1):
            p = s.params or {}
            txt = f"{i:02d}. {s.ts} | {s.op} | target='{s.target}' | {p}"
            if s.notes:
                txt += f" | notes={s.notes!r}"
            QListWidgetItem(txt, steps_list)

    # 🔥 LIVE UPDATES: refresh whenever steps change
    unsub = recorder.subscribe(refresh_list)
    root.destroyed.connect(lambda *_: unsub())

    refresh_list()

    def _confirm(msg: str) -> bool:
        r = QMessageBox.question(root, "Confirm", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return r == QMessageBox.Yes

    def on_replay():
        try:
            recorder.replay()
        except Exception as e:
            show_error(f"Replay failed: {e!r}")

    def on_save():
        path, _ = QFileDialog.getSaveFileName(root, "Save recipe JSON", "recipe.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(recorder.to_json())
            show_info(f"Saved recipe: {path}")
        except Exception as e:
            show_error(f"Save failed: {e!r}")

    def on_load():
        path, _ = QFileDialog.getOpenFileName(root, "Load recipe JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            recorder.load_json(txt)
            show_info(f"Loaded recipe: {path}")
        except Exception as e:
            show_error(f"Load failed: {e!r}")

    def on_clear():
        if not recorder.steps:
            return
        if _confirm("Clear the current recipe?"):
            recorder.clear()

    def on_reset():
        if _confirm("Reset the global recorder? (clears all steps)"):
            reset_global_recorder()

    btn_replay.clicked.connect(on_replay)
    btn_save.clicked.connect(on_save)
    btn_load.clicked.connect(on_load)
    btn_clear.clicked.connect(on_clear)
    btn_reset.clicked.connect(on_reset)

    return root
