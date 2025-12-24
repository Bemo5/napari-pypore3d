# napari_pypore3d/recorder_panel.py
from __future__ import annotations

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QFrame
)
from napari.utils.notifications import show_info, show_error

from .session_recorder import get_recorder, reset_global_recorder


def make_session_recorder_panel() -> QWidget:
    recorder = get_recorder()

    root = QWidget()
    outer = QVBoxLayout(root)
    # ✅ looser + nicer
    outer.setContentsMargins(14, 14, 14, 14)
    outer.setSpacing(12)

    title = QLabel("Session Recorder")
    title.setStyleSheet("font-size: 13pt; font-weight: 700;")
    outer.addWidget(title)

    sub = QLabel("Logs replayable actions (Functions, presets, etc.). Save / load / replay.")
    sub.setWordWrap(True)
    sub.setStyleSheet("color: rgba(255,255,255,0.80);")
    outer.addWidget(sub)

    # --- Buttons (keep same actions, just spaced better) ---
    row = QHBoxLayout()
    row.setSpacing(10)

    btn_replay = QPushButton("Replay")
    btn_save = QPushButton("Save JSON…")
    btn_load = QPushButton("Load JSON…")
    btn_clear = QPushButton("Clear")
    btn_reset = QPushButton("Reset (global)")

    for b in (btn_replay, btn_save, btn_load, btn_clear, btn_reset):
        b.setMinimumHeight(32)

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
    steps_list.setMinimumHeight(240)
    steps_list.setAlternatingRowColors(True)
    # ✅ readable list
    steps_list.setStyleSheet(
        "QListWidget {"
        "  font-family: Consolas, 'Courier New', monospace;"
        "  font-size: 10pt;"
        "}"
    )
    outer.addWidget(steps_list)

    def refresh_list() -> None:
        steps_list.clear()
        if not getattr(recorder, "steps", None):
            QListWidgetItem("(empty) Run a function and it will appear here.", steps_list)
            return

        for i, s in enumerate(recorder.steps, start=1):
            p = s.params or {}
            txt = f"{i:02d}. {s.ts} | {s.op} | target='{s.target}' | {p}"
            if s.notes:
                txt += f" | notes={s.notes!r}"
            QListWidgetItem(txt, steps_list)

    # 🔥 LIVE UPDATES
    try:
        unsub = recorder.subscribe(refresh_list)
    except Exception:
        unsub = None

    def _on_destroyed(*_):
        try:
            if callable(unsub):
                unsub()
        except Exception:
            pass

    try:
        root.destroyed.connect(_on_destroyed)
    except Exception:
        pass

    refresh_list()

    def _confirm(msg: str) -> bool:
        r = QMessageBox.question(
            root,
            "Confirm",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return r == QMessageBox.Yes

    def on_replay():
        try:
            recorder.replay()
            refresh_list()
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
            refresh_list()
        except Exception as e:
            show_error(f"Load failed: {e!r}")

    def on_clear():
        if not recorder.steps:
            return
        if _confirm("Clear the current recipe?"):
            recorder.clear()
            refresh_list()

    def on_reset():
        if _confirm("Reset the global recorder? (clears all steps)"):
            reset_global_recorder()
            refresh_list()

    btn_replay.clicked.connect(on_replay)
    btn_save.clicked.connect(on_save)
    btn_load.clicked.connect(on_load)
    btn_clear.clicked.connect(on_clear)
    btn_reset.clicked.connect(on_reset)

    return root
