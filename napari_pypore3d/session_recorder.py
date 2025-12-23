# napari_pypore3d/session_recorder.py
# ---------------------------------------------------------------------
# Minimal "Session Recipe" recorder + replay registry for napari-pypore3d
#
# Exports:
#   - Step (dataclass)
#   - register_handler(op, fn)
#   - RECORDER (global Recorder instance)
#
# Handlers signature:
#   handler(viewer, step: Step) -> None
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import json

from napari import current_viewer
from napari.utils.notifications import show_info, show_warning, show_error

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QFileDialog, QMessageBox, QFrame
)

Handler = Callable[[Any, "Step"], None]
_HANDLERS: Dict[str, Handler] = {}


@dataclass
class Step:
    op: str
    target: str
    params: Dict[str, Any]
    notes: str = ""
    ts: str = ""  # ISO-ish timestamp string


def register_handler(op: str, fn: Handler) -> None:
    """Register (or replace) a replay handler for a given op name."""
    _HANDLERS[str(op)] = fn


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Recorder:
    def __init__(self) -> None:
        self.steps: List[Step] = []

        # UI refs (optional)
        self._log: Optional[QPlainTextEdit] = None
        self._viewer_getter = current_viewer

    # -------------------------
    # Core API
    # -------------------------
    def add_step(self, op: str, target: str, params: Optional[Dict[str, Any]] = None, notes: str = "") -> None:
        st = Step(
            op=str(op),
            target=str(target),
            params=dict(params or {}),
            notes=str(notes or ""),
            ts=_now_iso(),
        )
        self.steps.append(st)
        self._ui_refresh()

    def clear(self) -> None:
        self.steps.clear()
        self._ui_refresh()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "created": _now_iso(),
            "steps": [asdict(s) for s in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def load_dict(self, d: Dict[str, Any]) -> None:
        steps_in = d.get("steps", [])
        out: List[Step] = []
        for s in steps_in:
            if not isinstance(s, dict):
                continue
            out.append(
                Step(
                    op=str(s.get("op", "")),
                    target=str(s.get("target", "")),
                    params=dict(s.get("params", {}) or {}),
                    notes=str(s.get("notes", "") or ""),
                    ts=str(s.get("ts", "") or ""),
                )
            )
        self.steps = out
        self._ui_refresh()

    def load_json(self, text: str) -> None:
        d = json.loads(text)
        if not isinstance(d, dict):
            raise ValueError("Invalid recipe JSON")
        self.load_dict(d)

    def replay(self, viewer=None) -> None:
        v = viewer
        if v is None:
            try:
                v = self._viewer_getter()
            except Exception:
                v = None
        if v is None:
            show_error("No viewer available for replay.")
            return

        if not self.steps:
            show_warning("Recipe is empty.")
            return

        # run steps in order
        for i, step in enumerate(self.steps, start=1):
            fn = _HANDLERS.get(step.op)
            if fn is None:
                raise RuntimeError(f"No handler registered for op '{step.op}'")
            fn(v, step)

        show_info(f"Replayed {len(self.steps)} step(s).")

    # -------------------------
    # UI
    # -------------------------
    def make_session_widget(self) -> QWidget:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        title = QLabel("Session Recipe")
        title.setStyleSheet("font-size: 12pt; font-weight: 600;")
        outer.addWidget(title)

        sub = QLabel("This is a replayable log of what you ran. Save / load / replay.")
        sub.setWordWrap(True)
        sub.setStyleSheet("opacity: 0.85;")
        outer.addWidget(sub)

        row = QHBoxLayout()
        row.setSpacing(8)

        btn_replay = QPushButton("Replay")
        btn_save = QPushButton("Save JSON…")
        btn_load = QPushButton("Load JSON…")
        btn_copy = QPushButton("Copy JSON")
        btn_clear = QPushButton("Clear")

        for b in (btn_replay, btn_save, btn_load, btn_copy, btn_clear):
            b.setMinimumHeight(30)

        row.addWidget(btn_replay)
        row.addWidget(btn_save)
        row.addWidget(btn_load)
        row.addWidget(btn_copy)
        row.addWidget(btn_clear)
        row.addStretch(1)
        outer.addLayout(row)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        outer.addWidget(line)

        log = QPlainTextEdit()
        log.setReadOnly(True)
        log.setMinimumHeight(220)
        log.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
        outer.addWidget(log)

        self._log = log
        self._ui_refresh()

        def _confirm(msg: str) -> bool:
            r = QMessageBox.question(root, "Confirm", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            return r == QMessageBox.Yes

        def on_replay():
            try:
                self.replay()
            except Exception as e:
                show_error(f"Replay failed: {e!r}")

        def on_save():
            path, _ = QFileDialog.getSaveFileName(root, "Save recipe JSON", "recipe.json", "JSON (*.json)")
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.to_json())
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
                self.load_json(txt)
                show_info(f"Loaded recipe: {path}")
            except Exception as e:
                show_error(f"Load failed: {e!r}")

        def on_copy():
            try:
                from qtpy.QtWidgets import QApplication
                QApplication.clipboard().setText(self.to_json())
                show_info("Copied recipe JSON to clipboard.")
            except Exception as e:
                show_error(f"Copy failed: {e!r}")

        def on_clear():
            if not self.steps:
                return
            if _confirm("Clear the current recipe?"):
                self.clear()

        btn_replay.clicked.connect(on_replay)
        btn_save.clicked.connect(on_save)
        btn_load.clicked.connect(on_load)
        btn_copy.clicked.connect(on_copy)
        btn_clear.clicked.connect(on_clear)

        return root

    def _ui_refresh(self) -> None:
        if self._log is None:
            return
        if not self.steps:
            self._log.setPlainText("(empty)\nRun a function and it will appear here.")
            return

        lines: List[str] = []
        for i, s in enumerate(self.steps, start=1):
            p = s.params or {}
            summary = f"{i:02d}. {s.ts} | {s.op} | target='{s.target}' | {p}"
            if s.notes:
                summary += f" | notes={s.notes!r}"
            lines.append(summary)
        self._log.setPlainText("\n".join(lines))


# Global singleton
RECORDER = Recorder()
