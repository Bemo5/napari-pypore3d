# napari_pypore3d/session_recorder.py
# ---------------------------------------------------------------------
# Minimal "Session Recipe" recorder + replay registry for napari-pypore3d
#
# Exports:
#   - Step (dataclass)
#   - register_handler(op, fn)
#   - RECORDER (global Recorder instance)
#   - get_recorder()
#   - reset_global_recorder()
#
# Live UI:
#   - Recorder supports listeners so external panels (like recorder_panel.py)
#     can refresh when steps change.
# ---------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import json

from napari import current_viewer
from napari.utils.notifications import show_info, show_warning, show_error

Handler = Callable[[Any, "Step"], None]
_HANDLERS: Dict[str, Handler] = {}


@dataclass
class Step:
    op: str
    target: str
    params: Dict[str, Any]
    notes: str = ""
    ts: str = ""  # timestamp string


def register_handler(op: str, fn: Handler) -> None:
    """Register (or replace) a replay handler for a given op name."""
    _HANDLERS[str(op)] = fn


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Recorder:
    def __init__(self) -> None:
        self.steps: List[Step] = []
        self._viewer_getter = current_viewer

        # External UI listeners (no args) – used by recorder_panel.py
        self._listeners: List[Callable[[], None]] = []

    # -------------------------
    # Listeners (for live UI)
    # -------------------------
    def subscribe(self, fn: Callable[[], None]) -> Callable[[], None]:
        """Subscribe a callback; returns an unsubscribe function."""
        if fn not in self._listeners:
            self._listeners.append(fn)

        def _unsub():
            try:
                self._listeners.remove(fn)
            except ValueError:
                pass

        return _unsub

    def _notify(self) -> None:
        dead: List[Callable[[], None]] = []
        for fn in list(self._listeners):
            try:
                fn()
            except Exception:
                dead.append(fn)
        for fn in dead:
            try:
                self._listeners.remove(fn)
            except Exception:
                pass

    # -------------------------
    # Core API
    # -------------------------
    def add_step(
        self,
        op: str,
        target: str,
        params: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> None:
        st = Step(
            op=str(op),
            target=str(target),
            params=dict(params or {}),
            notes=str(notes or ""),
            ts=_now_iso(),
        )
        self.steps.append(st)
        self._notify()

    def clear(self) -> None:
        self.steps.clear()
        self._notify()

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
        self._notify()

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

        for step in self.steps:
            fn = _HANDLERS.get(step.op)
            if fn is None:
                raise RuntimeError(f"No handler registered for op '{step.op}'")
            fn(v, step)

        show_info(f"Replayed {len(self.steps)} step(s).")


# -------------------------
# Global singleton API
# -------------------------
RECORDER = Recorder()
# ----------------------------
# Global singleton helpers
# ----------------------------

def get_recorder() -> Recorder:
    """Return the global recorder singleton used by the plugin."""
    return RECORDER


def reset_global_recorder() -> Recorder:
    """Clear the global recorder (used when the dock widget is re-created)."""
    try:
        RECORDER.clear()
    except Exception:
        pass
    return RECORDER


# ----------------------------
# Built-in replay handlers
# ----------------------------

def _find_layer_by_name(viewer, name: str):
    if not viewer:
        return None
    for L in viewer.layers:
        if getattr(L, "name", None) == name:
            return L
    return None


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(int(v), int(hi)))


def _parse_crop_params(params: dict, shape_zyx):
    """
    Supports multiple param styles:
      - z0,z1,y0,y1,x0,x1
      - start_zyx + size_zyx
      - crop_start_zyx + crop_size_zyx
      - z_range/y_range/x_range
    Returns slices (z0,z1,y0,y1,x0,x1) clamped to bounds.
    """
    Z, Y, X = map(int, shape_zyx)

    # style A: start + size
    for a, b in (("start_zyx", "size_zyx"), ("crop_start_zyx", "crop_size_zyx")):
        if a in params and b in params:
            sz = list(map(int, params[a]))
            ss = list(map(int, params[b]))
            z0, y0, x0 = sz
            dz, dy, dx = ss
            z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
            break
    else:
        # style B: ranges
        zr = params.get("z_range", None)
        yr = params.get("y_range", None)
        xr = params.get("x_range", None)
        if zr and yr and xr:
            z0, z1 = map(int, zr)
            y0, y1 = map(int, yr)
            x0, x1 = map(int, xr)
        else:
            # style C: explicit bounds (default full)
            z0 = int(params.get("z0", 0))
            z1 = int(params.get("z1", Z))
            y0 = int(params.get("y0", 0))
            y1 = int(params.get("y1", Y))
            x0 = int(params.get("x0", 0))
            x1 = int(params.get("x1", X))

    # normalise + clamp
    if z1 < z0: z0, z1 = z1, z0
    if y1 < y0: y0, y1 = y1, y0
    if x1 < x0: x0, x1 = x1, x0

    z0 = _clamp(z0, 0, Z)
    z1 = _clamp(z1, 0, Z)
    y0 = _clamp(y0, 0, Y)
    y1 = _clamp(y1, 0, Y)
    x0 = _clamp(x0, 0, X)
    x1 = _clamp(x1, 0, X)

    # avoid empty slices
    if z1 <= z0: z1 = min(z0 + 1, Z)
    if y1 <= y0: y1 = min(y0 + 1, Y)
    if x1 <= x0: x1 = min(x0 + 1, X)

    return z0, z1, y0, y1, x0, x1


def _op_crop_zyx(viewer, step: Step):
    """
    Replay handler for op='crop_zyx'.
    Expects step.target = image layer name.
    """
    layer = _find_layer_by_name(viewer, str(step.target))
    if layer is None:
        raise RuntimeError(f"crop_zyx: target layer not found: {step.target!r}")

    data = np.asarray(layer.data)
    if data.ndim == 2:
        # treat as (1,Y,X)
        data = data[None, :, :]

    if data.ndim != 3:
        raise RuntimeError(f"crop_zyx: expected 3D (Z,Y,X), got shape {data.shape}")

    z0, z1, y0, y1, x0, x1 = _parse_crop_params(step.params or {}, data.shape)

    cropped = np.ascontiguousarray(data[z0:z1, y0:y1, x0:x1])
    layer.data = cropped

    # best-effort contrast refresh
    try:
        if hasattr(layer, "reset_contrast_limits"):
            layer.reset_contrast_limits()
    except Exception:
        pass

    show_info(f"Replayed crop_zyx on '{layer.name}': Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}] -> {cropped.shape}")


# Register built-in op(s)
register_handler("crop_zyx", _op_crop_zyx)


def get_recorder() -> Recorder:
    return RECORDER


def reset_global_recorder() -> None:
    # IMPORTANT: do NOT replace RECORDER object (other modules keep a reference).
    RECORDER.clear()
