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

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import json

import numpy as np

from napari import current_viewer
from napari.utils.notifications import show_info, show_warning, show_error
import importlib

_PKG = __name__.rsplit(".", 1)[0]  # "napari_pypore3d"

def _lazy_import_for_op(op: str) -> None:
    # import modules that register handlers, based on op prefix
    if op.startswith("slice_compare_"):
        importlib.import_module(f"{_PKG}.slice_compare")
    elif op.startswith("crop_"):
        importlib.import_module(f"{_PKG}.crop")
    elif op.startswith("view3d_"):
        importlib.import_module(f"{_PKG}.view3d")
    elif op.startswith("brush_"):
        importlib.import_module(f"{_PKG}.brush")
    elif op.startswith("func_") or op.startswith("functions_"):
        importlib.import_module(f"{_PKG}.functions")

# Handler signature: handler(viewer, step)
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
    op = str(op).strip()
    if not op:
        return
    _HANDLERS[op] = fn


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Recorder:
    def __init__(self) -> None:
        self.steps: List[Step] = []
        self._viewer_getter = current_viewer

        # External UI listeners (no args) – used by recorder_panel.py
        self._listeners: List[Callable[[], None]] = []

        # Guard: don't re-record while replaying
        self._replaying: bool = False

    @property
    def is_replaying(self) -> bool:
        return bool(self._replaying)

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
        """Record a step (unless currently replaying)."""
        if self._replaying:
            return

        st = Step(
            op=str(op),
            target=str(target),
            params=dict(params or {}),
            notes=str(notes or ""),
            ts=_now_iso(),
        )
        self.steps.append(st)
        self._notify()

    # convenience alias (some files like to call .record)
    def record(self, op: str, target: str, params: Optional[Dict[str, Any]] = None, notes: str = "") -> None:
        self.add_step(op=op, target=target, params=params, notes=notes)

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

    def load_dict(self, d: Dict[str, Any], *, replace: bool = True) -> None:
        steps_in = d.get("steps", [])
        out: List[Step] = []
        if isinstance(steps_in, list):
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

        if replace:
            self.steps = out
        else:
            self.steps.extend(out)

        self._notify()

    def load_json(self, text: str, *, replace: bool = True) -> None:
        d = json.loads(text)
        if not isinstance(d, dict):
            raise ValueError("Invalid recipe JSON")
        self.load_dict(d, replace=replace)

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

        self._replaying = True
        ok = 0
        skipped = 0
        failed = 0

        try:
            for i, step in enumerate(self.steps, start=1):
                fn = _HANDLERS.get(step.op)
                if fn is None:
                    show_warning(f"Replay: skipping step {i:02d} op='{step.op}' (no handler registered).")
                    skipped += 1
                    continue

                # Helpful trace so you see EXACTLY which step died
                try:
                    show_info(f"Replay {i:02d}/{len(self.steps)}: op='{step.op}' target='{step.target}' params={step.params}")
                except Exception:
                    pass

                try:
                    fn(v, step)
                    ok += 1
                    continue
                except Exception as e:
                    msg = str(e or "")
                    lower = msg.lower()

                    # Auto-fallback: if target missing, retry with __ACTIVE__
                    if (
                        ("target layer" in lower or "target missing" in lower or "not found" in lower)
                        and str(step.target).strip() not in ("", "__ACTIVE__", "__ANY__", "__ANY_IMAGE__")
                    ):
                        try:
                            step2 = Step(
                                op=str(step.op),
                                target="__ACTIVE__",
                                params=dict(step.params or {}),
                                notes=str(getattr(step, "notes", "") or ""),
                                ts=str(getattr(step, "ts", "") or ""),
                            )
                            fn(v, step2)
                            show_warning(
                                f"Replay: step {i:02d} op='{step.op}' target '{step.target}' missing — used __ACTIVE__ instead."
                            )
                            ok += 1
                            continue
                        except Exception as e2:
                            show_error(
                                f"Replay: step {i:02d} FAILED op='{step.op}' target={step.target!r} "
                                f"(also failed with __ACTIVE__): {e2!r}"
                            )
                            failed += 1
                            continue

                    # Normal failure (don’t kill the whole replay)
                    show_error(
                        f"Replay: step {i:02d} FAILED op='{step.op}' target={step.target!r}: {e!r}"
                    )
                    failed += 1
                    continue

            show_info(f"Replay done: ok={ok}, skipped={skipped}, failed={failed}.")
        finally:
            self._replaying = False



# -------------------------
# Global singleton API
# -------------------------
RECORDER = Recorder()


def get_recorder() -> Recorder:
    """Return the global recorder singleton used by the plugin."""
    return RECORDER


def reset_global_recorder() -> None:
    """
    Clear the global recorder.
    IMPORTANT: do NOT replace RECORDER object (other modules keep references).
    """
    try:
        RECORDER.clear()
    except Exception:
        pass


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


def _active_image_layer(viewer):
    """Return the active *3D* Image layer (prefer non-slice_compare)."""
    if viewer is None:
        return None

    try:
        active = viewer.layers.selection.active
    except Exception:
        active = None

    try:
        from napari.layers import Image as NapariImage

        def _is_good_3d(layer):
            if not isinstance(layer, NapariImage):
                return False
            md = getattr(layer, "metadata", {}) or {}
            if md.get("_slice_compare", False):
                return False
            try:
                return np.asarray(layer.data).ndim >= 3
            except Exception:
                return False

        # 1) if active is a good 3D layer, use it
        if _is_good_3d(active):
            return active

        # 2) if active is a slice_compare plane, jump to its parent 3D volume
        if isinstance(active, NapariImage):
            md = getattr(active, "metadata", {}) or {}
            parent_name = md.get("_slice_parent")
            if parent_name:
                for L in viewer.layers:
                    if _is_good_3d(L) and getattr(L, "name", None) == parent_name:
                        return L

        # 3) fallback: last good 3D layer in the stack
        imgs = [L for L in viewer.layers if _is_good_3d(L)]
        return imgs[-1] if imgs else None

    except Exception:
        return active if (active is not None and hasattr(active, "data")) else None


def resolve_target_layer(viewer, target: str):
    """
    Resolve Step.target to a layer.
    - "__ACTIVE__" / "" -> active image layer
    - layer name -> that layer, else fallback to active image layer
    """
    t = (target or "").strip()

    if t in ("", "__ACTIVE__", "__ANY__", "__ANY_IMAGE__"):
        layer = _active_image_layer(viewer)
        if layer is None:
            raise RuntimeError("No active Image layer to apply the recipe to.")
        return layer

    layer = _find_layer_by_name(viewer, t)
    if layer is not None:
        return layer

    # fallback for old recipes when the named layer isn't present
    layer = _active_image_layer(viewer)
    if layer is None:
        raise RuntimeError(f"Target layer not found ({t!r}) and no active Image layer exists.")
    return layer


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
    z0 = y0 = x0 = 0
    z1, y1, x1 = Z, Y, X

    for a, b in (("start_zyx", "size_zyx"), ("crop_start_zyx", "crop_size_zyx")):
        if a in params and b in params:
            start = list(map(int, params[a]))
            size = list(map(int, params[b]))
            z0, y0, x0 = start
            dz, dy, dx = size
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
            # style C: explicit bounds
            z0 = int(params.get("z0", 0))
            z1 = int(params.get("z1", Z))
            y0 = int(params.get("y0", 0))
            y1 = int(params.get("y1", Y))
            x0 = int(params.get("x0", 0))
            x1 = int(params.get("x1", X))

    # normalise + clamp
    if z1 < z0:
        z0, z1 = z1, z0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 < x0:
        x0, x1 = x1, x0

    z0 = _clamp(z0, 0, Z)
    z1 = _clamp(z1, 0, Z)
    y0 = _clamp(y0, 0, Y)
    y1 = _clamp(y1, 0, Y)
    x0 = _clamp(x0, 0, X)
    x1 = _clamp(x1, 0, X)

    # avoid empty slices
    if z1 <= z0:
        z1 = min(z0 + 1, Z)
    if y1 <= y0:
        y1 = min(y0 + 1, Y)
    if x1 <= x0:
        x1 = min(x0 + 1, X)

    return z0, z1, y0, y1, x0, x1


def _op_crop_zyx(viewer, step: Step):
    """
    Replay handler for op='crop_zyx'.
    Expects step.target = image layer name.
    """
    layer = resolve_target_layer(viewer, str(step.target))
    if layer is None:
        raise RuntimeError(f"crop_zyx: target layer not found: {step.target!r}")

    data = np.asarray(layer.data)
    if data.ndim == 2:
        data = data[None, :, :]
    if data.ndim != 3:
        raise RuntimeError(f"crop_zyx: expected 3D (Z,Y,X), got shape {data.shape}")

    z0, z1, y0, y1, x0, x1 = _parse_crop_params(step.params or {}, data.shape)
    cropped = np.ascontiguousarray(data[z0:z1, y0:y1, x0:x1])
    layer.data = cropped

    try:
        if hasattr(layer, "reset_contrast_limits"):
            layer.reset_contrast_limits()
    except Exception:
        pass

    show_info(
        f"Replayed crop_zyx on '{layer.name}': "
        f"Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}] -> {cropped.shape}"
    )


# Register built-in op(s)
register_handler("crop_zyx", _op_crop_zyx)
