# napari_pypore3d/hud.py — r35 MIN
# Provides:
#   wire_caption_events_once()
#   refresh_all_captions(position="bottom-left")
#   toggle_ndisplay_single()
#   apply_rendering_current(mode, att=0.05, iso_thr=0.50)
#   apply_auto_grid()
#
# No external deps from our package; uses only napari APIs.

from __future__ import annotations
import math
from typing import Dict, Optional

import numpy as np
from napari import current_viewer
from napari.layers import Image as NapariImage

# ---- tiny helpers ------------------------------------------------------------
def _iter_images(v):
    return [l for l in (v.layers if v else []) if isinstance(l, NapariImage)]

def _active_image() -> tuple[Optional[NapariImage], object]:
    v = current_viewer()
    if v is None:
        return None, None
    L = v.layers.selection.active
    return (L, v) if isinstance(L, NapariImage) else (None, v)

def _last_zyx(a: np.ndarray):
    if a.ndim == 2:
        y, x = a.shape[-2:]
        return (1, int(y), int(x))
    if a.ndim >= 3:
        z, y, x = a.shape[-3:]
        return (int(z), int(y), int(x))
    return (1, 1, 1)

# ---- caption (HUD) -----------------------------------------------------------
def _caption_text(img: NapariImage) -> str:
    a = np.asarray(img.data)
    shape = tuple(a.shape)
    dtype = getattr(a.dtype, "name", str(a.dtype))
    ztxt = ""
    try:
        v = current_viewer()
        if a.ndim >= 3:
            ztxt = f" | z={int(v.dims.current_step[0])}"
    except Exception:
        pass
    try:
        lo, hi = (
            img.contrast_limits if img.contrast_limits is not None else (None, None)
        )
        cl = f" | CL=[{int(lo)},{int(hi)}]" if (lo is not None and hi is not None) else ""
    except Exception:
        cl = ""
    return f"{img.name}{ztxt} | shape={shape} | dtype={dtype}{cl}"

def refresh_all_captions(position: str = "bottom-left") -> None:
    v = current_viewer()
    if not v:
        return
    L, _ = _active_image()
    try:
        v.text_overlay.visible = True
        v.text_overlay.color = "white"
        v.text_overlay.border = "black"
        v.text_overlay.font_size = 12
    except Exception:
        pass
    if L is None:
        try:
            v.text_overlay.text = ""
        except Exception:
            pass
        return
    try:
        v.text_overlay.text = _caption_text(L)
    except Exception:
        pass

_HUD_WIRED = False
def wire_caption_events_once() -> None:
    """Connect napari events once to keep the overlay in sync."""
    global _HUD_WIRED
    v = current_viewer()
    if not v or _HUD_WIRED:
        return

    def _sync(*_args, **_kwargs):
        refresh_all_captions("bottom-left")

    try:
        v.layers.events.inserted.connect(_sync)
        v.layers.events.removed.connect(_sync)
        v.layers.events.reordered.connect(_sync)
        v.layers.selection.events.active.connect(_sync)
        v.layers.events.changed.connect(_sync)
        v.dims.events.current_step.connect(_sync)
        v.dims.events.ndisplay.connect(_sync)
    except Exception:
        pass

    _HUD_WIRED = True
    refresh_all_captions("bottom-left")

# ---- grid/contrast helpers ---------------------------------------------------
def apply_auto_grid() -> None:
    v = current_viewer()
    if not v:
        return
    imgs = _iter_images(v)
    v.grid.enabled = len(imgs) >= 2 and v.dims.ndisplay == 2

    if imgs and v.dims.ndisplay == 2:
        # smart rows×cols
        def _rc(n: int) -> tuple[int, int]:
            if n <= 1:
                return (1, 1)
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            if n == 2:
                return (2, 1)
            return (rows, cols)
        v.grid.shape = _rc(len(imgs))

        # sync contrast to first image
        try:
            cl = tuple(map(float, imgs[0].contrast_limits))
            for ly in imgs:
                ly.contrast_limits = cl
        except Exception:
            pass
    try:
        v.reset_view()
    except Exception:
        pass

# ---- 2D/3D single-layer toggle ----------------------------------------------
_SAVED_VIS: Dict[int, Dict[str, bool]] = {}

def _enter_3d_active():
    L, v = _active_image()
    if not (L and v):
        return
    vid = id(v)
    _SAVED_VIS[vid] = {ly.name: bool(ly.visible) for ly in _iter_images(v)}
    for ly in _iter_images(v):
        ly.visible = (ly is L)
    v.dims.ndisplay = 3
    v.grid.enabled = False
    try:
        v.reset_view()
        v.camera.zoom = float(v.camera.zoom) * 1.2
    except Exception:
        pass

def _exit_3d_active():
    L, v = _active_image()
    if not v:
        return
    vid = id(v)
    saved = _SAVED_VIS.pop(vid, None)
    if saved:
        by_name = {ly.name: ly for ly in _iter_images(v)}
        for nm, vis in saved.items():
            if nm in by_name:
                by_name[nm].visible = bool(vis)
    v.dims.ndisplay = 2
    apply_auto_grid()

def toggle_ndisplay_single() -> None:
    v = current_viewer()
    if not v:
        return
    if v.dims.ndisplay == 2:
        _enter_3d_active()
    else:
        _exit_3d_active()

# ---- rendering props on active image -----------------------------------------
def apply_rendering_current(mode: str, *, att: float = 0.05, iso_thr: float = 0.50) -> None:
    L, _ = _active_image()
    if not L:
        return
    try:
        L.rendering = mode
    except Exception:
        pass
    try:
        if hasattr(L, "attenuation") and mode == "attenuated_mip":
            L.attenuation = float(att)
    except Exception:
        pass
    try:
        if hasattr(L, "iso_threshold") and mode == "iso":
            L.iso_threshold = float(iso_thr)
    except Exception:
        pass
