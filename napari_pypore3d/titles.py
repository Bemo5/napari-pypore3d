"""
Per‑tile titles (2D only) using a tiny hidden Points layer per image.
API:
  - title_layer_name(img_name: str) -> str
  - ensure_title(img: NapariImage, visible: bool = True) -> None
  - remove_title_for(img_name: str) -> None
  - refresh_titles(show: bool = True) -> None

Design notes:
- Idempotent: safe to call repeatedly.
- Positions are derived from layer extent; falls back to small (y,x) offset.
- Keeps title layer immediately above its image in the layer stack so it
  never hides under the image.
- No global state; caller decides when to refresh (e.g. on layer insert/remove,
  reorder, dims change, etc.).
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from napari import current_viewer
from napari.layers import Image as NapariImage

__all__ = [
    "title_layer_name",
    "ensure_title",
    "remove_title_for",
    "refresh_titles",
]


def title_layer_name(img_name: str) -> str:
    return f"__title__:{img_name}"


def _get_layer_by_name(v, name: str):
    for ly in v.layers:
        if ly.name == name:
            return ly
    return None


def _title_text(img: NapariImage) -> str:
    return str(img.name)


def _title_pos(img: NapariImage):
    # Prefer real data extent if available (robust across transforms)
    try:
        mins, _maxs = img.extent.data
        if img.ndim >= 2:
            y0 = float(mins[-2]) + 2.0
            x0 = float(mins[-1]) + 2.0
            if img.ndim >= 3:
                z = float(current_viewer().dims.current_step[0])
                return np.array([[z, y0, x0]], dtype=float)
            return np.array([[y0, x0]], dtype=float)
    except Exception:
        pass
    # Fallback: midpoint z (if 3D) and a tiny (y,x) offset
    if img.ndim >= 3:
        zyx = np.asarray(img.data).shape[-3:]
        z = float(int(zyx[0]) // 2)
        return np.array([[z, 2.0, 2.0]], dtype=float)
    return np.array([[2.0, 2.0]], dtype=float)


def ensure_title(img: NapariImage, visible: bool = True) -> None:
    """Create/update the title Points layer for `img` if we are in 2D.

    Keeps the title immediately above `img` in the layer stack.
    Safe to call often.
    """
    v = current_viewer()
    if not v or v.dims.ndisplay != 2:
        return

    nm = title_layer_name(img.name)
    tl = _get_layer_by_name(v, nm)
    pos = _title_pos(img)

    if tl is None:
        tl = v.add_points(
            pos,
            name=nm,
            size=1,
            face_color=[0, 0, 0, 0],
            edge_color=[0, 0, 0, 0],
            opacity=1.0,
            blending="translucent",
            visible=True,
        )
        # Make it non‑interactive everywhere
        for attr, val in (
            ("editable", False),
            ("interactive", False),
            ("pickable", False),
            ("selected", False),
        ):
            try:
                setattr(tl, attr, val)
            except Exception:
                pass
    else:
        try:
            tl.data = pos
        except Exception:
            pass

    # Text styling (fallbacks for older napari)
    try:
        tl.text = {
            "string": _title_text(img),
            "size": 12,
            "color": "white",
            "anchor": "upper_left",
        }
    except Exception:
        tl.text = _title_text(img)

    # Keep title directly above its image (z-order)
    try:
        img_i = v.layers.index(img)
        tl_i = v.layers.index(tl)
        if tl_i < img_i:
            v.layers.move(tl_i, img_i + 1)
    except Exception:
        # Best effort fallback—push to top
        try:
            v.layers.move(v.layers.index(tl), len(v.layers) - 1)
        except Exception:
            pass

    tl.visible = bool(visible)


def remove_title_for(img_name: str) -> None:
    v = current_viewer()
    if not v:
        return
    nm = title_layer_name(img_name)
    ly = _get_layer_by_name(v, nm)
    if ly is not None:
        try:
            v.layers.remove(ly)
        except Exception:
            pass


def refresh_titles(show: bool = True) -> None:
    """Ensure titles for all images are present (2D) or cleared.

    Call this on layer insert/remove/reorder and on dims.ndisplay changes.
    """
    v = current_viewer()
    if not v:
        return

    if not show or v.dims.ndisplay != 2:
        # Remove all title layers when hidden or in 3D
        for ly in list(v.layers):
            if isinstance(getattr(ly, "name", None), str) and ly.name.startswith("__title__:"):
                try:
                    v.layers.remove(ly)
                except Exception:
                    pass
        return

    # Ensure/update for each image
    imgs = [ly for ly in v.layers if isinstance(ly, NapariImage)]
    present = {ly.name for ly in imgs}
    for img in imgs:
        try:
            ensure_title(img, visible=True)
        except Exception:
            continue

    # Remove stragglers (titles whose image was deleted/renamed)
    for ly in list(v.layers):
        nm = getattr(ly, "name", "")
        if isinstance(nm, str) and nm.startswith("__title__:"):
            base = nm.split(":", 1)[1]
            if base not in present:
                try:
                    v.layers.remove(ly)
                except Exception:
                    pass
