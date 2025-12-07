# napari_pypore3d/view3d.py — 3D View tab split out from _widget.py
from __future__ import annotations

from typing import Optional

import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QFrame,
    QFileDialog,
)
from magicgui.widgets import PushButton, CheckBox, ComboBox, FloatSpinBox
from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_warning, show_info

# NOTE: reuse helpers and globals from _widget so behaviour stays identical
from ._widget import _pad, _images, _apply_grid  # type: ignore[attr-defined]

# Optional SciPy import for connected-components labelling
try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None  # type: ignore[assignment]


def _make_view3d_tab() -> QWidget:
    # local imports so top-level stays unchanged
    try:
        from qtpy.QtWidgets import QAbstractItemView
    except Exception:
        QAbstractItemView = None  # type: ignore[assignment]

    # ---------- tiny helpers ----------
    def _card(title: str, inner: QWidget) -> QFrame:
        box = QFrame()
        box.setObjectName("card")
        box.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(16, 12, 16, 16)
        lay.setSpacing(12)
        ttl = QLabel(title)
        ttl.setStyleSheet("font-weight:600;")
        lay.addWidget(ttl)
        lay.addWidget(inner)
        box.setStyleSheet(
            """
            QFrame#card {
                border: 1px solid #4a4a4a;
                border-radius: 10px;
                background-color: rgba(255,255,255,0.03);
            }
        """
        )
        return box

    # ---------- Active viewer controls (apply to selected image) ----------
    btn_toggle = PushButton(text="Toggle 2D ↔ 3D")
    mode = ComboBox(choices=["mip", "attenuated_mip", "iso"], value="mip")
    att = FloatSpinBox(value=0.01, min=0.01, max=0.50, step=0.01)
    iso = FloatSpinBox(value=0.50, min=0.00, max=1.00, step=0.01)

    for w in (btn_toggle.native, mode.native, att.native, iso.native):
        try:
            w.setMinimumWidth(130)
        except Exception:
            pass
    try:
        btn_toggle.native.setMinimumHeight(36)
    except Exception:
        pass

    def _toggle(*_):
        v = current_viewer()
        if not v:
            return
        v.dims.ndisplay = 3 if v.dims.ndisplay == 2 else 2
        _apply_grid(v)

    btn_toggle.changed.connect(_toggle)

    def _apply_active(*_):
        v = current_viewer()
        lyr = v.layers.selection.active if v else None
        if isinstance(lyr, NapariImage):
            for k, val in (
                ("rendering", str(mode.value)),
                ("attenuation", float(att.value)),
                ("iso_threshold", float(iso.value)),
            ):
                try:
                    setattr(lyr, k, val)
                except Exception:
                    pass

    mode.changed.connect(_apply_active)
    att.changed.connect(_apply_active)
    iso.changed.connect(_apply_active)

    vc_row1 = QWidget()
    r1 = QHBoxLayout(vc_row1)
    r1.setContentsMargins(0, 0, 0, 0)
    r1.setSpacing(12)
    r1.addWidget(btn_toggle.native)
    r1.addStretch(1)

    vc_row2 = QWidget()
    r2 = QHBoxLayout(vc_row2)
    r2.setContentsMargins(0, 0, 0, 0)
    r2.setSpacing(14)
    r2.addWidget(QLabel("render"))
    r2.addWidget(mode.native)
    r2.addWidget(QLabel("atten"))
    r2.addWidget(att.native)
    r2.addWidget(QLabel("iso_thr"))
    r2.addWidget(iso.native)
    r2.addStretch(1)

    vc = QWidget()
    vcl = QVBoxLayout(vc)
    vcl.setContentsMargins(0, 0, 0, 0)
    vcl.setSpacing(10)
    vcl.addWidget(vc_row1)
    vcl.addWidget(vc_row2)
    viewer_card = _card("Active viewer (selected image)", vc)

    # ---------- 3D Mirror (separate window) ----------
    global _MIRROR_VIEWER, _MIRROR_LINKS
    try:
        _MIRROR_VIEWER  # type: ignore[name-defined]
    except NameError:
        _MIRROR_VIEWER = None  # type: ignore[assignment]
    try:
        _MIRROR_LINKS  # type: ignore[name-defined]
    except NameError:
        _MIRROR_LINKS = []  # type: ignore[assignment]

    lst = QListWidget()
    try:
        if QAbstractItemView is not None:
            lst.setSelectionMode(
                QAbstractItemView.ExtendedSelection  # type: ignore[arg-type]
            )
        lst.setMinimumHeight(360)
    except Exception:
        pass

    link_z = CheckBox(text="Link Z", value=True)
    btn_open_mirror = PushButton(text="Open 3D mirror (selected)")
    btn_close_mirror = PushButton(text="Close mirror")
    btn_sel_active = PushButton(text="Select active")
    btn_sel_all = PushButton(text="Select all")
    btn_clear_sel = PushButton(text="Clear")
    for w in (
        btn_open_mirror.native,
        btn_close_mirror.native,
        btn_sel_active.native,
        btn_sel_all.native,
        btn_clear_sel.native,
    ):
        try:
            w.setMinimumHeight(32)
        except Exception:
            pass

    def _refresh_list(select_active_if_empty=True):
        lst.clear()
        v = current_viewer()
        if not v:
            return
        active = v.layers.selection.active
        for L in _images(v):
            it = QListWidgetItem(L.name, lst)
            if select_active_if_empty and L is active:
                it.setSelected(True)

    def _cleanup_links():
        global _MIRROR_LINKS
        for emitter, attr, cb in list(_MIRROR_LINKS):
            try:
                getattr(emitter, attr).disconnect(cb)
            except Exception:
                pass
        _MIRROR_LINKS = []

    def _open_mirror(*_):
        global _MIRROR_VIEWER, _MIRROR_LINKS
        v = current_viewer()
        if not v:
            return
        try:
            import napari as _napari
        except Exception:
            show_warning("Could not import napari for mirror window.")
            return

        chosen = {it.text() for it in lst.selectedItems()}
        if not chosen:
            show_warning("Pick one or more images in the list first.")
            return

        _cleanup_links()
        if _MIRROR_VIEWER is None:
            _MIRROR_VIEWER = _napari.Viewer()
        else:
            for L in list(_MIRROR_VIEWER.layers):
                try:
                    _MIRROR_VIEWER.layers.remove(L)
                except Exception:
                    pass

        try:
            _MIRROR_VIEWER.dims.ndisplay = 3
        except Exception:
            pass

        for src in _images(v):
            if src.name not in chosen:
                continue
            try:
                dst = _MIRROR_VIEWER.add_image(
                    src.data,
                    name=src.name,
                    colormap=getattr(src, "colormap", "gray"),
                    contrast_limits=getattr(src, "contrast_limits", None),
                    rendering=getattr(src, "rendering", "mip"),
                )

                def _mk_sync(s=src, d=dst):
                    def _cb(*_):
                        try:
                            d.contrast_limits = s.contrast_limits
                        except Exception:
                            pass

                    return _cb

                try:
                    cb = _mk_sync()
                    src.events.contrast_limits.connect(cb)
                    _MIRROR_LINKS.append((src.events, "contrast_limits", cb))
                except Exception:
                    pass
            except Exception:
                pass

        if bool(link_z.value):
            def _src2mirror(*_):
                try:
                    _MIRROR_VIEWER.dims.set_current_step(0, v.dims.current_step[0])
                except Exception:
                    pass

            def _mirror2src(*_):
                try:
                    v.dims.set_current_step(0, _MIRROR_VIEWER.dims.current_step[0])
                except Exception:
                    pass

            try:
                v.dims.events.current_step.connect(_src2mirror)
                _MIRROR_LINKS.append((v.dims.events, "current_step", _src2mirror))
            except Exception:
                pass
            try:
                _MIRROR_VIEWER.dims.events.current_step.connect(_mirror2src)
                _MIRROR_LINKS.append(
                    (_MIRROR_VIEWER.dims.events, "current_step", _mirror2src)
                )
            except Exception:
                pass

        try:
            _MIRROR_VIEWER.window._qt_window.raise_()
        except Exception:
            pass

    def _close_mirror(*_):
        global _MIRROR_VIEWER
        _cleanup_links()
        if _MIRROR_VIEWER is not None:
            try:
                _MIRROR_VIEWER.close()
            except Exception:
                pass
        _MIRROR_VIEWER = None

    def _select_active(*_):
        v = current_viewer()
        if not v:
            return
        names = {L.name for L in _images(v)}
        cur = getattr(v.layers.selection.active, "name", None)
        for i in range(lst.count()):
            it = lst.item(i)
            it.setSelected(it.text() == cur and it.text() in names)

    def _select_all(*_):
        for i in range(lst.count()):
            lst.item(i).setSelected(True)

    def _clear_sel(*_):
        for i in range(lst.count()):
            lst.item(i).setSelected(False)

    btn_open_mirror.changed.connect(_open_mirror)
    btn_close_mirror.changed.connect(_close_mirror)
    btn_sel_active.changed.connect(_select_active)
    btn_sel_all.changed.connect(_select_all)
    btn_clear_sel.changed.connect(_clear_sel)

    v = current_viewer()
    if v:
        _refresh_list(select_active_if_empty=True)
        try:
            v.layers.events.inserted.connect(lambda *_: _refresh_list(False))
            v.layers.events.removed.connect(lambda *_: _refresh_list(False))
            v.layers.events.reordered.connect(lambda *_: _refresh_list(False))
            v.layers.selection.events.active.connect(
                lambda *_: _refresh_list(True)
            )
        except Exception:
            pass

    # layout for mirror controls
    mir_top = QWidget()
    mt = QHBoxLayout(mir_top)
    mt.setContentsMargins(0, 0, 0, 0)
    mt.setSpacing(10)
    mt.addWidget(btn_open_mirror.native)
    mt.addWidget(btn_close_mirror.native)
    mt.addWidget(link_z.native)
    mt.addStretch(1)

    mir_tools = QWidget()
    mtools = QHBoxLayout(mir_tools)
    mtools.setContentsMargins(0, 0, 0, 0)
    mtools.setSpacing(8)
    mtools.addWidget(btn_sel_active.native)
    mtools.addWidget(btn_sel_all.native)
    mtools.addWidget(btn_clear_sel.native)
    mtools.addStretch(1)

    mir_col = QWidget()
    mcol = QVBoxLayout(mir_col)
    mcol.setContentsMargins(0, 0, 0, 0)
    mcol.setSpacing(10)
    mcol.addWidget(mir_top)
    mcol.addWidget(mir_tools)
    mcol.addWidget(QLabel("Mirror these images in 3D:"))
    mcol.addWidget(lst)

    mirror_card = _card("3D mirror (separate window)", mir_col)

    # ---------- assemble page ----------
    page = QWidget()
    pv = QVBoxLayout(page)
    pv.setContentsMargins(18, 16, 18, 18)
    pv.setSpacing(16)
    pv.addWidget(viewer_card)
    pv.addWidget(mirror_card)

    # ---------- Segmentation Loader ----------
    seg_btn = PushButton(text="Load segmentation for active image…")
    seg_btn.native.setMinimumHeight(32)

    def _load_segmentation(*_):
        v = current_viewer()
        if not v:
            show_warning("Open napari viewer first.")
            return

        img = v.layers.selection.active
        if not isinstance(img, NapariImage):
            show_warning("Select a 3-D image layer first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            None,
            "Pick segmentation file (RAW/BIN, same shape as image)",
            "",
            "RAW/BIN (*.raw *.RAW *.bin *.BIN);;All files (*)",
        )
        if not path:
            return

        shp = tuple(int(s) for s in np.asarray(img.data).shape)

        try:
            raw = np.fromfile(path, dtype=np.uint8)
        except Exception as e:
            show_warning(f"Failed to load segmentation: {e}")
            return

        if raw.size != np.prod(shp):
            show_warning(
                f"Segmentation size mismatch.\n"
                f"Image shape = {shp}, file voxels = {raw.size}"
            )
            return

        seg = raw.reshape(shp)

        # ---------- NEW: turn binary mask into instance labels ----------
        if ndi is None:
            # no SciPy → just show what we have as labels
            labels = seg.astype(np.int32, copy=False)
            n = int(labels.max())
            show_warning(
                "SciPy is not installed – showing raw labels.\n"
                "Run `pip install scipy` in this environment for per-object labelling."
            )
        else:
            vals = np.unique(seg)
            vals = vals[vals != 0]  # ignore background
            if vals.size <= 1:
                # looks binary (0 / 1) → connected components in 3D
                labels, n = ndi.label(seg > 0)
            else:
                # already labelled (0,1,2,3,…) → just use as-is
                labels = seg.astype(np.int32, copy=False)
                n = int(labels.max())

        v.add_labels(
            labels,
            name=f"{img.name} [seg_cc {n}]",
            blending="translucent",
            opacity=0.7,
            rendering="iso_categorical",
        )

        show_info(f"Segmentation loaded with {n} connected component(s).")

    seg_btn.changed.connect(_load_segmentation)

    seg_box = QFrame()
    seg_box.setObjectName("seg_card")
    seg_lay = QVBoxLayout(seg_box)
    seg_lay.setContentsMargins(16, 12, 16, 16)
    seg_lay.setSpacing(12)
    seg_title = QLabel("Segmentation")
    seg_title.setStyleSheet("font-weight:600;")
    seg_lay.addWidget(seg_title)
    seg_lay.addWidget(seg_btn.native)
    seg_box.setStyleSheet(
        """
        QFrame#seg_card {
            border: 1px solid #4a4a4a;
            border-radius: 10px;
            background-color: rgba(255,255,255,0.03);
        }
    """
    )

    pv.addWidget(seg_box)
    pv.addStretch(1)

    return _pad(page)
