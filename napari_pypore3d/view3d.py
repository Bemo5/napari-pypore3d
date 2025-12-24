# napari_pypore3d/view3d.py — 3D View tab split out from _widget.py
from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
from skimage.filters import threshold_otsu, threshold_multiotsu
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QFormLayout,
)
from magicgui.widgets import PushButton, ComboBox, FloatSpinBox
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

# ---------------- Session Recorder integration (optional) ---------------------
try:
    from .session_recorder import get_recorder, register_handler
except Exception:
    get_recorder = None
    register_handler = None

_REC = get_recorder() if callable(get_recorder) else None

def _find_image_by_name(v, name: str):
    if not v or not name:
        return None
    for L in v.layers:
        if isinstance(L, NapariImage) and getattr(L, "name", None) == name:
            return L
    return None

def _make_view3d_tab() -> QWidget:
    # ---------- tiny helpers ----------
    def _card(title: str, inner: QWidget) -> QFrame:
        box = QFrame()
        box.setObjectName("card")
        box.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(20, 16, 20, 20)
        lay.setSpacing(14)
        ttl = QLabel(title)
        ttl.setStyleSheet("font-weight:600; font-size: 14px;")
        lay.addWidget(ttl)
        lay.addWidget(inner)
        box.setStyleSheet(
            """
            QFrame#card {
                border: 2px solid #5a5a5a;
                border-radius: 12px;
                background-color: rgba(255,255,255,0.05);
            }
        """
        )
        return box

    # ---------- Log widget for messages (moved to bottom) ----------
    log_text = QLabel("")
    log_text.setWordWrap(True)
    log_text.setMinimumHeight(120)
    log_text.setStyleSheet(
        """
        QLabel {
            background-color: rgba(0, 0, 0, 0.8);
            color: #e0e0e0;
            font-family: monospace;
            font-size: 12px;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #4a4a4a;
        }
    """
    )

    def _log_message(message: str):
        """Add a timestamped message to the log."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        current_text = log_text.text()
        new_text = f"[{timestamp}] {message}"
        if current_text:
            new_text = current_text + "\n" + new_text
        log_text.setText(new_text)

        # best-effort: keep latest visible (QLabel doesn't truly scroll)
        try:
            log_text.setAlignment(log_text.alignment() | 0x0080)  # AlignBottom
        except Exception:
            pass

    # ---------- Active viewer controls (apply to selected image) ----------
    btn_toggle = PushButton(text="Toggle 2D ↔ 3D")
    mode = ComboBox(choices=["mip", "attenuated_mip", "iso"], value="mip")
    att = FloatSpinBox(value=0.01, min=0.01, max=0.50, step=0.01)
    iso = FloatSpinBox(value=0.50, min=0.00, max=1.00, step=0.01)

    for w in (btn_toggle.native, mode.native, att.native, iso.native):
        try:
            w.setMinimumWidth(130)
            w.setMinimumHeight(30)
        except Exception:
            pass
    try:
        btn_toggle.native.setMinimumHeight(40)
        btn_toggle.native.setStyleSheet("font-weight: bold;")
    except Exception:
        pass

    def _toggle(*_):
        v = current_viewer()
        if not v:
            return
        v.dims.ndisplay = 3 if v.dims.ndisplay == 2 else 2
        _apply_grid(v)
        _log_message(f"Toggled display to {v.dims.ndisplay}D")

        # record
        if _REC is not None:
            try:
                _REC.add_step(
                    "view3d_toggle_ndisplay",
                    target="viewer",
                    params={"ndisplay": int(v.dims.ndisplay)},
                )
            except Exception:
                pass

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
            _log_message(
                f"Applied {mode.value} render with attenuation={att.value:.3f}, iso={iso.value:.3f}"
            )

            # record
            if _REC is not None:
                try:
                    _REC.add_step(
                        "view3d_apply_active_render",
                        target=str(getattr(lyr, "name", "")),
                        params={
                            "rendering": str(mode.value),
                            "attenuation": float(att.value),
                            "iso_threshold": float(iso.value),
                        },
                    )
                except Exception:
                    pass

    mode.changed.connect(_apply_active)
    att.changed.connect(_apply_active)
    iso.changed.connect(_apply_active)

    vc_row1 = QWidget()
    r1 = QHBoxLayout(vc_row1)
    r1.setContentsMargins(0, 4, 0, 4)
    r1.setSpacing(20)
    r1.addWidget(btn_toggle.native)
    r1.addStretch(1)

    vc_row2 = QWidget()
    r2 = QFormLayout(vc_row2)
    r2.setContentsMargins(0, 0, 0, 0)
    r2.setSpacing(8)
    r2.addRow("Render:", mode.native)
    r2.addRow("Atten:", att.native)
    r2.addRow("Iso Thr:", iso.native)

    vc = QWidget()
    vcl = QVBoxLayout(vc)
    vcl.setContentsMargins(0, 0, 0, 0)
    vcl.setSpacing(12)
    vcl.addWidget(vc_row1)
    vcl.addWidget(vc_row2)
    viewer_card = _card("Active viewer (selected image)", vc)

    # ---------- Thresholding / Transfer Function ----------
    thresh_low = FloatSpinBox(value=50.0, min=0.0, max=255.0, step=1.0)
    thresh_high = FloatSpinBox(value=200.0, min=0.0, max=255.0, step=1.0)

    for w in (thresh_low.native, thresh_high.native):
        try:
            w.setMinimumWidth(120)
            w.setMinimumHeight(30)
        except Exception:
            pass

    btn_threshold = PushButton(text="Apply threshold & segment")
    btn_threshold.native.setMinimumHeight(40)
    btn_threshold.native.setStyleSheet("font-weight: bold;")

    def _apply_threshold(*_):
        v = current_viewer()
        if not v:
            show_warning("Open napari viewer first.")
            return

        img = v.layers.selection.active
        if not isinstance(img, NapariImage):
            show_warning("Select an image layer first.")
            return

        raw_data = np.asarray(img.data)

        lo = float(thresh_low.value)
        hi = float(thresh_high.value)

        if lo >= hi:
            show_warning("Low threshold must be less than high threshold.")
            return

        mask = (raw_data >= lo) & (raw_data <= hi)

        if ndi is None:
            labels = mask.astype(np.int32)
            n = int(labels.max())
            show_warning(
                "SciPy not installed – showing raw binary mask.\n"
                "Install scipy for connected-component labeling."
            )
            _log_message("WARNING: SciPy not installed - using binary mask only")
        else:
            labels, n = ndi.label(mask)

        layer_name = f"{img.name} [threshold {lo}-{hi}, {n} objects]"
        v.add_labels(
            labels,
            name=layer_name,
            blending="translucent",
            opacity=0.7,
            rendering="iso_categorical",
        )

        show_info(f"Segmentation: {n} connected component(s) found.")
        _log_message(f"Applied threshold [{lo:.1f}-{hi:.1f}] on '{img.name}'")
        _log_message(f"→ Found {n} connected components")
        _log_message(f"→ Added labels layer '{layer_name}'")

        # record
        if _REC is not None:
            try:
                _REC.add_step(
                    "view3d_threshold_segment",
                    target=str(getattr(img, "name", "")),
                    params={"lo": float(lo), "hi": float(hi)},
                )
            except Exception:
                pass

    btn_threshold.changed.connect(_apply_threshold)

    thresh_row1 = QWidget()
    tr1 = QHBoxLayout(thresh_row1)
    tr1.setContentsMargins(0, 0, 0, 0)
    tr1.setSpacing(16)
    tr1.addWidget(QLabel("Low:"))
    tr1.addWidget(thresh_low.native)
    tr1.addWidget(QLabel("High:"))
    tr1.addWidget(thresh_high.native)
    tr1.addStretch(1)

    thresh_row2 = QWidget()
    tr2 = QHBoxLayout(thresh_row2)
    tr2.setContentsMargins(0, 0, 0, 0)
    tr2.setSpacing(10)
    tr2.addWidget(btn_threshold.native)
    tr2.addStretch(1)

    thresh_col = QWidget()
    tcol = QVBoxLayout(thresh_col)
    tcol.setContentsMargins(0, 0, 0, 0)
    tcol.setSpacing(14)
    tcol.addWidget(thresh_row1)
    tcol.addWidget(thresh_row2)

    threshold_card = _card("Threshold (Transfer Function)", thresh_col)

    # ---------- Otsu Automatic Segmentation ----------
    otsu_classes = FloatSpinBox(value=2, min=2, max=5, step=1)
    blur_sigma = FloatSpinBox(value=1.0, min=0.0, max=5.0, step=0.5)

    for w in (otsu_classes.native, blur_sigma.native):
        try:
            w.setMinimumWidth(120)
            w.setMinimumHeight(30)
        except Exception:
            pass

    btn_otsu = PushButton(text="Apply Otsu segmentation")
    btn_otsu.native.setMinimumHeight(40)
    btn_otsu.native.setStyleSheet("font-weight: bold;")

    def _apply_otsu(*_):
        v = current_viewer()
        if not v:
            show_warning("Open napari viewer first.")
            return

        img = v.layers.selection.active
        if not isinstance(img, NapariImage):
            show_warning("Select an image layer first.")
            return

        raw_data = np.asarray(img.data, dtype=np.float32)
        sigma = float(blur_sigma.value)
        num_classes = int(otsu_classes.value)

        if sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter

                raw_data = gaussian_filter(raw_data, sigma=sigma)
                _log_message(f"Applied Gaussian blur (σ={sigma:.1f})")
            except Exception as e:
                show_warning(f"Blur failed: {e}")
                _log_message(f"WARNING: Blur failed - {e}")
                return

        try:
            if num_classes == 2:
                thresh = threshold_otsu(raw_data)
                mask = raw_data >= thresh
                _log_message(f"Otsu threshold: {thresh:.1f}")
            else:
                thresholds = threshold_multiotsu(raw_data, classes=num_classes)
                _log_message(
                    f"Otsu thresholds ({num_classes} classes): "
                    f"{[f'{t:.1f}' for t in thresholds]}"
                )
                mask = np.zeros_like(raw_data, dtype=bool)
                for t in thresholds:
                    mask |= raw_data >= t
        except Exception as e:
            show_warning(f"Otsu failed: {e}")
            _log_message(f"ERROR: Otsu segmentation failed - {e}")
            return

        if ndi is None:
            labels = mask.astype(np.int32)
            n = int(labels.max())
            show_warning("SciPy not installed – showing binary mask.")
            _log_message("WARNING: SciPy not installed - binary mask only")
        else:
            labels, n = ndi.label(mask)

        layer_name = f"{img.name} [otsu_{num_classes}class, {n} objects]"
        v.add_labels(
            labels,
            name=layer_name,
            blending="translucent",
            opacity=0.7,
            rendering="iso_categorical",
        )

        show_info(f"Otsu segmentation: {n} connected component(s) found.")
        _log_message(f"Applied Otsu ({num_classes} classes) on '{img.name}'")
        _log_message(f"→ Found {n} connected components")
        _log_message(f"→ Added labels layer '{layer_name}'")

        # record
        if _REC is not None:
            try:
                _REC.add_step(
                    "view3d_otsu_segment",
                    target=str(getattr(img, "name", "")),
                    params={"classes": int(num_classes), "blur_sigma": float(sigma)},
                )
            except Exception:
                pass

    btn_otsu.changed.connect(_apply_otsu)

    otsu_row1 = QWidget()
    or1 = QHBoxLayout(otsu_row1)
    or1.setContentsMargins(0, 0, 0, 0)
    or1.setSpacing(16)
    or1.addWidget(QLabel("Classes:"))
    or1.addWidget(otsu_classes.native)
    or1.addWidget(QLabel("Blur σ:"))
    or1.addWidget(blur_sigma.native)
    or1.addStretch(1)

    otsu_row2 = QWidget()
    or2 = QHBoxLayout(otsu_row2)
    or2.setContentsMargins(0, 0, 0, 0)
    or2.setSpacing(10)
    or2.addWidget(btn_otsu.native)
    or2.addStretch(1)

    otsu_col = QWidget()
    ocol = QVBoxLayout(otsu_col)
    ocol.setContentsMargins(0, 0, 0, 0)
    ocol.setSpacing(14)
    ocol.addWidget(otsu_row1)
    ocol.addWidget(otsu_row2)

    otsu_card = _card("Otsu Segmentation (automatic)", otsu_col)

    # ---------- Activity Log card (NOW AT BOTTOM) ----------
    log_wrap = QWidget()
    lw = QVBoxLayout(log_wrap)
    lw.setContentsMargins(0, 0, 0, 0)
    lw.setSpacing(10)
    hint = QLabel("Shows actions taken in this tab (and will replay via Session Recorder).")
    hint.setStyleSheet("color:#9aa0a6; font-size:11px;")
    hint.setWordWrap(True)
    lw.addWidget(hint)
    lw.addWidget(log_text)
    log_card = _card("Activity Log", log_wrap)

    # ---------- assemble page ----------
    page = QWidget()
    pv = QVBoxLayout(page)
    pv.setContentsMargins(24, 20, 24, 24)
    pv.setSpacing(20)
    pv.addWidget(viewer_card)
    pv.addWidget(threshold_card)
    pv.addWidget(otsu_card)
    pv.addWidget(log_card)     # ✅ moved down here
    pv.addStretch(1)

    # Initial log message
    _log_message("3D View tab initialized")
    _log_message("Ready for threshold segmentation")

    return _pad(page)


# ------------------- Replay handlers (Session Recorder) -----------------------
def _replay_view3d_toggle(v, step):
    params = getattr(step, "params", {}) or {}
    ndisplay = int(params.get("ndisplay", 2))
    try:
        v.dims.ndisplay = 3 if int(ndisplay) == 3 else 2
    except Exception:
        pass
    try:
        _apply_grid(v)
    except Exception:
        pass


def _replay_view3d_apply_active_render(v, step):
    target = getattr(step, "target", "") or ""
    params = getattr(step, "params", {}) or {}
    lyr = _find_image_by_name(v, target)
    if not isinstance(lyr, NapariImage):
        return
    for k, val in (
        ("rendering", str(params.get("rendering", "mip"))),
        ("attenuation", float(params.get("attenuation", 0.01))),
        ("iso_threshold", float(params.get("iso_threshold", 0.5))),
    ):
        try:
            setattr(lyr, k, val)
        except Exception:
            pass


def _replay_view3d_threshold_segment(v, step):
    target = getattr(step, "target", "") or ""
    params = getattr(step, "params", {}) or {}
    img = _find_image_by_name(v, target)
    if not isinstance(img, NapariImage):
        return

    raw_data = np.asarray(img.data)
    lo = float(params.get("lo", 50.0))
    hi = float(params.get("hi", 200.0))
    if lo >= hi:
        return

    mask = (raw_data >= lo) & (raw_data <= hi)
    if ndi is None:
        labels = mask.astype(np.int32)
        n = int(labels.max())
    else:
        labels, n = ndi.label(mask)

    layer_name = f"{img.name} [threshold {lo}-{hi}, {n} objects]"
    v.add_labels(
        labels,
        name=layer_name,
        blending="translucent",
        opacity=0.7,
        rendering="iso_categorical",
    )


def _replay_view3d_otsu_segment(v, step):
    target = getattr(step, "target", "") or ""
    params = getattr(step, "params", {}) or {}
    img = _find_image_by_name(v, target)
    if not isinstance(img, NapariImage):
        return

    raw_data = np.asarray(img.data, dtype=np.float32)
    sigma = float(params.get("blur_sigma", 1.0))
    num_classes = int(params.get("classes", 2))

    if sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            raw_data = gaussian_filter(raw_data, sigma=sigma)
        except Exception:
            return

    try:
        if num_classes == 2:
            thresh = threshold_otsu(raw_data)
            mask = raw_data >= thresh
        else:
            thresholds = threshold_multiotsu(raw_data, classes=num_classes)
            mask = np.zeros_like(raw_data, dtype=bool)
            for t in thresholds:
                mask |= raw_data >= t
    except Exception:
        return

    if ndi is None:
        labels = mask.astype(np.int32)
        n = int(labels.max())
    else:
        labels, n = ndi.label(mask)

    layer_name = f"{img.name} [otsu_{num_classes}class, {n} objects]"
    v.add_labels(
        labels,
        name=layer_name,
        blending="translucent",
        opacity=0.7,
        rendering="iso_categorical",
    )


if callable(register_handler):
    try:
        register_handler("view3d_toggle_ndisplay", _replay_view3d_toggle)
        register_handler("view3d_apply_active_render", _replay_view3d_apply_active_render)
        register_handler("view3d_threshold_segment", _replay_view3d_threshold_segment)
        register_handler("view3d_otsu_segment", _replay_view3d_otsu_segment)
    except Exception:
        pass
