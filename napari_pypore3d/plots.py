# napari_pypore3d/plots.py — r67 PLOT LAB
# Histogram + profiles + box/scatter, simple mask brush, zoomable plot.
# Mask brush: single "mask" labels layer, multi-class (Air/Bone/Pores/Other).
# Export: full 2D/3D labels volume, legend JSON, and CSV with voxel info.
#
# Files on "Export mask":
#   - <base>.csv           : z,y,x,value,label,class,color
#   - <base>_mask.tif/npy : full labels volume (0,1,2,...)
#   - <base>_legend.json  : {"1": {"class": "...", "color": "#rrggbb"}, ...}

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import os
import json

import numpy as np

from magicgui.widgets import (
    Container, Label, PushButton, CheckBox, ComboBox, SpinBox, LineEdit
)
from napari import current_viewer
from napari.layers import Image as NapariImage, Labels as NapariLabels
from napari.utils.notifications import show_warning, show_info

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFrame,
    QFileDialog,
    QSpacerItem,
    QSizePolicy,
    QColorDialog,
)
from qtpy.QtGui import QColor

# --- optional matplotlib -------------------------------------------------------
HAVE_MPL = True
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
except Exception:
    HAVE_MPL = False
    Figure = object          # type: ignore
    FigureCanvas = object    # type: ignore
    NavigationToolbar = object  # type: ignore


# --- helpers / settings --------------------------------------------------------
try:
    from .helpers import (
        AppSettings,
        array_stats,
        iter_images,
        active_image,
        unique_layer_name,
        debug,
    )
except Exception:  # fallback for standalone testing
    @dataclass
    class AppSettings:
        default_bins: int = 256
        clip_1_99: bool = True

    def array_stats(a: np.ndarray):
        a = np.asarray(a)
        return dict(
            min=float(np.nanmin(a)),
            max=float(np.nanmax(a)),
            mean=float(np.nanmean(a)),
            std=float(np.nanstd(a)),
        )

    def iter_images(v):  # type: ignore
        return []

    def active_image():  # type: ignore
        v = current_viewer()
        if v is None:
            return None, None
        try:
            L = v.layers.selection.active
        except Exception:
            L = None
        if isinstance(L, NapariImage):
            return L, v
        for lay in v.layers:
            if isinstance(lay, NapariImage):
                return lay, v
        return None, v

    def unique_layer_name(base: str) -> str:  # type: ignore
        v = current_viewer()
        if v is None:
            return base
        names = {l.name for l in v.layers}
        if base not in names:
            return base
        i = 1
        while f"{base}-{i}" in names:
            i += 1
        return f"{base}-{i}"

    def debug(msg: str) -> None:
        print("[napari-pypore3d]", msg)


# label indices per class
CLASS_TO_LABEL: Dict[str, int] = {
    "Air": 1,
    "Bone": 2,
    "Pores": 3,
    "Other": 4,
}


# --- small Qt card helper ------------------------------------------------------
def _card(title: str, inner: QWidget) -> QFrame:
    """Simple card-style frame with a title and inner QWidget."""
    box = QFrame()
    box.setObjectName("card")
    box.setFrameShape(QFrame.StyledPanel)

    lay = QVBoxLayout(box)
    lay.setContentsMargins(14, 10, 14, 12)
    lay.setSpacing(10)

    ttl = Label(value=title)
    try:
        ttl.native.setStyleSheet("font-weight:600; font-size: 13px;")
    except Exception:
        pass

    lay.addWidget(ttl.native if hasattr(ttl, "native") else ttl)
    lay.addWidget(inner)

    box.setStyleSheet("""
        QFrame#card {
            border: 1px solid #4a4a4a;
            border-radius: 10px;
            background-color: rgba(255,255,255,0.03);
        }
    """)
    return box


# ==============================================================================
# PlotLab
# ==============================================================================

class PlotLab:
    """Plot dock: histogram / profiles / box / scatter + simple mask brush.

    Brush usage:
      1. Choose the image layer.
      2. Choose a class: Air / Bone / Pores / Other (mapped to labels 1–4).
      3. Pick colour (optional).
      4. Set brush size.
      5. Click "Brush" → it switches to 'mask' Labels layer in paint mode.
      6. Paint in the viewer.
      7. Click "Brush (on)" again to turn it off.

    Export:
      - "Export mask" writes ALL voxels where mask > 0 (whole volume):
          • CSV:   z,y,x,value,label,class,color
          • TIFF:  full labels volume (0,1,2,3,...) for segmentation
          • JSON:  legend mapping label → {class, color}
    """

    def __init__(self, settings: AppSettings) -> None:
        self._alive = True
        self._live_on = True
        self._last_series: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (x, y)

        # brush state
        self._brush_on = False
        self._brush_prev_layer = None
        self._brush_prev_mode: Optional[str] = None

        # mask colour (RGBA, 0–1)
        self._mask_rgba: Tuple[float, float, float, float] = (1.0, 0.4, 0.0, 1.0)
        self._current_label: int = 1  # updated from class dropdown

        # ------------------------------------------------------------------ controls
        self.pick = ComboBox(choices=[], value=None, label="layer", nullable=True)
        self.plot_type = ComboBox(
            choices=["histogram", "profile X", "profile Y", "profile Z", "box", "scatter"],
            value="histogram",
            label="kind",
        )
        self.bins = SpinBox(
            value=int(getattr(settings, "default_bins", 256) or 256),
            min=1,
            max=4096,
            step=1,
            label="bins",
        )
        self.clip = CheckBox(
            text="clip 1–99%",
            value=bool(getattr(settings, "clip_1_99", True)),
        )
        self.logy = CheckBox(text="log-Y", value=False)
        self.live = CheckBox(text="live", value=True)

        self.max_points = SpinBox(
            value=200_000,
            min=10_000,
            max=2_000_000,
            step=10_000,
            label="max samples",
        )

        self.btn_plot = PushButton(text="Plot")
        self.btn_png = PushButton(text="Export PNG")
        self.btn_csv = PushButton(text="Export CSV")

        # ------------------------------------------------------------------ mask brush (multi-class + export)
        self.mask_name = ComboBox(
            choices=list(CLASS_TO_LABEL.keys()),
            value="Air",
            label="class",
        )
        self.brush_size = SpinBox(value=5, min=1, max=100, step=1, label="size")
        self.btn_mask_color = PushButton(text="Pick colour")
        self.btn_brush = PushButton(text="Brush")
        self.btn_mask_export = PushButton(text="Export mask")

        self.save_mode = ComboBox(
            choices=["Ask", "Auto (near image)"],
            value="Ask",
            label="save",
        )

        # initial label from class
        self._update_current_label()

        # ------------------------------------------------------------------ rows
        # ------------------------------------------------------------------ rows (less squished)
        # Controls:
        #   layer
        #   kind
        #   bins | max samples
        #   clip | log-Y | live
        #   Plot | Export PNG | Export CSV
        row_layer = Container(
            widgets=[self.pick],
            layout="horizontal",
            labels=True,
        )
        row_kind = Container(
            widgets=[self.plot_type],
            layout="horizontal",
            labels=True,
        )
        row_hist = Container(
            widgets=[self.bins, self.max_points],
            layout="horizontal",
            labels=True,
        )
        row_flags = Container(
            widgets=[self.clip, self.logy, self.live],
            layout="horizontal",
            labels=True,
        )
        row_actions = Container(
            widgets=[self.btn_plot, self.btn_png, self.btn_csv],
            layout="horizontal",
            labels=True,
        )


        # --- mask brush rows (less cramped) ---------------------------------
        mask_row1 = Container(
            widgets=[self.mask_name, self.brush_size, self.btn_mask_color],
            layout="horizontal",
            labels=True,
        )
        mask_row2 = Container(
            widgets=[self.btn_brush, self.btn_mask_export, self.save_mode],
            layout="horizontal",
            labels=True,
        )


        # tighten rows (slightly more margin for mask rows)
        # tighten rows (slightly more margin for readability)
        for row in (
            row_layer, row_kind, row_hist, row_flags, row_actions,
            mask_row1, mask_row2,
        ):
            try:
                lay = row.native.layout()
                if lay is not None:
                    lay.setSpacing(10)
                    lay.setContentsMargins(6, 4, 6, 4)
            except Exception:
                pass



        def _minw(w, px: int) -> None:
            try:
                w.native.setMinimumWidth(px)
            except Exception:
                pass

        for w in (self.pick,):
            _minw(w, 220)   # layer combo gets full width
        for w in (self.plot_type,):
            _minw(w, 160)
        for w in (self.bins, self.max_points, self.brush_size, self.save_mode):
            _minw(w, 130)
        for w in (self.btn_plot, self.btn_png, self.btn_csv,
                  self.btn_mask_color, self.btn_brush, self.btn_mask_export):
            _minw(w, 110)


        # ------------------------------------------------------------------ Qt wrapper
        self._q = QWidget()
        root = QVBoxLayout(self._q)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(20)

        # Controls card
        # Controls card
        controls = QWidget()
        c_lay = QVBoxLayout(controls)
        c_lay.setContentsMargins(10, 8, 10, 8)
        c_lay.setSpacing(6)
        c_lay.addWidget(row_layer.native)
        c_lay.addWidget(row_kind.native)
        c_lay.addWidget(row_hist.native)
        c_lay.addWidget(row_flags.native)
        c_lay.addWidget(row_actions.native)
        root.addWidget(_card("Controls", controls))


        # Mask brush card
        mask_widget = QWidget()
        m_lay = QVBoxLayout(mask_widget)
        m_lay.setContentsMargins(10, 8, 10, 8)
        m_lay.setSpacing(6)
        m_lay.addWidget(mask_row1.native)
        m_lay.addWidget(mask_row2.native)
        root.addWidget(_card("Mask brush", mask_widget))


        # small gap before plot
        root.addSpacerItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # Plot card
        if HAVE_MPL:
            self.fig = Figure(figsize=(7.5, 4.8))
            self.canvas = FigureCanvas(self.fig)
            try:
                self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.canvas.setMinimumHeight(340)
            except Exception:
                pass

            self.toolbar = NavigationToolbar(self.canvas, self._q)

            plot_wrap = QWidget()
            p_lay = QVBoxLayout(plot_wrap)
            p_lay.setContentsMargins(10, 8, 10, 10)
            p_lay.setSpacing(6)
            p_lay.addWidget(self.toolbar)
            p_lay.addWidget(self.canvas)

            plot_card = _card("Plot", plot_wrap)
            root.addWidget(plot_card, 2)
        else:
            self.fig = None
            self.canvas = None
            self.toolbar = None
            dummy = QWidget()
            d_lay = QVBoxLayout(dummy)
            d_lay.addWidget(Label(value="Matplotlib not available").native)
            plot_card = _card("Plot", dummy)
            root.addWidget(plot_card, 2)

        try:
            self._q.destroyed.connect(lambda *_: self._mark_dead())
        except Exception:
            pass

        # ------------------------------------------------------------------ wiring
        try:
            self.btn_plot.clicked.connect(lambda *_: self.plot())
            self.btn_png.clicked.connect(lambda *_: self._export_png())
            self.btn_csv.clicked.connect(lambda *_: self._export_csv())

            self.btn_mask_color.clicked.connect(lambda *_: self._pick_mask_color())
            self.btn_brush.clicked.connect(lambda *_: self._toggle_mask_brush())
            self.btn_mask_export.clicked.connect(lambda *_: self._export_mask_full())
        except Exception:
            pass

        self.plot_type.changed.connect(lambda *_: self._on_kind_change())
        self.bins.changed.connect(lambda *_: self._maybe_plot_live())
        self.clip.changed.connect(lambda *_: self._maybe_plot_live())
        self.logy.changed.connect(lambda *_: self._maybe_plot_live())
        self.max_points.changed.connect(lambda *_: self._maybe_plot_live())
        self.live.changed.connect(lambda *_: self._toggle_live())

        # class change -> label/colour/metadata update
        self.mask_name.changed.connect(lambda *_: self._on_class_changed())
        # brush size live update
        self.brush_size.changed.connect(lambda *_: self._update_mask_brush_size())

        v = current_viewer()
        if v is not None:
            try:
                v.layers.selection.events.active.connect(
                    lambda *_: (self.refresh(), self._maybe_plot_live())
                )
                v.layers.events.inserted.connect(
                    lambda *_: (self.refresh(), self._maybe_plot_live())
                )
                v.layers.events.removed.connect(
                    lambda *_: (self.refresh(), self._maybe_plot_live())
                )
                v.layers.events.reordered.connect(
                    lambda *_: (self.refresh(), self._maybe_plot_live())
                )
                v.dims.events.current_step.connect(
                    lambda *_: self._maybe_plot_live()
                )
            except Exception:
                pass

        self.refresh()
        self._on_kind_change()

    # ------------------------------------------------------------------ lifecycle
    def _mark_dead(self) -> None:
        self._alive = False

    def is_alive(self) -> bool:
        return bool(self._alive)

    def as_qwidget(self) -> QWidget:
        return self._q

    # ------------------------------------------------------------------ public
    def refresh(self) -> None:
        """Refresh the image layer list and keep active one selected."""
        if not self._alive:
            return

        v = current_viewer()
        names: List[str] = []

        if v is not None:
            for layer in v.layers:
                if isinstance(layer, NapariImage):
                    names.append(layer.name)

        try:
            if not names:
                self.pick.choices = [("— no images —", None)]
                self.pick.value = None
                return

            self.pick.choices = names

            current = None
            try:
                active = v.layers.selection.active  # type: ignore[assignment]
                if isinstance(active, NapariImage):
                    current = active.name
            except Exception:
                pass

            if current in names:
                self.pick.value = current
            elif self.pick.value not in names:
                self.pick.value = names[0]

        except Exception:
            return

    # ------------------------------------------------------------------ internals
    def _toggle_live(self) -> None:
        self._live_on = bool(self.live.value)
        self._maybe_plot_live()

    def _maybe_plot_live(self) -> None:
        if self._live_on:
            self.plot()

    def _on_kind_change(self) -> None:
        kind = (self.plot_type.value or "").lower()
        hist_like = kind.startswith("hist")
        self.bins.visible = bool(hist_like)
        self.logy.visible = bool(hist_like)
        self._maybe_plot_live()

    def _current_image(self) -> Optional[NapariImage]:
        v = current_viewer()
        if v is None:
            return None

        nm = self.pick.value if isinstance(self.pick.value, str) else None
        if nm:
            for layer in v.layers:
                if isinstance(layer, NapariImage) and layer.name == nm:
                    return layer

        try:
            active = v.layers.selection.active
            return active if isinstance(active, NapariImage) else None
        except Exception:
            return None

    def _view_plane(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim < 2:
            return a

        v = current_viewer()
        if a.ndim == 2 or v is None:
            return a

        try:
            z = int(v.dims.current_step[0])
        except Exception:
            z = 0
        z = max(0, min(z, a.shape[0] - 1))
        return a[z]

    def _safe_sample(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a).ravel()
        nmax = int(self.max_points.value or 200_000)
        n = a.size
        if n <= nmax:
            return a
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=nmax, replace=False)
        return a[idx]

    def _get_hist_data(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if bool(self.clip.value):
            try:
                lo, hi = np.nanpercentile(a, [1.0, 99.0])
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    a = np.clip(a, lo, hi)
            except Exception:
                pass
        return self._safe_sample(a)

    # ------------------------------------------------------------------ mask helpers (multi-class)

    def _update_current_label(self) -> None:
        name = str(self.mask_name.value or "Air")
        self._current_label = CLASS_TO_LABEL.get(name, 1)

    def _on_class_changed(self) -> None:
        """When the class dropdown changes, update label and mask layer brush."""
        self._update_current_label()
        lab = self._find_mask_layer()
        if lab is not None:
            self._apply_mask_colourmap(lab)
            self._update_mask_brush_size()

    def _find_mask_layer(self) -> Optional[NapariLabels]:
        v = current_viewer()
        if v is None:
            return None
        for L in v.layers:
            if isinstance(L, NapariLabels) and L.name == "mask":
                return L  # type: ignore[return-value]
        return None

    def _current_class_name(self) -> str:
        name = str(self.mask_name.value or "Air")
        if name not in CLASS_TO_LABEL:
            name = "Other"
        return name

    def _rgba_to_hex(self, rgba: Tuple[float, float, float, float]) -> str:
        r = int(np.clip(rgba[0] * 255, 0, 255))
        g = int(np.clip(rgba[1] * 255, 0, 255))
        b = int(np.clip(rgba[2] * 255, 0, 255))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _apply_mask_colourmap(self, lab: NapariLabels) -> None:
        """Apply colormap:
        - 0 → transparent
        - current label → current colour
        Store mapping in metadata['classes'].
        """
        rgba = self._mask_rgba
        label = int(self._current_label)
        class_name = self._current_class_name()
        hex_color = self._rgba_to_hex(rgba)

        try:
            cmap = dict(getattr(lab, "color", {}) or {})
        except Exception:
            cmap = {}
        cmap[0] = (0.0, 0.0, 0.0, 0.0)  # background fully transparent
        cmap[label] = rgba
        try:
            lab.color = cmap
        except Exception:
            debug("Could not set mask colour map")

        try:
            lab.selected_label = label
        except Exception:
            pass

        # store mapping label -> {class, color} in metadata
        try:
            meta = dict(getattr(lab, "metadata", {}) or {})
            classes = dict(meta.get("classes", {}) or {})
            classes[str(label)] = {
                "class": class_name,
                "color": hex_color,
            }
            meta["classes"] = classes
            lab.metadata = meta
        except Exception:
            pass

    def _update_mask_brush_size(self) -> None:
        lab = self._find_mask_layer()
        if lab is None:
            return
        try:
            lab.brush_size = int(self.brush_size.value or 5)
        except Exception:
            pass

    def _ensure_mask_layer(self) -> Optional[NapariLabels]:
        """Return a 'mask' labels layer, create if missing, and sync brush size/colour."""
        v = current_viewer()
        if v is None:
            show_warning("No viewer available for mask.")
            return None

        lab = self._find_mask_layer()
        img = self._current_image()

        if lab is None:
            if img is None:
                show_warning("Need an image to size the mask.")
                return None

            try:
                data = np.zeros_like(img.data, dtype=np.uint16)
            except Exception:
                show_warning("Could not create mask layer (bad image data).")
                return None

            lab = v.add_labels(data, name="mask")

            try:
                lab.opacity = 0.7
                lab.blending = "translucent"
            except Exception:
                pass

        # colour mapping + brush size
        self._apply_mask_colourmap(lab)
        self._update_mask_brush_size()

        return lab

    def _pick_mask_color(self) -> None:
        """Open colour dialog and update mask label colour for current class."""
        try:
            col0 = QColor(255, 102, 0)
            col = QColorDialog.getColor(col0, None, "Pick mask colour")
        except Exception:
            show_warning("Colour dialog not available.")
            return

        if not col.isValid():
            return

        rgba = (
            col.red() / 255.0,
            col.green() / 255.0,
            col.blue() / 255.0,
            1.0,
        )
        self._mask_rgba = rgba

        # visual feedback on button
        try:
            self.btn_mask_color.native.setStyleSheet(
                f"background-color: {col.name()};"
            )
        except Exception:
            try:
                self.btn_mask_color.text = f"Colour {col.name()}"
            except Exception:
                pass

        lab = self._ensure_mask_layer()
        if lab is not None:
            self._apply_mask_colourmap(lab)

        show_info("Mask colour updated for current class.")

    def _toggle_mask_brush(self) -> None:
        """Toggle painting on the 'mask' labels layer."""
        v = current_viewer()
        if v is None:
            show_warning("No viewer available for brush.")
            return

        # turn ON
        if not self._brush_on:
            lab = self._ensure_mask_layer()
            if lab is None:
                return

            # remember previous active layer + mode
            try:
                active = v.layers.selection.active
            except Exception:
                active = None

            self._brush_prev_layer = active
            self._brush_prev_mode = getattr(active, "mode", None) if active is not None else None

            # activate mask + paint mode
            try:
                v.layers.selection.active = lab
                lab.mode = "paint"
            except Exception:
                show_warning("Could not switch to brush mode.")
                return

            self._update_mask_brush_size()

            self._brush_on = True
            self.btn_brush.text = "Brush (on)"
            show_info("Brush ON — paint on 'mask' in the viewer, click again to turn off.")
            return

        # turn OFF
        self._brush_on = False
        self.btn_brush.text = "Brush"

        prev_layer = self._brush_prev_layer
        prev_mode = self._brush_prev_mode

        try:
            layers = list(v.layers)
        except Exception:
            layers = []

        # try to go back to previous layer + mode
        if prev_layer is not None and prev_layer in layers:
            try:
                v.layers.selection.active = prev_layer
                if prev_mode is not None and hasattr(prev_layer, "mode"):
                    prev_layer.mode = prev_mode
                show_info("Brush OFF — returned to previous layer.")
                return
            except Exception:
                pass

        # fallback: keep current active layer, but pan/zoom mode
        try:
            active = v.layers.selection.active
            if hasattr(active, "mode"):
                active.mode = "pan_zoom"
        except Exception:
            pass

        show_info("Brush OFF.")

    def _export_mask_full(self) -> None:
        """Export CURRENT SLICE ONLY into a single ZIP:

        ZIP contents:
          - mask.png         : RGB mask (each label coloured)
          - mask_legend.png  : RGB mask with legend drawn on it
          - overlay.png      : original slice + mask overlay
          - legend.json      : label -> {class, color}
        """
        import os, json, zipfile

        v = current_viewer()
        if v is None:
            show_warning("No viewer available for mask export.")
            return

        lab = self._find_mask_layer()
        img = self._current_image()
        if lab is None or img is None:
            show_warning("Need both an image and a 'mask' labels layer.")
            return

        try:
            mask_data_full = np.asarray(lab.data)
            img_data_full = np.asarray(img.data)
        except Exception:
            show_warning("Could not read image/mask data.")
            return

        # ---------- pick current slice (2D only) ----------
        if mask_data_full.ndim == 3:
            try:
                z_idx = int(v.dims.current_step[0])
            except Exception:
                z_idx = 0
            z_idx = max(0, min(z_idx, mask_data_full.shape[0] - 1))
            mask_slice = mask_data_full[z_idx]
        elif mask_data_full.ndim == 2:
            z_idx = 0
            mask_slice = mask_data_full
        else:
            show_warning("Mask must be 2D or 3D.")
            return

        if img_data_full.ndim == 3:
            if z_idx >= img_data_full.shape[0]:
                show_warning("Image and mask Z do not match.")
                return
            img_slice = img_data_full[z_idx]
        elif img_data_full.ndim == 2:
            img_slice = img_data_full
        else:
            show_warning("Image must be 2D or 3D.")
            return

        if mask_slice.shape != img_slice.shape:
            show_warning("Mask and image slice shapes do not match.")
            return

        # ---------- collect labels on this slice ----------
        yy, xx = np.where(mask_slice > 0)
        if yy.size == 0:
            show_warning("Mask is empty on this slice (no label>0).")
            return

        labels_here = mask_slice[yy, xx].astype(np.int64)
        uniq_labels = sorted(int(lv) for lv in np.unique(labels_here) if lv > 0)
        if not uniq_labels:
            show_warning("Mask has no labels > 0 on this slice.")
            return

        # ---------- legend (classes + colours) ------------
        meta = dict(getattr(lab, "metadata", {}) or {})
        classes_meta = dict(meta.get("classes", {}) or {})

        base_palette = [
            "#ff0000", "#00ff00", "#0000ff", "#ffff00",
            "#ff00ff", "#00ffff", "#ff8000", "#8000ff",
            "#00ff80", "#ff0080", "#808000", "#008080",
        ]

        # detect if all existing colours are the same → force palette
        existing_colors = []
        for lv in uniq_labels:
            info = classes_meta.get(str(lv), {})
            col = str(info.get("color", "")).lower()
            if col:
                existing_colors.append(col)
        unique_existing = set(existing_colors)
        force_palette = len(uniq_labels) > 1 and (len(unique_existing) <= 1)

        label_to_class = {}
        label_to_color = {}

        for idx, lv in enumerate(uniq_labels):
            key = str(lv)
            info = dict(classes_meta.get(key, {}) or {})
            class_name = info.get("class") or f"class_{lv}"

            if force_palette or not info.get("color"):
                color_hex = base_palette[idx % len(base_palette)]
            else:
                color_hex = str(info["color"])

            info["class"] = class_name
            info["color"] = color_hex
            classes_meta[key] = info
            label_to_class[lv] = class_name
            label_to_color[lv] = color_hex

        # push legend back into metadata so it persists
        try:
            meta["classes"] = classes_meta
            lab.metadata = meta
        except Exception:
            pass

        # ---------- ask for ZIP path ----------------------
        start_dir = ""
        try:
            start_dir = getattr(getattr(v, "window", None), "workspace_dir", "") or ""
        except Exception:
            pass

        zip_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save 2D mask export (ZIP)",
            start_dir,
            "Zip files (*.zip)",
        )
        if not zip_path:
            return
        if not zip_path.lower().endswith(".zip"):
            zip_path = zip_path + ".zip"

        base_no_ext = os.path.splitext(zip_path)[0]
        tmp_mask_png      = base_no_ext + "_mask.png"
        tmp_mask_legend   = base_no_ext + "_mask_legend.png"
        tmp_overlay_png   = base_no_ext + "_overlay.png"
        tmp_legend_json   = base_no_ext + "_legend.json"

        tmp_files = [tmp_mask_png, tmp_mask_legend, tmp_overlay_png, tmp_legend_json]

        # ---------- helper: hex -> rgb --------------------
        def _parse_hex(c: str) -> Tuple[int, int, int]:
            c = str(c or "").strip()
            if c.startswith("#"):
                c = c[1:]
            if len(c) != 6:
                return (255, 255, 255)
            try:
                r = int(c[0:2], 16)
                g = int(c[2:4], 16)
                b = int(c[4:6], 16)
                return (r, g, b)
            except Exception:
                return (255, 255, 255)

        # ---------- build RGB mask (no legend) ------------
        h, w = mask_slice.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for lv in uniq_labels:
            color_hex = label_to_color.get(lv, "#ffffff")
            r, g, b = _parse_hex(color_hex)
            sel = (mask_slice == lv)
            if not np.any(sel):
                continue
            rgb_mask[sel, 0] = r
            rgb_mask[sel, 1] = g
            rgb_mask[sel, 2] = b

        # ---------- make overlay (image + mask) ----------
        try:
            img_arr = np.asarray(img_slice, dtype=np.float32)
            if img_arr.size == 0:
                raise ValueError("Empty image slice")

            step = max(1, img_arr.size // 256_000)
            flat = img_arr.ravel()[::step]
            lo, hi = np.nanpercentile(flat, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))

            base_norm = np.clip((img_arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            base_u8 = (base_norm * 255).astype(np.uint8)
            base_rgb = np.stack([base_u8] * 3, axis=-1)

            overlay = base_rgb.copy()
            alpha = 0.4
            mask_nonzero = mask_slice > 0
            overlay[mask_nonzero] = (
                (1.0 - alpha) * overlay[mask_nonzero].astype(np.float32)
                + alpha * rgb_mask[mask_nonzero].astype(np.float32)
            ).astype(np.uint8)
        except Exception:
            # fallback: just reuse rgb_mask if intensity scaling fails
            overlay = rgb_mask.copy()

        # ---------- write legend.json ---------------------
        try:
            with open(tmp_legend_json, "w", encoding="utf-8") as f:
                json.dump(classes_meta, f, indent=2)
        except Exception as e:
            show_warning(f"Legend export failed: {e}")
            # still continue; PNGs are useful

        # ---------- write PNGs (plain + overlay) ---------
        try:
            from tifffile import imwrite as _imwrite_png  # type: ignore

            _imwrite_png(tmp_mask_png, rgb_mask, photometric="rgb")
            _imwrite_png(tmp_overlay_png, overlay, photometric="rgb")
        except Exception as e:
            show_warning(f"PNG export (mask/overlay) failed: {e}")
            return

        # ---------- make mask_legend.png -----------------
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from matplotlib.patches import Patch  # type: ignore

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(rgb_mask)
            ax.axis("off")

            handles = []
            labels_txt = []
            for lv in uniq_labels:
                color_hex = label_to_color.get(lv, "#ffffff")
                class_name = label_to_class.get(lv, f"class_{lv}")
                handles.append(Patch(facecolor=color_hex, edgecolor="none"))
                labels_txt.append(class_name)

            if handles:
                ax.legend(
                    handles,
                    labels_txt,
                    loc="lower right",
                    framealpha=0.75,
                    fontsize=8,
                )

            fig.savefig(tmp_mask_legend, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            # fallback: if mpl missing, reuse plain mask
            try:
                import shutil
                shutil.copyfile(tmp_mask_png, tmp_mask_legend)
            except Exception:
                pass

        # ---------- pack everything into ZIP -------------
        try:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                if os.path.exists(tmp_mask_png):
                    zf.write(tmp_mask_png, arcname="mask.png")
                if os.path.exists(tmp_mask_legend):
                    zf.write(tmp_mask_legend, arcname="mask_legend.png")
                if os.path.exists(tmp_overlay_png):
                    zf.write(tmp_overlay_png, arcname="overlay.png")
                if os.path.exists(tmp_legend_json):
                    zf.write(tmp_legend_json, arcname="legend.json")
        except Exception as e:
            show_warning(f"Creating ZIP failed: {e}")
            return
        finally:
            # clean temporary files
            for p in tmp_files:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

        show_info(
            f"Saved 2D mask ZIP for slice z={z_idx}:\n{zip_path}\n"
            "Contents: mask.png, mask_legend.png, overlay.png, legend.json"
        )

    # ------------------------------------------------------------------ plotting
    def plot(self) -> None:
        if not HAVE_MPL or self.fig is None or self.canvas is None:
            show_warning("Matplotlib not available.")
            return
        if not self._alive:
            return

        layer = self._current_image()
        if layer is None:
            show_warning("Pick an image layer.")
            return

        try:
            data = np.asarray(layer.data)
        except Exception:
            show_warning("Could not read layer data.")
            return

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        self._last_series = None

        kind = str(self.plot_type.value or "histogram").lower()

        if kind == "histogram":
            bins = int(max(1, int(self.bins.value or 256)))
            vals = self._get_hist_data(data)

            ax.hist(vals.ravel(), bins=bins, log=bool(self.logy.value))
            s = array_stats(vals)

            ax.set_title(
                f"{layer.name} — hist  "
                f"(min={s['min']:.3g}, max={s['max']:.3g}, "
                f"μ={s['mean']:.3g}, σ={s['std']:.3g})"
            )
            ax.set_xlabel("value")
            ax.set_ylabel("count")

            try:
                counts, edges = np.histogram(vals.ravel(), bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2.0
                self._last_series = (centers.astype(float), counts.astype(float))
            except Exception:
                pass

        elif kind in ("profile x", "profile y"):
            plane = self._view_plane(data)
            if plane.ndim != 2:
                plane = np.squeeze(plane)
            if plane.ndim != 2:
                show_warning("Profile requires a 2D image plane.")
                return

            if kind.endswith("x"):
                y = plane.mean(axis=0)
                x = np.arange(y.size, dtype=float)
                ax.set_xlabel("x")
                ax.set_ylabel("mean intensity")
                ax.set_title(f"{layer.name} — profile X (mean over rows) at current Z")
            else:
                y = plane.mean(axis=1)
                x = np.arange(y.size, dtype=float)
                ax.set_xlabel("y")
                ax.set_ylabel("mean intensity")
                ax.set_title(f"{layer.name} — profile Y (mean over cols) at current Z")

            ax.plot(x, y)
            self._last_series = (x, y.astype(float))

        elif kind == "profile z":
            if data.ndim < 3:
                y = np.array([float(np.mean(data))], dtype=float)
                x = np.array([0.0], dtype=float)
            else:
                y = data.reshape(data.shape[0], -1).mean(axis=1)
                x = np.arange(y.size, dtype=float)

            ax.plot(x, y)
            ax.set_xlabel("z")
            ax.set_ylabel("mean intensity")
            ax.set_title(f"{layer.name} — profile Z (mean over Y×X)")
            self._last_series = (x, y.astype(float))

        elif kind == "box":
            vals = self._get_hist_data(data)
            ax.boxplot(vals, vert=True, showfliers=False)
            s = array_stats(vals)
            try:
                q1 = float(np.nanpercentile(vals, 25))
                med = float(np.nanpercentile(vals, 50))
                q3 = float(np.nanpercentile(vals, 75))
            except Exception:
                q1 = med = q3 = float("nan")

            ax.set_title(
                f"{layer.name} — box "
                f"(min={s['min']:.3g}, Q1≈{q1:.3g}, median≈{med:.3g}, "
                f"Q3≈{q3:.3g}, max={s['max']:.3g})"
            )
            ax.set_ylabel("value")

            idx = np.arange(vals.size, dtype=float)
            self._last_series = (idx, vals.astype(float))

        elif kind == "scatter":
            vals = self._get_hist_data(data)
            x = np.arange(vals.size, dtype=float)
            ax.scatter(x, vals, s=2, alpha=0.6)
            ax.set_xlabel("index")
            ax.set_ylabel("value")
            ax.set_title(f"{layer.name} — scatter (sampled {vals.size:n} pts)")
            self._last_series = (x, vals.astype(float))

        else:
            ax.text(
                0.5,
                0.5,
                f"'{kind}' not implemented",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{layer.name} — {kind}")

        try:
            self.fig.tight_layout()
            self.canvas.draw_idle()
        except Exception:
            pass

    # ------------------------------------------------------------------ export (plot series)
    def _export_png(self) -> None:
        if not HAVE_MPL or self.fig is None:
            show_warning("Cannot export: matplotlib not available.")
            return

        v = current_viewer()
        start = ""
        try:
            start = getattr(getattr(v, "window", None), "workspace_dir", "") or ""
        except Exception:
            pass

        path, _ = QFileDialog.getSaveFileName(
            None, "Save plot as PNG", start, "PNG images (*.png)"
        )
        if not path:
            return

        try:
            self.fig.savefig(path, dpi=150)
            show_info(f"Saved PNG: {path}")
        except Exception as e:
            show_warning(f"Save failed: {e}")

    def _export_csv(self) -> None:
        if not self._last_series:
            show_warning("Nothing to export — plot a series first.")
            return

        x, y = self._last_series

        v = current_viewer()
        start = ""
        try:
            start = getattr(getattr(v, "window", None), "workspace_dir", "") or ""
        except Exception:
            pass

        path, _ = QFileDialog.getSaveFileName(
            None, "Save series as CSV", start, "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            arr = np.column_stack([x, y])
            np.savetxt(path, arr, delimiter=",", header="x,y", comments="")
            show_info(f"Saved CSV: {path}")
        except Exception as e:
            show_warning(f"Save failed: {e}")


# ---------------------------------------------------------------------- factory
def create_plot_lab_widget(settings: Optional[AppSettings] = None) -> QWidget:
    """Factory: return a ready-to-dock QWidget for the plot lab."""
    if settings is None:
        settings = AppSettings()
    lab = PlotLab(settings)
    return lab.as_qwidget()
