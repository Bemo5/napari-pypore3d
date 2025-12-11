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
    """Plot dock: histogram / profiles / box / scatter.

    Features:
      - Multiple plot types: histogram, profile X/Y/Z, box, scatter
      - Live plotting with configurable bin count and max samples
      - Clip 1–99% percentile option
      - Log-Y scale for histograms
      - Export plots as PNG and series data as CSV
    """

    def __init__(self, settings: AppSettings) -> None:
        self._alive = True
        self._live_on = True
        self._last_series: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (x, y)

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

        # ------------------------------------------------------------------ rows
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

        # tighten rows
        for row in (row_layer, row_kind, row_hist, row_flags, row_actions):
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
            _minw(w, 220)
        for w in (self.plot_type,):
            _minw(w, 160)
        for w in (self.bins, self.max_points):
            _minw(w, 130)
        for w in (self.btn_plot, self.btn_png, self.btn_csv):
            _minw(w, 110)

        # ------------------------------------------------------------------ Qt wrapper
        self._q = QWidget()
        root = QVBoxLayout(self._q)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(20)

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
        except Exception:
            pass

        self.plot_type.changed.connect(lambda *_: self._on_kind_change())
        self.bins.changed.connect(lambda *_: self._maybe_plot_live())
        self.clip.changed.connect(lambda *_: self._maybe_plot_live())
        self.logy.changed.connect(lambda *_: self._maybe_plot_live())
        self.max_points.changed.connect(lambda *_: self._maybe_plot_live())
        self.live.changed.connect(lambda *_: self._toggle_live())

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