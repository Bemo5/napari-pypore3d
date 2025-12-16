from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Callable
import numpy as np

from magicgui.widgets import Container, Label, PushButton, CheckBox, ComboBox, SpinBox
from napari import current_viewer
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_warning, show_info

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFrame,
    QFileDialog,
    QSpacerItem,
    QSizePolicy,
    QDialog,
)
from qtpy.QtCore import QTimer

# napari worker (runs compute off the UI thread)
try:
    from napari.qt.threading import thread_worker
except Exception:
    thread_worker = None  # type: ignore

# --- optional matplotlib -------------------------------------------------------
HAVE_MPL = True
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.ticker import FuncFormatter
except Exception:
    HAVE_MPL = False
    Figure = object  # type: ignore
    FigureCanvas = object  # type: ignore
    NavigationToolbar = object  # type: ignore
    FuncFormatter = None  # type: ignore

# --- optional hover cursor ----------------------------------------------------
# NOTE: mplcursors can lag on big scatters; we will only use it for small artists.
try:
    import mplcursors  # type: ignore
    HAVE_CURSOR = True
except Exception:
    mplcursors = None  # type: ignore
    HAVE_CURSOR = False

# --- helpers / settings --------------------------------------------------------
try:
    from .helpers import AppSettings, array_stats
except Exception:
    @dataclass
    class AppSettings:
        default_bins: int = 256
        clip_1_99: bool = True

    def array_stats(a: np.ndarray) -> Dict[str, float]:
        a = np.asarray(a)
        return dict(
            min=float(np.nanmin(a)),
            max=float(np.nanmax(a)),
            mean=float(np.nanmean(a)),
            std=float(np.nanstd(a)),
        )


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


# ==============================================================================
# Compute helpers
# ==============================================================================

def _rng0_choice(n: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.choice(n, size=k, replace=False)


def _get_current_z() -> int:
    v = current_viewer()
    if v is None:
        return 0
    try:
        return int(v.dims.current_step[0])
    except Exception:
        return 0


def _slice_plane(arr: np.ndarray, z: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim < 2:
        return a
    z = max(0, min(int(z), int(a.shape[0]) - 1))
    return a[z]


def _clip_1_99_full(a: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(a, [1.0, 99.0])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return np.clip(a, lo, hi)
    return a


def _fmt_plain(v: float) -> str:
    """Plain, human-readable number (no scientific), with commas."""
    if not np.isfinite(v):
        return "nan"
    # integer-ish -> show as int
    if abs(v - round(v)) < 1e-12:
        return f"{int(round(v)):,}"
    # otherwise 6 sig figs, no sci
    return f"{v:,.6g}"


def _compute_series(
    data: Any,
    kind: str,
    bins: int,
    clip: bool,
    logy: bool,
    max_points: int,
    z_index: int,
) -> Dict[str, Any]:
    """
    Heavy compute done OFF the UI thread.

    Returns dict containing:
      - 'kind'
      - 'title'
      - 'xlabel', 'ylabel'
      - 'plot': instructions for matplotlib
      - '_last_series': (x, y) for CSV export
      - 'hover': info for hover fallback
    """
    a = np.asarray(data)
    out: Dict[str, Any] = dict(kind=kind)
    k = kind.lower().strip()

    if k == "histogram":
        vals = a.ravel()
        if clip:
            vals = _clip_1_99_full(vals)

        s = array_stats(vals)
        counts, edges = np.histogram(vals, bins=int(max(1, bins)))
        centers = (edges[:-1] + edges[1:]) / 2.0

        out["xlabel"] = "value"
        out["ylabel"] = "count"
        out["title"] = (
            f"hist (min={_fmt_plain(s['min'])}, max={_fmt_plain(s['max'])}, "
            f"μ={_fmt_plain(s['mean'])}, σ={_fmt_plain(s['std'])})"
        )
        out["plot"] = dict(
            type="hist_prebinned",
            centers=centers.astype(float),
            counts=counts.astype(float),
            edges=edges.astype(float),
            log=bool(logy),
        )
        out["_last_series"] = (centers.astype(float), counts.astype(float))
        out["hover"] = dict(mode="xy", x=centers.astype(float), y=counts.astype(float))
        return out

    if k in ("profile x", "profile y"):
        plane = _slice_plane(a, z_index)
        plane = np.asarray(plane)
        if plane.ndim != 2:
            plane = np.squeeze(plane)
        if plane.ndim != 2:
            raise ValueError("Profile requires a 2D image plane.")

        if k.endswith("x"):
            y = plane.mean(axis=0).astype(float)
            x = np.arange(y.size, dtype=float)
            out["xlabel"] = "x"
            out["ylabel"] = "mean intensity"
            out["title"] = "profile X (mean over rows) at current Z"
        else:
            y = plane.mean(axis=1).astype(float)
            x = np.arange(y.size, dtype=float)
            out["xlabel"] = "y"
            out["ylabel"] = "mean intensity"
            out["title"] = "profile Y (mean over cols) at current Z"

        out["plot"] = dict(type="line", x=x, y=y)
        out["_last_series"] = (x, y)
        out["hover"] = dict(mode="xy", x=x, y=y)
        return out

    if k == "profile z":
        if a.ndim < 3:
            y = np.array([float(np.mean(a))], dtype=float)
            x = np.array([0.0], dtype=float)
        else:
            y = a.reshape(a.shape[0], -1).mean(axis=1).astype(float)
            x = np.arange(y.size, dtype=float)

        out["xlabel"] = "z"
        out["ylabel"] = "mean intensity"
        out["title"] = "profile Z (mean over Y×X)"
        out["plot"] = dict(type="line", x=x, y=y)
        out["_last_series"] = (x, y)
        out["hover"] = dict(mode="xy", x=x, y=y)
        return out

    if k == "box":
        # Accurate stats on FULL data; render later via bxp (fast draw).
        vals = a.ravel()
        if clip:
            vals = _clip_1_99_full(vals)

        # exact percentiles:
        q1, med, q3 = np.nanpercentile(vals, [25.0, 50.0, 75.0])
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

        # whiskers: classic Tukey 1.5*IQR (accurate)
        iqr = float(q3 - q1)
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        # clamp whiskers to data range
        whislo = float(np.nanmin(vals[vals >= lo])) if np.any(vals >= lo) else vmin
        whishi = float(np.nanmax(vals[vals <= hi])) if np.any(vals <= hi) else vmax

        out["xlabel"] = ""
        out["ylabel"] = "value"
        out["title"] = (
            f"box (min={_fmt_plain(vmin)}, Q1={_fmt_plain(float(q1))}, "
            f"median={_fmt_plain(float(med))}, Q3={_fmt_plain(float(q3))}, "
            f"max={_fmt_plain(vmax)})"
        )

        out["plot"] = dict(
            type="box_stats",
            stats=dict(
                label="",
                med=float(med),
                q1=float(q1),
                q3=float(q3),
                whislo=float(whislo),
                whishi=float(whishi),
                fliers=np.array([], dtype=float),
            ),
        )

        # export series (warning: huge). Still accurate.
        idx = np.arange(vals.size, dtype=float)
        out["_last_series"] = (idx, vals.astype(float))
        out["hover"] = dict(mode="none")
        return out

    if k == "scatter":
        vals_full = a.ravel()
        if clip:
            vals_full = _clip_1_99_full(vals_full)

        n = int(vals_full.size)
        kmax = int(max_points) if int(max_points) > 0 else 100_000
        if n > kmax:
            idx = _rng0_choice(n, kmax)
            vals = vals_full[idx]
            x = np.arange(vals.size, dtype=float)
        else:
            vals = vals_full
            x = np.arange(n, dtype=float)

        out["xlabel"] = "index"
        out["ylabel"] = "value"
        out["title"] = f"scatter (displaying {vals.size:,} pts)"
        out["plot"] = dict(type="scatter", x=x.astype(float), y=vals.astype(float))
        out["_last_series"] = (x.astype(float), vals.astype(float))
        out["hover"] = dict(mode="xy", x=x.astype(float), y=vals.astype(float))
        return out

    raise ValueError(f"'{kind}' not implemented")


# ==============================================================================
# Hover + formatting helpers
# ==============================================================================

def _format_xy(x: float, y: float) -> str:
    return f"x = {_fmt_plain(x)}\ny = {_fmt_plain(y)}"


def _apply_plain_axis_format(ax: Any) -> None:
    """Force plain (non-scientific) tick labels with commas."""
    if FuncFormatter is None:
        return
    try:
        fmt = FuncFormatter(lambda v, _pos: _fmt_plain(float(v)))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
    except Exception:
        pass
    # also disable offsets like "1e6"
    try:
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    except Exception:
        pass


def _install_mplcursors(artists: List[Any]) -> None:
    """Use mplcursors for small-ish artists only (avoid huge-lag cases)."""
    if not HAVE_CURSOR or mplcursors is None:
        return
    try:
        cur = mplcursors.cursor(artists, hover=True)

        @cur.connect("add")
        def _on_add(sel):
            try:
                x, y = sel.target
                sel.annotation.set_text(_format_xy(float(x), float(y)))
            except Exception:
                pass
    except Exception:
        pass


def _install_hover_fallback(canvas: Any, ax: Any, hover: Dict[str, Any]) -> None:
    """
    Pure-matplotlib hover fallback.
    For sampled series (<= ~100k) it's fine.
    """
    if canvas is None or ax is None or not isinstance(hover, dict):
        return
    if hover.get("mode", "xy") in ("none", None):
        return

    mode = hover.get("mode", "xy")
    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="black", ec="none", alpha=0.6),
        color="white",
    )
    ann.set_visible(False)

    x = None
    y = None
    if mode == "xy":
        try:
            x = np.asarray(hover["x"], dtype=float)
            y = np.asarray(hover["y"], dtype=float)
        except Exception:
            return
    elif mode == "y_only":
        try:
            y = np.asarray(hover["y"], dtype=float)
            x = np.arange(y.size, dtype=float)
        except Exception:
            return
    else:
        return

    if x is None or y is None or x.size == 0 or y.size == 0:
        return

    # if gigantic, don't hover (too slow)
    if x.size > 200_000:
        return

    def _on_move(event):
        if event.inaxes != ax:
            if ann.get_visible():
                ann.set_visible(False)
                canvas.draw_idle()
            return
        if event.xdata is None:
            return

        xd = float(event.xdata)
        i = int(np.argmin(np.abs(x - xd)))
        xi = float(x[i])
        yi = float(y[i])

        ann.xy = (xi, yi)
        ann.set_text(_format_xy(xi, yi))
        ann.set_visible(True)
        canvas.draw_idle()

    def _on_leave(_event):
        if ann.get_visible():
            ann.set_visible(False)
            canvas.draw_idle()

    try:
        canvas.mpl_connect("motion_notify_event", _on_move)
        canvas.mpl_connect("axes_leave_event", _on_leave)
    except Exception:
        pass


# ==============================================================================
# Pop-out dialog
# ==============================================================================

class _PlotPopup(QDialog):
    def __init__(self, parent: QWidget, title: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(980, 680)

        self.fig = Figure(figsize=(11.0, 7.0), constrained_layout=False)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)


# ==============================================================================
# PlotLab (UI)
# ==============================================================================

class PlotLab:
    """Plot dock: histogram / profiles / box / scatter + hover + pop-out."""

    def __init__(self, settings: AppSettings) -> None:
        self._alive = True
        self._live_on = False
        self._last_series: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self._plot_token: int = 0
        self._worker = None

        self._last_res: Optional[Dict[str, Any]] = None
        self._last_layer_name: str = ""
        self._last_kind: str = ""

        self._popup: Optional[_PlotPopup] = None

        # controls
        self.pick = ComboBox(choices=[], value=None, label="layer", nullable=True)
        self.plot_type = ComboBox(
            choices=["histogram", "profile X", "profile Y", "profile Z", "box", "scatter"],
            value="histogram",
            label="kind",
        )
        self.bins = SpinBox(
            value=int(getattr(settings, "default_bins", 256) or 256),
            min=1, max=4096, step=1,
            label="bins",
        )
        self.clip = CheckBox(text="clip 1–99%", value=bool(getattr(settings, "clip_1_99", True)))
        self.logy = CheckBox(text="log-Y", value=False)
        self.live = CheckBox(text="live", value=False)

        # IMPORTANT: lower default to reduce scatter lag
        self.max_points = SpinBox(
            value=30_000,
            min=10_000, max=2_000_000, step=10_000,
            label="max samples",
        )

        self.btn_plot = PushButton(text="Plot")
        self.btn_png = PushButton(text="Export PNG")
        self.btn_csv = PushButton(text="Export CSV")
        self.btn_pop = PushButton(text="Pop-out")

        # rows
        row_layer = Container(widgets=[self.pick], layout="horizontal", labels=True)
        row_kind = Container(widgets=[self.plot_type], layout="horizontal", labels=True)
        row_hist = Container(widgets=[self.bins, self.max_points], layout="horizontal", labels=True)
        row_flags = Container(widgets=[self.clip, self.logy, self.live], layout="horizontal", labels=True)
        row_actions = Container(
            widgets=[self.btn_plot, self.btn_pop, self.btn_png, self.btn_csv],
            layout="horizontal",
            labels=True,
        )

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

        _minw(self.pick, 220)
        _minw(self.plot_type, 160)
        _minw(self.bins, 130)
        _minw(self.max_points, 130)
        for w in (self.btn_plot, self.btn_pop, self.btn_png, self.btn_csv):
            _minw(w, 110)

        # Qt wrapper
        self._q = QWidget()
        root = QVBoxLayout(self._q)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(18)

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

        root.addSpacerItem(QSpacerItem(0, 8, QSizePolicy.Minimum, QSizePolicy.Fixed))

        if HAVE_MPL:
            self.fig = Figure(figsize=(9.5, 6.2), constrained_layout=False)
            self.canvas = FigureCanvas(self.fig)
            try:
                self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.canvas.setMinimumHeight(520)
            except Exception:
                pass

            self.toolbar = NavigationToolbar(self.canvas, self._q)

            plot_wrap = QWidget()
            p_lay = QVBoxLayout(plot_wrap)
            p_lay.setContentsMargins(10, 8, 10, 10)
            p_lay.setSpacing(6)
            p_lay.addWidget(self.toolbar)
            p_lay.addWidget(self.canvas)

            root.addWidget(_card("Plot (double-click to pop out)", plot_wrap), 2)

            try:
                self.canvas.mpl_connect("button_press_event", self._on_mpl_click)
            except Exception:
                pass
        else:
            self.fig = None
            self.canvas = None
            self.toolbar = None
            dummy = QWidget()
            d_lay = QVBoxLayout(dummy)
            d_lay.addWidget(Label(value="Matplotlib not available").native)
            root.addWidget(_card("Plot", dummy), 2)

        try:
            self._q.destroyed.connect(lambda *_: self._mark_dead())
        except Exception:
            pass

        # debounce
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self.plot)

        # wiring
        def _connect_btn(btn, fn: Callable[[], None]) -> None:
            try:
                btn.changed.connect(lambda *_: fn())
                return
            except Exception:
                pass
            try:
                btn.clicked.connect(lambda *_: fn())  # type: ignore[attr-defined]
            except Exception:
                pass

        _connect_btn(self.btn_plot, self.plot)
        _connect_btn(self.btn_pop, self.pop_out)
        _connect_btn(self.btn_png, self._export_png)
        _connect_btn(self.btn_csv, self._export_csv)

        self.plot_type.changed.connect(lambda *_: self._on_kind_change())
        self.bins.changed.connect(lambda *_: self._maybe_plot_live())
        self.clip.changed.connect(lambda *_: self._maybe_plot_live())
        self.logy.changed.connect(lambda *_: self._maybe_plot_live())
        self.max_points.changed.connect(lambda *_: self._maybe_plot_live())
        self.live.changed.connect(lambda *_: self._toggle_live())

        v = current_viewer()
        if v is not None:
            try:
                v.layers.selection.events.active.connect(lambda *_: (self.refresh(), self._maybe_plot_live()))
                v.layers.events.inserted.connect(lambda *_: (self.refresh(), self._maybe_plot_live()))
                v.layers.events.removed.connect(lambda *_: (self.refresh(), self._maybe_plot_live()))
                v.layers.events.reordered.connect(lambda *_: (self.refresh(), self._maybe_plot_live()))
            except Exception:
                pass

        self.refresh()
        self._on_kind_change()

    # lifecycle
    def _mark_dead(self) -> None:
        self._alive = False
        self._cancel_worker()
        try:
            if self._popup is not None:
                self._popup.close()
        except Exception:
            pass
        self._popup = None

    def as_qwidget(self) -> QWidget:
        return self._q

    # public
    def refresh(self) -> None:
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
                active = v.layers.selection.active
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

    # internals
    def _toggle_live(self) -> None:
        self._live_on = bool(self.live.value)
        self._maybe_plot_live()

    def _maybe_plot_live(self) -> None:
        if not self._live_on:
            return
        try:
            self._debounce.start(150)
        except Exception:
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

    def _cancel_worker(self) -> None:
        w = self._worker
        self._worker = None
        if w is None:
            return
        try:
            if hasattr(w, "quit"):
                w.quit()
            if hasattr(w, "cancel"):
                w.cancel()
        except Exception:
            pass

    # plotting
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

        kind = str(self.plot_type.value or "histogram")
        bins = int(max(1, int(self.bins.value or 256)))
        clip = bool(self.clip.value)
        logy = bool(self.logy.value)
        max_pts = int(self.max_points.value or 30_000)
        z_idx = _get_current_z()

        self._plot_token += 1
        token = int(self._plot_token)
        self._cancel_worker()

        # quick UI feedback
        try:
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Computing…", ha="center", va="center", transform=ax.transAxes)
            self._apply_margins(self.fig)
            self.canvas.draw_idle()
        except Exception:
            pass

        if thread_worker is None:
            try:
                res = _compute_series(layer.data, kind, bins, clip, logy, max_pts, z_idx)
                self._apply_plot_result(layer.name, kind, res)
            except Exception as e:
                show_warning(str(e))
            return

        @thread_worker
        def _work():
            return _compute_series(layer.data, kind, bins, clip, logy, max_pts, z_idx)

        w = _work()
        self._worker = w

        def _on_returned(res: Dict[str, Any]) -> None:
            if not self._alive or token != self._plot_token:
                return
            self._apply_plot_result(layer.name, kind, res)

        def _on_error(err: BaseException) -> None:
            if token != self._plot_token:
                return
            show_warning(f"Plot failed: {err}")

        try:
            w.returned.connect(_on_returned)
        except Exception:
            pass
        try:
            w.errored.connect(_on_error)  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            w.start()
        except Exception:
            try:
                res = _compute_series(layer.data, kind, bins, clip, logy, max_pts, z_idx)
                self._apply_plot_result(layer.name, kind, res)
            except Exception as e:
                show_warning(f"Plot failed: {e}")

    def _apply_margins(self, fig: Any) -> None:
        try:
            fig.subplots_adjust(left=0.12, right=0.985, bottom=0.14, top=0.90)
        except Exception:
            pass

    def _apply_plot_result(self, layer_name: str, kind: str, res: Dict[str, Any]) -> None:
        if not HAVE_MPL or self.fig is None or self.canvas is None:
            return
        if not self._alive:
            return

        self._last_res = res
        self._last_layer_name = layer_name
        self._last_kind = kind

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        self._last_series = None

        title = res.get("title", "")
        xlabel = res.get("xlabel", "")
        ylabel = res.get("ylabel", "")
        plot = res.get("plot", {})
        hover = res.get("hover", {})

        artists: List[Any] = []

        try:
            ptype = plot.get("type", "")

            if ptype == "hist_prebinned":
                centers = np.asarray(plot["centers"])
                counts = np.asarray(plot["counts"])
                edges = np.asarray(plot.get("edges", []))
                log = bool(plot.get("log", False))

                if edges.size >= 2:
                    width = float(edges[1] - edges[0])
                elif centers.size > 1:
                    width = float(centers[1] - centers[0])
                else:
                    width = 1.0

                bars = ax.bar(centers, counts, width=width, align="center")
                # bars are fine to hover (<= bins)
                artists.extend(list(bars))
                if log:
                    ax.set_yscale("log")

            elif ptype == "line":
                x = np.asarray(plot["x"])
                y = np.asarray(plot["y"])
                (ln,) = ax.plot(x, y)
                artists.append(ln)

            elif ptype == "box_stats":
                # FAST render, scientifically accurate stats (computed on full data)
                stats = plot.get("stats", None)
                if isinstance(stats, dict):
                    ax.bxp([stats], showfliers=True, vert=True)
                else:
                    ax.text(0.5, 0.5, "Box stats missing", ha="center", va="center", transform=ax.transAxes)

            elif ptype == "scatter":
                x = np.asarray(plot["x"])
                y = np.asarray(plot["y"])
                sc = ax.scatter(x, y, s=2, alpha=0.6)
                # DO NOT use mplcursors on big scatters; hover fallback is lighter.
                artists.append(sc)

            else:
                ax.text(0.5, 0.5, f"'{kind}' not implemented", ha="center", va="center", transform=ax.transAxes)

            ax.set_title(f"{layer_name} — {kind.lower()}  {title}".strip())
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)

            self._last_series = res.get("_last_series", None)

        except Exception as e:
            ax.text(0.5, 0.5, f"Render error: {e}", ha="center", va="center", transform=ax.transAxes)

        # force plain formatting (no 1e6, no e+07)
        _apply_plain_axis_format(ax)

        # margins
        self._apply_margins(self.fig)
        try:
            self.fig.tight_layout()
        except Exception:
            pass

        # hover tooltips:
        # - use mplcursors only if it won't hurt (avoid big scatter)
        if HAVE_CURSOR and artists and self._last_kind.lower().strip() != "scatter":
            _install_mplcursors(artists)
        else:
            _install_hover_fallback(self.canvas, ax, hover)

        try:
            self.canvas.draw_idle()
        except Exception:
            pass

        # If popup open, refresh it too
        if self._popup is not None and self._popup.isVisible():
            try:
                self._render_into_popup()
            except Exception:
                pass

    # pop-out
    def _on_mpl_click(self, event) -> None:
        try:
            if getattr(event, "dblclick", False):
                self.pop_out()
        except Exception:
            pass

    def pop_out(self) -> None:
        if not HAVE_MPL or self.fig is None or self.canvas is None:
            show_warning("Matplotlib not available.")
            return
        if not self._last_res:
            show_warning("Nothing to pop out — plot first.")
            return

        if self._popup is None:
            self._popup = _PlotPopup(self._q, f"Plot — {self._last_layer_name}")

        self._render_into_popup()
        self._popup.show()
        self._popup.raise_()
        self._popup.activateWindow()

    def _render_into_popup(self) -> None:
        if self._popup is None or not self._last_res:
            return

        fig = self._popup.fig
        canvas = self._popup.canvas

        fig.clf()
        ax = fig.add_subplot(111)

        res = self._last_res
        layer_name = self._last_layer_name
        kind = self._last_kind

        title = res.get("title", "")
        xlabel = res.get("xlabel", "")
        ylabel = res.get("ylabel", "")
        plot = res.get("plot", {})
        hover = res.get("hover", {})

        artists: List[Any] = []
        ptype = plot.get("type", "")

        if ptype == "hist_prebinned":
            centers = np.asarray(plot["centers"])
            counts = np.asarray(plot["counts"])
            edges = np.asarray(plot.get("edges", []))
            log = bool(plot.get("log", False))
            if edges.size >= 2:
                width = float(edges[1] - edges[0])
            elif centers.size > 1:
                width = float(centers[1] - centers[0])
            else:
                width = 1.0
            bars = ax.bar(centers, counts, width=width, align="center")
            artists.extend(list(bars))
            if log:
                ax.set_yscale("log")

        elif ptype == "line":
            x = np.asarray(plot["x"])
            y = np.asarray(plot["y"])
            (ln,) = ax.plot(x, y)
            artists.append(ln)

        elif ptype == "box_stats":
            stats = plot.get("stats", None)
            if isinstance(stats, dict):
                ax.bxp([stats], showfliers=False, vert=True)

        elif ptype == "scatter":
            x = np.asarray(plot["x"])
            y = np.asarray(plot["y"])
            sc = ax.scatter(x, y, s=6, alpha=0.7)
            artists.append(sc)

        ax.set_title(f"{layer_name} — {kind.lower()}  {title}".strip())
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        _apply_plain_axis_format(ax)

        try:
            fig.subplots_adjust(left=0.10, right=0.99, bottom=0.12, top=0.90)
        except Exception:
            pass
        try:
            fig.tight_layout()
        except Exception:
            pass

        if HAVE_CURSOR and artists and kind.lower().strip() != "scatter":
            _install_mplcursors(artists)
        else:
            _install_hover_fallback(canvas, ax, hover)

        try:
            canvas.draw_idle()
        except Exception:
            pass

    # export
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

        path, _ = QFileDialog.getSaveFileName(None, "Save plot as PNG", start, "PNG images (*.png)")
        if not path:
            return

        try:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
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

        path, _ = QFileDialog.getSaveFileName(None, "Save series as CSV", start, "CSV files (*.csv)")
        if not path:
            return

        try:
            arr = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
            np.savetxt(path, arr, delimiter=",", header="x,y", comments="")
            show_info(f"Saved CSV: {path}")
        except Exception as e:
            show_warning(f"Save failed: {e}")


# ---------------------------------------------------------------------- factory
def create_plot_lab_widget(settings: Optional[AppSettings] = None) -> QWidget:
    if settings is None:
        settings = AppSettings()
    lab = PlotLab(settings)
    return lab.as_qwidget()
