# napari_pypore3d/_widget.py — r51 (adds Session Recorder tab, global recorder reset)
from __future__ import annotations
import os, re, math, pathlib
from typing import Optional, Tuple, List
from contextlib import contextmanager
import numpy as np

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QScrollArea, QSizePolicy,
    QFileDialog, QLabel, QFormLayout, QSpacerItem, QSizePolicy as QSP, QFrame,
    QListWidget, QListWidgetItem, QPushButton
)
from magicgui.widgets import PushButton, CheckBox, ComboBox, FloatSpinBox, SpinBox
from napari import current_viewer
from napari.layers import Image as NapariImage, Points as NapariPoints
from napari.utils.notifications import show_info, show_warning

# ---------------- helpers that might be missing in your tree ------------------
try:
    from .helpers import AppSettings, cap_width
except Exception:
    class AppSettings:
        def __init__(self, dock_max_width=560, last_dir="", default_dtype="auto",
                     default_bins=256, clip_1_99=True, prefer_memmap=False):
            self.dock_max_width = dock_max_width
            self.last_dir = last_dir
            self.default_dtype = default_dtype
            self.default_bins = default_bins
            self.clip_1_99 = clip_1_99
            self.prefer_memmap = prefer_memmap
        def save(self): pass
        @classmethod
        def load(cls): return cls()
    def cap_width(*_, **__): pass

from .functions import functions_widget
from .brush import brush_widget

# --- Session Recorder (global singleton) ---
try:
    from .session_recorder import reset_global_recorder
    from .recorder_panel import make_session_recorder_panel
except Exception:
    reset_global_recorder = None
    make_session_recorder_panel = None

# --- MONAI inference tab (lazy) ---
try:
    from .monai_infer_widget import make_monai_infer_widget
except Exception:
    make_monai_infer_widget = None

def function_page_widget():
    return functions_widget()

try:
    from .slice_compare import make_slice_compare_panel
except Exception:
    def make_slice_compare_panel(): return None, QWidget()

try:
    from .crop import make_crop_panel
except Exception:
    def make_crop_panel(): return None, QWidget()

try:
    from .plots import PlotLab
except Exception:
    class PlotLab:
        def __init__(self, *_): pass
        def as_qwidget(self): return QWidget()

try:
    from .info_export import build_info_export_panel
except Exception:
    def build_info_export_panel(*_): return QWidget(), (lambda : None)

try:
    from .titles import refresh_titles
except Exception:
    def refresh_titles(*_): pass

try:
    from .hud import wire_caption_events_once, refresh_all_captions
except Exception:
    def wire_caption_events_once(): pass
    def refresh_all_captions(*_): pass

from .state import set_active_tab, active_tab, is_tab_active

# ---------------- constants ---------------------------------------------------
PLUGIN_BUILD = "napari-pypore3d plugin enjoy :)"
_DTYPE_CHOICES = ["auto","uint8","uint16","int16","int32","float32","float64","int8"]
_MIN_DOCK_WIDTH = 560
CENTER_TAG = "[center] "
SHOW_CENTERS_DEFAULT = True
CENTERS_ENABLED = SHOW_CENTERS_DEFAULT

os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

# ---------------- tiny UI helpers --------------------------------------------
def _pad(w: QWidget, L=18,T=14,R=16,B=16):
    holder = QWidget(); lay = QVBoxLayout(holder)
    lay.setContentsMargins(L,T,R,B); lay.setSpacing(10); lay.addWidget(w); return holder

def _wrap_scroll(w: QWidget):
    sc = QScrollArea(); sc.setFrameShape(QFrame.NoFrame); sc.setWidget(w); sc.setWidgetResizable(True)
    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    out = QWidget(); v = QVBoxLayout(out); v.setContentsMargins(0,0,0,0); v.addWidget(sc)
    out.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding); return out

def _spacer(): return QSpacerItem(1,1,QSP.Expanding,QSP.Minimum)

# ---------------- layer queries ----------------------------------------------
def _images(v) -> List[NapariImage]:
    """Return only true 3-D volume images (no 2-D compare views)."""
    if not v:
        return []
    out: List[NapariImage] = []
    for L in v.layers:
        if not isinstance(L, NapariImage):
            continue
        md = getattr(L, "metadata", {}) or {}
        if md.get("_slice_compare", False):
            continue
        if np.asarray(L.data).ndim != 3:
            continue
        out.append(L)
    return out

def _centers(v) -> List[NapariPoints]:
    return [L for L in (v.layers if v else []) if isinstance(L, NapariPoints) and L.name.startswith(CENTER_TAG)]

def _center_name(im: NapariImage) -> str:
    return f"{CENTER_TAG}{im.name}"

# ---------------- filenames / dtype / shape ----------------------------------
_AXBXB = re.compile(r"(\d+)[xX](\d+)[xX](\d+)")
_NCUBE = re.compile(r"(\d+)\s*(?:\^?\s*3|³)", re.IGNORECASE)

def _dtype_from(name: str) -> np.dtype:
    return {
        "uint8":np.uint8,"uint16":np.uint16,"int16":np.int16,"int32":np.int32,
        "float32":np.float32,"float64":np.float64,"int8":np.int8,"auto":np.uint8
    }.get(name,np.uint8)

def _hints(path: str) -> Optional[Tuple[int,int,int]]:
    stem = pathlib.Path(path).stem
    m = _AXBXB.search(stem);  m2 = _NCUBE.search(stem)
    if m:  z,y,x = map(int, m.groups());  return (z,y,x)
    if m2: n = int(m2.group(1));          return (n,n,n)
    return None

def _remember_dir(settings: AppSettings, any_path: str):
    try:
        p = pathlib.Path(any_path).expanduser().resolve()
        settings.last_dir = str(p.parent); settings.save()
    except Exception: pass

# ---------------- IO ----------------------------------------------------------
def _dtype_ok_quick(p: pathlib.Path, dt: np.dtype, fsize: int) -> bool:
    try:
        n = min(64*1024, fsize)
        a = np.frombuffer(open(p,"rb").read(n), dtype=np.dtype(dt).newbyteorder("<"))
        if a.size == 0: return True
        if np.issubdtype(np.dtype(dt), np.floating):
            bad = np.isnan(a).sum() + np.isinf(a).sum()
            return bad <= 0.20 * a.size
        return True
    except Exception:
        return True

def _read_array(path: str, dtype: np.dtype, shape: Tuple[int,int,int], prefer_memmap: bool):
    try:
        if prefer_memmap:
            mm = np.memmap(path, dtype=np.dtype(dtype).newbyteorder("<"), mode="r")
            need = int(np.prod(shape));  assert mm.size >= need
            return mm[:need].reshape(shape)
        arr = np.fromfile(path, dtype=np.dtype(dtype).newbyteorder("<"))
        need = int(np.prod(shape));      assert arr.size >= need
        return arr[:need].reshape(shape)
    except Exception as e:
        show_warning(f"Load failed: {e}"); return None

def _infer_shape_dtype(path: str, dtype_name: str, Z:int,Y:int,X:int) -> Optional[Tuple[np.dtype,Tuple[int,int,int]]]:
    p = pathlib.Path(path); fsize = p.stat().st_size
    if fsize <= 0: return None
    cands = [np.uint8,np.uint16,np.int16,np.uint32,np.int32,np.float32,np.float64,np.int8]
    if dtype_name != "auto": cands = [np.dtype(_dtype_from(dtype_name))]
    hint = _hints(path); planes = (hint[1]*hint[2]) if hint else None

    out: List[Tuple[np.dtype,Tuple[int,int,int]]] = []
    for dt in cands:
        item = np.dtype(dt).itemsize
        if fsize % item: continue
        if planes:
            Zc = (fsize//item)//planes
            if Zc>0 and _dtype_ok_quick(p,dt,fsize): out.append((np.dtype(dt),(int(Zc),int(hint[1]),int(hint[2])))); continue
        if Y>0 and X>0:
            pl = Y*X
            if pl>0 and (fsize//item)>=pl and ((fsize//item)%pl)==0:
                Zc = (fsize//item)//pl
                if Zc>0 and _dtype_ok_quick(p,dt,fsize): out.append((np.dtype(dt),(int(Zc),int(Y),int(X)))); continue
        vox = fsize//item; n = int(round(vox**(1/3))) if vox>0 else 0
        if n>0 and n*n*n==vox and _dtype_ok_quick(p,dt,fsize): out.append((np.dtype(dt),(n,n,n)))

    if not out: return None
    order = {np.dtype(t):i for i,t in enumerate([np.uint8,np.uint16,np.int16,np.uint32,np.int32,np.float32,np.float64,np.int8])}
    out.sort(key=lambda t: order.get(np.dtype(t[0]),999))
    dt, shp = out[0]

    if Z>0 and Y>0 and X>0:
        item = np.dtype(dt).itemsize; pl = Y*X
        if pl>0 and (fsize//item)>=pl:
            Zc = (fsize//item)//pl
            if Zc != Z: show_warning(f"{p.name}: adjusted Z {Z}→{Zc} to match bytes.")
            return dt,(int(Zc),int(Y),int(X))
    return dt, shp

# ---------------- crop & contrast --------------------------------------------
def _sampled_percentiles(a, lo=0.5, hi=99.5):
    try:
        a = np.asarray(a);  n = min(a.size, 256_000)
        idx = np.linspace(0, a.size-1, n, dtype=np.int64)
        samp = a.ravel()[idx]
        lo_v = float(np.nanpercentile(samp, lo)); hi_v = float(np.nanpercentile(samp, hi))
        if not np.isfinite(lo_v) or not np.isfinite(hi_v) or hi_v <= lo_v:
            lo_v, hi_v = float(np.nanmin(samp)), float(np.nanmax(samp))
            if not np.isfinite(lo_v) or not np.isfinite(hi_v) or hi_v <= lo_v: hi_v = lo_v + 1.0
        return lo_v, hi_v
    except Exception:
        return 0.0, 1.0

def _safe_contrast(layer: NapariImage):
    try:
        if hasattr(layer, "reset_contrast_limits"): layer.reset_contrast_limits()
        else: layer.contrast_limits = _sampled_percentiles(layer.data)
        layer.visible = True; layer.opacity = 1.0
        try: layer.colormap = "gray"
        except Exception: pass
    except Exception: pass

def _backup_orig(layer: NapariImage):
    md = layer.metadata
    if "_orig_data" not in md:
        md["_orig_data"] = layer.data
        shp = tuple(int(s) for s in np.asarray(layer.data.shape))
        md["_orig_shape_zyx"]  = shp
        md["_orig_center_zyx"] = tuple((np.asarray(shp)//2).tolist())
        md["_crop_start_zyx"]  = (0,0,0)

def _apply_center_crop(layer: NapariImage, target_zyx: Tuple[int,int,int]):
    _backup_orig(layer)
    src = layer.metadata["_orig_data"]
    SZ,SY,SX = map(int, np.asarray(src.shape)); TZ,TY,TX = map(int, target_zyx)
    if not (TZ<=SZ and TY<=SY and TX<=SX): return
    zs,ys,xs = max((SZ-TZ)//2,0), max((SY-TY)//2,0), max((SX-TX)//2,0)
    layer.metadata["_crop_start_zyx"] = (int(zs),int(ys),int(xs))
    layer.data = src[zs:zs+TZ, ys:ys+TY, xs:xs+TX]
    _safe_contrast(layer)

def _smallest_shape(imgs: List[NapariImage]) -> Optional[Tuple[int, int, int]]:
    if not imgs:
        return None
    shapes3d: List[Tuple[int,int,int]] = []
    for L in imgs:
        shp = tuple(int(s) for s in np.asarray(L.data).shape)
        if len(shp) < 3:
            continue
        z, y, x = shp[-3:]
        shapes3d.append((z, y, x))
    if not shapes3d:
        return None
    TZ = min(z for (z, _, _) in shapes3d)
    TY = min(y for (_, y, _) in shapes3d)
    TX = min(x for (_, _, x) in shapes3d)
    return (TZ, TY, TX)

# ---------------- centers (points) -------------------------------------------
def _ensure_center(v, im: NapariImage) -> NapariPoints:
    _backup_orig(im)
    oz,oy,ox = im.metadata.get("_orig_center_zyx", (np.asarray(im.data)//2))
    zs,ys,xs = im.metadata.get("_crop_start_zyx", (0,0,0))
    rz = int(np.clip(int(oz)-int(zs), 0, max(int(im.data.shape[0])-1,0)))
    ry = int(np.clip(int(oy)-int(ys), 0, max(int(im.data.shape[1])-1,0)))
    rx = int(np.clip(int(ox)-int(xs), 0, max(int(im.data.shape[2])-1,0)))

    try: curZ = int(v.dims.current_step[0])
    except Exception: curZ = rz

    name = _center_name(im)
    data = np.asarray([(float(curZ), float(ry), float(rx))], dtype=float)
    pts = next((L for L in v.layers if isinstance(L, NapariPoints) and L.name == name), None)
    if pts is None:
        pts = v.add_points(data, name=name, size=10, face_color="red")
        for a,val in (("edge_color","white"),("edge_width",0.5),("blending","translucent")):
            try: setattr(pts,a,val)
            except Exception: pass
        for p in ("out_of_slice_display","n_dimensional"):
            try: setattr(pts,p,False)
            except Exception: pass
    else:
        pts.data = data
    return pts

def _update_dot_plane(v):
    try: curZ = float(v.dims.current_step[0])
    except Exception: return
    for pts in _centers(v):
        d = np.asarray(pts.data, dtype=float)
        if d.ndim==2 and d.shape[0]>=1 and d.shape[1]>=3:
            d[:,0] = curZ; pts.data = d

def _remove_all_centers(v):
    for L in list(v.layers):
        if isinstance(L, NapariPoints) and L.name.startswith(CENTER_TAG):
            try: v.layers.remove(L)
            except Exception: pass

def _force_pair_order(v):
    imgs = _images(v)
    for im in imgs:
        name = _center_name(im)
        pt = next((p for p in _centers(v) if p.name == name), None)
        if pt is None: continue
        try:
            ii, ip = v.layers.index(im), v.layers.index(pt)
            if ip != ii + 1: v.layers.move(ip, ii + 1)
        except Exception: pass

# ---------------- grid logic --------------------------------------------------
_LAST_GRID = None
def _grid_shape(n: int) -> Tuple[int,int]:
    if n <= 1: return (1,1)
    r = int(math.ceil(math.sqrt(n)))
    c = int(math.ceil(n / r))
    return (r,c)

def _apply_grid(v):
    global _LAST_GRID
    if not v or v.dims.ndisplay != 2:
        return

    all_imgs = [L for L in (v.layers or []) if isinstance(L, NapariImage)]
    n = len(all_imgs)

    v.grid.enabled = (n >= 2)
    v.grid.shape   = _grid_shape(n)

    imgs = _images(v)
    cents = _centers(v)
    all_paired = (
        len(cents) == len(imgs)
        and all(
            next((p for p in cents if p.name == _center_name(im)), None) is not None
            for im in imgs
        )
    )
    try:
        v.grid.stride = 2 if CENTERS_ENABLED and all_paired else 1
    except Exception:
        pass

    new_state = (n, v.grid.shape, getattr(v.grid, "stride", 1))
    if new_state != _LAST_GRID:
        try:
            v.reset_view()
        except Exception:
            pass
        _LAST_GRID = new_state

# ---------------- bulk guard --------------------------------------------------
_IN_SYNC = False
@contextmanager
def _bulk(v):
    global _IN_SYNC; _IN_SYNC = True
    try: yield
    finally: _IN_SYNC = False

# ---------------- core sync ---------------------------------------------------
def _auto_match_and_fix(v):
    imgs = _images(v)
    tgt = _smallest_shape(imgs)
    if not tgt:
        return
    TZ, TY, TX = map(int, tgt)

    for L in imgs:
        md = getattr(L, "metadata", {}) or {}
        if md.get("_slice_compare"):
            continue

        cur = tuple(int(s) for s in np.asarray(L.data.shape))
        if cur != (TZ, TY, TX):
            _apply_center_crop(L, (TZ, TY, TX))
        else:
            _safe_contrast(L)

    try:
        z = int(v.dims.current_step[0])
        if z >= TZ:
            v.dims.set_current_step(0, TZ - 1)
    except Exception:
        pass

def _rebuild_centers(v):
    imgs = [
        im
        for im in _images(v)
        if not (getattr(im, "metadata", {}) or {}).get("_slice_compare")
    ]
    if not imgs:
        _remove_all_centers(v)
        return

    for im in imgs:
        _ensure_center(v, im)

    names = {_center_name(im) for im in imgs}
    for L in list(v.layers):
        if isinstance(L, NapariPoints) and L.name.startswith(CENTER_TAG) and L.name not in names:
            try:
                v.layers.remove(L)
            except Exception:
                pass
    _force_pair_order(v)

def _sync(centers_on: bool):
    v = current_viewer()
    if not v or _IN_SYNC: return
    with _bulk(v):
        _auto_match_and_fix(v)
        if centers_on: _rebuild_centers(v); _update_dot_plane(v)
        else: _remove_all_centers(v)
        _force_pair_order(v)
        _apply_grid(v)
        try: refresh_titles(True)
        except Exception: pass
        try: refresh_all_captions("bottom")
        except Exception: pass

# ---------------- loader / UI -------------------------------------------------
def _add_image_from_array(arr: np.ndarray, path: str, dtype_name: str):
    v = current_viewer()
    if not v: return None
    shp = arr.shape
    name = f"{pathlib.Path(path).stem} [{shp[0]}×{shp[1]}×{shp[2]} {dtype_name}]"
    L = v.add_image(arr, name=name)
    _backup_orig(L); _safe_contrast(L)
    v.layers.selection.active = L
    if CENTERS_ENABLED: _ensure_center(v, L)
    return L

def _load_one(path: str, dtype_name: str, Z:int,Y:int,X:int, prefer_memmap: bool):
    dt_shp = _infer_shape_dtype(path, dtype_name, Z, Y, X)
    if not dt_shp: show_warning(f"{pathlib.Path(path).name}: could not infer shape/dtype."); return None
    dt, shp = dt_shp
    arr = _read_array(path, dt, shp, prefer_memmap)
    if arr is None: return None
    return _add_image_from_array(arr, path, dtype_name if dtype_name!="auto" else dt.name)

class _LoadState:
    def __init__(self):
        self.files: List[str] = []; self.folder_files: List[str] = []
    @property
    def all(self): return [*self.files, *self.folder_files]

class _LoadUI:
    def __init__(self, settings: AppSettings):
        self.s = settings; self.state = _LoadState()
        self.btn_add_files  = PushButton(text="Add files…")
        self.btn_add_folder = PushButton(text="Add folder…")
        self.show_titles    = CheckBox(text="show titles (2D)", value=True)

        self.dtype   = ComboBox(choices=_DTYPE_CHOICES, value=settings.default_dtype or "auto")
        self.memmap  = CheckBox(text="memmap", value=bool(getattr(settings,"prefer_memmap",False)))
        self.use_hints    = CheckBox(text="use filename hints (Z×Y×X / N^3)", value=True)
        self.show_centers = CheckBox(text="show center dots", value=False)

        self.z = SpinBox(value=0,min=0,max=2_000_000); self.y = SpinBox(value=0,min=0,max=2_000_000); self.x = SpinBox(value=0,min=0,max=2_000_000)
        self.btn_load = PushButton(text="Load → layers")
        self.btn_del  = QPushButton("Delete selected"); self.btn_del_all = QPushButton("Delete ALL images")
        self.pick = QListWidget(); self.loaded = QListWidget()

        self.btn_add_files.changed.connect(lambda *_: self._pick_files())
        self.btn_add_folder.changed.connect(lambda *_: self._pick_folder())
        self.btn_load.changed.connect(self._on_load)
        self.show_centers.changed.connect(self._toggle_centers)
        self.btn_del.clicked.connect(self._on_delete_selected)
        self.btn_del_all.clicked.connect(self._on_delete_all)

        self.widget = self._build()

    def _build(self):
        for w in (self.dtype.native,self.z.native,self.y.native,self.x.native):
            w.setMinimumWidth(120)
        for w in (self.btn_add_files.native,self.btn_add_folder.native,self.btn_load.native):
            w.setMinimumHeight(32)

        root = QWidget(); v = QVBoxLayout(root); v.setContentsMargins(14,14,14,14); v.setSpacing(12)
        title = QLabel("Load RAW (auto-infer dtype & shape; supports 256×512×512 or 512^3 in filename)")
        title.setStyleSheet("font-weight:600; font-size:14px;"); v.addWidget(title)

        r0 = QHBoxLayout(); r0.addWidget(self.btn_add_files.native); r0.addWidget(self.btn_add_folder.native)
        r0.addWidget(self.show_titles.native); r0.addStretch(1); v.addLayout(r0)

        form = QFormLayout(); form.setHorizontalSpacing(14); form.setVerticalSpacing(8)
        form.addRow("dtype", self.dtype.native); form.addRow("", self.memmap.native)
        form.addRow("", self.use_hints.native); form.addRow("", self.show_centers.native)
        form.addRow("Z (0 = infer)", self.z.native); form.addRow("Y (0 = infer)", self.y.native); form.addRow("X (0 = infer)", self.x.native)
        v.addLayout(form)

        v.addWidget(QLabel("Picked files:")); v.addWidget(self.pick)
        r1 = QHBoxLayout(); r1.addStretch(1); r1.addWidget(self.btn_load.native); r1.addStretch(1); v.addLayout(r1)

        v.addWidget(QLabel("Loaded layers (images only):")); v.addWidget(self.loaded)
        r2 = QHBoxLayout(); r2.addWidget(self.btn_del); r2.addWidget(self.btn_del_all); r2.addStretch(1); v.addLayout(r2)

        tip = QLabel("Images auto-crop to the smallest Z×Y×X (view-only), contrast is safe, and Z is clamped.\n"
                     "Grid stays stable; each image pairs with its red center when dots are ON.")
        tip.setWordWrap(True); v.addWidget(tip)
        return root

    def _pick_files(self):
        start = self.s.last_dir if self.s.last_dir and pathlib.Path(self.s.last_dir).exists() else ""
        paths,_ = QFileDialog.getOpenFileNames(None,"Add RAW files",start,
                    "RAW (*.raw *.RAW *.bin *.BIN);;All files (*)")
        if paths:
            self.state.files = list(paths); self.pick.clear()
            for p in paths: QListWidgetItem(pathlib.Path(p).name, self.pick)
            _remember_dir(self.s, paths[0])

    def _pick_folder(self):
        start = self.s.last_dir if self.s.last_dir and pathlib.Path(self.s.last_dir).exists() else ""
        folder = QFileDialog.getExistingDirectory(None,"Pick folder with RAW",start)
        if not folder: return
        exts = {".raw",".RAW",".bin",".BIN"}
        files = [str(pathlib.Path(folder)/f) for f in sorted(os.listdir(folder)) if pathlib.Path(f).suffix in exts]
        if not files: show_warning("No RAW files in that folder."); return
        self.state.folder_files = files; self.pick.clear()
        for p in files: QListWidgetItem(pathlib.Path(p).name, self.pick)
        _remember_dir(self.s, files[0])

    def _refresh_loaded_list(self):
        v = current_viewer();  self.loaded.clear()
        if not v: return
        for L in _images(v): QListWidgetItem(L.name, self.loaded)

    def _on_load(self, *_):
        v = current_viewer()
        if not v: show_warning("Open napari viewer first."); return
        paths = self.state.all
        if not paths: show_warning("Pick files or a folder first."); return

        dtype = str(self.dtype.value); Z,Y,X = int(self.z.value), int(self.y.value), int(self.x.value)
        for p in paths:
            _ = _load_one(p, dtype if (self.use_hints.value or dtype!="auto") else "auto", Z, Y, X, bool(self.memmap.value))

        try: self.s.default_dtype = dtype; self.s.prefer_memmap = bool(self.memmap.value); self.s.save()
        except Exception: pass

        _sync(bool(self.show_centers.value))
        self._refresh_loaded_list()

    def _toggle_centers(self, *_):
        global CENTERS_ENABLED; CENTERS_ENABLED = bool(self.show_centers.value)
        _sync(CENTERS_ENABLED)

    def _on_delete_selected(self, *_):
        v = current_viewer();  items = self.loaded.selectedItems()
        if not v or not items: return
        with _bulk(v):
            for it in items:
                for L in list(v.layers):
                    if isinstance(L, NapariImage) and L.name == it.text():
                        try: v.layers.remove(L)
                        except Exception: pass
                for P in list(v.layers):
                    if isinstance(P, NapariPoints) and P.name == f"{CENTER_TAG}{it.text()}":
                        try: v.layers.remove(P)
                        except Exception: pass
        self._refresh_loaded_list(); _sync(bool(self.show_centers.value))

    def _on_delete_all(self, *_):
        v = current_viewer();  cnt = 0
        if not v: return
        with _bulk(v):
            for L in list(v.layers):
                if isinstance(L, NapariImage):
                    try: v.layers.remove(L); cnt += 1
                    except Exception: pass
            _remove_all_centers(v)
        self._refresh_loaded_list(); _sync(bool(self.show_centers.value))
        show_info(f"Deleted {cnt} image(s).")

# ---------------- entry -------------------------------------------------------
def raw_loader_widget() -> QWidget:
    s = AppSettings.load()
    try:
        if not getattr(s, "dock_max_width", None) or s.dock_max_width < _MIN_DOCK_WIDTH:
            s.dock_max_width = _MIN_DOCK_WIDTH
            s.save()
    except Exception:
        pass

    show_info(PLUGIN_BUILD)

    # 🔥 Start fresh recipe each time widget is created
    try:
        if reset_global_recorder is not None:
            reset_global_recorder()
    except Exception:
        pass

    tabs = QTabWidget()

    # --- Load tab (always built) ---
    ui = _LoadUI(s)
    load_tab = ui.widget
    load_tab_index = tabs.addTab(load_tab, "Raw Loader")

    # --- Crop tab (eager) ---
    crop_ctrl, crop_panel = make_crop_panel()
    crop_tab = _pad(crop_panel)
    crop_tab_index = tabs.addTab(crop_tab, "Crop")

    # --- Slice compare tab (eager) ---
    _, slice_panel = make_slice_compare_panel()
    slice_tab = _pad(slice_panel)
    slice_tab_index = tabs.addTab(slice_tab, "Slice compare")

    # --- 3D View tab (eager) ---
    from .view3d import _make_view3d_tab
    view3d_tab = _make_view3d_tab()
    view3d_tab_index = tabs.addTab(view3d_tab, "3D View")

    # ------------------------------------------------------------------
    # HEAVY TABS: Plot / Export / Functions / Brush / MONAI → LAZY BUILT
    # ------------------------------------------------------------------
    plot_tab = QWidget()
    plot_tab_layout = QVBoxLayout(plot_tab)
    plot_tab_layout.setContentsMargins(0, 0, 0, 0)
    plot_tab_layout.setSpacing(0)
    plot_tab_index = tabs.addTab(plot_tab, "Plotting")

    info_tab = QWidget()
    info_tab_layout = QVBoxLayout(info_tab)
    info_tab_layout.setContentsMargins(0, 0, 0, 0)
    info_tab_layout.setSpacing(0)
    info_tab_index = tabs.addTab(info_tab, "Export")

    fn_tab = QWidget()
    fn_tab_layout = QVBoxLayout(fn_tab)
    fn_tab_layout.setContentsMargins(0, 0, 0, 0)
    fn_tab_layout.setSpacing(0)
    fn_tab_index = tabs.addTab(fn_tab, "Functions")

    brush_tab = QWidget()
    brush_tab_layout = QVBoxLayout(brush_tab)
    brush_tab_layout.setContentsMargins(0, 0, 0, 0)
    brush_tab_layout.setSpacing(0)
    brush_tab_index = tabs.addTab(brush_tab, "Brushi")

    monai_tab = QWidget()
    monai_tab_layout = QVBoxLayout(monai_tab)
    monai_tab_layout.setContentsMargins(0, 0, 0, 0)
    monai_tab_layout.setSpacing(0)
    monai_tab_index = tabs.addTab(monai_tab, "MONAI Segment")

        # --- Session tab (always built, very light) ---
    try:
        session_panel = make_session_recorder_panel() if make_session_recorder_panel else QWidget()
    except Exception:
        session_panel = QWidget()
    session_tab = _pad(session_panel)
    session_tab_index = tabs.addTab(session_tab, "Session")

    plotlab = None
    info_widget = None
    info_refresh = lambda: None
    fn_page = None
    brush_page = None
    monai_page = None
    built_monai = False

    built_plot = False
    built_info = False
    built_fn = False
    built_brush = False

    outer = QWidget()
    pl = QVBoxLayout(outer)
    pl.setContentsMargins(0, 0, 0, 0)
    pl.setSpacing(0)
    pl.addWidget(tabs)
    wrapper = _wrap_scroll(outer)

    def _apply_cap_width():
        try:
            cap_width(
                getattr(s, "dock_max_width", _MIN_DOCK_WIDTH),
                info_widget=info_widget,
                plotlab_widget=plotlab,
                crop_panel=crop_panel,
                tabs=tabs,
                wrapper=wrapper,
            )
        except Exception:
            pass

    def _ensure_plotlab():
        nonlocal plotlab, built_plot
        if built_plot:
            return
        try:
            plotlab_local = PlotLab(s)
            w = _pad(plotlab_local.as_qwidget())
            plot_tab_layout.addWidget(w)
            plotlab = plotlab_local
            built_plot = True
            _apply_cap_width()
        except Exception as e:
            show_warning(f"Plot Lab failed to build: {e!r}")

    def _ensure_info():
        nonlocal info_widget, info_refresh, built_info
        if built_info:
            return
        try:
            info_w, info_ref = build_info_export_panel(s)
            info_widget_local = info_w
            info_refresh_local = info_ref
            info_tab_layout.addWidget(_pad(info_widget_local))
            info_widget = info_widget_local
            info_refresh = info_refresh_local
            built_info = True
            _apply_cap_width()
        except Exception as e:
            show_warning(f"Info / Export page failed to build: {e!r}")

    def _ensure_functions():
        nonlocal fn_page, built_fn
        if built_fn:
            return
        try:
            fn_pg = functions_widget()
            fn_qwidget = fn_pg.native if hasattr(fn_pg, "native") else fn_pg
            fn_tab_layout.addWidget(_pad(fn_qwidget))
            fn_page = fn_pg
            built_fn = True
        except Exception as e:
            show_warning(f"Functions page failed to build: {e!r}")

    def _ensure_brush():
        nonlocal brush_page, built_brush
        if built_brush:
            return
        try:
            brush_pg = brush_widget()
            brush_qwidget = brush_pg.native if hasattr(brush_pg, "native") else brush_pg
            brush_tab_layout.addWidget(_pad(brush_qwidget))
            brush_page = brush_pg
            built_brush = True
        except Exception as e:
            show_warning(f"Brush page failed to build: {e!r}")

    def _ensure_monai():
        nonlocal monai_page, built_monai
        if built_monai:
            return
        if make_monai_infer_widget is None:
            show_warning("MONAI tab not available (missing monai_infer_widget.py or deps).")
            built_monai = True
            return
        try:
            monai_pg = make_monai_infer_widget()
            monai_qwidget = monai_pg.native if hasattr(monai_pg, "native") else monai_pg
            monai_tab_layout.addWidget(_pad(monai_qwidget))
            monai_page = monai_pg
            built_monai = True
        except Exception as e:
            show_warning(f"MONAI page failed to build: {e!r}")

    _apply_cap_width()

    v = current_viewer()
    if v:
        wire_caption_events_once()

        def _on_layers(event=None, *_):
            if _IN_SYNC:
                return

            current = tabs.currentIndex()

            if current == load_tab_index:
                ui._refresh_loaded_list()

            if current == info_tab_index and built_info:
                try:
                    info_refresh()
                except Exception:
                    pass

        v.layers.events.inserted.connect(_on_layers)
        v.layers.events.removed.connect(_on_layers)
        v.layers.events.reordered.connect(_on_layers)

        # guard: crop_ctrl could be None if crop import failed
        try:
            if crop_ctrl is not None:
                v.layers.selection.events.active.connect(crop_ctrl.on_layers_changed)
        except Exception:
            pass

        v.layers.selection.events.active.connect(
            lambda *_: refresh_all_captions("bottom")
        )

        set_active_tab("load")

        def _on_tab_changed(idx: int):
            if idx == load_tab_index:
                set_active_tab("load")
            elif idx == session_tab_index:
                set_active_tab("session")
            elif idx == plot_tab_index:
                set_active_tab("plot")
            elif idx == info_tab_index:
                set_active_tab("info")
            elif idx == fn_tab_index:
                set_active_tab("functions")
            elif idx == brush_tab_index:
                set_active_tab("brush")
            elif idx == crop_tab_index:
                set_active_tab("crop")
            elif idx == slice_tab_index:
                set_active_tab("slice")
            elif idx == view3d_tab_index:
                set_active_tab("view3d")
            elif idx == monai_tab_index:
                set_active_tab("monai")
            else:
                set_active_tab(f"tab_{idx}")

            if idx == plot_tab_index:
                _ensure_plotlab()
            elif idx == info_tab_index:
                _ensure_info()
            elif idx == fn_tab_index:
                _ensure_functions()
            elif idx == brush_tab_index:
                _ensure_brush()
            elif idx == monai_tab_index:
                _ensure_monai()

            _on_layers()

        tabs.currentChanged.connect(_on_tab_changed)
        _on_layers()

    return wrapper
