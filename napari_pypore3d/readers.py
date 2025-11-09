# napari_pypore3d/readers.py
# npe2-compatible RAW/BIN reader:
# - Works when filename contains ZxYxX or Z_Y_X (case-insensitive)
# - Infers dtype from file size against common types
# - Otherwise: raise a clear message to use the dock widget

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

# (data, meta) is sufficient for npe2 readers; layer_type is optional.
LayerData = Tuple[np.ndarray, dict]

_SHAPE_RE = re.compile(r"(\d+)[x_](\d+)[x_](\d+)", re.IGNORECASE)

def _parse_shape_from_name(name: str) -> Optional[Tuple[int, int, int]]:
    m = _SHAPE_RE.search(name)
    if not m:
        return None
    z, y, x = map(int, m.groups())
    return z, y, x

def _infer_dtype_from_size(file_size: int, zyx: Tuple[int, int, int]) -> Optional[np.dtype]:
    nvox = int(np.prod(zyx))
    if nvox <= 0:
        return None
    if file_size % nvox != 0:
        return None
    bpv = file_size // nvox
    mapping = {
        1: np.uint8,
        2: np.uint16,   # (could also be int16; we check more below)
        4: np.float32,  # (could also be uint32/int32)
        8: np.float64,
    }
    return mapping.get(int(bpv))

def read_raw_volume(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> Optional[List[LayerData]]:
    # Accept a single path or an iterable; enforce single-file behavior.
    if isinstance(paths, (list, tuple)):
        if len(paths) != 1:
            return None
        path = Path(paths[0])
    else:
        path = Path(paths)

    if not path.exists() or path.is_dir():
        return None
    if path.suffix.lower() not in {".raw", ".bin"}:
        return None

    fname = path.name
    shp = _parse_shape_from_name(fname)
    if shp is None:
        # We matched *.raw/*.bin by napari.yaml but cannot infer dimensions:
        # guiding message is more helpful than silently returning None here.
        raise ValueError(
            "Cannot infer shape from filename. "
            "Rename like '..._ZxYxX.raw' (e.g., _700x700x700.raw), "
            "or use the plugin panel: Plugins → napari-pypore3d → Load RAW…"
        )

    z, y, x = map(int, shp)
    fsize = path.stat().st_size
    nvox = z * y * x

    # Try common dtypes by exact size match (preference order).
    candidates = [np.uint8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.float64, np.int8]
    dtype: Optional[np.dtype] = None
    for dt in candidates:
        if fsize == nvox * np.dtype(dt).itemsize:
            dtype = np.dtype(dt)
            break

    if dtype is None:
        raise ValueError(
            f"File size {fsize} B doesn't match shape {shp} "
            "for u8/u16/i16/u32/i32/f32/f64/i8. "
            "Check shape in name or load via the dock to set dtype explicitly."
        )

    # memmap keeps RAM low and is fine with napari
    vol = np.memmap(path, dtype=dtype, mode="r", shape=(z, y, x), order="C")

    # Minimal, helpful metadata
    meta = {"name": f"{Path(fname).stem} {z}x{y}x{x}"}
    return [(np.asarray(vol), meta)]
