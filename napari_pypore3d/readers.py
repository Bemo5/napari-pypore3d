# Reader factory for npe2. Supports drag&drop of *.raw / *.bin files whose
# filename contains ZxYxX or Z_Y_X shape, e.g., sample_700x700x700.raw

import os
import re
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np

LayerData = Tuple[Any, dict, str]  # (data, meta, layer_type)


def _parse_shape_from_name(name: str):
    m = re.search(r"(\d+)[x_](\d+)[x_](\d+)", name)
    if not m:
        return None
    # return z, y, x
    return tuple(map(int, m.groups()))


def napari_get_reader(path) -> Callable | None:
    """Return a reader callable if we can read the given path, else None."""
    p = path[0] if isinstance(path, (list, tuple)) else path
    if not str(p).lower().endswith((".raw", ".bin")):
        return None

    def _reader(_path) -> List[LayerData]:
        q = _path[0] if isinstance(_path, (list, tuple)) else _path
        q = str(q)
        fname = os.path.basename(q)
        shape = _parse_shape_from_name(fname)
        if shape is None:
            raise ValueError(
                "Cannot infer shape from filename. "
                "Use a name like '..._700x700x700.raw' or load via the plugin panel."
            )

        z, y, x = shape
        nvox = z * y * x
        fsize = os.path.getsize(q)

        if fsize == nvox:
            dtype = np.uint8
        elif fsize == nvox * 2:
            dtype = np.uint16
        elif fsize == nvox * 4:
            dtype = np.float32
        else:
            raise ValueError(
                f"File size {fsize} B doesn't match shape {shape} with "
                "uint8/uint16/float32."
            )

        vol = np.memmap(q, dtype=dtype, mode="r", shape=(z, y, x))
        meta = {"name": Path(q).stem}
        return [(vol, meta, "image")]

    return _reader
