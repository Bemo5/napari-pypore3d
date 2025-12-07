import ctypes
import numpy as np
from pypore3d import _p3dFilt  # low-level C extension

def median_filter_uint8(vol: np.ndarray, width: int = 3) -> np.ndarray:
    # Ensure uint8 + contiguous
    vol = np.ascontiguousarray(vol, dtype=np.uint8)

    # PyPore3D expects dims in (x, y, z)
    z, y, x = vol.shape

    out = np.empty_like(vol)

    # Build C pointers for SWIG
    ptr_in = vol.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    ptr_out = out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    # Call the real C function:
    # int p3dMedianFilter3D_8(unsigned char *in,
    #                         unsigned char *out,
    #                         int dimx, int dimy, int dimz,
    #                         int width,
    #                         void *wr_log, void *wr_progress);
    err = _p3dFilt.p3dMedianFilter3D_8(
        ptr_in,
        ptr_out,
        x, y, z,
        width,
        None,  # wr_log
        None,  # wr_progress
    )

    if err != 0:
        raise RuntimeError(f"p3dMedianFilter3D_8 failed with code {err}")

    return out


if __name__ == "__main__":
    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    vol[10, 10, 10] = 255  # tiny test signal
    out = median_filter_uint8(vol, width=3)
    print("OK, result shape:", out.shape, "dtype:", out.dtype)
