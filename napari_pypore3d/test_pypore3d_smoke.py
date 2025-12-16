import os, sys, tempfile
import numpy as np

PYPOR3D_ROOT = r"C:\Users\adam\Downloads\bachelor-1"
sys.path.append(PYPOR3D_ROOT)

RAW_PATH = r"C:\Users\ABC\Desktop\Thesis\napari-pypore3d\napari_pypore3d\data\SC1_700x700x700.raw"

Z, Y, X = 700, 700, 700
DTYPE = np.uint8

# ✅ choose a small cube
CROP = 128  # try 96/128/160 depending on speed
z0, y0, x0 = (Z - CROP)//2, (Y - CROP)//2, (X - CROP)//2  # centre crop


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(RAW_PATH)

    # memmap avoids loading full 700^3 into RAM
    mm = np.memmap(RAW_PATH, dtype=DTYPE, mode="r", shape=(Z, Y, X))

    sub = np.array(mm[z0:z0+CROP, y0:y0+CROP, x0:x0+CROP], copy=True)
    sub = np.ascontiguousarray(sub)  # important
    z, y, x = sub.shape

    print("Subvolume:", sub.shape, "min/max:", int(sub.min()), int(sub.max()))

    # write subvolume to temp RAW
    with tempfile.TemporaryDirectory() as tmp:
        sub_path = os.path.join(tmp, f"sub_{z}x{y}x{x}.raw")
        sub.tofile(sub_path)

        from pypore3d.p3dSITKPy import py_p3dReadRaw8
        from pypore3d.p3dSkelPy import py_p3dLKCSkeletonization

        raw_obj = py_p3dReadRaw8(sub_path, x, y, dimz=z)
        print("✅ Read sub RAW OK")

        skl_obj = py_p3dLKCSkeletonization(raw_obj, x, y, dimz=z)
        print("✅ Skeletonisation on subvolume OK:", type(skl_obj))

    print("DONE ✅")


if __name__ == "__main__":
    main()
