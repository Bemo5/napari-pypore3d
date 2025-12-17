import os, tempfile
import numpy as np

# popup
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

RAW_PATH = r"C:\Users\ABC\Desktop\Thesis\napari-pypore3d\napari_pypore3d\data\SC1_700x700x700.raw"
Z, Y, X = 700, 700, 700
DTYPE = np.uint8

CROP = 128
z0, y0, x0 = (Z - CROP)//2, (Y - CROP)//2, (X - CROP)//2


def show_popup(before2d: np.ndarray, after2d: np.ndarray, title: str, note: str = ""):
    root = tk.Tk()
    root.title(title)

    fig = Figure(figsize=(9, 4.5), dpi=110)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.imshow(before2d, cmap="gray")
    ax1.set_title("Before")
    ax1.axis("off")

    ax2.imshow(after2d, cmap="gray")
    ax2.set_title("After (Otsu mask)")
    ax2.axis("off")

    if note:
        fig.suptitle(note, fontsize=10)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    root.mainloop()


def find_best_otsu_for_uint8():
    """
    Find PyPore3D otsu function, preferring 8-bit versions and avoiding _16.
    """
    import importlib

    # Otsu is likely in p3dFiltPy
    mods = ["pypore3d.p3dFiltPy", "pypore3d.p3dSITKPy", "pypore3d.p3dBlobPy"]
    candidates = []

    for m in mods:
        try:
            mod = importlib.import_module(m)
        except Exception:
            continue
        for name in dir(mod):
            if "otsu" in name.lower():
                fn = getattr(mod, name)
                if callable(fn):
                    candidates.append((m, name, fn))

    if not candidates:
        raise RuntimeError("No Otsu-like function found in PyPore3D modules.")

    # Prefer _8, then names containing '8', then anything NOT containing _16
    def score(item):
        m, name, _ = item
        n = name.lower()
        s = 0
        if "_8" in n or n.endswith("8"):
            s += 100
        if "8" in n:
            s += 10
        if "_16" in n or n.endswith("16"):
            s -= 1000
        return s

    candidates.sort(key=score, reverse=True)

    # Print what we found (useful debug)
    print("Found Otsu candidates:")
    for m, name, _ in candidates[:10]:
        print(" -", f"{m}.{name}")

    best = candidates[0]
    print("✅ Using:", f"{best[0]}.{best[1]}")
    return best[2], f"{best[0]}.{best[1]}"


def call_otsu(fn, raw_obj, x: int, y: int, z: int):
    """
    Try a few likely SWIG signatures.
    IMPORTANT: your error showed no keyword 'dimz', so we use positional z.
    """
    # Try simplest first: (raw, x, y, z)
    tries = [
        ("(raw,x,y,z)", (raw_obj, x, y, z)),
        # Some wrappers want extra numeric params:
        ("(raw,x,y,z,0,0)", (raw_obj, x, y, z, 0, 0)),
        ("(raw,x,y,z,0,0,0)", (raw_obj, x, y, z, 0, 0, 0)),
        # Some wrappers want log/progress placeholders at end:
        ("(raw,x,y,z,0,0,'','')", (raw_obj, x, y, z, 0, 0, "", "")),
        ("(raw,x,y,z,0,0,[],[])", (raw_obj, x, y, z, 0, 0, [], [])),
    ]

    last = None
    for label, args in tries:
        try:
            out = fn(*args)
            print("✅ Otsu succeeded with", label)
            return out, label
        except TypeError as e:
            print("⚠️ TypeError", label, "->", e)
            last = e

    raise TypeError(f"All Otsu call patterns failed. Last: {last}")


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(RAW_PATH)

    # Crop 128^3
    mm = np.memmap(RAW_PATH, dtype=DTYPE, mode="r", shape=(Z, Y, X))
    sub = np.array(mm[z0:z0+CROP, y0:y0+CROP, x0:x0+CROP], copy=True)
    sub = np.ascontiguousarray(sub.astype(np.uint8, copy=False))
    z, y, x = sub.shape
    zmid = z // 2

    print("Subvolume:", sub.shape, "min/max:", int(sub.min()), int(sub.max()))

    # PyPore3D read RAW into native SWIG object
    from pypore3d.p3dSITKPy import py_p3dReadRaw8
    try:
        from pypore3d.p3dSITKPy import py_p3dWriteRaw8
        HAVE_WRITE = True
    except Exception:
        HAVE_WRITE = False

    with tempfile.TemporaryDirectory() as tmp:
        sub_path = os.path.join(tmp, f"sub_{z}x{y}x{x}.raw")
        sub.tofile(sub_path)

        raw_obj = py_p3dReadRaw8(sub_path, x, y, dimz=z)
        print("✅ Read sub RAW OK")

        otsu_fn, otsu_name = find_best_otsu_for_uint8()
        out, used = call_otsu(otsu_fn, raw_obj, x, y, z)

        # Convert output to a NumPy mask for display
        if isinstance(out, (int, float, np.integer, np.floating)):
            thr = float(out)
            mask = ((sub > thr) * 255).astype(np.uint8)
            note = f"{otsu_name} returned threshold={thr:.3f} using {used}"
        else:
            # SWIG object mask/image: write + read back if possible
            if not HAVE_WRITE:
                raise RuntimeError(
                    f"{otsu_name} returned a SWIG object, but py_p3dWriteRaw8 isn't available, "
                    "so we can't display it as NumPy."
                )
            out_path = os.path.join(tmp, f"otsu_{z}x{y}x{x}.raw")
            py_p3dWriteRaw8(out, out_path, x, y, dimz=z)

            mask_mm = np.memmap(out_path, dtype=np.uint8, mode="r", shape=(z, y, x))
            mask = np.array(mask_mm, copy=True)
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            note = f"{otsu_name} returned mask object (written+read) using {used}"

    show_popup(sub[zmid], mask[zmid], "PyPore3D Otsu (cropped)", note)


if __name__ == "__main__":
    main()
