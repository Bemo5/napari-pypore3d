import importlib, inspect, pkgutil, pathlib, sys

def list_public(modname):
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        return modname, f"(not found: {e})", []
    res = []
    for k, obj in vars(mod).items():
        if k.startswith("_"):
            continue
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(...)"
            res.append((k, sig))
    return modname, None, sorted(res)

# where is pypore3d?
import pypore3d
pkg_dir = pathlib.Path(pypore3d.__file__).parent
print("pypore3d at:", pkg_dir)

# candidates: common pypore3d modules + any *.py sitting next to it in site-packages
candidates = {
    "pypore3d","pypore3d.filters","p3dFiltPy","p3dFilt","p3dBlobPy","p3dBlob",
    "p3dSkelPy","p3dSkel","p3dSITKPy","p3dSITKPy_16","p3d_common_lib",
    "p3d_SITK_common_lib","p3d_SITK_common_lib_16","p3dSITK_read_raw",
}

for py in pkg_dir.parent.glob("*.py"):
    candidates.add(py.stem)

found = False
for name in sorted(candidates):
    modname, err, funcs = list_public(name)
    if err:
        print(f"{modname} -> {err}")
        continue
    if funcs:
        found = True
        print(f"\n[{modname}]")
        for fname, sig in funcs:
            print(f" - {modname}.{fname}{sig}")

if not found:
    print("No public functions found.")
