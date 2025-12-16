# list_pypore3d_functions.py
import inspect
import pypore3d
import pypore3d._p3dBlob as B
import pypore3d._p3dSkel as S
import pypore3d._p3dFilt as F

modules = {
    "pypore3d (python)": pypore3d,
    "pypore3d._p3dBlob": B,
    "pypore3d._p3dSkel": S,
    "pypore3d._p3dFilt": F,
}

def is_useful_swig_name(name: str) -> bool:
    """Try to remove pure SWIG boilerplate (get/set/swig helpers)."""
    # ignore dunder
    if name.startswith("__") and name.endswith("__"):
        return False

    bad_suffixes = ("_get", "_set", "_swiginit", "_swigregister")
    bad_prefixes = ("new_", "delete_", "Swig", "SWIG")

    if name.endswith(bad_suffixes):
        return False
    if name.startswith(bad_prefixes):
        return False

    # you can add more prefixes if you see noisy stuff
    return True


def collect_functions(mod):
    # all callables (raw)
    raw = sorted(
        name for name in dir(mod)
        if callable(getattr(mod, name))
    )

    # filtered "more likely to be real algorithms"
    filtered = [
        name for name in raw
        if is_useful_swig_name(name)
    ]

    return raw, filtered


def main():
    for label, mod in modules.items():
        raw, filtered = collect_functions(mod)
        print("=" * 80)
        print(f"{label}")
        print(f"  Total callables: {len(raw)}")
        print(f"  Filtered 'useful' functions: {len(filtered)}")
        print("  Useful function names:")
        for name in filtered:
            print("   -", name)


if __name__ == "__main__":
    main()
