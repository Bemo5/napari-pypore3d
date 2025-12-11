# napari_pypore3d/state.py
from __future__ import annotations

_ACTIVE_TAB = "load"   # or whatever you want as default


def set_active_tab(name: str) -> None:
    global _ACTIVE_TAB
    _ACTIVE_TAB = str(name)


def active_tab() -> str:
    return _ACTIVE_TAB


def is_tab_active(name: str) -> bool:
    return _ACTIVE_TAB == name
