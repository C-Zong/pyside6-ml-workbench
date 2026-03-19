from __future__ import annotations

import sys
from pathlib import Path


def resource_path(relative_path: str) -> str:
    """Resolve bundled-resource paths for both source and PyInstaller runs."""
    if hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS) / relative_path)
    return str(Path(__file__).resolve().parent.parent / relative_path)
