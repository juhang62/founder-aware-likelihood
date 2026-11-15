"""Amplitude-mixture pilots; see the no-death and birth-death specifications."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Keep generated caches separate from the completed no-death artifacts.
_cache_root = ROOT / ("data/founder-mixture-birth-death/cache" if any(
    "founder-mixture-birth-death" in arg for arg in sys.argv) else "data/founder-mixture-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache_root / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
__version__ = "0.1.0"
