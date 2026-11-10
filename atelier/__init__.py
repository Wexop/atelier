"""atelier — a procedural image studio driven by short Python scripts."""
import sys as _sys

from .core import *  # noqa: F401,F403
from .field import *  # noqa: F401,F403
from .vector import *  # noqa: F401,F403
from .fx import *  # noqa: F401,F403
from .brush import *  # noqa: F401,F403
from .sdf import *  # noqa: F401,F403
from .pt import *  # noqa: F401,F403
from .stylize import *  # noqa: F401,F403
from .terrain import *  # noqa: F401,F403
from .nature import *  # noqa: F401,F403
from .figure import *  # noqa: F401,F403
from .anime import *  # noqa: F401,F403
from .inspect import *  # noqa: F401,F403

__version__ = "0.4.0"
_MODULES = ("core", "field", "vector", "fx", "brush", "sdf", "pt", "stylize", "terrain", "nature", "figure", "anime", "inspect")
__all__ = list(dict.fromkeys(n for m in _MODULES for n in getattr(_sys.modules[f"{__name__}.{m}"], "__all__", [])))
