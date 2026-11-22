"""cspaps: heliostat field layout and aim-point strategy optimization.

Coordinate convention (used everywhere):
    x = East, y = North, z = Up, metres. Origin at the tower base.
    Azimuths are compass azimuths: 0 deg = North, 90 deg = East, clockwise.
"""

from .config import Case, load_case, case_from_dict

__version__ = "0.7.1"

__all__ = ["Case", "load_case", "case_from_dict", "__version__"]
