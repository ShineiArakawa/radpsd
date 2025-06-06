"""
radpsd
======

Radial PSD (Power Spectral Density) calculation library.

Contact
-------

- Author: Shinei Arakawa
- Email: sarakawalab@gmail.com
"""

# ----------------------------------------------------------------------------
# Check Python version

import sys

if sys.version_info < (3, 11):
    raise ImportError("Python 3.11 or higher is required.")

# ----------------------------------------------------------------------------
# Check the version of this package

import importlib.metadata

try:
    __version__ = importlib.metadata.version("radpsd")
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

# ----------------------------------------------------------------------------
# Import modules

from .signal import compute_psd, compute_radial_psd, radial_freq

__all__ = [
    "__version__",
    "compute_psd",
    "compute_radial_psd",
    "radial_freq",
]

# ----------------------------------------------------------------------------
