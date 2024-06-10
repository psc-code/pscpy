"""
Copyright (c) 2024 Kai Germaschewski. All rights reserved.

pscpy: Python utilities for reading PSC data
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = ["__version__"]


import numpy as np
import pscpy.adios2py
import os
import xarray as xr
from collections import namedtuple

from . import pscadios2
from . import pschdf5
