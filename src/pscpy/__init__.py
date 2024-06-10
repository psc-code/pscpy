"""
Copyright (c) 2024 Kai Germaschewski. All rights reserved.

pscpy: Python utilities for reading PSC data
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = ["__version__"]


import os
from collections import namedtuple

import numpy as np
import xarray as xr

import pscpy.adios2py

from . import pscadios2, pschdf5
