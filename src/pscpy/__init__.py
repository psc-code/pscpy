"""
Copyright (c) 2024 Kai Germaschewski. All rights reserved.

pscpy: Python utilities for reading PSC data
"""

from __future__ import annotations

import pathlib

from ._version import version as __version__
from .psc import decode_psc

sample_dir = pathlib.Path(__file__).parent / "sample"


__all__ = [
    "__version__",
    "decode_psc",
    "sample_dir",
]
