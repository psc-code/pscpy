"""
Copyright (c) 2024 Kai Germaschewski. All rights reserved.

pscpy: Python utilities for reading PSC data
"""

from __future__ import annotations

import pathlib

from ._version import version as __version__

sample_dir = pathlib.Path(__file__).parent / "sample"


__all__ = ["__version__"]
