"""
Copyright (c) 2024 Kai Germaschewski. All rights reserved.

pscpy: Python utilities for reading PSC data
"""

from __future__ import annotations

import pathlib

from pscpy import pscadios2  # noqa: F401

from ._version import version as __version__

sample_dir = pathlib.Path(__file__).parent / "sample"


__all__ = ["__version__"]
