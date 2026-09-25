from __future__ import annotations

import pathlib

import pytest

import pscpy
from pscpy.psc import SUPPORTED_MAJOR_VERSION


@pytest.fixture
def latest_sample_dir() -> pathlib.Path:
    """Sample psc output in the psc_output_version that pscpy supports."""
    return pscpy.sample_dir / f"v{SUPPORTED_MAJOR_VERSION}"
