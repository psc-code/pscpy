from __future__ import annotations

import importlib.metadata

import pscpy as m


def test_version():
    assert importlib.metadata.version("pscpy") == m.__version__
