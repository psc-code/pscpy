from __future__ import annotations

import pscpy
from pscpy import adios2py


def test_adios2py_open():
    adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")
