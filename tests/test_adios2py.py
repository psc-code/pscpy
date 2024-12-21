from __future__ import annotations

import numpy as np

import pscpy
from pscpy import adios2py


def test_open_close():
    file = adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")
    file.close()


def test_open_twice():
    file1 = adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")  # noqa: F841
    file2 = adios2py.File(pscpy.sample_dir / "pfd.000000400.bp")  # noqa: F841


def test_with():
    with adios2py.File(pscpy.sample_dir / "pfd.000000400.bp"):
        pass


def test_variable_names():
    with adios2py.File(pscpy.sample_dir / "pfd.000000400.bp") as file:
        assert file.variable_names == set({"jeh"})
        assert file.attribute_names == set({"ib", "im", "step", "time"})


def test_get_variable():
    with adios2py.File(pscpy.sample_dir / "pfd.000000400.bp") as file:
        var = file.get_variable("jeh")
        assert var.name == "jeh"
        assert var.shape == (1, 128, 512, 9)
        assert var.dtype == np.float32


def test_get_attribute():
    with adios2py.File(pscpy.sample_dir / "pfd.000000400.bp") as file:
        assert all(file.get_attribute("ib") == (0, 0, 0))
        assert all(file.get_attribute("im") == (1, 128, 128))
        assert np.isclose(file.get_attribute("time"), 109.38)
        assert file.get_attribute("step") == 400
