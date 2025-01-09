from __future__ import annotations

import adios2py
import numpy as np
import pytest

import pscpy
from pscpy.pscadios2 import Adios2Store


# FIXME, duplicated
@pytest.fixture
def test_filename(tmp_path):
    filename = tmp_path / "test_file.bp"
    with adios2py.File(filename, mode="w") as file:
        for n, step in zip(range(5), file.steps):
            step["scalar"] = n
            step["arr1d"] = np.arange(10)

    return filename


@pytest.fixture
def test_store(test_filename):
    return Adios2Store.open(test_filename, mode="r")


def test_open_close(test_store):
    test_store.close()


def test_open_with_parameters(test_store):
    filename = test_store.ds.filename
    test_store.close()

    params = {"OpenTimeoutSecs": "20"}
    with Adios2Store.open(filename, parameters=params) as store:
        assert store.ds.parameters == params


def test_open_with_engine():
    with Adios2Store.open(
        str(pscpy.sample_dir / "pfd.000000400.bp"), engine_type="BP4"
    ) as store:
        assert store.ds.engine_type == "BP4"


def test_vars_attrs(test_store):
    vars, attrs = test_store.load()
    assert vars == {}
    assert attrs == {}

    for step in test_store.ds:
        vars, attrs = step.load()
        assert vars.keys() == set({"scalar", "arr1d"})
        assert attrs == {}


def test_rra(test_filename):
    with Adios2Store.open(test_filename, mode="rra") as store:
        vars, _ = store.load()
        assert np.all(vars["scalar"] == np.arange(5))
