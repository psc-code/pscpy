from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import pscpy


@pytest.fixture
def test_dataarray():
    return xr.DataArray([4, 5, 6, 5], coords={"x": [0, 1, 2, 3]})


def test_recenter_periodic_left(test_dataarray):
    recentered = pscpy.get_recentered(test_dataarray, "x", -1)
    assert np.array_equal(recentered.coords, test_dataarray.coords)
    assert np.array_equal(recentered, [4.5, 4.5, 5.5, 5.5])


def test_recenter_periodic_right(test_dataarray):
    recentered = pscpy.get_recentered(test_dataarray, "x", 1)
    assert np.array_equal(recentered.coords, test_dataarray.coords)
    assert np.array_equal(recentered, [4.5, 5.5, 5.5, 4.5])


def test_recenter_pad_left(test_dataarray):
    recentered = pscpy.get_recentered(test_dataarray, "x", -1, boundary="pad")
    assert np.array_equal(recentered.coords, test_dataarray.coords)
    assert np.array_equal(recentered, [4, 4.5, 5.5, 5.5])


def test_recenter_pad_right(test_dataarray):
    recentered = pscpy.get_recentered(test_dataarray, "x", 1, boundary="pad")
    assert np.array_equal(recentered.coords, test_dataarray.coords)
    assert np.array_equal(recentered, [4.5, 5.5, 5.5, 5])


def test_recenter_zero_left(test_dataarray):
    recentered = pscpy.get_recentered(test_dataarray, "x", -1, boundary="zero")
    assert np.array_equal(recentered.coords, test_dataarray.coords)
    assert np.array_equal(recentered, [2, 4.5, 5.5, 5.5])


def test_recenter_zero_right(test_dataarray):
    recentered = pscpy.get_recentered(test_dataarray, "x", 1, boundary="zero")
    assert np.array_equal(recentered.coords, test_dataarray.coords)
    assert np.array_equal(recentered, [4.5, 5.5, 5.5, 2.5])


@pytest.fixture
def test_dataset_ec():
    coords = [[0, 1], [0, 1, 2]]
    dims = ["x", "y"]
    ex_ec = xr.DataArray([[0, 1, 2], [3, 4, 5]], coords, dims)
    ey_ec = xr.DataArray([[0, 2, 4], [1, 3, 5]], coords, dims)
    return xr.Dataset({"ex_ec": ex_ec, "ey_ec": ey_ec})


def test_autorecenter_ec_to_nc(test_dataset_ec):
    pscpy.auto_recenter(test_dataset_ec, "nc", x="pad", y="pad")
    assert np.array_equal(test_dataset_ec.ex_nc, [[0, 1, 2], [1.5, 2.5, 3.5]])
    assert np.array_equal(test_dataset_ec.ey_nc, [[0, 1, 3], [1, 2, 4]])


def test_autorecenter_ec_to_cc(test_dataset_ec):
    pscpy.auto_recenter(test_dataset_ec, "cc", x="pad", y="pad")
    assert np.array_equal(test_dataset_ec.ex_cc, [[0.5, 1.5, 2], [3.5, 4.5, 5]])
    assert np.array_equal(test_dataset_ec.ey_cc, [[0.5, 2.5, 4.5], [1, 3, 5]])
