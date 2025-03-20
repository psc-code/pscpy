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
