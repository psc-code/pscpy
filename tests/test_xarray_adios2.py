from __future__ import annotations

import os
from typing import Any

import adios2py
import numpy as np
import pytest
import xarray as xr
from xarray_adios2 import Adios2Store

import pscpy


@pytest.fixture
def test_filename(tmp_path):
    filename = tmp_path / "test_file.bp"
    with adios2py.File(filename, mode="w") as file:
        file.attrs["step_dimension"] = "step"
        for n, step in zip(range(5), file.steps, strict=False):
            step["scalar"] = n
            step["arr1d"] = np.arange(10)
            step["arr1d"].attrs["dimensions"] = "x"

    return filename


@pytest.fixture
def test_filename_2(tmp_path):
    filename = tmp_path / "test_file_2.bp"
    with adios2py.File(filename, mode="w") as file:
        file.attrs["step_dimension"] = "time"
        for n, step in zip(range(5), file.steps, strict=False):
            step["step"] = n
            step["time"] = 10.0 * n

            step["x"] = np.linspace(0, 1, 10)
            step["x"].attrs["dimensions"] = "x"

            step["arr1d"] = np.arange(10)
            step["arr1d"].attrs["dimensions"] = "x"

    return filename


@pytest.fixture
def test_filename_3(tmp_path):
    filename = tmp_path / "test_file_3.bp"
    with adios2py.File(filename, mode="w") as file:
        file.attrs["step_dimension"] = "time"
        for n, step in zip(range(5), file.steps, strict=False):
            step["step"] = n
            # step["step"].attrs["dimensions"] = "step"

            step["time"] = 100.0 + 10 * n
            step["time"].attrs["units"] = "second since 2020-01-01"

    return filename


@pytest.fixture
def test_filename_4(tmp_path):
    filename = tmp_path / "test_file_4.bp"
    with adios2py.File(filename, mode="w") as file:
        file.attrs["step_dimension"] = "time"
        for n, step in zip(range(5), file.steps, strict=False):
            step["time"] = n
            step["time"].attrs["units"] = "seconds since 1970-01-01"

    return filename


def _open_dataset(filename: os.PathLike[Any]) -> xr.Dataset:
    ds = xr.open_dataset(filename)
    return pscpy.decode_psc(
        ds,
        species_names=["e", "i"],
        length=[1, 12.8, 51.2],
        corner=[0, -6.4, -25.6],
    )


def test_open_dataset():
    ds = _open_dataset(pscpy.sample_dir / "pfd.000000400.bp")
    assert "jx_ec" in ds
    assert ds.coords.keys() == set({"x", "y", "z"})
    assert ds.jx_ec.sizes == dict(x=1, y=128, z=512)  # noqa: C408
    assert np.allclose(ds.jx_ec.z, np.linspace(-25.6, 25.6, 512, endpoint=False))


def test_component():
    ds = _open_dataset(pscpy.sample_dir / "pfd.000000400.bp")
    assert ds.jeh.sizes == dict(x=1, y=128, z=512, comp_jeh=9)  # noqa: C408
    assert np.all(ds.jeh.isel(comp_jeh=0) == ds.jx_ec)


def test_selection():
    ds = _open_dataset(pscpy.sample_dir / "pfd.000000400.bp")
    assert ds.jeh.sizes == dict(x=1, y=128, z=512, comp_jeh=9)  # noqa: C408
    assert np.all(
        ds.jeh.isel(comp_jeh=0, y=slice(0, 10), z=slice(0, 40))
        == ds.jx_ec.isel(y=slice(0, 10), z=slice(0, 40))
    )


def test_computed():
    ds = _open_dataset(pscpy.sample_dir / "pfd.000000400.bp")
    ds = ds.assign(jx=ds.jeh.isel(comp_jeh=0))
    assert np.all(ds.jx == ds.jx_ec)


def test_computed_via_lambda():
    ds = _open_dataset(pscpy.sample_dir / "pfd.000000400.bp")
    ds = ds.assign(jx=lambda ds: ds.jeh.isel(comp_jeh=0))
    assert np.all(ds.jx == ds.jx_ec)


def test_pfd_moments():
    ds = _open_dataset(pscpy.sample_dir / "pfd_moments.000000400.bp")
    assert "all_1st" in ds
    assert ds.all_1st.sizes == dict(x=1, y=128, z=512, comp_all_1st=26)  # noqa: C408
    assert "rho_i" in ds
    assert np.all(ds.rho_i == ds.all_1st.isel(comp_all_1st=13))


def test_open_dataset_steps(test_filename):
    ds = xr.open_dataset(test_filename)
    assert ds.keys() == set({"scalar", "arr1d"})


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_open_dataset_steps_from_Step(test_filename, mode):
    with adios2py.File(test_filename, mode) as file:
        for n, step in enumerate(file.steps):
            store = Adios2Store(step)
            ds = xr.open_dataset(store)
            assert ds.keys() == set({"scalar", "arr1d"})
            assert ds["scalar"] == n


def test_open_dataset_2(test_filename_2):
    ds = xr.open_dataset(test_filename_2)
    assert ds.keys() == set({"step", "arr1d"})
    assert ds.step.shape == (5,)
    assert ds.arr1d.shape == (5, 10)
    assert ds.coords.keys() == set({"time", "x"})
    assert ds.time.shape == (5,)


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_open_dataset_2_step(test_filename_2, mode):
    with adios2py.File(test_filename_2, mode=mode) as file:
        for _, step in enumerate(file.steps):
            ds = xr.open_dataset(Adios2Store(step))
            assert ds.keys() == set({"step", "time", "arr1d"})
            assert ds.coords.keys() == set({"x"})


def test_open_dataset_3(test_filename_3):
    ds = xr.open_dataset(test_filename_3)
    assert ds.time.shape == (5,)
    assert ds.time[0] == np.datetime64("2020-01-01T00:01:40")
    assert ds.time[1] == np.datetime64("2020-01-01T00:01:50")


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_open_dataset_3_step(test_filename_3, mode):
    with adios2py.File(test_filename_3, mode=mode) as file:
        for n, step in enumerate(file.steps):
            ds = xr.open_dataset(Adios2Store(step))
            assert ds.time == np.datetime64("2020-01-01T00:01:40") + np.timedelta64(
                10 * n, "s"
            )


def test_open_dataset_4(test_filename_4):
    ds = xr.open_dataset(test_filename_4)
    assert ds.time[0] == np.datetime64("1970-01-01T00:00:00.000")
    assert ds.time[1] == np.datetime64("1970-01-01T00:00:01.000")
