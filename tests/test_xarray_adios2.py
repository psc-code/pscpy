from __future__ import annotations

import os
from typing import Any

import adios2
import numpy as np
import pytest
import xarray as xr

import pscpy
from pscpy import adios2py
from pscpy.pscadios2 import Adios2Store


@pytest.fixture
def test_filename(tmp_path):
    filename = tmp_path / "test_file.bp"
    with adios2.Stream(str(filename), mode="w") as file:
        for step, _ in enumerate(file.steps(5)):
            file.write("scalar", step)
            arr1d = np.arange(10)
            file.write("arr1d", arr1d, arr1d.shape, [0], arr1d.shape)

    return filename


@pytest.fixture
def test_filename_2(tmp_path):
    filename = tmp_path / "test_file_2.bp"
    with adios2.Stream(str(filename), mode="w") as file:
        for step, _ in enumerate(file.steps(5)):
            file.write("step", step)
            file.write("time", 10.0 * step)

            x = np.linspace(0, 1, 10)
            file.write("x", x, x.shape, [0], x.shape)
            file.write_attribute("dimensions", "redundant x", variable_name="x")
            arr1d = np.arange(10)
            file.write("arr1d", arr1d, arr1d.shape, [0], arr1d.shape)
            file.write_attribute("dimensions", "time x", variable_name="arr1d")

    return filename


@pytest.fixture
def test_filename_3(tmp_path):
    filename = tmp_path / "test_file_3.bp"
    with adios2.Stream(str(filename), mode="w") as file:
        for step, _ in enumerate(file.steps(5)):
            file.write("step", step)
            time = np.array([2013, 3, 17, 13, 0, step, 200], dtype=np.int32)
            file.write("time", time, time.shape, [0], time.shape)

    return filename


def _open_dataset(filename: os.PathLike[Any]) -> xr.Dataset:
    return xr.open_dataset(
        filename,
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
            store = Adios2Store.open(step)
            ds = xr.open_dataset(store, engine="pscadios2_engine")
            assert ds.keys() == set({"scalar", "arr1d"})
            assert ds["scalar"] == n


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_open_dataset_from_Step(test_filename, mode):
    with adios2py.File(test_filename, mode) as file:
        for n, step in enumerate(file.steps):
            ds = xr.open_dataset(step)
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
            ds = xr.open_dataset(Adios2Store.open(step))
            assert ds.keys() == set({"step", "time", "arr1d"})
            assert ds.coords.keys() == set({"x"})


def test_open_dataset_3(test_filename_3):
    ds = xr.open_dataset(test_filename_3, decode_openggcm=True)
    assert ds.time.shape == (5,)
    assert ds.time[0] == np.datetime64("2013-03-17T13:00:00.000200000")
    assert ds.time[1] == np.datetime64("2013-03-17T13:00:01.000200000")


# def test_ggcm_i2c():
#     ds = xr.open_dataset(
#         "/workspaces/openggcm/ggcm-gitm-coupling-tools/data/iono_to_sigmas.bp"
#     )
#     assert ds.sizes == dict(lats=181, longs=61)
#     assert np.isclose(ds.dacttime, 1.4897556e09)
