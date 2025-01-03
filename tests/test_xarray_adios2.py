from __future__ import annotations

import os
from typing import Any

import adios2
import numpy as np
import pytest
import xarray as xr

import pscpy
from pscpy import adios2py, pscadios2
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
            file.write_attribute("dimensions", "step", "step")
            time = np.array([2013, 3, 17, 13, 0, step, 200], dtype=np.int32)
            file.write("time", time, time.shape, [0], time.shape)
            file.write_attribute("dimensions", "step time_array", "time")
            file.write_attribute("units", "time_array", "time")

    return filename


@pytest.fixture
def test_filename_4(tmp_path):
    filename = tmp_path / "test_file_4.bp"
    with adios2.Stream(str(filename), mode="w") as file:
        for step, _ in enumerate(file.steps(5)):
            time = np.array(step)
            file.write("time", time)
            file.write_attribute("units", "seconds since 1970-01-01", "time")

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
    assert ds.time[0] == np.datetime64("2013-03-17T13:00:00.200")
    assert ds.time[1] == np.datetime64("2013-03-17T13:00:01.200")


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_open_dataset_3_step(test_filename_3, mode):
    with adios2py.File(test_filename_3, mode=mode) as file:
        for n, step in enumerate(file.steps):
            ds = xr.open_dataset(Adios2Store.open(step), decode_openggcm=True)
            assert ds.time == np.datetime64("2013-03-17T13:00:00.200") + np.timedelta64(
                n, "s"
            )


def test_open_dataset_4(test_filename_4):
    ds = xr.open_dataset(test_filename_4, decode_openggcm=True)
    print(ds)
    assert ds.time[0] == np.datetime64("1970-01-01T00:00:00.000")
    assert ds.time[1] == np.datetime64("1970-01-01T00:00:01.000")


def test_encode_time_array_0d():
    time = xr.Variable(
        dims=(),
        data=np.datetime64("1970-01-02T03:04:05", "ns"),
        encoding=dict(dtype="int32", units="time_array"),  # noqa: C408
    )

    time = pscadios2._encode_openggcm_variable(time)

    assert time.attrs["units"] == "time_array"
    assert time.sizes == dict(time_array=7)  # noqa: C408
    assert time.dtype == np.int32
    assert np.all(time == [1970, 1, 2, 3, 4, 5, 0])


def test_encode_time_array_1d():
    time1d = xr.Variable(
        dims=("time",),
        data=[
            np.datetime64("1970-01-02T03:04:05", "ns"),
            np.datetime64("1970-01-02T03:04:05.600", "ns"),
        ],
        encoding=dict(dtype="int32", units="time_array"),  # noqa: C408
    )

    time1d = pscadios2._encode_openggcm_variable(time1d)

    assert time1d.attrs["units"] == "time_array"
    assert time1d.sizes == dict(time=2, time_array=7)  # noqa: C408
    assert np.all(time1d == [[1970, 1, 2, 3, 4, 5, 0], [1970, 1, 2, 3, 4, 5, 600]])


def test_decode_time_array_0d():
    time = xr.Variable(
        dims=("time_array",),
        data=np.array([1970, 1, 2, 3, 4, 5, 0], dtype=np.int32),
        attrs=dict(units="time_array"),  # noqa: C408
    )
    time = pscadios2._decode_openggcm_variable(time, "time")

    assert "units" not in time.attrs
    assert np.all(time.to_numpy() == np.datetime64("1970-01-02T03:04:05", "ns"))


def test_decode_time_array_1d():
    time = xr.Variable(
        dims=("time_array", "time"),
        data=np.array(
            [[1970, 1, 2, 3, 4, 5, 0], [1970, 1, 2, 3, 4, 5, 600]], dtype=np.int32
        ),
        attrs=dict(units="time_array"),  # noqa: C408
    )
    time = pscadios2._decode_openggcm_variable(time, "time")

    assert "units" not in time.attrs
    assert np.all(
        time
        == [
            np.datetime64("1970-01-02T03:04:05", "ns"),
            np.datetime64("1970-01-02T03:04:05.600", "ns"),
        ]
    )


# def test_ggcm_i2c():
#     ds = xr.open_dataset(
#         "/workspaces/openggcm/ggcm-gitm-coupling-tools/data/iono_to_sigmas.bp"
#     )
#     assert ds.sizes == dict(lats=181, longs=61)
#     assert np.isclose(ds.dacttime, 1.4897556e09)
