from __future__ import annotations

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


@pytest.fixture
def ds_pfd_raw() -> xr.Dataset:
    return xr.open_dataset(pscpy.sample_dir / "v1" / "pfd.000000001.bp")


@pytest.fixture
def ds_pfd_moments_raw() -> xr.Dataset:
    return xr.open_dataset(pscpy.sample_dir / "v1" / "pfd_moments.000000001.bp")


@pytest.fixture
def ds_pfd_decoded(ds_pfd_raw) -> xr.Dataset:
    return pscpy.decode_psc(ds_pfd_raw)


@pytest.fixture
def ds_pfd_moments_decoded(ds_pfd_moments_raw) -> xr.Dataset:
    return pscpy.decode_psc(ds_pfd_moments_raw)


def _cell_centers(corner: float, length: float, n: int) -> np.ndarray:
    return corner + (np.arange(n) + 0.5) * (length / n)


def test_open_dataset(ds_pfd_decoded):
    assert set(ds_pfd_decoded.data_vars) == {
        "jx_ec", "jy_ec", "jz_ec", "ex_ec", "ey_ec", "ez_ec", "hx_fc", "hy_fc", "hz_fc"
    }  # fmt: skip
    assert ds_pfd_decoded.coords.keys() == set({"x", "y", "z", "time"})
    assert ds_pfd_decoded.jx_ec.dims == ("z", "y", "x")
    assert ds_pfd_decoded.jx_ec.sizes == dict(x=1, y=8, z=4)  # noqa: C408


def test_coords(ds_pfd_decoded):
    assert np.allclose(ds_pfd_decoded.x, _cell_centers(0.0, 1.0, 1))
    assert np.allclose(ds_pfd_decoded.y, _cell_centers(-5.0, 10.0, 8))
    assert np.allclose(ds_pfd_decoded.z, _cell_centers(-2.5, 5.0, 4))


def test_time(ds_pfd_decoded):
    assert ds_pfd_decoded.time.ndim == 0
    assert ds_pfd_decoded.time == 0.7954951288348661
    assert "t" not in ds_pfd_decoded.coords


def test_time_multiple_steps(ds_pfd_raw):
    ds = xr.concat(
        [ds_pfd_raw, ds_pfd_raw.assign_coords(time=ds_pfd_raw.time + 1.0)], dim="time"
    )
    ds_decoded = pscpy.decode_psc(ds)
    assert ds_decoded.sizes["time"] == 2
    assert ds_decoded.jx_ec.dims == ("time", "z", "y", "x")


def test_data_unchanged(ds_pfd_raw, ds_pfd_decoded):
    for name, var in ds_pfd_decoded.data_vars.items():
        assert np.array_equal(var.data, ds_pfd_raw[name].isel(time=0).data)


def test_selection(ds_pfd_raw, ds_pfd_decoded):
    data_raw = ds_pfd_raw.jx_ec.isel(time=0, y=slice(0, 10), z=slice(0, 1)).data
    data_decoded = ds_pfd_decoded.jx_ec.isel(y=slice(0, 10), z=slice(0, 1)).data
    assert np.all(data_raw == data_decoded)


def _get_nbytes(ds: xr.Dataset) -> int:
    return sum(arr.nbytes for arr in ds.data_vars.values())


def test_nbytes(ds_pfd_raw, ds_pfd_decoded):
    assert _get_nbytes(ds_pfd_raw) == _get_nbytes(ds_pfd_decoded)


def test_computed(ds_pfd_decoded):
    ds = ds_pfd_decoded.assign(jx=ds_pfd_decoded.jx_ec * 2)
    assert np.all(ds.jx.data == 2 * ds_pfd_decoded.jx_ec.data)


def test_computed_via_lambda(ds_pfd_decoded):
    ds = ds_pfd_decoded.assign(jx=lambda ds: ds.jx_ec * 2)
    assert np.all(ds.jx.data == 2 * ds_pfd_decoded.jx_ec.data)


def test_pfd_moments(ds_pfd_moments_decoded):
    moments = ["rho", "jx", "jy", "jz", "px", "py", "pz", "txx", "tyy", "tzz", "txy", "tyz", "tzx"]  # fmt: skip
    expected = {f"{moment}_{species}" for species in "ei" for moment in moments}
    assert set(ds_pfd_moments_decoded.data_vars) == expected
    assert np.all(ds_pfd_moments_decoded.rho_e < 0)
    assert np.all(ds_pfd_moments_decoded.rho_i > 0)


@pytest.mark.parametrize(
    "filename", ["pfd.000000400.bp", "pfd_moments.000000400.bp", "pfd.000000000.bp"]
)
def test_legacy_rejected(filename):
    ds = xr.open_dataset(pscpy.sample_dir / filename)
    with pytest.raises(
        ValueError, match=r"psc_output_version.*pscpy\.convert\.legacy_to_v1"
    ):
        pscpy.decode_psc(ds)


@pytest.mark.parametrize("version", ["0.9.0", "2.0.0", "garbage"])
def test_unsupported_version_rejected(ds_pfd_raw, version):
    ds = ds_pfd_raw.assign_attrs(psc_output_version=version)
    with pytest.raises(ValueError, match=r"Unsupported psc_output_version"):
        pscpy.decode_psc(ds)


def test_older_version_points_to_converters(ds_pfd_raw):
    ds = ds_pfd_raw.assign_attrs(psc_output_version="0.9.0")
    with pytest.raises(ValueError, match=r"pscpy\.convert"):
        pscpy.decode_psc(ds)


@pytest.mark.parametrize("version", ["2.0.0", "garbage"])
def test_newer_version_has_no_converter_hint(ds_pfd_raw, version):
    ds = ds_pfd_raw.assign_attrs(psc_output_version=version)
    with pytest.raises(ValueError, match=r"supported\.$"):
        pscpy.decode_psc(ds)


def test_compatible_version_accepted(ds_pfd_raw):
    ds = ds_pfd_raw.assign_attrs(psc_output_version="1.2.3")
    assert "jx_ec" in pscpy.decode_psc(ds)


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
