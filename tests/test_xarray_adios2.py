from __future__ import annotations

import os
from typing import Any

import numpy as np
import xarray as xr

import pscpy

# import adios2
# from pscpy.pscadios2 import Adios2Store


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


# def test_ggcm_i2c():
#     ds = xr.open_dataset(
#         "/workspaces/openggcm/ggcm-gitm-coupling-tools/data/iono_to_sigmas.bp"
#     )
#     assert ds.sizes == dict(lats=181, longs=61)
#     assert np.isclose(ds.dacttime, 1.4897556e09)
