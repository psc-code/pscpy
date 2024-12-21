from __future__ import annotations

import numpy as np
import xarray as xr

import pscpy


def _open_dataset() -> xr.Dataset:
    return xr.open_dataset(
        pscpy.sample_dir / "pfd.000000400.bp",
        species_names=[],
        length=[1, 12.8, 51.2],
        corner=[0, -6.4, -25.6],
    )


def test_open_dataset():
    ds = _open_dataset()
    assert "jx_ec" in ds
    assert ds.coords.keys() == set({"x", "y", "z"})
    assert ds.jx_ec.shape == (1, 128, 512)
    assert np.allclose(ds.jx_ec.z, np.linspace(-25.6, 25.6, 512, endpoint=False))


def test_component():
    ds = _open_dataset()
    assert ds.jeh.shape == (1, 128, 512, 9)
    assert np.all(ds.jeh[..., 0] == ds.jx_ec)


def test_selection():
    ds = _open_dataset()
    assert ds.jeh.shape == (1, 128, 512, 9)
    assert np.all(ds.jeh[:, :10, :40, 0] == ds.jx_ec[:, :10, :40])


def test_partial_selection():
    ds = _open_dataset()
    assert ds.jeh.shape == (1, 128, 512, 9)
    assert np.all(ds.jeh[:, :10, ..., 0] == ds.jx_ec[:, :10, :])
    assert np.all(ds.jeh[:, :10, ..., 0] == ds.jx_ec[:, :10])


def test_computed():
    ds = _open_dataset()
    ds = ds.assign(jx=ds.jeh.isel(comp_9=0))
    assert np.all(ds.jx == ds.jx_ec)


def test_computed_via_lambda():
    ds = _open_dataset()
    ds = ds.assign(jx=lambda ds: ds.jeh.isel(comp_9=0))
    assert np.all(ds.jx == ds.jx_ec)
