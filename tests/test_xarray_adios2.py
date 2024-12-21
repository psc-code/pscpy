from __future__ import annotations

import numpy as np
import xarray as xr

import pscpy


def test_open_dataset():
    ds = xr.open_dataset(pscpy.sample_dir / "pfd.000000400.bp", species_names=[], length=[1, 12.8, 51.2], corner=[0, -6.4, -25.6])
    assert "jx_ec" in ds
    assert ds.coords.keys() == set({"x", "y", "z"})
    assert ds.jx_ec.shape == (1, 128, 512)
    assert np.allclose(ds.jx_ec.z, np.linspace(-25.6, 25.6, 512, endpoint=False))
