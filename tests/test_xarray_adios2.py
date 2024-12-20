from __future__ import annotations

import xarray as xr

import pscpy


def test_open_dataset():
    xr.open_dataset(pscpy.sample_dir / "pfd.000000400.bp", species_names=[], length=[10, 10, 10], corner=[0, 0, 0])
