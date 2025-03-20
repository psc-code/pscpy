from __future__ import annotations

from typing import Literal

import xarray as xr


def recenter(
    da: xr.DataArray,
    dim: str,
    interp_dir: Literal[-1, 1],
    *,
    boundary: Literal["periodic", "pad", "zero"] = "periodic",
) -> xr.DataArray:
    """
    Returns a new array with values along `dim` recentered in the direction `interp_dir`.

    For example, `interp_dir=-1` interpolates node-centered values from cell-centered values, because each node center is at a lesser coordinate than the cell center of the same index.
    """

    return da
