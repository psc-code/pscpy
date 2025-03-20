from __future__ import annotations

from typing import Literal

import xarray as xr

type BoundaryInterpMethod = Literal["periodic", "pad", "zero"]

def get_recentered(
    da: xr.DataArray,
    dim: str,
    interp_dir: Literal[-1, 1],
    *,
    boundary: BoundaryInterpMethod = "periodic",
) -> xr.DataArray:
    """
    Returns a new array with values along `dim` recentered in the direction `interp_dir`.

    For example, `interp_dir=-1` interpolates node-centered values from cell-centered values, because each node center is at a lesser coordinate than the cell center of the same index.
    """

    shifted = da.roll({dim: -interp_dir}, roll_coords=False)
    boundary_idx = {-1: 0, 1: -1}[interp_dir]

    if boundary == "periodic":
        pass  # this case is already handled by the behavior of roll()
    elif boundary == "pad":
        shifted[{dim: boundary_idx}] = da[{dim: boundary_idx}]
    elif boundary == "zero":
        shifted[{dim: boundary_idx}] = 0

    return 0.5 * (da + shifted)
