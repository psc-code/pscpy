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


def auto_recenter(
    ds: xr.Dataset,
    to_centering: Literal["nc", "cc"],
    **boundaries: BoundaryInterpMethod,
):
    """
    Recenters variables with names matching a particular pattern to the given centering.

    In particular, variable name ending in `"{dim}_ec"`, `"{dim}_fc"`, `"_nc"`, or `"_cc"`, where `dim` is a dimension name and a key in `boundaries`, is recentered appropriately. For example, if `to_centering="nc"` (node-centered), a variable ending in "x_ec" (i.e., the x-component of an edge-centered field) will be recentered along x, but not y or z, since it is already node-centered in those dimensions.

    Variables are also renamed appropriately. In the example above, `ex_ec` would be renamed to `ex_nc`.
    """

    interp_dir = {"cc": 1, "nc": -1}[to_centering]

    for var_name in ds:
        if not isinstance(var_name, str):
            continue

        for dim, boundary_method in boundaries.items():
            if to_centering == "nc" and var_name.endswith(f"{dim}_ec"):
                ds[var_name] = get_recentered(ds[var_name], dim, interp_dir, boundary=boundary_method)

        new_name = var_name[:-3] + "_" + to_centering
        ds[new_name] = ds[var_name].rename(new_name)
        del ds[var_name]
