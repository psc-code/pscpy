from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray


class RunInfo:
    """Global information about the PSC run

    Currently stores domain info.
    TODO: Should also know about timestep, species, whatever...
    """

    def __init__(
        self,
        ds: xr.Dataset,
        length: ArrayLike | None = None,
        corner: ArrayLike | None = None,
    ) -> None:
        first_var = ds[next(iter(ds))]
        self.gdims = np.asarray(first_var.shape)[::-1][:3]

        self.length = ds.attrs.get("length", length)
        self.corner = ds.attrs.get("corner", corner)

        self.x = self._get_coord(0)
        self.y = self._get_coord(1)
        self.z = self._get_coord(2)

    def _get_coord(self, coord_idx: int) -> NDArray[Any]:
        return np.linspace(
            start=self.corner[coord_idx],
            stop=self.corner[coord_idx] + self.length[coord_idx],
            num=self.gdims[coord_idx],
            endpoint=False,
        )

    def __repr__(self) -> str:
        return f"Psc(gdims={self.gdims}, length={self.length}, corner={self.corner})"


def get_field_to_component(species_names: Iterable[str]) -> dict[str, dict[str, int]]:
    field_to_component: dict[str, dict[str, int]] = {}
    field_to_component["jeh"] = {
        "jx_ec": 0,
        "jy_ec": 1,
        "jz_ec": 2,
        "ex_ec": 3,
        "ey_ec": 4,
        "ez_ec": 5,
        "hx_fc": 6,
        "hy_fc": 7,
        "hz_fc": 8,
    }
    field_to_component["dive"] = {"dive": 0}
    field_to_component["rho"] = {"rho": 0}
    field_to_component["d_rho"] = {"d_rho": 0}
    field_to_component["div_j"] = {"div_j": 0}

    # keeping 'all_1st' for backwards compatibility
    field_to_component["all_1st"] = {}
    field_to_component["all_1st_cc"] = {}
    moments = [
        "rho",
        "jx",
        "jy",
        "jz",
        "px",
        "py",
        "pz",
        "txx",
        "tyy",
        "tzz",
        "txy",
        "tyz",
        "tzx",
    ]
    for species_idx, species_name in enumerate(species_names):
        for moment_idx, moment in enumerate(moments):
            field_to_component["all_1st"][f"{moment}_{species_name}"] = (
                moment_idx + 13 * species_idx
            )
            field_to_component["all_1st_cc"][f"{moment}_{species_name}"] = (
                moment_idx + 13 * species_idx
            )

    return field_to_component


def decode_psc(
    ds: xr.Dataset,
    species_names: Iterable[str],
    length: ArrayLike | None = None,
    corner: ArrayLike | None = None,
) -> xr.Dataset:
    da = ds[next(iter(ds))]  # first dataset
    if da.dims[0] == "dim_0_1":
        # for compatibility, if dimensions weren't saved as attribute in the .bp file,
        # fix them up here
        ds = ds.rename_dims(
            {
                da.dims[0]: "step",
                da.dims[1]: f"comp_{da.name}",
                da.dims[2]: "z",
                da.dims[3]: "y",
                da.dims[4]: "x",
            }
        )
    ds = ds.squeeze("step")
    field_to_component = get_field_to_component(species_names)

    data_vars = {}
    for var_name in ds:
        if var_name in field_to_component:
            for field, component in field_to_component[var_name].items():  # type: ignore[index]
                data_vars[field] = ds[var_name].isel({f"comp_{var_name}": component})
        ds = ds.drop_vars([var_name])
    ds = ds.assign(data_vars)

    if length is not None:
        run_info = RunInfo(ds, length=length, corner=corner)
        coords = {
            "x": ("x", run_info.x),
            "y": ("y", run_info.y),
            "z": ("z", run_info.z),
        }
        ds = ds.assign_coords(coords)

    return ds
