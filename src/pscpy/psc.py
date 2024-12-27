from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import adios2py


class RunInfo:
    """Global information about the PSC run

    Currently stores domain info.
    TODO: Should also know about timestep, species, whatever...
    """

    def __init__(
        self,
        file: adios2py.Group,
        length: ArrayLike | None = None,
        corner: ArrayLike | None = None,
    ) -> None:
        first_var = next(iter(file.values()))
        self.gdims = np.asarray(first_var.shape)[::-1][:3]

        self.length = file.attrs.get("length", length)
        self.corner = file.attrs.get("corner", corner)

        self.x = self._get_coord(0)
        self.y = self._get_coord(1)
        self.z = self._get_coord(2)

    def _get_coord(self, coord_idx: int) -> NDArray[np.floating[Any]]:
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
