from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from .adios2py import File


class RunInfo:
    """Global information about the PSC run

    Currently stores domain info.
    TODO: Should also know about timestep, species, whatever...
    """

    def __init__(self, file: File, length: ArrayLike | None = None, corner: ArrayLike | None = None):
        assert len(file.variable_names) > 0
        var = next(iter(file.variable_names))
        self.gdims = np.asarray(file.get_variable(var).shape)[0:3]

        maybe_length_attr = file.get_attribute("length")
        if maybe_length_attr:
            self.corner = file.get_attribute("corner")
        elif length is not None:
            self.length = np.asarray(length)
            if corner is not None:
                self.corner = np.asarray(corner)
            else:
                self.corner = -0.5 * self.length
        else:
            self.length = self.gdims
            self.corner = np.array([0.0, 0.0, 0.0])

        self.x = np.linspace(
            self.corner[0],
            self.corner[0] + self.length[0],
            self.gdims[0],
            endpoint=False,
        )
        self.y = np.linspace(
            self.corner[1],
            self.corner[1] + self.length[1],
            self.gdims[1],
            endpoint=False,
        )
        self.z = np.linspace(
            self.corner[2],
            self.corner[2] + self.length[2],
            self.gdims[2],
            endpoint=False,
        )

    def __repr__(self):
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
            field_to_component["all_1st"][f"{moment}_{species_name}"] = moment_idx + 13 * species_idx
            field_to_component["all_1st_cc"][f"{moment}_{species_name}"] = moment_idx + 13 * species_idx

    return field_to_component
