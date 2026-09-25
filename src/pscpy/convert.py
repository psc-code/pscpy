"""Convert legacy psc .bp output to psc_output_version 1.0.0.

Legacy (pre-1.0.0) psc output stores all components of a field in a single
4-d variable (e.g. "jeh" with shape (9, z, y, x)). Starting with 1.0.0, psc
writes each component as its own (z, y, x) variable, together with
cell-centered x/y/z coordinates and a scalar "time" variable.

Usage::

    python -m pscpy.convert -o OUTDIR [--species e i] [--length LX LY LZ]
        [--corner CX CY CZ] SRC.bp [SRC.bp ...]

Each SRC.bp is written to OUTDIR with the same file name.
"""

from __future__ import annotations

import argparse
import os
import pathlib
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

import adios2py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .psc import PSC_OUTPUT_VERSION

JEH_COMPONENTS = ["jx_ec", "jy_ec", "jz_ec", "ex_ec", "ey_ec", "ez_ec", "hx_fc", "hy_fc", "hz_fc"]  # fmt: skip
FIELD_COMPONENTS = {
    "jeh": JEH_COMPONENTS,
    "dive": ["dive"],
    "rho": ["rho"],
    "d_rho": ["d_rho"],
    "dt_divj": ["dt_divj"],
}
MOMENT_FIELDS = ["all_1st", "all_1st_cc"]
MOMENTS = ["rho", "jx", "jy", "jz", "px", "py", "pz", "txx", "tyy", "tzz", "txy", "tyz", "tzx"]  # fmt: skip


def legacy_component_names(
    field: str, n_components: int, species_names: Sequence[str] | None = None
) -> list[str]:
    """Return the names of the components stored in legacy variable `field`."""
    if field in FIELD_COMPONENTS:
        names = FIELD_COMPONENTS[field]
    elif field in MOMENT_FIELDS:
        if species_names is None:
            message = f"species_names are required to convert moments ({field!r})"
            raise ValueError(message)
        names = [
            f"{moment}_{species}" for species in species_names for moment in MOMENTS
        ]
    else:
        message = f"Don't know how to convert legacy variable {field!r}"
        raise ValueError(message)

    if len(names) != n_components:
        message = (
            f"Legacy variable {field!r} has {n_components} components, "
            f"expected {len(names)} ({', '.join(names)})"
        )
        raise ValueError(message)
    return names


def _unwrap_scalar(value: Any) -> Any:
    """Legacy files sometimes store scalar attributes as 1-element arrays."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr[()]
    return arr[0]


def _fma(a: float, b: float, c: float) -> float:
    """a * b + c with a single rounding, like math.fma (Python >= 3.13)."""
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _cell_centers(corner: float, length: float, n: int) -> NDArray[np.float64]:
    # same expression as psc's writer, which the compiler contracts into an
    # fma, so the result is bitwise identical
    dx = length / n
    return np.array([_fma(i + 0.5, dx, corner) for i in range(n)], dtype=np.float64)


def _get_domain_attr(
    attrs: Any, name: str, override: ArrayLike | None
) -> NDArray[np.float64]:
    if override is not None:
        value = override
    elif name in attrs:
        value = attrs[name]
    else:
        message = f"{name} is missing from the file and must be provided."
        raise ValueError(message)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        message = f"{name} must have 3 entries, got {arr.shape}"
        raise ValueError(message)
    return arr


def convert_file(
    src: os.PathLike[Any] | str,
    dst: os.PathLike[Any] | str,
    *,
    species_names: Sequence[str] | None = None,
    length: ArrayLike | None = None,
    corner: ArrayLike | None = None,
) -> None:
    """Convert the legacy psc output file `src` to psc_output_version 1.0.0 at `dst`.

    `length` and `corner` override the values stored in `src` (older files
    don't have them). `species_names` is required for moment output.
    """
    if pathlib.Path(dst).exists():
        message = f"{dst} already exists"
        raise FileExistsError(message)

    with adios2py.File(src, mode="rra") as file:
        attrs = file.attrs
        if "psc_output_version" in attrs:
            message = f"{src} is not in the legacy format (psc_output_version={attrs['psc_output_version']!r})"
            raise ValueError(message)
        if len(file.steps) != 1:
            message = f"{src} has {len(file.steps)} steps, expected 1"
            raise ValueError(message)

        length_arr = _get_domain_attr(attrs, "length", length)
        corner_arr = _get_domain_attr(attrs, "corner", corner)
        step_num = np.int32(_unwrap_scalar(attrs["step"]))
        time = np.float64(_unwrap_scalar(attrs["time"]))

        step_in = file.steps[0]
        fields: dict[str, tuple[NDArray[Any], dict[str, Any]]] = {}
        for field in step_in:
            data = step_in[field][...]
            if data.ndim != 4:
                message = f"Legacy variable {field!r} has shape {data.shape}, expected (component, z, y, x)"
                raise ValueError(message)
            var_attrs = {
                name: np.asarray(attrs[key], dtype=np.int32)
                for name in ["ib", "im"]
                for key in [f"{field}::{name}", name]
                if key in attrs
            }
            names = legacy_component_names(field, data.shape[0], species_names)
            for idx, name in enumerate(names):
                fields[name] = (data[idx], var_attrs)

    gdims_zyx = next(iter(fields.values()))[0].shape
    gdims = gdims_zyx[::-1]

    with adios2py.File(dst, mode="w") as out, out.steps.next() as step:
        out.attrs["psc_output_version"] = PSC_OUTPUT_VERSION
        out.attrs["step"] = step_num
        out.attrs["time"] = time
        out.attrs["length"] = length_arr
        out.attrs["corner"] = corner_arr
        out.attrs["step_dimension"] = "time"

        step["time"] = time
        for d, crd_name in enumerate("xyz"):
            step[crd_name] = _cell_centers(corner_arr[d], length_arr[d], gdims[d])
            step[crd_name].attrs["dimensions"] = crd_name

        for name, (data, var_attrs) in fields.items():
            if data.shape != gdims_zyx:
                message = (
                    f"Component {name!r} has shape {data.shape}, expected {gdims_zyx}"
                )
                raise ValueError(message)
            step[name] = data
            step[name].attrs["dimensions"] = "z y x"
            for attr_name, value in var_attrs.items():
                step[name].attrs[attr_name] = value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pscpy.convert",
        description=f"Convert legacy psc .bp output to psc_output_version {PSC_OUTPUT_VERSION}.",
    )
    parser.add_argument("src", nargs="+", type=pathlib.Path, help="legacy .bp files")
    parser.add_argument(
        "-o", "--outdir", required=True, type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "--species", nargs="+", help="species names, required for moments (e.g. e i)"
    )
    parser.add_argument(
        "--length", nargs=3, type=float, help="domain length, if missing from the file"
    )
    parser.add_argument(
        "--corner", nargs=3, type=float, help="domain corner, if missing from the file"
    )
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for src in args.src:
        dst = args.outdir / src.name
        convert_file(
            src,
            dst,
            species_names=args.species,
            length=args.length,
            corner=args.corner,
        )
        print(f"{src} -> {dst}")  # noqa: T201


if __name__ == "__main__":
    main()
