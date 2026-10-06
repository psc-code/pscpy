from __future__ import annotations

import warnings
from typing import Any

import xarray as xr
from typing_extensions import deprecated

SUPPORTED_MAJOR_VERSION = 1


def check_psc_output_version(attrs: dict[str, object]) -> None:
    """Raise ValueError unless `attrs` has a supported "psc_output_version"."""
    version = attrs.get("psc_output_version")
    if version is None:
        message = (
            "Dataset has no psc_output_version attribute, so it was written in the "
            "legacy (pre-1.0.0) psc output format. Convert it first with "
            "`python -m pscpy.convert.legacy_to_v1`."
        )
        raise ValueError(message)

    major = str(version).split(".", maxsplit=1)[0]
    if major != str(SUPPORTED_MAJOR_VERSION):
        message = (
            f"Unsupported psc_output_version {version!r}; "
            f"only {SUPPORTED_MAJOR_VERSION}.x.y is supported."
        )
        if major.isdigit() and int(major) < SUPPORTED_MAJOR_VERSION:
            message += " Convert older output with the modules in pscpy.convert."
        raise ValueError(message)


@deprecated(
    "As of PSC output version v1, `decode_psc`'s only transformative action is to squeeze `time` and rename it to `t` to preserve backwards compatibility. From now on, the officially-supported way to get usable PSC data is to call `xr.open_dataset()` or `xr.open_mfdataset()` with default options."
)
def decode_psc(ds: xr.Dataset, **kwargs: Any) -> xr.Dataset:
    """Decode a dataset written by psc (psc_output_version 1.x.y).

    Each field component is already its own variable with (z, y, x) dims and
    cell-centered coordinates, so this only validates the format version,
    renames "time" to "t" and drops the "t" dimension if the dataset contains
    a single step.
    """
    check_psc_output_version(ds.attrs)

    if kwargs:
        warnings.warn(
            f"Other args ({', '.join(kwargs.keys())}) to `decode_psc` are no longer used.",
            stacklevel=2,
        )

    if "time" in ds.variables or "time" in ds.dims:
        ds = ds.rename(time="t")
    if ds.sizes.get("t") == 1:
        ds = ds.squeeze("t")

    return ds
