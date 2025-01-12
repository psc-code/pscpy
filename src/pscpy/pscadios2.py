from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray
from numpy.typing import ArrayLike, DTypeLike

logger = logging.getLogger(__name__)


def _decode_openggcm(
    ds: xarray.Dataset,
) -> xarray.Dataset:
    # add colats and mlts as coordinates
    # FIXME? not clear that this is the best place to do this
    if (
        not {"colats", "mlts"} <= ds.coords.keys()
        and {"lats", "longs"} <= ds.coords.keys()
    ):
        ds = ds.assign_coords(colats=90 - ds.lats, mlts=(ds.longs + 180) * 24 / 360)

    for name, var in ds.variables.items():
        ds[name] = _decode_openggcm_variable(var, name)  # type: ignore[arg-type]

    return ds


def _dt64_to_time_array(times: ArrayLike, dtype: DTypeLike) -> ArrayLike:
    dt_times = pd.to_datetime(times)
    return np.array(
        [
            dt_times.year,
            dt_times.month,
            dt_times.day,
            dt_times.hour,
            dt_times.minute,
            dt_times.second,
            dt_times.microsecond // 1000,
        ],
        dtype=dtype,
    ).T


def _time_array_to_dt64(times: Iterable[Sequence[int]]) -> Sequence[np.datetime64]:
    return [
        np.datetime64(
            dt.datetime(
                year=time[0],
                month=time[1],
                day=time[2],
                hour=time[3],
                minute=time[4],
                second=time[5],
                microsecond=time[6] * 1000,
            ),
            "ns",
        )
        for time in times
    ]


def _decode_openggcm_variable(var: xarray.Variable, name: str) -> xarray.Variable:  # noqa: ARG001
    if var.attrs.get("units") == "time_array":
        times: Any = var.to_numpy().tolist()
        if var.ndim == 1:
            dt_times = _time_array_to_dt64([times])[0]
        else:
            dt_times = _time_array_to_dt64(times)  # type: ignore[assignment]

        attrs = var.attrs.copy()
        attrs.pop("units")
        new_var = xarray.Variable(dims=var.dims[1:], data=dt_times, attrs=attrs)
    else:
        new_var = var
    return new_var


def _encode_openggcm_variable(var: xarray.Variable) -> xarray.Variable:
    if var.encoding.get("units") == "time_array":
        attrs = var.attrs.copy()
        attrs["units"] = "time_array"
        if "_FillValue" in attrs:
            attrs.pop("_FillValue")

        new_var = xarray.Variable(
            dims=(*var.dims, "time_array"),
            data=_dt64_to_time_array(
                var.to_numpy(),
                var.encoding.get("dtype", "int32"),
            ),
            attrs=attrs,
        )
    else:
        new_var = var
    return new_var


def _encode_openggcm(
    vars: Mapping[str, xarray.Variable], attrs: Mapping[str, Any]
) -> tuple[Mapping[str, xarray.Variable], Mapping[str, Any]]:
    new_vars = {name: _encode_openggcm_variable(var) for name, var in vars.items()}

    return new_vars, attrs


def _decode_openggcm_vars_attrs(
    vars: Mapping[str, xarray.Variable], attrs: Mapping[str, Any]
) -> tuple[Mapping[str, xarray.Variable], Mapping[str, Any]]:
    new_vars = {
        name: _decode_openggcm_variable(var, name) for name, var in vars.items()
    }

    return new_vars, attrs
