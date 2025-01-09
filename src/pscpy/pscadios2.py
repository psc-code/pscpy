from __future__ import annotations

import datetime as dt
import logging
import os
import pathlib
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing_extensions import Never, override
from xarray.backends import CachingFileManager, DummyFileManager, FileManager
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    WritableCFDataStore,
    _normalize_path,
)
from xarray.backends.locks import (
    SerializableLock,
    combine_locks,
    ensure_lock,
    get_write_lock,
)
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.datatree import DataTree
from xarray.core.types import ReadBuffer
from xarray.core.utils import FrozenDict

from . import adios2py, psc

logger = logging.getLogger(__name__)


class Lock(Protocol):
    """Provides duck typing for xarray locks, which do not inherit from a common base class."""

    def acquire(self, blocking: bool = True) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
    def locked(self) -> bool: ...


# adios2 is not thread safe
ADIOS2_LOCK = SerializableLock()


class Adios2Array(BackendArray):
    """Lazy evaluation of a variable stored in an adios2 file.

    This also takes care of slicing out the specific component of the data stored as 4-d array.
    """

    def __init__(
        self,
        variable_name: str,
        datastore: Adios2Store,
    ) -> None:
        self.variable_name = variable_name
        self.datastore = datastore
        array = self.get_array()
        self.shape = array.shape
        self.dtype = array.dtype

    def get_array(self, needs_lock: bool = True) -> adios2py.Variable:
        return self.datastore.acquire(needs_lock)[self.variable_name]

    def __getitem__(self, key: indexing.ExplicitIndexer) -> Any:
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key) -> NDArray[np.floating[Any]]:  # type: ignore [no-untyped-def]
        with self.datastore.lock:
            return self.get_array(needs_lock=False)[key]


class Adios2Store(WritableCFDataStore):
    """DataStore to facilitate loading an Adios2 file."""

    def __init__(
        self,
        manager: FileManager,
        mode: str | None = None,
        lock: Lock = ADIOS2_LOCK,
    ) -> None:
        self._manager = manager
        self._mode = mode
        self.lock: Lock = ensure_lock(lock)  # type: ignore[no-untyped-call]
        # keep track of attributes that belong with a variable
        self._var_attrs: set[str] = set()

    @classmethod
    def open(
        cls,
        filename_or_obj: str | os.PathLike[Any] | adios2py.Group | None,
        mode: str = "r",
        lock: Lock | None = None,
        parameters: dict[str, str] | None = None,
        engine_type: str | None = None,
    ) -> Adios2Store:
        if lock is None:
            if mode == "r":
                lock = ADIOS2_LOCK
            else:
                lock = combine_locks([ADIOS2_LOCK, get_write_lock(filename_or_obj)])  # type: ignore[no-untyped-call]

        if isinstance(filename_or_obj, (str, os.PathLike)):
            kwargs: dict[str, Any] = {}
            if parameters is not None:
                kwargs["parameters"] = tuple(sorted(parameters.items()))
            if engine_type is not None:
                kwargs["engine_type"] = engine_type
            manager: FileManager = CachingFileManager(
                adios2py.File, filename_or_obj, mode=mode, kwargs=kwargs
            )
        elif isinstance(filename_or_obj, adios2py.Group):
            manager = DummyFileManager(filename_or_obj)  # type: ignore[no-untyped-call]
        elif filename_or_obj is None:
            assert mode == "w"
            manager = DummyFileManager(filename_or_obj)  # type: ignore[no-untyped-call]
        else:
            msg = f"Adios2Store: unknown filename_or_obj {filename_or_obj}"  # type: ignore[unreachable]
            raise TypeError(msg)
        return cls(manager, mode=mode, lock=lock)

    def acquire(self, needs_lock: bool = True) -> adios2py.Group:
        with self._manager.acquire_context(needs_lock) as root:  # type: ignore[no-untyped-call]
            ds = root
        assert isinstance(ds, adios2py.Group)
        return ds

    @property
    def ds(self) -> adios2py.Group:
        return self.acquire()

    @override
    def get_variables(self) -> Mapping[str, xarray.Variable]:
        return FrozenDict((k, self.open_store_variable(k)) for k in self.ds)

    def open_store_variable(self, var_name: str) -> xarray.Variable:
        data = indexing.LazilyIndexedArray(Adios2Array(var_name, self))
        attr_names = [name for name in self.ds.attrs if name.startswith(f"{var_name}/")]
        self._var_attrs |= set(attr_names)
        attrs = {
            name.removeprefix(f"{var_name}/"): self.ds.attrs[name]  # type: ignore[attr-defined]
            for name in attr_names
        }
        if "dimensions" in attrs:
            dims: tuple[str, ...] = attrs["dimensions"].split()
            del attrs["dimensions"]
            if len(dims) == data.ndim + 1:
                dims = dims[1:]
            return xarray.Variable(dims, data, attrs)

        if data.ndim == 5:  # for psc compatibility
            dims = ("step", f"comp_{var_name}", "z", "y", "x")
            return xarray.Variable(dims, data, attrs)

        # if we have no info, not much we can do...
        # print(f"Variable without dimensions: {var_name}")
        dims = tuple(f"len_{dim}" for dim in data.shape)
        return xarray.Variable(dims, data, attrs)

    @override
    def get_attrs(self) -> Mapping[str, Any]:
        attrs_remaining = self.ds.attrs.keys() - self._var_attrs
        return FrozenDict((name, self.ds.attrs[name]) for name in attrs_remaining)

    @override
    def get_dimensions(self) -> Never:
        raise NotImplementedError()

    @override
    def load(
        self,
    ) -> tuple[Mapping[str, xarray.Variable], Mapping[str, Any]]:
        self._var_attrs = set()
        vars, attrs = super().load()  # type: ignore[no-untyped-call]
        # TODO, this isn't really the right place to do this -- more of a hack
        # to get the decoding hooked in while we still have vars, attrs
        return _decode_openggcm_vars_attrs(vars, attrs)

    def store(
        self,
        variables: Mapping[str, xarray.Variable],
        attributes: Mapping[str, Any],
        check_encoding_set: Any = frozenset(),  # noqa: ARG002
        writer: Any = None,
        unlimited_dims: bool | None = None,  # noqa: ARG002
    ) -> None:
        variables, attributes = self.encode(variables, attributes)  # type:ignore[no-untyped-call]

        writer._begin_step()
        for var_name, var in variables.items():
            writer._write(var_name, var)

        for attr_name, attr in attributes.items():
            writer._write_attribute(attr_name, attr)
        writer._end_step()


class PscAdios2BackendEntrypoint(BackendEntrypoint):
    """Entrypoint that lets xarray recognize and read adios2 output."""

    open_dataset_parameters = ("filename_or_obj", "drop_variables")
    available = True

    @override
    def open_dataset(  # type: ignore[no-untyped-def]
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime=None,
        decode_timedelta=None,
        length: ArrayLike | None = None,
        corner: ArrayLike | None = None,
        species_names: Iterable[str]
        | None = None,  # e.g. ['e', 'i']; FIXME should be readable from file
        decode_openggcm=False,
    ) -> xarray.Dataset:
        if isinstance(filename_or_obj, Adios2Store):
            store = filename_or_obj
        else:
            filename = _normalize_path(filename_or_obj)
            if not isinstance(filename, str):
                raise NotImplementedError()

            store = Adios2Store.open(filename, mode="rra")

        store_entrypoint = StoreBackendEntrypoint()

        ds = store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        if "redundant" in ds.dims:
            ds = ds.isel(redundant=0)

        if species_names is not None:
            ds = _decode_psc(ds, store.ds, species_names, length, corner)

        if decode_openggcm:
            ds = _decode_openggcm(ds)

        return ds

    @override
    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
    ) -> bool:
        if isinstance(filename_or_obj, (str, os.PathLike)):
            ext = pathlib.Path(filename_or_obj).suffix
            return ext in {".bp"}

        return isinstance(filename_or_obj, (Adios2Store, adios2py.Group))

    @override
    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
        **kwargs: Any,
    ) -> DataTree:
        raise NotImplementedError()


def _decode_psc(
    ds: xarray.Dataset,
    file: adios2py.Group,
    species_names: Iterable[str],
    length: ArrayLike | None = None,
    corner: ArrayLike | None = None,
) -> xarray.Dataset:
    ds = ds.squeeze("step")
    field_to_component = psc.get_field_to_component(species_names)

    data_vars = {}
    for var_name in ds:
        if var_name in field_to_component:
            for field, component in field_to_component[var_name].items():  # type: ignore[index]
                data_vars[field] = ds[var_name].isel({f"comp_{var_name}": component})
    ds = ds.assign(data_vars)

    if length is not None:
        run_info = psc.RunInfo(file, length=length, corner=corner)
        coords = {
            "x": ("x", run_info.x),
            "y": ("y", run_info.y),
            "z": ("z", run_info.z),
        }
        ds = ds.assign_coords(coords)

    return ds


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
