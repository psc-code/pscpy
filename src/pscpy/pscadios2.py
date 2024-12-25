from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Iterable, Protocol

import numpy as np
import xarray
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Never, override
from xarray.backends import CachingFileManager, DummyFileManager, FileManager
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
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
from xarray.core.utils import Frozen, FrozenDict

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
        step: int | None,
        datastore: Adios2Store,
    ) -> None:
        self.variable_name = variable_name
        self.datastore = datastore
        self._step = step
        array = self.get_array()
        self.shape = array.shape
        self.dtype = array.dtype

    def get_array(self, needs_lock: bool = True) -> adios2py.Variable:
        return self.datastore.acquire(needs_lock).get_variable(
            self.variable_name, step=self._step
        )

    def __getitem__(self, key: indexing.ExplicitIndexer) -> Any:
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key) -> NDArray[np.floating[Any]]:  # type: ignore [no-untyped-def]
        with self.datastore.lock:
            return self.get_array(needs_lock=False)[key]


class Adios2Store(AbstractDataStore):
    """DataStore to facilitate loading an Adios2 file."""

    def __init__(
        self,
        manager: FileManager,
        mode: str | None = None,
        lock: Lock = ADIOS2_LOCK,
        step: int | None = None,
    ) -> None:
        self._manager = manager
        self._mode = mode
        self.lock: Lock = ensure_lock(lock)  # type: ignore[no-untyped-call]
        # keep track of attributes that belong with a variable
        self._var_attrs: set[str] = set()
        self._step = step

    @classmethod
    def open(
        cls,
        filename: str,
        mode: str = "r",
        lock: Lock | None = None,
        parameters: dict[str, str] | None = None,
        engine_type: str | None = None,
    ) -> Adios2Store:
        if lock is None:
            if mode == "r":
                lock = ADIOS2_LOCK
            else:
                lock = combine_locks([ADIOS2_LOCK, get_write_lock(filename)])  # type: ignore[no-untyped-call]

        kwargs: dict[str, Any] = {}
        if parameters is not None:
            kwargs["parameters"] = tuple(sorted(parameters.items()))
        if engine_type is not None:
            kwargs["engine_type"] = engine_type
        manager = CachingFileManager(adios2py.File, filename, mode=mode, kwargs=kwargs)
        return cls(manager, mode=mode, lock=lock)

    @classmethod
    def open_existing(
        cls,
        io_engine: tuple[Any, Any],
        mode: str = "r",
        lock: Lock | None = None,
        step: int | None = None,
    ) -> Adios2Store:
        assert mode == "r"
        if lock is None:
            lock = ADIOS2_LOCK

        file = adios2py.File(io_engine, mode=mode)
        manager = DummyFileManager(file)  # type: ignore[no-untyped-call]
        return cls(manager, mode=mode, lock=lock, step=step)

    def acquire(self, needs_lock: bool = True) -> adios2py.File:
        with self._manager.acquire_context(needs_lock) as root:  # type: ignore[no-untyped-call]
            ds = root
        assert isinstance(ds, adios2py.File)
        return ds

    @property
    def ds(self) -> adios2py.File:
        return self.acquire()

    @override
    def get_variables(self) -> Frozen[str, xarray.Variable]:
        return FrozenDict((k, self.open_store_variable(k)) for k in self.ds.keys())  # noqa: SIM118

    def open_store_variable(self, var_name: str) -> xarray.Variable:
        data = indexing.LazilyIndexedArray(Adios2Array(var_name, self._step, self))
        attr_names = [
            name for name in self.ds.attribute_names if name.startswith(f"{var_name}::")
        ]
        self._var_attrs |= set(attr_names)
        attrs = {
            name.removeprefix(f"{var_name}::"): self.ds.get_attribute(name)  # type: ignore[attr-defined]
            for name in attr_names
        }
        if "xr-dims" in attrs:
            dims: tuple[str, ...] = attrs["xr-dims"].split(";")
        elif data.ndim == 4:  # for psc compatibility
            dims = ("x", "y", "z", f"comp_{var_name}")
        else:  # if we have no info, not much we can do...
            dims = tuple(f"len_{dim}" for dim in data.shape)
        return xarray.Variable(dims, data, attrs)

    @override
    def get_attrs(self) -> Frozen[str, Any]:
        attrs_remaining = set(self.ds.attribute_names) - self._var_attrs
        return FrozenDict(
            (name, self.ds.get_attribute(name)) for name in attrs_remaining
        )

    @override
    def get_dimensions(self) -> Never:
        raise NotImplementedError()

    @override
    def load(self):  # type: ignore[no-untyped-def]
        self._var_attrs = set()
        return super().load()  # type:ignore[no-untyped-call]


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
    ) -> xarray.Dataset:
        if isinstance(filename_or_obj, Adios2Store):
            store = filename_or_obj
        else:
            filename = _normalize_path(filename_or_obj)
            if not isinstance(filename, str):
                raise NotImplementedError()

            store = Adios2Store.open(filename)

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

        if species_names is not None:
            field_to_component = psc.get_field_to_component(species_names)

            data_vars = {}
            for var_name in ds:
                if var_name in field_to_component:
                    for field, component in field_to_component[var_name].items():  # type: ignore[index]
                        data_vars[field] = ds[var_name].isel(
                            {f"comp_{var_name}": component}
                        )
            ds = ds.assign(data_vars)

        if length is not None:
            run_info = psc.RunInfo(store.ds, length=length, corner=corner)
            coords = {
                "x": ("x", run_info.x),
                "y": ("y", run_info.y),
                "z": ("z", run_info.z),
            }
            ds = ds.assign_coords(coords)

        return ds

    @override
    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
    ) -> bool:
        if isinstance(filename_or_obj, (str, os.PathLike)):
            ext = pathlib.Path(filename_or_obj).suffix
            return ext in {".bp"}

        return isinstance(filename_or_obj, tuple) and len(filename_or_obj) == 2

    @override
    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
        **kwargs: Any,
    ) -> DataTree:
        raise NotImplementedError()
