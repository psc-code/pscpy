from __future__ import annotations

import io
import os
from typing import Any, Iterable, Protocol, SupportsInt

import numpy as np
import xarray
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Never, override
from xarray.backends import CachingFileManager
from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
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
from xarray.core import indexing
from xarray.core.datatree import DataTree
from xarray.core.utils import Frozen, FrozenDict

from .adios2py import File, Variable
from .psc import RunInfo, get_field_to_component


class Lock(Protocol):
    """Provides duck typing for xarray locks, which do not inherit from a common base class."""

    def acquire(self, blocking: bool = True) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
    def locked(self) -> bool: ...


# adios2 is not thread safe
ADIOS2_LOCK = SerializableLock()


class PscAdios2Array(BackendArray):
    """Lazy evaluation of a variable stored in PSC's adios2 field output.

    This also takes care of slicing out the specific component of the data stored as 4-d array.
    """

    def __init__(self, variable_name: str, datastore: PscAdios2Store, orig_varname: str, component: int) -> None:
        self.variable_name = variable_name
        self.datastore = datastore
        self._orig_varname = orig_varname
        self._component = component
        array = self.get_array()
        self.shape = array.shape[:-1]
        self.dtype = array.dtype

    def get_array(self, needs_lock: bool = True) -> Variable:
        return self.datastore.acquire(needs_lock).get_variable(self._orig_varname)

    def __getitem__(self, key: indexing.ExplicitIndexer) -> Any:
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.BASIC, self._getitem)

    def _getitem(self, args: tuple[SupportsInt | slice, ...]) -> NDArray[np.floating[Any]]:
        with self.datastore.lock:
            return self.get_array(needs_lock=False)[(*args, self._component)]  # FIXME add ... in between


class PscAdios2Store(AbstractDataStore):
    """DataStore to facilitate loading an Adios2 file."""

    def __init__(
        self,
        manager: CachingFileManager,
        species_names: Iterable[str],
        mode: str | None = None,
        lock: Lock = ADIOS2_LOCK,
        length: ArrayLike | None = None,
        corner: ArrayLike | None = None,
    ) -> None:
        self._manager = manager
        self._mode = mode
        self.lock: Lock = ensure_lock(lock)
        self.psc = RunInfo(self.ds, length=length, corner=corner)
        self._species_names = species_names

    @staticmethod
    def open(
        filename: str,
        species_names: Iterable[str],
        mode: str = "r",
        lock: Lock | None = None,
        length: ArrayLike | None = None,
        corner: ArrayLike | None = None,
    ) -> PscAdios2Store:
        if lock is None:
            if mode == "r":
                lock = ADIOS2_LOCK
            else:
                lock = combine_locks([ADIOS2_LOCK, get_write_lock(filename)])

        manager = CachingFileManager(File, filename, mode=mode)
        return PscAdios2Store(manager, species_names, mode=mode, lock=lock, length=length, corner=corner)

    def acquire(self, needs_lock: bool = True) -> File:
        with self._manager.acquire_context(needs_lock) as root:
            ds = root
        assert isinstance(ds, File)
        return ds

    @property
    def ds(self) -> File:
        return self.acquire()

    @override
    def get_variables(self) -> Frozen[str, xarray.DataArray]:
        field_to_component = get_field_to_component(self._species_names)

        variables: dict[str, tuple[str, int]] = {}
        for orig_varname in self.ds.variable_names:
            for field, component in field_to_component[orig_varname].items():
                variables[field] = (orig_varname, component)

        return FrozenDict((field, self.open_store_variable(field, *tup)) for field, tup in variables.items())

    def open_store_variable(self, field: str, orig_varname: str, component: int) -> xarray.DataArray:
        data = indexing.LazilyIndexedArray(PscAdios2Array(field, self, orig_varname, component))
        dims = ["x", "y", "z"]
        coords = {"x": self.psc.x, "y": self.psc.y, "z": self.psc.z}
        return xarray.DataArray(data, dims=dims, coords=coords)

    @override
    def get_attrs(self) -> Frozen[str, Any]:
        return FrozenDict((name, self.ds.get_attribute(name)) for name in self.ds.attribute_names)

    @override
    def get_dimensions(self) -> Never:
        raise NotImplementedError()


def psc_open_dataset(
    filename_or_obj: Any,
    species_names: Iterable[str],
    length: ArrayLike | None = None,
    corner: ArrayLike | None = None,
) -> xarray.Dataset:
    filename = _normalize_path(filename_or_obj)
    store = PscAdios2Store.open(filename, species_names, length=length, corner=corner)

    data_vars, attrs = store.load()
    ds = xarray.Dataset(data_vars=data_vars, attrs=attrs)
    ds.set_close(store.close)
    return ds


class PscAdios2BackendEntrypoint(BackendEntrypoint):
    """Entrypoint that lets xarray recognize and read (PSC's) Adios2 output."""

    open_dataset_parameters = ("filename_or_obj", "drop_variables")
    available = True

    @override
    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | io.BufferedIOBase | AbstractDataStore,
        *,
        drop_variables: str | Iterable[str] | None = None,
        length: ArrayLike | None = None,
        corner: ArrayLike | None = None,
        species_names: Iterable[str] | None = None,  # e.g. ['e', 'i']; FIXME should be readable from file
        **kwargs: Any,
    ) -> xarray.Dataset:
        if not isinstance(filename_or_obj, (str, os.PathLike)):
            raise NotImplementedError()

        if species_names is None:
            raise ValueError(f"Missing required keyword argument: '{species_names=}'")

        return psc_open_dataset(
            filename_or_obj,
            species_names,
            length=length,
            corner=corner,
        )

    @override
    def guess_can_open(self, filename_or_obj: str | os.PathLike[Any] | io.BufferedIOBase | AbstractDataStore) -> bool:
        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            return ext in {".bp"}
        return False

    @override
    def open_datatree(self, filename_or_obj: str | os.PathLike[Any] | io.BufferedIOBase | AbstractDataStore, **kwargs: Any) -> DataTree[Any]:
        raise NotImplementedError()


BACKEND_ENTRYPOINTS["pscadios2"] = ("pscpy", PscAdios2BackendEntrypoint)
