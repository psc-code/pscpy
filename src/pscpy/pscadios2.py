from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Iterable, Protocol, SupportsInt

import numpy as np
import xarray
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Never, override
from xarray.backends import CachingFileManager
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
from xarray.core import indexing
from xarray.core.datatree import DataTree
from xarray.core.types import ReadBuffer
from xarray.core.utils import Frozen, FrozenDict

from . import psc
from .adios2py import File, Variable

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


class PscAdios2Array(BackendArray):
    """Lazy evaluation of a variable stored in PSC's adios2 field output.

    This also takes care of slicing out the specific component of the data stored as 4-d array.
    """

    def __init__(
        self,
        variable_name: str,
        datastore: PscAdios2Store,
        orig_varname: str,
        component: int | None,
    ) -> None:
        self.variable_name = variable_name
        self.datastore = datastore
        self._orig_varname = orig_varname
        self._component = component
        array = self.get_array()
        self.shape = array.shape[:-1] if self._component is not None else array.shape
        self.dtype = array.dtype

    def get_array(self, needs_lock: bool = True) -> Variable:
        return self.datastore.acquire(needs_lock).get_variable(self._orig_varname)

    def __getitem__(self, key: indexing.ExplicitIndexer) -> Any:
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(
        self, args: tuple[SupportsInt | slice, ...]
    ) -> NDArray[np.floating[Any]]:
        with self.datastore.lock:
            if self._component is not None:
                logger.debug("_get_item component [%s, comp]", args)
                return self.get_array(needs_lock=False)[
                    (*args, self._component)
                ]  # FIXME add ... in between

            logger.debug("_get_item component [%s]", args)
            return self.get_array(needs_lock=False)[(*args,)]


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
        self.lock: Lock = ensure_lock(lock)  # type: ignore[no-untyped-call]
        self.psc = psc.RunInfo(self.ds, length=length, corner=corner)
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
                lock = combine_locks([ADIOS2_LOCK, get_write_lock(filename)])  # type: ignore[no-untyped-call]

        manager = CachingFileManager(File, filename, mode=mode)
        return PscAdios2Store(
            manager, species_names, mode=mode, lock=lock, length=length, corner=corner
        )

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
        return FrozenDict(
            (k, self.open_store_variable(k)) for k in self.ds.variable_names
        )

    def open_store_variable(self, var_name: str) -> xarray.DataArray:
        data = indexing.LazilyIndexedArray(
            PscAdios2Array(var_name, self, var_name, None)
        )
        dims: tuple[str, ...] = ("x", "y", "z", f"comp_{data.shape[3]}")
        coords = {
            "x": ("x", self.psc.x),
            "y": ("y", self.psc.y),
            "z": ("z", self.psc.z),
        }
        return xarray.DataArray(data, dims=dims, coords=coords)

    @override
    def get_attrs(self) -> Frozen[str, Any]:
        return FrozenDict(
            (name, self.ds.get_attribute(name)) for name in self.ds.attribute_names
        )

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

    data_vars, attrs = store.load()  # type: ignore[no-untyped-call]
    ds = xarray.Dataset(data_vars=data_vars, attrs=attrs)
    ds.set_close(store.close)

    field_to_component = psc.get_field_to_component(species_names)

    data_vars = {}
    for var_name in ds:
        if var_name not in field_to_component:
            continue
        for field, component in field_to_component[var_name].items():  # type: ignore[index]
            data_vars[field] = ds[var_name][..., component]

    return ds.assign(data_vars)


class PscAdios2BackendEntrypoint(BackendEntrypoint):
    """Entrypoint that lets xarray recognize and read (PSC's) Adios2 output."""

    open_dataset_parameters = ("filename_or_obj", "drop_variables")
    available = True

    @override
    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
        *,
        drop_variables: str | Iterable[str] | None = None,
        length: ArrayLike | None = None,
        corner: ArrayLike | None = None,
        species_names: Iterable[str]
        | None = None,  # e.g. ['e', 'i']; FIXME should be readable from file
    ) -> xarray.Dataset:
        if not isinstance(filename_or_obj, (str, os.PathLike)):
            raise NotImplementedError()

        if species_names is None:
            error_message = f"Missing required keyword argument: '{species_names=}'"
            raise ValueError(error_message)

        return psc_open_dataset(
            filename_or_obj,
            species_names,
            length=length,
            corner=corner,
        )

    @override
    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
    ) -> bool:
        if isinstance(filename_or_obj, (str, os.PathLike)):
            ext = pathlib.Path(filename_or_obj).suffix
            return ext in {".bp"}
        return False

    @override
    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
        **kwargs: Any,
    ) -> DataTree:
        raise NotImplementedError()
