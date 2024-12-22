from __future__ import annotations

import itertools
import logging
import os
import pathlib
from collections.abc import Collection
from types import TracebackType
from typing import Any, SupportsInt

import adios2  # type: ignore[import-untyped]
import adios2.stream  # type: ignore[import-untyped]
import numpy as np
from adios2.adios import Adios  # type: ignore[import-untyped]
from numpy.typing import NDArray
from typing_extensions import TypeGuard

logger = logging.getLogger(__name__)


class Variable:
    """Wrapper for an `adios2.Variable` object to facilitate loading and indexing into it."""

    def __init__(
        self,
        var: adios2.Variable,
        engine: adios2.Engine,
        reverse_dims: bool | None = None,
    ) -> None:
        self._var = var
        self._engine = engine
        self.name = self._name()
        self.is_reverse_dims = self._is_reverse_dims()
        self._reverse_dims = self.is_reverse_dims
        if reverse_dims is not None:
            self._reverse_dims = reverse_dims
        self.shape = self._shape()
        self.dtype = self._dtype()
        logger.debug("variable __init__ var %s engine %s", var, engine)

    def close(self) -> None:
        logger.debug("adios2py.variable close")
        self._var = None
        self._engine = None

    def _assert_not_closed(self) -> None:
        if not self._var:
            error_message = "adios2py: variable is closed"
            raise ValueError(error_message)

    def _maybe_reverse(self, dims: tuple[int, ...]) -> tuple[int, ...]:
        return dims[::-1] if self._reverse_dims else dims

    def _set_selection(
        self, start: NDArray[np.integer[Any]], count: NDArray[np.integer[Any]]
    ) -> None:
        self._assert_not_closed()

        self._var.set_selection(
            (self._maybe_reverse(start), self._maybe_reverse(count))
        )

    def _shape(self) -> tuple[int, ...]:
        self._assert_not_closed()

        return self._maybe_reverse(tuple(self._var.shape()))

    def _name(self) -> str:
        self._assert_not_closed()

        return self._var.name()  # type: ignore[no-any-return]

    def _dtype(self) -> np.dtype[Any]:
        self._assert_not_closed()

        return np.dtype(adios2.type_adios_to_numpy(self._var.type()))  # type: ignore[no-any-return]

    def _is_reverse_dims(self) -> bool:
        infos = self._engine.blocks_info(self.name, self._engine.current_step())
        return infos[0]["IsReverseDims"] == "True"  # type: ignore[no-any-return]

    def __getitem__(
        self, args: SupportsInt | slice | tuple[SupportsInt | slice, ...]
    ) -> NDArray[Any]:
        self._assert_not_closed()

        if not isinstance(args, tuple):
            args = (args,)

        shape = self.shape
        sel_start = np.zeros_like(shape)
        sel_count = np.zeros_like(shape)
        arr_shape = []

        for d, arg in enumerate(args):
            if isinstance(arg, slice):
                start, stop, step = arg.indices(shape[d])
                assert stop > start
                assert step == 1
                sel_start[d] = start
                sel_count[d] = stop - start
                arr_shape.append(sel_count[d])
                continue

            try:
                idx = int(arg)
            except ValueError:
                pass
            else:
                if idx < 0:
                    idx += shape[d]
                sel_start[d] = idx
                sel_count[d] = 1
                continue

            error_message = f"invalid args to __getitem__: {args}"
            raise RuntimeError(error_message)

        for d in range(len(args), len(shape)):
            sel_start[d] = 0
            sel_count[d] = shape[d]
            arr_shape.append(sel_count[d])

        logger.debug(
            "arr_shape = %s, sel_start = %s, sel_count = %s",
            arr_shape,
            sel_start,
            sel_count,
        )
        self._set_selection(sel_start, sel_count)

        order = "F" if self._reverse_dims else "C"
        arr = np.empty(arr_shape, dtype=self.dtype, order=order)
        self._engine.get(self._var, arr, adios2.bindings.Mode.Sync)
        return arr

    def __repr__(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__name__}(name={self.name}, shape={self.shape}, dtype={self.dtype}"


class FileState:
    """Collects the state of a `File` to reflect the fact that they are coupled."""

    _ad = Adios()
    _io_count = itertools.count()

    def __init__(self, filename: str | os.PathLike[Any]) -> None:
        self.io_name = f"io-adios2py-{next(self._io_count)}"
        logger.debug("io_name = %s", self.io_name)
        self.io = self._ad.declare_io(self.io_name)
        self.engine = self.io.open(str(filename), adios2.bindings.Mode.Read)

    def close(self) -> None:
        self.engine.close()
        self._ad.remove_io(self.io_name)

    @staticmethod
    def is_open(maybe_state: FileState | None) -> TypeGuard[FileState]:
        return maybe_state is not None


class File:
    """Wrapper for an `adios2.IO` object to facilitate variable and attribute reading."""

    _state: FileState | None

    def __init__(self, filename: str | os.PathLike[Any], mode: str = "r") -> None:
        logger.debug("File.__init__(%s, %s)", filename, mode)
        assert mode == "r"
        self._state = FileState(filename)
        self._filename = filename
        self._open_vars: dict[str, Variable] = {}

        self.variable_names: Collection[str] = (
            self._state.io.available_variables().keys()
        )
        self.attribute_names: Collection[str] = (
            self._state.io.available_attributes().keys()
        )

    def __enter__(self) -> File:
        logger.debug("File.__enter__()")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        logger.debug("File.__exit__()")
        self.close()

    def __del__(self) -> None:
        logger.debug("File.__del__()")
        if FileState.is_open(self._state):
            self.close()

    def close(self) -> None:
        assert FileState.is_open(self._state)

        logger.debug("File.close(): open vars %s", self._open_vars)
        for var in self._open_vars.values():
            var.close()

        self._state.close()
        self._state = None

    def get_variable(self, variable_name: str) -> Variable:
        assert FileState.is_open(self._state)

        if variable_name not in self._open_vars:
            reverse_dims = None
            filename = pathlib.Path(self._filename).name
            if filename.startswith(("pfd", "tfd")):
                reverse_dims = True

            var = Variable(
                self._state.io.inquire_variable(variable_name),
                self._state.engine,
                reverse_dims,
            )
            self._open_vars[variable_name] = var
            return var

        return self._open_vars[variable_name]

    def get_attribute(self, attribute_name: str) -> Any:
        assert FileState.is_open(self._state)

        attr = self._state.io.inquire_attribute(attribute_name)
        if attr.type() == "string":
            return attr.data_string()

        return attr.data()
