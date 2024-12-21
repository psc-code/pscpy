from __future__ import annotations

import logging
import os
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

_ad = Adios()


class Variable:
    """Wrapper for an `adios2.Variable` object to facilitate loading and indexing into it."""

    def __init__(self, var: adios2.Variable, engine: adios2.Engine) -> None:
        self._var = var
        self._engine = engine
        self.name = self._name()
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

    def _set_selection(
        self, start: NDArray[np.integer[Any]], count: NDArray[np.integer[Any]]
    ) -> None:
        self._assert_not_closed()

        self._var.set_selection((start[::-1], count[::-1]))

    def _shape(self) -> tuple[int, ...]:
        self._assert_not_closed()

        return tuple(self._var.shape())[::-1]

    def _name(self) -> str:
        self._assert_not_closed()

        return self._var.name()  # type: ignore[no-any-return]

    def _dtype(self) -> np.dtype[Any]:
        self._assert_not_closed()

        return np.dtype(adios2.type_adios_to_numpy(self._var.type()))  # type: ignore[no-any-return]

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

        self._set_selection(sel_start, sel_count)

        arr = np.empty(
            arr_shape, dtype=self.dtype, order="F"
        )  # FIXME is column-major correct?
        self._engine.get(self._var, arr, adios2.bindings.Mode.Sync)
        return arr

    def __repr__(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__name__}(name={self.name}, shape={self.shape}, dtype={self.dtype}"


class FileState:
    """Collects the state of a `File` to reflect the fact that they are coupled."""

    def __init__(self, filename: str | os.PathLike[Any]) -> None:
        self.io_name = f"io-{filename}"
        self.io = _ad.declare_io(self.io_name)
        self.engine = self.io.open(str(filename), adios2.bindings.Mode.Read)

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

        self._state.engine.close()
        _ad.remove_io(self._state.io_name)
        self._state = None

    def get_variable(self, variable_name: str) -> Variable:
        assert FileState.is_open(self._state)

        var = Variable(
            self._state.io.inquire_variable(variable_name), self._state.engine
        )
        self._open_vars[variable_name] = var
        return var

    def get_attribute(self, attribute_name: str) -> Any:
        assert FileState.is_open(self._state)

        return self._state.io.inquire_attribute(attribute_name).data()
