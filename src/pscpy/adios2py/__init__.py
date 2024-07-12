from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any

import adios2  # type: ignore[import-untyped]
import adios2.stream  # type: ignore[import-untyped]
import numpy as np
from adios2.adios import Adios  # type: ignore[import-untyped]
from numpy.typing import NDArray
from typing_extensions import TypeGuard

_ad = Adios()


class Variable:
    """Wrapper for an `adios2.Variable` object to facilitate loading and indexing into it."""

    def __init__(self, var: adios2.Variable, engine: adios2.Engine) -> None:
        self._var = var
        self._engine = engine
        self.name = self._name()
        self.shape = self._shape()
        self.dtype = self._dtype()
        logging.debug("variable __init__ var %s engine %s", var, engine)

    def close(self) -> None:
        logging.debug("adios2py.variable close")
        self._var = None
        self._engine = None

    def _assert_not_closed(self) -> None:
        if not self._var:
            raise ValueError("adios2py: variable is closed")

    def _set_selection(self, start: NDArray[np.integer[Any]], count: NDArray[np.integer[Any]]) -> None:
        self._assert_not_closed()

        self._var.set_selection((start[::-1], count[::-1]))

    def _shape(self) -> list[int]:
        self._assert_not_closed()

        shape_reversed: list[int] = self._var.shape()
        return shape_reversed[::-1]

    def _name(self) -> str:
        self._assert_not_closed()

        return self._var.name()  # type: ignore[no-any-return]

    def _dtype(self) -> np.dtype:
        self._assert_not_closed()

        return adios2.type_adios_to_numpy(self._var.type())()

    def __getitem__(self, args: Any) -> NDArray:
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

            raise RuntimeError(f"invalid args to __getitem__: {args}")

        for d in range(len(args), len(shape)):
            sel_start[d] = 0
            sel_count[d] = shape[d]
            arr_shape.append(sel_count[d])

        self._set_selection(sel_start, sel_count)

        arr = np.empty(arr_shape, dtype=self.dtype, order="F")  # FIXME is column-major correct?
        self._engine.get(self._var, arr, adios2.bindings.Mode.Sync)
        return arr

    def __repr__(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__name__}(name={self.name}, shape={self.shape}, dtype={self.dtype}"


class FileState:
    """Collects the state of a `File` to reflect the fact that they are coupled."""

    def __init__(self, filename: str) -> None:
        self.io_name = f"io-{filename}"
        self.io = _ad.declare_io(self.io_name)
        self.engine = self.io.open(filename, adios2.bindings.Mode.Read)

    @staticmethod
    def is_open(maybe_state: FileState | None) -> TypeGuard[FileState]:
        return maybe_state is not None


class File:
    """Wrapper for an `adios2.IO` object to facilitate variable and attribute reading."""

    _state: FileState | None

    def __init__(self, filename: str, mode: str = "r") -> None:
        logging.debug("adios2py: __init__ %s", filename)
        assert mode == "r"
        self._state = FileState(filename)
        self._open_vars: dict[str, Variable] = {}

        self.variable_names: Collection[str] = self._state.io.available_variables().keys()
        self.attribute_names: Collection[str] = self._state.io.available_attributes().keys()

    def __enter__(self) -> File:
        logging.debug("adios2py: __enter__")
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        logging.debug("adios2py: __exit__")
        self.close()

    def __del__(self) -> None:
        logging.debug("adios2py: __del__")
        if FileState.is_open(self._state):
            self.close()

    def close(self) -> None:
        assert FileState.is_open(self._state)

        logging.debug("adios2py: close")
        logging.debug("open vars %s", self._open_vars)
        for var in self._open_vars.values():
            var.close()

        self._state.engine.close()
        _ad.remove_io(self._state.io_name)
        self._state = None

    def get_variable(self, variable_name: str) -> Variable:
        assert FileState.is_open(self._state)

        var = Variable(self._state.io.inquire_variable(variable_name), self._state.engine)
        self._open_vars[variable_name] = var
        return var

    def get_attribute(self, attribute_name: str) -> Any:
        assert FileState.is_open(self._state)

        adios2_attr = self._state.io.inquire_attribute(attribute_name)
        data = adios2_attr.data()
        # FIXME use SingleValue when writing data to avoid doing this (?)
        if len(data) == 1:
            return data[0]
        return data
