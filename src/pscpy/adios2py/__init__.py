from __future__ import annotations

import itertools
import logging
import os
from collections.abc import Generator, Mapping
from types import TracebackType
from typing import Any, Iterable, Iterator, SupportsInt

import adios2  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Variable:
    """Wrapper for an `adios2.Variable` object to facilitate loading and indexing into it."""

    def __init__(
        self,
        name: str,
        file: adios2.File,
        step: int | None = None,
    ) -> None:
        logger.debug("Variable.__init__(name=%s, file=%s)", name, file)
        if not file.io.inquire_variable(name):
            msg = f"Variable '{name}' not found in {file}"
            raise KeyError(msg)

        self._name = name
        self._file = file
        self._step = step
        self._reverse_dims = self._is_reverse_dims()

    def __bool__(self) -> bool:
        return bool(self._file)

    @property
    def var(self) -> adios2.Variable:
        self._assert_not_closed()

        var = self._file.io.inquire_variable(self._name)
        if not var:
            msg = f"Variable '{self._name}' not found"
            raise ValueError(msg)
        return var

    def _assert_not_closed(self) -> None:
        if not self:
            error_message = "adios2py: variable is closed"
            raise ValueError(error_message)

    def _maybe_reverse(self, dims: tuple[int, ...]) -> tuple[int, ...]:
        return dims[::-1] if self._reverse_dims else dims

    @property
    def shape(self) -> tuple[int, ...]:
        return self._maybe_reverse(tuple(self.var.shape()))

    @property
    def name(self) -> str:
        return self.var.name()  # type: ignore[no-any-return]

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(adios2.type_adios_to_numpy(self.var.type()))  # type: ignore[no-any-return]

    def _is_reverse_dims(self) -> bool:
        self._assert_not_closed()

        infos = self._file.engine.blocks_info(
            self.name, self._file.engine.current_step()
        )
        return infos[0]["IsReverseDims"] == "True"  # type: ignore[no-any-return]

    def __array__(self) -> np.ndarray[Any, Any]:
        return self[()]  # type: ignore [no-any-return]

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
            "arr_shape = %s, sel_start = %s, sel_count = %s step=%s",
            arr_shape,
            sel_start,
            sel_count,
            self._step,
        )

        var = self.var
        if self._step is not None:
            var.set_step_selection([self._step, 1])

        if len(sel_start) > 0:
            var.set_selection(
                (self._maybe_reverse(sel_start), self._maybe_reverse(sel_count))
            )

        order = "F" if self._reverse_dims else "C"
        arr = np.empty(arr_shape, dtype=self.dtype, order=order)
        self._file.engine.get(var, arr, adios2.bindings.Mode.Sync)
        return arr

    def __repr__(self) -> str:
        if not self:
            return f"{type(self)} (closed)"

        return f"{type(self)}(name={self.name}, shape={self.shape}, dtype={self.dtype}"


def _io_generator(ad: adios2.Adios) -> Generator[adios2.IO]:
    for io_count in itertools.count():
        io = ad.declare_io(f"io-adios2py-{io_count}")
        yield io


_ad = adios2.Adios()
_generate_io = _io_generator(_ad)


# TODO: It'd be nice to have some "io.close()" kind of functionality
def _close_io(io: adios2.IO) -> None:
    _ad.remove_io(io._name)


class File(Mapping[str, Any]):
    """Wrapper for an `adios2.IO` object to facilitate variable and attribute reading."""

    _own_io_engine: bool = False
    _io: adios2.IO | None = None
    _engine: adios2.Engine | None = None
    _step: int | None = None

    def __init__(
        self,
        filename_or_obj: str | os.PathLike[Any],
        mode: str = "r",
        parameters: dict[str, str] | None = None,
        engine_type: str | None = None,
    ) -> None:
        logger.debug("File.__init__(%s, %s)", filename_or_obj, mode)
        assert mode in ("r", "rra")
        self._mode = mode
        self._filename = filename_or_obj

        self._own_io_engine = True
        self._io = next(_generate_io)
        if parameters is not None:
            # CachingFileManager needs to pass something hashable, so convert back to dict
            self.io.set_parameters(dict(parameters))
        if engine_type is not None:
            self.io.set_engine(engine_type)
        if mode == "r":
            openmode = adios2.bindings.Mode.Read
        elif mode == "rra":
            openmode = adios2.bindings.Mode.ReadRandomAccess
        else:
            msg = f"adios2py: invalid mode {mode}"
            raise ValueError(msg)
        self._engine = self.io.open(str(filename_or_obj), openmode)

    def __bool__(self) -> bool:
        return self._engine is not None and self._io is not None

    @property
    def attrs(self) -> AttrsProxy:
        return AttrsProxy(self)

    def __repr__(self) -> str:
        return f"{type(self)}(filename='{self._filename}')"

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
        if self:  # is open?
            self.close()

    def close(self) -> None:
        assert self  # is_open

        if self._own_io_engine:  # if we created the io/engine ourselves
            self.engine.close()
            _close_io(self.io)
        self._engine = None
        self._io = None

    @property
    def engine(self) -> adios2.Engine:
        assert self._engine
        return self._engine

    @property
    def io(self) -> adios2.IO:
        assert self._io
        return self._io

    def current_step(self) -> int:
        return self.engine.current_step()  # type: ignore[no-any-return]

    def begin_step(self) -> adios2.StepStatus:
        return self.engine.begin_step()

    def end_step(self) -> None:
        return self.engine.end_step()  # type: ignore[no-any-return]

    @property
    def steps(self) -> StepsProxy:
        return StepsProxy(self)

    def set_step(self, step: int | None) -> None:
        self._step = step

    def _keys(self) -> set[str]:
        return self.io.available_variables().keys()  # type: ignore[no-any-return]

    def __getitem__(self, name: str) -> Variable:
        if self._mode == "r":
            return Variable(name, self)

        assert self._mode == "rra"
        return Variable(name, self, step=self._step)

    def __len__(self) -> int:
        return len(self._keys())

    def __iter__(self) -> Iterator[str]:
        yield from self._keys()


class AttrsProxy(Mapping[str, Any]):
    def __init__(self, file: File) -> None:
        self._file = file

    def __getitem__(self, name: str) -> Any:
        attr = self._file.io.inquire_attribute(name)
        if not attr:
            raise KeyError()

        if attr.type() == "string":
            return attr.data_string()

        return attr.data()

    def __len__(self) -> int:
        return len(self._keys())

    def __iter__(self) -> Iterator[str]:
        yield from self._keys()

    def _keys(self) -> set[str]:
        return set(self._file.io.available_attributes().keys())


class StepsProxy(Iterable[File]):
    _file: File | None

    def __init__(self, file: File) -> None:
        self._file = file

    @property
    def file(self) -> File:
        assert self._file
        return self._file

    def __iter__(self) -> Iterator[File]:
        # FIXME, should prevent giving out more than one iterator at a time in streaming mode
        file = self.file
        if file._mode == "r":
            while True:
                status = file.begin_step()
                if status == adios2.bindings.StepStatus.EndOfStream:
                    break
                assert status == adios2.bindings.StepStatus.OK

                yield file
                file.end_step()
        elif file._mode == "rra":
            for step in range(len(self)):
                file.set_step(step)
                yield file

    def __len__(self) -> int:
        return self.file.engine.steps()  # type: ignore[no-any-return]
