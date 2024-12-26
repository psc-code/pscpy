from __future__ import annotations

import itertools
import logging
import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
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


def _mode_to_adios2_openmode(mode: str) -> adios2.bindings.Mode:
    if mode == "r":
        return adios2.bindings.Mode.Read
    if mode == "rra":
        return adios2.bindings.Mode.ReadRandomAccess

    msg = f"adios2py: invalid mode {mode}"
    raise ValueError(msg)


class FileState:
    _io: adios2.IO | None = None
    _engine: adios2.Engine | None = None
    _step_status: adios2.bindings.StepStatus | None = None
    _step: int = -1

    def __init__(
        self,
        filename_or_obj: str | os.PathLike[Any],
        mode: str,
        parameters: dict[str, str] | None = None,
        engine_type: str | None = None,
    ) -> None:
        self._filename = filename_or_obj
        self._mode = mode
        self._io = next(_generate_io)
        if parameters is not None:
            # CachingFileManager needs to pass something hashable, so convert back to dict
            self._io.set_parameters(dict(parameters))
        if engine_type is not None:
            self._io.set_engine(engine_type)

        openmode = _mode_to_adios2_openmode(mode)
        # FIXME use os.fspath
        self._engine = self._io.open(str(filename_or_obj), openmode)

    def __bool__(self) -> bool:
        return self._io is not None and self._engine is not None

    def close(self) -> None:
        self.engine.close()
        _close_io(self.io)
        self._engine = None
        self._io = None

    @property
    def filename(self) -> str | os.PathLike[Any]:
        return self._filename

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def io(self) -> adios2.IO:
        assert self
        return self._io

    @property
    def engine(self) -> adios2.Engine:
        assert self
        return self._engine

    def begin_step(self) -> adios2.bindings.StepStatus:
        assert self._step_status is None
        self._step_status = self.engine.begin_step()
        if self._step_status == adios2.bindings.StepStatus.EndOfStream:
            raise EOFError
        if self._step_status != adios2.bindings.StepStatus.OK:
            msg = f"adios2 StepStatus = {self._step_status}"
            raise RuntimeError(msg)
        self._step += 1
        return self._step_status

    def end_step(self) -> None:
        assert self._step_status == adios2.bindings.StepStatus.OK
        self.engine.end_step()
        self._step_status = None

    def current_step(self) -> int | None:
        if self._step_status is None:
            return None
        assert self._step == self.engine.current_step()
        return self._step

    def __repr__(self) -> str:
        if not self:
            return "adios2py.Filestate (closed)"

        return f"adios2py.FileState(filename={self.filename}, mode={self.mode})"


class Group(Mapping[str, Any]):
    _state: FileState

    def __init__(self, state: FileState) -> None:
        self._state = state

    def __bool__(self) -> bool:
        return bool(self._state)

    @property
    def engine(self) -> adios2.Engine:
        return self._state.engine

    @property
    def io(self) -> adios2.IO:
        return self._state.io

    def _keys(self) -> set[str]:
        return self.io.available_variables().keys()  # type: ignore[no-any-return]

    def __getitem__(self, name: str) -> Variable:
        if self._state.mode == "r":
            return Variable(name, self)

        assert self._state.mode == "rra"
        return Variable(name, self)

    def __len__(self) -> int:
        return len(self._keys())

    def __iter__(self) -> Iterator[str]:
        yield from self._keys()


class Step(Group):
    def __init__(self, state: FileState, step: int) -> None:
        super().__init__(state)
        self._step = step

    def __getitem__(self, name: str) -> Variable:
        if self._state.mode == "r":
            return Variable(name, self)

        assert self._state.mode == "rra"
        return Variable(name, self, step=self._step)

    def step(self) -> int:
        return self._step


class File(Group):
    """Wrapper for an `adios2.IO` object to facilitate variable and attribute reading."""

    _step: int | None = None

    def __init__(
        self,
        filename_or_obj: str | os.PathLike[Any],
        mode: str = "r",
        parameters: dict[str, str] | None = None,
        engine_type: str | None = None,
    ) -> None:
        logger.debug("File.__init__(%s, %s)", filename_or_obj, mode)
        super().__init__(FileState(filename_or_obj, mode, parameters, engine_type))

    @property
    def attrs(self) -> AttrsProxy:
        return AttrsProxy(self)

    def __repr__(self) -> str:
        return f"adios2py.File(state={self._state})"

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
        self._state.close()

    def current_step(self) -> int:
        return self.engine.current_step()  # type: ignore[no-any-return]

    @property
    def steps(self) -> StepsProxy:
        return StepsProxy(self)


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


class StepsProxy(Iterable[Step]):
    _file: File | None

    def __init__(self, file: File) -> None:
        self._file = file

    @property
    def file(self) -> File:
        assert self._file
        return self._file

    def __iter__(self) -> Iterator[Step]:
        # FIXME, should prevent giving out more than one iterator at a time in streaming mode
        file = self.file
        if file._state.mode == "r":
            try:
                while True:
                    with next(self) as step:
                        yield step
            except EOFError:
                pass
        elif file._state.mode == "rra":
            for n in range(len(self)):
                yield self[n]

    def __getitem__(self, step: int) -> Step:
        if self.file._state.mode != "rra" and step != self.file._state.current_step():
            msg = f"Failed to get steps({step} in streaming mode."
            raise TypeError(msg)

        return Step(self.file._state, step)

    def __len__(self) -> int:
        return self.file.engine.steps()  # type: ignore[no-any-return]

    @contextmanager
    def __next__(self) -> Generator[Step]:
        status = None
        try:
            status = self.file._state.begin_step()
            step = self.file._state.current_step()
            assert step is not None
            yield self[step]
        finally:
            if status == adios2.bindings.StepStatus.OK:
                self.file._state.end_step()
