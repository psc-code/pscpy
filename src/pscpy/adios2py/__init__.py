from __future__ import annotations

import itertools
import logging
import os
from collections.abc import Generator, Mapping, Sequence
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
        state: FileState,
        step: int | None = None,
    ) -> None:
        logger.debug("Variable.__init__(name=%s, state=%s)", name, state)
        self._name = name
        self._state = state
        if not state.io.inquire_variable(name):
            msg = f"Variable '{name}' not found in {state}"
            raise KeyError(msg)

        self._step = step
        self._reverse_dims = self._is_reverse_dims()

    def __bool__(self) -> bool:
        if not self._state:  # file closed?
            return False

        if self._state.mode == "rra":
            return True

        return bool(
            (self._step is None and self._state.current_step() is None)
            or self._step == self._state.current_step()
        )

    @property
    def var(self) -> adios2.Variable:
        var = self._state.io.inquire_variable(self._name)
        if not var:
            msg = f"Variable '{self._name}' not found"
            raise ValueError(msg)
        return var

    def _maybe_reverse(self, dims: Sequence[int]) -> Sequence[int]:
        return dims[::-1] if self._reverse_dims else dims

    def _steps(self) -> int | None:
        if self._state.mode != "rra":
            return None

        if self._step is not None:
            return None

        steps: int = self.var.steps()
        assert steps == self._state.engine.steps()

        return steps

    @property
    def shape(self) -> tuple[int, ...]:
        shape = tuple(self._maybe_reverse(self.var.shape()))
        if (steps := self._steps()) is not None:
            shape = (steps, *shape)
        return shape

    @property
    def name(self) -> str:
        return self.var.name()  # type: ignore[no-any-return]

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(adios2.type_adios_to_numpy(self.var.type()))  # type: ignore[no-any-return]

    def _is_reverse_dims(self) -> bool:
        step = self._state.current_step() or 0
        infos = self._state.engine.blocks_info(self.name, step)
        return infos[0]["IsReverseDims"] == "True"  # type: ignore[no-any-return]

    def __array__(self) -> np.ndarray[Any, Any]:
        return self[()]  # type: ignore [no-any-return]

    def __getitem__(
        self,
        args: SupportsInt | slice | tuple[SupportsInt | slice, ...],
    ) -> NDArray[Any]:
        if not isinstance(args, tuple):
            args = (args,)

        if self._state.mode == "r":
            assert self._step is not None
            if self._step != self._state.current_step():
                msg = (
                    f"cannot access step {self._step} in streaming mode, "
                    f"current_step = {self._state.current_step()}"
                )
                raise KeyError(msg)

            return self._getitem((0, *args))

        # rra mode
        steps = self._steps()
        if steps is None:
            step = self._step if self._step is not None else 0
            return self._getitem((step, *args))

        return self._getitem(args)

    def _getitem(
        self,
        args: tuple[SupportsInt | slice, ...],
    ) -> NDArray[Any]:
        var_shape = (1, *self.shape) if self._steps() is None else self.shape
        sel: list[tuple[int, int]] = []  # list of (start, count)
        arr_shape: list[int] = []

        for arg, length in itertools.zip_longest(args, var_shape):
            if arg is None:
                # if too fewer slices/indices were passed, pad with full slices
                sel.append((0, length))
                arr_shape.append(length)
            elif isinstance(arg, slice):
                assert isinstance(arg, slice)
                start, stop, step = arg.indices(length)
                assert start < stop
                assert step == 1
                sel.append((start, stop - start))
                arr_shape.append(stop - start)
            else:
                idx = int(arg)
                if idx < 0:
                    idx += length
                sel.append((idx, 1))

        logger.debug("arr_shape = %s, sel = %s", arr_shape, sel)

        sel_start, sel_count = zip(*sel)

        if self._state.mode == "rra":
            self.var.set_step_selection((sel_start[0], sel_count[0]))
        else:  # mode == "r"
            assert (sel_start[0], sel_count[0]) == (0, 1)

        if len(sel) > 1:
            self.var.set_selection(
                (self._maybe_reverse(sel_start[1:]), self._maybe_reverse(sel_count[1:]))
            )

        order = "F" if self._reverse_dims else "C"
        arr = np.empty(arr_shape, dtype=self.dtype, order=order)
        assert arr.size == np.prod(sel_count)
        self._state.engine.get(self.var, arr, adios2.bindings.Mode.Sync)
        return arr

    def __repr__(self) -> str:
        if not self:
            return f"adios2py.Variable(name={self._name}) (closed)"

        return f"adios2py.Variable(name={self._name}, shape={self.shape}, dtype={self.dtype}"


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

    def __del__(self) -> None:
        if self:
            self.close()

    def _assert_is_not_closed(self) -> None:
        if not self:
            msg = f"{self} is closed."
            raise ValueError(msg)

    @property
    def filename(self) -> str | os.PathLike[Any]:
        return self._filename

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def io(self) -> adios2.IO:
        self._assert_is_not_closed()
        return self._io

    @property
    def engine(self) -> adios2.Engine:
        self._assert_is_not_closed()
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

    def _keys(self) -> set[str]:
        return self._state.io.available_variables().keys()  # type: ignore[no-any-return]

    def __getitem__(self, name: str) -> Variable:
        if self._state.mode == "r":
            msg = "in streaming mode, need to read by step"
            raise RuntimeError(msg)
        return Variable(name, self._state)

    def __len__(self) -> int:
        return len(self._keys())

    def __iter__(self) -> Iterator[str]:
        yield from self._keys()

    @property
    def attrs(self) -> AttrsProxy:
        return AttrsProxy(self._state)


class Step(Group):
    def __init__(self, state: FileState, step: int) -> None:
        super().__init__(state)
        self._step = step

    def __getitem__(self, name: str) -> Variable:
        return Variable(name, self._state, step=self._step)

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

    def close(self) -> None:
        self._state.close()

    @property
    def steps(self) -> StepsProxy:
        return StepsProxy(self._state)


class AttrsProxy(Mapping[str, Any]):
    def __init__(self, state: FileState) -> None:
        self._state = state

    def __getitem__(self, name: str) -> Any:
        attr = self._state.io.inquire_attribute(name)
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
        return set(self._state.io.available_attributes().keys())


class StepsProxy(Iterable[Step]):
    def __init__(self, state: FileState) -> None:
        self._state = state

    def __iter__(self) -> Iterator[Step]:
        if self._state.mode == "r":
            try:
                while True:
                    with next(self) as step:
                        yield step
            except EOFError:
                pass
        elif self._state.mode == "rra":
            for n in range(len(self)):
                yield self[n]

    def __getitem__(self, step: int) -> Step:
        if self._state.mode != "rra" and step != self._state.current_step():
            msg = f"Failed to get steps({step} in streaming mode."
            raise TypeError(msg)

        return Step(self._state, step)

    def __len__(self) -> int:
        return self._state.engine.steps()  # type: ignore[no-any-return]

    @contextmanager
    def __next__(self) -> Generator[Step]:
        status = None
        try:
            status = self._state.begin_step()
            step = self._state.current_step()
            assert step is not None
            yield self[step]
        finally:
            if status == adios2.bindings.StepStatus.OK:
                self._state.end_step()
