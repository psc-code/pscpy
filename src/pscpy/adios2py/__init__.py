from __future__ import annotations

import logging

import adios2
import adios2.stream
import numpy as np

_ad = adios2.Adios()


class Variable:
    def __init__(self, var: adios2.Variable, engine: adios2.Engine):
        self._var = var
        self._engine = engine
        self.name = self._name()
        self.shape = self._shape()
        self.dtype = self._dtype()
        logging.debug("variable __init__ var %s engine %s", var, engine)

    def close(self):
        logging.debug("adios2py.variable close")
        self._var = None
        self._engine = None

    def _assert_not_closed(self):
        if not self._var:
            raise ValueError("adios2py: variable is closed")

    def _set_selection(self, start, count):
        self._assert_not_closed()

        self._var.set_selection((start[::-1], count[::-1]))

    def _shape(self):
        self._assert_not_closed()

        return self._var.shape()[::-1]

    def _name(self) -> str:
        self._assert_not_closed()

        return self._var.name()

    def _dtype(self):
        self._assert_not_closed()

        return adios2.type_adios_to_numpy(self._var.type())

    def __getitem__(self, args):
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

        # print("A sel_start", sel_start, "sel_count",
        #       sel_count, "arr_shape", arr_shape)

        for d in range(len(args), len(shape)):
            sel_start[d] = 0
            sel_count[d] = shape[d]
            arr_shape.append(sel_count[d])

        # print("B sel_start", sel_start, "sel_count",
        #       sel_count, "arr_shape", arr_shape)

        self._set_selection(sel_start, sel_count)

        arr = np.empty(arr_shape, dtype=self.dtype, order="F")
        # print("reading ", self.name, args)
        self._engine.get(self._var, arr, adios2.bindings.Mode.Sync)
        return arr

    def __repr__(self):
        return f"adios2py.variable(name={self.name}, shape={self.shape}, dtype={self.dtype}"


class File:
    def __init__(self, filename, mode="r"):
        logging.debug("adios2py: __init__ %s", filename)
        assert mode == "r"
        self._io_name = f"io-{filename}"
        self._io = _ad.declare_io(self._io_name)
        self._engine = self._io.open(filename, adios2.bindings.Mode.Read)
        self._open_vars: dict[str, Variable] = {}

        self.variables = self._io.available_variables().keys()
        self.attributes = self._io.available_attributes().keys()

    def __enter__(self):
        logging.debug("adios2py: __enter__")
        return self

    def __exit__(self, type, value, traceback):
        logging.debug("adios2py: __exit__")
        self.close()

    def __del__(self):
        logging.debug("adios2py: __del__")
        if self._engine:
            self.close()

    def close(self):
        logging.debug("adios2py: close")
        logging.debug("open vars %s", self._open_vars)
        for varname, var in self._open_vars.items():
            var.close()

        self._engine.close()
        self._engine = None

        _ad.remove_io(self._io_name)
        self._io = None
        self._io_name = None

    def __getitem__(self, varname):
        var = Variable(self._io.inquire_variable(varname), self._engine)
        self._open_vars[varname] = var
        return var
